import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.cluster import DBSCAN
import seaborn as sns
import pandas as pd
from torch.cuda.amp import GradScaler, autocast

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IPv6WordDataset(Dataset):
    """实现原论文的Sample Generation（Section 4.2）"""
    def __init__(self, file_path: str, vocab_path: str, window_size: int = 5, max_samples: int = None):
        self.samples = []
        self.word2id = {}
        
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        
        # 加载地址词序列
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()
            if max_samples:
                lines = lines[:max_samples]
                
            # 使用原位更新的进度条
            for line in tqdm(lines, desc="生成训练样本", ncols=100, position=0, leave=True):
                words = line.strip().split()
                for i in range(len(words)):
                    # 输入词+上下文词（原论文图2）
                    input_word = words[i]
                    context = words[max(0,i-window_size):i] + words[i+1:min(len(words),i+window_size+1)]
                    if context:  # 确保有上下文词
                        self.samples.append((input_word, context))
        
        logging.info(f"生成了 {len(self.samples)} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_word, context = self.samples[idx]
        input_id = self.word2id.get(input_word, self.word2id["[UNK]"])
        context_ids = [self.word2id.get(word, self.word2id["[UNK]"]) for word in context]
        
        return {
            "input_id": input_id,
            "context_ids": context_ids
        }

class Word2VecModel(nn.Module):
    """实现类似Word2Vec的CBOW模型"""
    def __init__(self, vocab_size, embedding_dim=100):
        super(Word2VecModel, self).__init__()
        # 输入词嵌入层 (类似Word2Vec的输入矩阵)
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 输出词嵌入层 (类似Word2Vec的输出矩阵)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化嵌入层权重，类似Word2Vec的初始化方式"""
        initrange = 0.5 / self.in_embeddings.embedding_dim
        self.in_embeddings.weight.data.uniform_(-initrange, initrange)
        self.out_embeddings.weight.data.uniform_(-0, 0)
        
    def forward(self, input_ids, context_ids, negative_ids=None):
        # 获取目标词的嵌入向量
        input_embeds = self.in_embeddings(input_ids)  # [batch_size, embed_dim]
        
        # 获取上下文词的嵌入向量
        if context_ids.dim() == 2:  # 批处理模式
            # 对上下文词嵌入取平均，实现CBOW
            context_embeds = self.in_embeddings(context_ids)  # [batch_size, context_size, embed_dim]
            context_embeds = torch.mean(context_embeds, dim=1)  # [batch_size, embed_dim]
        else:  # 单个样本
            context_embeds = self.in_embeddings(context_ids).mean(0, keepdim=True)  # [1, embed_dim]
        
        # 计算正样本得分
        pos_output = self.out_embeddings(input_ids)  # [batch_size, embed_dim]
        pos_score = torch.sum(context_embeds * pos_output, dim=1)  # [batch_size]
        pos_score = F.logsigmoid(pos_score)  # [batch_size]
        
        # 如果有负样本，计算负样本得分
        neg_score = 0
        if negative_ids is not None:
            neg_output = self.out_embeddings(negative_ids)  # [batch_size, num_neg, embed_dim]
            neg_score = torch.bmm(neg_output, context_embeds.unsqueeze(2)).squeeze()  # [batch_size, num_neg]
            neg_score = F.logsigmoid(-neg_score).sum(1)  # [batch_size]
            
        return -(pos_score + neg_score).mean()
    
    def get_embedding(self, word_id):
        """获取指定词ID的嵌入向量"""
        with torch.no_grad():
            word_tensor = torch.tensor([word_id], device=self.in_embeddings.weight.device)
            # 使用输入嵌入作为最终的词向量表示
            return self.in_embeddings(word_tensor).squeeze(0).cpu().numpy()

def collate_fn(batch):
    """自定义批处理函数，处理变长的上下文序列"""
    input_ids = torch.tensor([item["input_id"] for item in batch])
    
    # 获取每个样本的上下文ID列表
    context_ids_list = [item["context_ids"] for item in batch]
    
    # 计算最大上下文长度
    max_context_len = max(len(context) for context in context_ids_list)
    
    # 填充上下文序列
    padded_context_ids = []
    for context in context_ids_list:
        if len(context) < max_context_len:
            # 填充到最大长度
            padded = context + [0] * (max_context_len - len(context))
        else:
            padded = context
        padded_context_ids.append(padded)
    
    context_ids = torch.tensor(padded_context_ids)
    
    return {
        "input_ids": input_ids,
        "context_ids": context_ids
    }

def train(config):
    """训练Word2Vec模型"""
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 打印详细的GPU信息
        gpu_props = torch.cuda.get_device_properties(device)
        logging.info(f"使用GPU: {gpu_props.name}")
        logging.info(f"GPU内存: {gpu_props.total_memory/1024**3:.2f} GB")
        logging.info(f"GPU计算能力: {gpu_props.major}.{gpu_props.minor}")
        
        # 设置CUDA性能优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("已启用CUDA性能优化")
    else:
        device = torch.device("cpu")
        logging.warning("未检测到GPU，将使用CPU进行训练，这可能会很慢")
    
    # 加载词汇表
    with open(config["vocab_path"], "r", encoding="utf-8") as f:
        word2id = json.load(f)
    id2word = {v: k for k, v in word2id.items()}
    vocab_size = len(word2id)
    logging.info(f"词汇表大小: {vocab_size}")
    
    # 创建数据集和数据加载器
    dataset = IPv6WordDataset(
        file_path=config["train_data_path"],
        vocab_path=config["vocab_path"],
        window_size=config["window_size"],
        max_samples=config.get("max_samples", None)
    )
    
    # 优化数据加载器配置
    pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=min(8, os.cpu_count()),  # 增加工作进程数
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=4 if pin_memory else None,  # 增加预取因子
        persistent_workers=pin_memory
    )
    
    # 创建模型
    model = Word2VecModel(
        vocab_size=vocab_size,
        embedding_dim=config["embedding_dim"]
    ).to(device)
    
    # 打印模型是否在GPU上
    logging.info(f"模型是否在GPU上: {next(model.parameters()).is_cuda}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    
    # 创建梯度缩放器，用于混合精度训练
    scaler = GradScaler(enabled=config["use_amp"])
    
    # 负采样器
    def get_negative_samples(batch_size, num_samples, vocab_size, exclude_ids=None):
        """生成负样本ID"""
        # 批量生成负样本，减少GPU调用次数
        if exclude_ids is None:
            return torch.randint(2, vocab_size, (batch_size, num_samples), device=device)
        
        # 使用向量化操作替代循环
        neg_ids = torch.randint(2, vocab_size, (batch_size, num_samples), device=device)
        mask = (neg_ids == exclude_ids.unsqueeze(1))
        while mask.any():
            new_samples = torch.randint(2, vocab_size, (batch_size, num_samples), device=device)
            neg_ids = torch.where(mask, new_samples, neg_ids)
            mask = (neg_ids == exclude_ids.unsqueeze(1))
        return neg_ids
    
    # 训练循环
    logging.info("开始训练...")
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        batch_count = 0
        epoch_start_time = time.time()
        
        # 使用原位更新的进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}", 
                          ncols=100, position=0, leave=True, dynamic_ncols=False)
        
        for batch in progress_bar:
            batch_start_time = time.time()
            
            # 将数据移至GPU
            input_ids = batch["input_ids"].to(device, non_blocking=pin_memory)
            context_ids = batch["context_ids"].to(device, non_blocking=pin_memory)
            
            # 生成负样本
            negative_ids = get_negative_samples(
                input_ids.size(0), 
                config["num_negative"], 
                vocab_size,
                input_ids
            )
            
            # 使用混合精度训练
            with autocast(enabled=config["use_amp"]):
                # 前向传播
                loss = model(input_ids, context_ids, negative_ids)
            
            # 反向传播和优化
            optimizer.zero_grad(set_to_none=True)
            
            # 使用梯度缩放器
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            
            # 更新权重
            scaler.step(optimizer)
            scaler.update()
            
            # 更新进度条
            total_loss += loss.item()
            batch_count += 1
            batch_time = time.time() - batch_start_time
            batch_speed = config["batch_size"] / batch_time if batch_time > 0 else 0
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "gpu_mem": f"{torch.cuda.memory_allocated(0)/1024**3:.2f}GB",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "batch/s": f"{batch_speed:.1f}"  # 使用手动计算的批处理速度
            })
        
        # 计算平均损失
        avg_loss = total_loss / batch_count
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1}/{config['epochs']}, 平均损失: {avg_loss:.4f}, 耗时: {epoch_time:.2f}秒")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config["model_save_path"])
            logging.info(f"保存最佳模型，损失: {best_loss:.4f}")
    
    # 训练完成
    elapsed_time = time.time() - start_time
    logging.info(f"训练完成，耗时: {elapsed_time/60:.2f}分钟")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(config["model_save_path"]))
    
    # 提取词嵌入
    word_embeddings = {}
    model.eval()
    
    # 批量提取词嵌入以加速
    batch_size = 2048  # 增大批量提取大小
    word_ids = list(range(vocab_size))
    num_batches = (vocab_size + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="提取词嵌入", ncols=100, position=0, leave=True):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, vocab_size)
            batch_ids = word_ids[start_idx:end_idx]
            
            # 批量获取嵌入
            batch_tensor = torch.tensor(batch_ids, device=device)
            batch_embeds = model.in_embeddings(batch_tensor).cpu().numpy()
            
            # 保存到字典
            for j, idx in enumerate(batch_ids):
                if idx in id2word:
                    word_embeddings[id2word[idx]] = batch_embeds[j]
    
    # 保存词嵌入
    np.save(config["embeddings_save_path"], word_embeddings)
    logging.info(f"词嵌入已保存到 {config['embeddings_save_path']}")
    
    return model, word_embeddings, word2id, id2word

def visualize_embeddings(word_embeddings, word2id, id2word, output_path, n_components=2, perplexity=30):
    """可视化词嵌入"""
    logging.info("开始可视化词嵌入...")
    
    # 提取词向量和对应的词
    words = []
    vectors = []
    
    for word, vector in word_embeddings.items():
        if word not in ["[PAD]", "[UNK]"]:  # 排除特殊标记
            words.append(word)
            vectors.append(vector)
    
    # 将列表转换为NumPy数组
    vectors = np.array(vectors)  # 添加这行转换
    
    # 使用t-SNE降维
    logging.info(f"使用t-SNE将词向量降至{n_components}维...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)
    
    # 创建DataFrame以便于绘图
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'word': words,
        'nybble': [word[0] for word in words],  # 提取nybble值
        'position': [word[1:] for word in words]  # 提取位置信息
    })
    
    # 绘制散点图
    plt.figure(ffigsize=(12, 10))
    
    # 按nybble值着色
    sns.scatterplot(x='x', y='y', hue='nybble', data=df, palette='tab20', s=50, alpha=0.7)
    
    # 添加标题和图例
    plt.title('IPv6地址词向量t-SNE可视化', fontsize=16)
    plt.xlabel('t-SNE维度1', fontsize=12)
    plt.ylabel('t-SNE维度2', fontsize=12)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"可视化结果已保存到 {output_path}")
    
    # 使用DBSCAN进行聚类分析
    logging.info("使用DBSCAN进行聚类分析...")
    clustering = DBSCAN(eps=3, min_samples=5).fit(reduced_vectors)
    df['cluster'] = clustering.labels_
    
    # 绘制聚类结果
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='x', y='y', hue='cluster', data=df, palette='tab20', s=50, alpha=0.7)
    plt.title('IPv6地址词向量聚类结果', fontsize=16)
    plt.xlabel('t-SNE维度1', fontsize=12)
    plt.ylabel('t-SNE维度2', fontsize=12)
    
    # 保存聚类图像
    cluster_output_path = output_path.replace('.png', '_clusters.png')
    plt.savefig(cluster_output_path, dpi=300, bbox_inches='tight')
    logging.info(f"聚类结果已保存到 {cluster_output_path}")
    
    # 输出聚类统计信息
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    logging.info(f"聚类数量: {n_clusters}")
    logging.info(f"噪声点数量: {list(clustering.labels_).count(-1)}")
    
    return df

def main():
    """主函数"""
    config = {
        "train_data_path": "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/word_sequences.txt",
                "vocab_path": "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/vocabulary.json",
        "model_save_path": "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/models/ipv6_word2vec.pt",
        "embeddings_save_path": "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/models/ipv6_embeddings.npy",
        "visualization_path": "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/models/embeddings_tsne.png",
        "batch_size": 2048,  # 增大批处理大小以提高GPU利用率
        "embedding_dim": 100,  # 与原论文一致
        "window_size": 5,      # 与原论文一致
        "num_negative": 10,    # 增加负采样数量
        "learning_rate": 0.001, # 更稳定的学习率
        "epochs": 10,          # 增加训练轮次
        "clip_grad": 1.0,      # 更严格的梯度裁剪
        "num_workers": 8,      # 最大化数据加载工作进程
        "use_amp": True        # 使用自动混合精度
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["embeddings_save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["visualization_path"]), exist_ok=True)
    
    # 训练模型
    model, word_embeddings, word2id, id2word = train(config)
    
    # 可视化词嵌入
    visualize_embeddings(
        word_embeddings, 
        word2id, 
        id2word, 
        config["visualization_path"]
    )

if __name__ == "__main__":
    main()