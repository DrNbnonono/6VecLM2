import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import logging
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import math

# 添加项目根目录到路径
sys.path.append("d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/src")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#######################
# 模型定义部分 (原model.py)
#######################

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x的形状: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MixtureOfExperts(nn.Module):
    """专家混合层"""
    def __init__(self, input_dim, output_dim, num_experts=8):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 门控网络，决定每个专家的权重
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 4),
                nn.GELU(),
                nn.Linear(input_dim * 4, output_dim)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # 输入形状: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # 计算门控权重
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_weights = torch.softmax(gate_logits, dim=-1)
        
        # 初始化输出
        output = torch.zeros(batch_size, seq_len, self.output_dim, device=x.device)
        
        # 对每个专家的输出进行加权求和
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)  # [batch_size, seq_len, output_dim]
            # 获取当前专家的权重并扩展维度以便广播
            expert_weight = gate_weights[:, :, i].unsqueeze(-1)  # [batch_size, seq_len, 1]
            # 加权求和
            output += expert_weight * expert_output
        
        return output

class IPv6Generator(nn.Module):
    """IPv6地址生成器模型"""
    def __init__(self, vocab_size, d_model=100, num_heads=10, num_experts=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=4*d_model, 
            dropout=dropout,
            batch_first=True  # 设置为True，使输入形状为 [batch_size, seq_len, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.moe = MixtureOfExperts(d_model, d_model, num_experts)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        
        # Transformer编码器
        output = self.transformer_encoder(src, src_key_padding_mask=None, mask=src_mask)
        
        # 专家混合层
        output = self.moe(output)
        
        # 输出层
        output = self.fc_out(output)
        
        return output

#######################
# 数据处理部分 (原train.py)
#######################

class IPv6SequenceDataset(Dataset):
    """IPv6地址序列数据集"""
    def __init__(self, file_path: str, vocab_path: str, seq_len: int = 32, input_len: int = 16):
        self.seq_len = seq_len
        self.input_len = input_len
        
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        
        # 加载序列数据
        self.sequences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                if len(words) == seq_len:
                    # 将词转换为ID
                    word_ids = [self.word2id.get(w, self.word2id["[UNK]"]) for w in words]
                    self.sequences.append(word_ids)
        
        logging.info(f"加载了 {len(self.sequences)} 个序列")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 输入序列（前input_len个词）
        src = torch.tensor(seq[:self.input_len], dtype=torch.long)
        
        # 目标序列（后seq_len-input_len个词）
        tgt = torch.tensor(seq[self.input_len:], dtype=torch.long)
        
        return src, tgt

def create_masks(src):
    """创建Transformer所需的掩码"""
    # 修改掩码创建方式，使其与PyTorch Transformer期望的维度匹配
    # 对于batch_first=True的设置，掩码应该是2D或3D的
    # 我们这里不需要掩码，因为我们没有使用padding，所以返回None
    return None

def cosine_distance_loss(pred, target, word_embeddings):
    """余弦距离损失函数"""
    # 获取目标词的嵌入
    target_embeds = word_embeddings[target]
    
    # 计算余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(pred, target_embeds, dim=1)
    
    # 转换为余弦距离
    cos_dist = 1.0 - cos_sim
    
    # 返回平均距离
    return torch.mean(cos_dist)

def train_epoch(model, dataloader, optimizer, criterion, word_embeddings, device, clip=1.0, use_amp=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    # 创建混合精度训练的scaler
    scaler = torch.amp.GradScaler() if use_amp else None
    
    progress_bar = tqdm(dataloader, desc="训练中", ncols=100)
    for src, tgt in progress_bar:
        src, tgt = src.to(device), tgt.to(device)
        
        # 创建掩码
        src_mask = create_masks(src)
        
        # 前向传播
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                output = model(src, src_mask)
                # 只取最后一个位置的输出用于预测下一个词
                last_output = output[:, -1, :]
                # 计算损失（余弦距离）
                loss = criterion(last_output, tgt[:, 0], word_embeddings)
            
            # 使用scaler进行反向传播和优化
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(src, src_mask)
            # 只取最后一个位置的输出用于预测下一个词
            last_output = output[:, -1, :]
            # 计算损失（余弦距离）
            loss = criterion(last_output, tgt[:, 0], word_embeddings)
            
            # 反向传播
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            # 更新参数
            optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, word_embeddings, device, use_amp=False):
    """验证模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="验证中", ncols=100):
            src, tgt = src.to(device), tgt.to(device)
            
            # 创建掩码
            src_mask = create_masks(src)
            
            # 前向传播
            if use_amp:
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    output = model(src, src_mask)
                    # 只取最后一个位置的输出
                    last_output = output[:, -1, :]
                    # 计算损失
                    loss = criterion(last_output, tgt[:, 0], word_embeddings)
            else:
                output = model(src, src_mask)
                # 只取最后一个位置的输出
                last_output = output[:, -1, :]
                # 计算损失
                loss = criterion(last_output, tgt[:, 0], word_embeddings)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    """主函数"""
    config = {
        'train_data_path': "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/train_sequences.txt",
        'val_data_path': "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/val_sequences.txt",
        'vocab_path': "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/vocabulary.json",
        'embeddings_path': "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/models/ipv6_embeddings.npy",
        'output_dir': "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/models/transformer",
        'batch_size': 64,
        'd_model': 100,       # 与原论文一致
        'num_heads': 10,      # 与原论文一致
        'num_experts': 8,     # MoE专家数量
        'num_layers': 6,      # 与原论文一致
        'dropout': 0.1,
        'learning_rate': 0.0001,
        'epochs': 20,
        'save_every': 5,
        'seq_len': 32,        # 地址总长度
        'input_len': 16,      # 输入序列长度
        'clip': 1.0,          # 梯度裁剪阈值
        'patience': 5,        # 早停耐心值
        'use_amp': True       # 是否使用混合精度训练
    }
    
    # 确保输出目录存在
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 检查CUDA是否可用
    # 设置设备 - 强制使用GPU如果可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 打印GPU信息
        gpu_props = torch.cuda.get_device_properties(device)
        logging.info(f"使用GPU: {gpu_props.name}")
        logging.info(f"GPU内存: {gpu_props.total_memory/1024**3:.2f} GB")
        logging.info(f"GPU计算能力: {gpu_props.major}.{gpu_props.minor}")
        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True
        logging.info("已启用CUDA性能优化")
    else:
        device = torch.device("cpu")
        logging.warning("未检测到可用GPU，将使用CPU训练，速度会较慢")

        
        # 如果使用混合精度训练
        if config['use_amp']:
            logging.info("启用混合精度训练 (AMP)")
    
    # 加载词汇表和词嵌入
    with open(config['vocab_path'], 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    id2word = {v: k for k, v in word2id.items()}
    vocab_size = len(word2id)
    logging.info(f"词汇表大小: {vocab_size}")
    
    # 加载预训练的词嵌入
    logging.info(f"加载词嵌入: {config['embeddings_path']}")
    try:
        word_embeddings = np.load(config['embeddings_path'], allow_pickle=True).item()
        logging.info(f"成功加载词嵌入，包含 {len(word_embeddings)} 个词向量")
    except Exception as e:
        logging.error(f"加载词嵌入失败: {e}")
        logging.error("请确保已经运行了embedding/train.py来生成词嵌入")
        return
    
    # 创建词嵌入矩阵
    embedding_dim = config['d_model']
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)
    
    # 统计找到的词嵌入数量
    found_embeddings = 0
    for word, idx in word2id.items():
        if word in word_embeddings:
            embedding_matrix[idx] = torch.tensor(word_embeddings[word])
            found_embeddings += 1
    
    logging.info(f"在词嵌入中找到 {found_embeddings}/{vocab_size} 个词 ({found_embeddings/vocab_size*100:.2f}%)")
    
    # 将词嵌入矩阵移至设备
    embedding_matrix = embedding_matrix.to(device)
    
    # 初始化数据集和数据加载器
    logging.info("初始化训练数据集...")
    train_dataset = IPv6SequenceDataset(
        config['train_data_path'], 
        config['vocab_path'],
        seq_len=config['seq_len'],
        input_len=config['input_len']
    )
    
    # 优化数据加载器以提高GPU利用率
    pin_memory = torch.cuda.is_available()
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=True if pin_memory else False
    )
    
    logging.info("初始化验证数据集...")
    val_dataset = IPv6SequenceDataset(
        config['val_data_path'], 
        config['vocab_path'],
        seq_len=config['seq_len'],
        input_len=config['input_len']
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )
    
    # 初始化模型
    logging.info(f"初始化模型，维度: {config['d_model']}, 头数: {config['num_heads']}, 层数: {config['num_layers']}")
    model = IPv6Generator(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_experts=config['num_experts'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # 使用预训练的词嵌入初始化模型的嵌入层
    logging.info("使用预训练词嵌入初始化模型嵌入层")
    model.embedding.weight.data.copy_(embedding_matrix)
    
    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 训练循环
    logging.info("=" * 60)
    logging.info(f"开始训练 - 共 {config['epochs']} 轮")
    logging.info("=" * 60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_time = time.time()
    epochs_no_improve = 0
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # 训练一个epoch
        train_loss = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            cosine_distance_loss, 
            embedding_matrix, 
            device, 
            config['clip'],
            config['use_amp']
        )
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(
            model, 
            val_dataloader, 
            cosine_distance_loss, 
            embedding_matrix, 
            device,
            config['use_amp']
        )
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 计算本轮用时
        epoch_time = time.time() - epoch_start
        
        logging.info(f"Epoch {epoch+1}/{config['epochs']} - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 用时: {epoch_time:.2f}秒")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config['output_dir'], "ipv6_generator_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            logging.info(f"最佳模型已保存到 {best_model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(f"验证损失未改善。已连续 {epochs_no_improve} 轮未改善")
        
        # 早停
        if epochs_no_improve >= config['patience']:
            logging.info(f"早停! {config['patience']} 轮验证损失未改善")
            break
        
        # 定期保存检查点
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_path = os.path.join(config['output_dir'], f"ipv6_generator_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            logging.info(f"模型检查点已保存到 {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(config['output_dir'], "ipv6_generator_final.pt")
    torch.save({
        'epoch': config['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, final_model_path)
    logging.info(f"最终模型已保存到 {final_model_path}")
    
    # 训练完成统计
    total_time = time.time() - start_time
    logging.info("=" * 60)
    logging.info(f"训练完成! 总用时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    logging.info(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(config['output_dir'], "loss_plot.png")
    plt.savefig(loss_plot_path)
    logging.info(f"损失曲线已保存到 {loss_plot_path}")

if __name__ == "__main__":
    main()