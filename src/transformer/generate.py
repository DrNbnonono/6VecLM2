import torch
import torch.nn as nn
import numpy as np
import json
import os
import logging
import ipaddress
from tqdm import tqdm
import argparse
import sys
import math

# 确保能找到项目根目录
sys.path.append("d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/src")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#######################
# 模型定义部分 (直接从train.py复制)
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
        # 修改这里，使用正确的掩码格式
        output = self.transformer_encoder(src, mask=src_mask)
        
        # 专家混合层
        output = self.moe(output)
        
        # 输出层
        output = self.fc_out(output)
        
        return output

#######################
# 生成地址部分
#######################

def load_model_and_data(config):
    """加载模型和数据"""
    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.warning("没有可用的GPU，将使用CPU进行计算")
    
    # 加载词汇表
    with open(config['vocab_path'], 'r') as f:
        word2id = json.load(f)
    id2word = {v: k for k, v in word2id.items()}
    vocab_size = len(word2id)
    
    # 加载词嵌入
    word_embeddings = np.load(config['embeddings_path'], allow_pickle=True).item()
    
    # 创建词嵌入矩阵
    embedding_dim = config['d_model']
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)
    for word, idx in word2id.items():
        if word in word_embeddings:
            embedding_matrix[idx] = torch.tensor(word_embeddings[word])
    
    # 将嵌入矩阵移至GPU
    embedding_matrix = embedding_matrix.to(device)
    
    # 初始化模型
    model = IPv6Generator(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_experts=config['num_experts'],
        num_layers=config['num_layers'],
        dropout=0.0  # 推理时不需要dropout
    ).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(config['model_path'], map_location=device)
    # 检查checkpoint的结构并相应地加载模型
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, word2id, id2word, embedding_matrix, device

def create_masks(src):
    """创建Transformer所需的掩码"""
    # 修改掩码维度，从4D改为2D
    # 原来的代码: src_mask = torch.ones((src.shape[0], 1, 1, src.shape[1])).to(src.device)
    
    # 创建一个2D掩码，适用于transformer_encoder
    src_mask = None  # 对于没有填充的序列，可以使用None
    
    return src_mask
def generate_address(model, seed_address, word2id, id2word, embedding_matrix, device, temperature=0.01, max_len=32):
    """生成单个IPv6地址"""
    # 将种子地址转换为词序列
    exploded = ipaddress.ip_address(seed_address).exploded.replace(":", "")
    words = [f"{nybble}{chr(97 + pos)}" for pos, nybble in enumerate(exploded)]
    
    # 将词转换为ID
    word_ids = [word2id.get(w, word2id["[UNK]"]) for w in words[:16]]  # 只取前16个词
    src = torch.tensor([word_ids]).to(device)
    
    # 创建掩码
    src_mask = create_masks(src)
    
    # 生成地址的后半部分
    generated_words = []
    for i in range(16):  # 生成后16个词
        # 获取模型输出
        with torch.no_grad():  # 添加这一行，确保不计算梯度
            output = model(src, src_mask)
            
            # 获取最后一个位置的输出
            last_hidden = output[:, -1, :]
            
            # 计算与所有词嵌入的余弦相似度
            # 只考虑当前位置的词（例如，如果生成第17个位置，只考虑形如"Xq"的词）
            position = 16 + i
            position_char = chr(97 + position)
            
            # 找出所有当前位置的词ID
            position_word_ids = []
            position_embeddings = []
            
            for word, idx in word2id.items():
                if len(word) >= 2 and word[-1] == position_char:
                    position_word_ids.append(idx)
                    position_embeddings.append(embedding_matrix[idx])
            
            if not position_embeddings:
                # 如果没有找到当前位置的词，随机选择
                next_word_id = np.random.choice(list(id2word.keys()))
                generated_words.append(id2word[next_word_id])
                continue
            
            # 转换为张量
            position_embeddings = torch.stack(position_embeddings).to(device)
            
            # 计算余弦相似度
            cos_similarities = torch.nn.functional.cosine_similarity(
                last_hidden, position_embeddings, dim=1
            )
            
            # 应用温度缩放
            scaled_similarities = cos_similarities / temperature
            
            # 转换为概率分布 - 使用detach()分离梯度
            probs = torch.nn.functional.softmax(scaled_similarities, dim=0).detach().cpu().numpy()
            
            # 采样下一个词
            next_word_idx = np.random.choice(len(position_word_ids), p=probs)
            next_word_id = position_word_ids[next_word_idx]
            next_word = id2word[next_word_id]
            
            # 添加到生成的词列表
            generated_words.append(next_word)
            
            # 更新输入序列
            new_src = torch.cat([src, torch.tensor([[next_word_id]]).to(device)], dim=1)
            src = new_src
            src_mask = create_masks(src)
    
    # 将生成的词转换回IPv6地址
    all_words = words[:16] + generated_words
    nybbles = [word[0] for word in all_words]
    
    # 将nybbles转换为IPv6地址格式
    ipv6_parts = []
    for i in range(0, 32, 4):
        part = ''.join(nybbles[i:i+4])
        ipv6_parts.append(part)
    
    ipv6_address = ':'.join(ipv6_parts)
    
    return ipv6_address


def generate_addresses(config):
    """生成多个IPv6地址"""
    # 加载模型和数据
    model, word2id, id2word, embedding_matrix, device = load_model_and_data(config)
    
    # 加载种子地址
    with open(config['seed_path'], 'r') as f:
        seed_addresses = [line.strip() for line in f.readlines()]
    
    # 随机选择种子地址
    if config['num_seeds'] < len(seed_addresses):
        seed_addresses = np.random.choice(seed_addresses, config['num_seeds'], replace=False)
    
    # 生成地址
    generated_addresses = set()
    
    logging.info(f"使用温度 {config['temperature']} 生成 {config['num_candidates']} 个候选地址...")
    
    with tqdm(total=config['num_candidates']) as pbar:
        while len(generated_addresses) < config['num_candidates']:
            # 随机选择一个种子地址
            seed_address = np.random.choice(seed_addresses)
            
            # 生成新地址
            try:
                new_address = generate_address(
                    model, 
                    seed_address, 
                    word2id, 
                    id2word, 
                    embedding_matrix, 
                    device, 
                    temperature=config['temperature']
                )
                
                # 确保地址有效且不重复
                if new_address not in generated_addresses:
                    generated_addresses.add(new_address)
                    pbar.update(1)
            except Exception as e:
                logging.warning(f"生成地址时出错: {e}")
    
    # 保存生成的地址
    with open(config['output_path'], 'w') as f:
        for addr in generated_addresses:
            f.write(f"{addr}\n")
    
    logging.info(f"已生成 {len(generated_addresses)} 个候选地址并保存到 {config['output_path']}")

def generate_addresses_batch(config):
    """使用批处理方式生成多个IPv6地址，加速生成过程"""
    # 加载模型和数据
    model, word2id, id2word, embedding_matrix, device = load_model_and_data(config)
    
    # 加载种子地址
    with open(config['seed_path'], 'r') as f:
        seed_addresses = [line.strip() for line in f.readlines()]
    
    # 随机选择种子地址
    if config['num_seeds'] < len(seed_addresses):
        seed_addresses = np.random.choice(seed_addresses, config['num_seeds'], replace=False)
    
    # 生成地址
    generated_addresses = set()
    batch_size = min(32, config['num_candidates'])  # 批处理大小，根据GPU内存调整
    
    logging.info(f"使用温度 {config['temperature']} 生成 {config['num_candidates']} 个候选地址...")
    logging.info(f"使用批处理大小: {batch_size}")
    
    with tqdm(total=config['num_candidates']) as pbar:
        while len(generated_addresses) < config['num_candidates']:
            # 确定当前批次大小
            current_batch_size = min(batch_size, config['num_candidates'] - len(generated_addresses))
            
            # 随机选择种子地址
            batch_seeds = np.random.choice(seed_addresses, current_batch_size, replace=True)
            
            # 批量生成地址
            try:
                with torch.no_grad():
                    # 准备批量输入
                    batch_inputs = []
                    batch_words = []
                    
                    for seed_address in batch_seeds:
                        # 将种子地址转换为词序列
                        exploded = ipaddress.ip_address(seed_address).exploded.replace(":", "")
                        words = [f"{nybble}{chr(97 + pos)}" for pos, nybble in enumerate(exploded)]
                        batch_words.append(words[:16])  # 只取前16个词
                        
                        # 将词转换为ID
                        word_ids = [word2id.get(w, word2id["[UNK]"]) for w in words[:16]]
                        batch_inputs.append(word_ids)
                    
                    # 转换为张量
                    src = torch.tensor(batch_inputs).to(device)
                    src_mask = None  # 对于没有填充的序列，可以使用None
                    
                    # 生成地址的后半部分
                    for i in range(16):  # 生成后16个词
                        # 获取模型输出
                        output = model(src, src_mask)
                        
                        # 获取最后一个位置的输出
                        last_hidden = output[:, -1, :]  # [batch_size, d_model]
                        
                        # 当前位置
                        position = 16 + i
                        position_char = chr(97 + position)
                        
                        # 找出所有当前位置的词ID
                        position_word_ids = []
                        position_embeddings = []
                        
                        for word, idx in word2id.items():
                            if len(word) >= 2 and word[-1] == position_char:
                                position_word_ids.append(idx)
                                position_embeddings.append(embedding_matrix[idx])
                        
                        if not position_embeddings:
                            # 如果没有找到当前位置的词，随机选择
                            next_word_ids = np.random.choice(list(id2word.keys()), current_batch_size)
                            next_words = [id2word[idx] for idx in next_word_ids]
                            
                            # 更新输入序列
                            next_word_tensor = torch.tensor(next_word_ids).view(-1, 1).to(device)
                            src = torch.cat([src, next_word_tensor], dim=1)
                            
                            # 更新批次词列表
                            for j, next_word in enumerate(next_words):
                                if len(batch_words) > j and i < len(batch_words[j]):
                                    batch_words[j].append(next_word)
                            
                            continue
                        
                        # 转换为张量
                        position_embeddings = torch.stack(position_embeddings).to(device)  # [num_position_words, d_model]
                        
                        # 计算每个样本与所有位置词的余弦相似度
                        # 扩展维度以便进行批量计算
                        last_hidden_expanded = last_hidden.unsqueeze(1)  # [batch_size, 1, d_model]
                        position_embeddings_expanded = position_embeddings.unsqueeze(0)  # [1, num_position_words, d_model]
                        
                        # 计算余弦相似度 [batch_size, num_position_words]
                        cos_similarities = torch.nn.functional.cosine_similarity(
                            last_hidden_expanded, position_embeddings_expanded, dim=2
                        )
                        
                        # 应用温度缩放
                        scaled_similarities = cos_similarities / config['temperature']
                        
                        # 转换为概率分布
                        probs = torch.nn.functional.softmax(scaled_similarities, dim=1).cpu().numpy()
                        
                        # 为每个样本采样下一个词
                        next_word_indices = [np.random.choice(len(position_word_ids), p=prob) for prob in probs]
                        next_word_ids = [position_word_ids[idx] for idx in next_word_indices]
                        next_words = [id2word[idx] for idx in next_word_ids]
                        
                        # 更新输入序列
                        next_word_tensor = torch.tensor(next_word_ids).view(-1, 1).to(device)
                        src = torch.cat([src, next_word_tensor], dim=1)
                        
                        # 更新批次词列表
                        for j, next_word in enumerate(next_words):
                            if j < len(batch_words):
                                batch_words[j].append(next_word)
                    
                    # 将生成的词转换回IPv6地址
                    new_addresses = []
                    for j, words in enumerate(batch_words):
                        if len(words) >= 32:  # 确保有足够的词
                            all_words = words[:32]  # 取前32个词
                            nybbles = [word[0] for word in all_words]
                            
                            # 将nybbles转换为IPv6地址格式
                            ipv6_parts = []
                            for k in range(0, 32, 4):
                                part = ''.join(nybbles[k:k+4])
                                ipv6_parts.append(part)
                            
                            ipv6_address = ':'.join(ipv6_parts)
                            new_addresses.append(ipv6_address)
                    
                    # 添加到生成的地址集合
                    for addr in new_addresses:
                        try:
                            # 验证地址有效性
                            ipaddress.IPv6Address(addr)
                            if addr not in generated_addresses:
                                generated_addresses.add(addr)
                                pbar.update(1)
                                if len(generated_addresses) >= config['num_candidates']:
                                    break
                        except Exception:
                            continue
                    
            except Exception as e:
                logging.warning(f"批量生成地址时出错: {e}")
                # 如果批处理失败，回退到单个生成
                try:
                    seed_address = np.random.choice(seed_addresses)
                    new_address = generate_address(
                        model, 
                        seed_address, 
                        word2id, 
                        id2word, 
                        embedding_matrix, 
                        device, 
                        temperature=config['temperature']
                    )
                    
                    if new_address not in generated_addresses:
                        generated_addresses.add(new_address)
                        pbar.update(1)
                except Exception as e2:
                    logging.warning(f"单个生成地址时也出错: {e2}")
    
    # 保存生成的地址
    with open(config['output_path'], 'w') as f:
        for addr in generated_addresses:
            f.write(f"{addr}\n")
    
    logging.info(f"已生成 {len(generated_addresses)} 个候选地址并保存到 {config['output_path']}")

def main():
    parser = argparse.ArgumentParser(description='生成IPv6候选地址')
    parser.add_argument('--model_path', type=str, 
                        default="d:\\bigchuang\\ipv6地址论文\\10-6VecLM\\6VecLM2\\models\\transformer\\ipv6_generator_best.pt",
                        help='模型路径')
    parser.add_argument('--vocab_path', type=str, 
                        default="d:\\bigchuang\\ipv6地址论文\\10-6VecLM\\6VecLM2\\data\\processed\\vocabulary.json",
                        help='词汇表路径')
    parser.add_argument('--embeddings_path', type=str, 
                        default="d:\\bigchuang\\ipv6地址论文\\10-6VecLM\\6VecLM2\\models\\ipv6_embeddings.npy",
                        help='词嵌入路径')
    parser.add_argument('--seed_path', type=str, 
                        default="d:\\bigchuang\\ipv6地址论文\\10-6VecLM\\6VecLM2\\data\\seeds\\candidate_s6_e10_t0015.txt",
                        help='种子地址路径')
    parser.add_argument('--output_path', type=str, 
                        default="d:\\bigchuang\\ipv6地址论文\\10-6VecLM\\6VecLM2\\data\\generated\\candidate.txt",
                        help='输出地址路径')
    parser.add_argument('--num_candidates', type=int, default=10000, help='生成的候选地址数量')
    parser.add_argument('--num_seeds', type=int, default=100, help='使用的种子地址数量')
    parser.add_argument('--temperature', type=float, default=0.01, help='采样温度')
    parser.add_argument('--d_model', type=int, default=100, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=10, help='注意力头数')
    parser.add_argument('--num_experts', type=int, default=8, help='MoE专家数量')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='是否使用GPU')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    config = vars(args)
    
    # 使用批处理方式生成地址
    generate_addresses_batch(config)

if __name__ == "__main__":
    main()