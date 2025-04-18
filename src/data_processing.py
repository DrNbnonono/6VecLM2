import ipaddress
from typing import List, Dict, Tuple, Set
import numpy as np
import pandas as pd
import logging
import os
import json
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_address_chunk(addresses: List[str], position: int = 0) -> Tuple[Set[str], List[str], Dict]:
    """处理一批IPv6地址，返回词汇集合、词序列和统计信息"""
    vocab = set()
    word_sequences = []
    stats = {"地址": [], "长度": [], "前缀": []}
    
    # 使用与原论文一致的位置编码
    location_alpha = '0123456789abcdefghijklmnopqrstuv'
    
    for addr in addresses:
        try:
            # 直接处理地址，移除冒号
            ip = ipaddress.ip_address(addr.strip())
            exploded = ip.exploded.replace(":", "")
            
            # 确保地址长度为32个十六进制数字
            if len(exploded) != 32:
                exploded = exploded.zfill(32)
            
            # 构建词汇 - 使用原论文的方式
            words = []
            for pos, nybble in enumerate(exploded):
                if pos < len(location_alpha):  # 确保位置在有效范围内
                    word = f"{nybble}{location_alpha[pos]}"
                    vocab.add(word)
                    words.append(word)
            
            word_sequences.append(" ".join(words) + "\n")
            
            # 收集统计信息
            stats["地址"].append(addr)
            stats["长度"].append(len(exploded))
            stats["前缀"].append(exploded[:8])  # 假设前8个nybble是前缀
        except Exception as e:
            continue
    
    return vocab, word_sequences, stats

def build_vocabulary(all_vocabs: List[Set[str]]) -> Dict[str, int]:
    """从多个词汇集合构建统一的词汇表"""
    # 合并所有词汇集合
    combined_vocab = set()
    for vocab_set in all_vocabs:
        combined_vocab.update(vocab_set)
    
    # 添加特殊标记
    special_tokens = {"[PAD]": 0, "[UNK]": 1}
    word2id = {word: i+len(special_tokens) for i, word in enumerate(sorted(combined_vocab))}
    word2id.update(special_tokens)
    
    logging.info(f"构建词汇表完成，大小: {len(word2id)}")
    return word2id

def process_to_words(input_path: str, output_path: str, stats_path: str = None, 
                     chunk_size: int = 100000, num_workers: int = None) -> Tuple[Dict[str, int], pd.DataFrame]:
    """将原始地址转换为词序列文件，返回词汇表和统计信息，使用并行处理"""
    start_time = time.time()
    logging.info(f"从 {input_path} 读取地址...")
    
    # 确定CPU核心数
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # 保留一个核心给系统
    
    logging.info(f"使用 {num_workers} 个工作进程进行并行处理")
    
    # 计算文件总行数以初始化进度条
    total_lines = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    logging.info(f"文件包含 {total_lines} 行数据")
    
    # 创建进程池
    pool = mp.Pool(processes=num_workers)
    
    # 准备结果收集器
    all_vocabs = []
    all_word_sequences = []
    all_stats = {"地址": [], "长度": [], "前缀": []}
    
    # 分块读取文件并并行处理
    with open(input_path, 'r', encoding='utf-8') as f:
        # 使用tqdm创建总体进度条
        pbar = tqdm(total=total_lines, desc="处理IPv6地址", unit="地址")
        
        chunk = []
        chunk_position = 0
        
        for line in f:
            chunk.append(line.strip())
            
            if len(chunk) >= chunk_size:
                # 提交当前批次进行处理
                chunk_position += 1
                result = pool.apply_async(
                    process_address_chunk, 
                    args=(chunk, chunk_position),
                    callback=lambda x: pbar.update(len(chunk))
                )
                
                # 收集结果
                vocab_chunk, word_sequences_chunk, stats_chunk = result.get()
                all_vocabs.append(vocab_chunk)
                all_word_sequences.extend(word_sequences_chunk)
                
                for key in all_stats:
                    all_stats[key].extend(stats_chunk[key])
                
                # 重置批次
                chunk = []
        
        # 处理最后一个不完整的批次
        if chunk:
            vocab_chunk, word_sequences_chunk, stats_chunk = process_address_chunk(chunk, chunk_position + 1)
            all_vocabs.append(vocab_chunk)
            all_word_sequences.extend(word_sequences_chunk)
            
            for key in all_stats:
                all_stats[key].extend(stats_chunk[key])
            
            pbar.update(len(chunk))
        
        pbar.close()
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 构建词汇表
    vocab = build_vocabulary(all_vocabs)
    
    # 写入词序列
    logging.info(f"写入 {len(all_word_sequences)} 个词序列到 {output_path}")
    with open(output_path, "w", encoding='utf-8') as f:
        f.writelines(all_word_sequences)
    
    # 保存统计信息
    stats_df = pd.DataFrame(all_stats)
    if stats_path:
        stats_df.to_csv(stats_path, index=False, encoding='utf-8')
        logging.info(f"统计信息已保存到 {stats_path}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"处理完成，耗时: {elapsed_time:.2f}秒")
    
    return vocab, stats_df

def parallel_process_to_words(input_path: str, output_path: str, stats_path: str = None, 
                             chunk_size: int = 100000, num_workers: int = None) -> Tuple[Dict[str, int], pd.DataFrame]:
    """使用更高效的并行处理方法处理大文件"""
    start_time = time.time()
    logging.info(f"从 {input_path} 读取地址...")
    
    # 确定CPU核心数
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # 保留一个核心给系统
    
    logging.info(f"使用 {num_workers} 个工作进程进行并行处理")
    
    # 计算文件总行数以初始化进度条
    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    logging.info(f"文件包含 {total_lines} 行数据")
    
    # 分块读取文件
    chunks = []
    addresses = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="读取文件", unit="行"):
            addresses.append(line.strip())
            if len(addresses) >= chunk_size:
                chunks.append(addresses)
                addresses = []
    
    # 添加最后一个不完整的块
    if addresses:
        chunks.append(addresses)
    
    logging.info(f"文件已分为 {len(chunks)} 个块进行处理")
    
    # 创建进程池并并行处理
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_address_chunk, chunks),
            total=len(chunks),
            desc="并行处理数据块",
            unit="块"
        ))
    
    # 合并结果
    all_vocabs = [result[0] for result in results]
    all_word_sequences = []
    all_stats = {"地址": [], "长度": [], "前缀": []}
    
    for _, word_sequences, stats in results:
        all_word_sequences.extend(word_sequences)
        for key in all_stats:
            all_stats[key].extend(stats[key])
    
    # 构建词汇表
    vocab = build_vocabulary(all_vocabs)
    
    # 写入词序列
    logging.info(f"写入 {len(all_word_sequences)} 个词序列到 {output_path}")
    with open(output_path, "w", encoding='utf-8') as f:
        f.writelines(all_word_sequences)
    
    # 保存统计信息
    stats_df = pd.DataFrame(all_stats)
    if stats_path:
        stats_df.to_csv(stats_path, index=False, encoding='utf-8')
        logging.info(f"统计信息已保存到 {stats_path}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"处理完成，耗时: {elapsed_time:.2f}秒")
    
    return vocab, stats_df

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed", exist_ok=True)
    
    # 使用并行处理方法
    vocab, stats = parallel_process_to_words(
        input_path="d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/public_database/sample_address.txt",
        output_path="d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/word_sequences.txt",
        stats_path="d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/address_stats.csv",
        chunk_size=100000,  # 每个块处理10万行
        num_workers=None    # 自动检测CPU核心数
    )
    
    # 保存词汇表
    with open("d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/vocabulary.json", "w", encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    
    # 划分训练集和验证集
    logging.info("划分训练集和验证集...")
    with open("d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/word_sequences.txt", 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    # 随机打乱数据
    np.random.seed(42)  # 设置随机种子以确保可重复性
    np.random.shuffle(all_lines)
    
    # 划分比例：90%训练，10%验证
    split_idx = int(len(all_lines) * 0.9)
    train_lines = all_lines[:split_idx]
    val_lines = all_lines[split_idx:]
    
    # 保存训练集
    train_path = "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/train_sequences.txt"
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    logging.info(f"训练集已保存到 {train_path}，包含 {len(train_lines)} 个序列")
    
    # 保存验证集
    val_path = "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/val_sequences.txt"
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    logging.info(f"验证集已保存到 {val_path}，包含 {len(val_lines)} 个序列")
    
    print(f"生成词汇表大小: {len(vocab)}")
    print(f"处理的地址数量: {len(stats)}")
    print(f"前缀分布示例:\n{stats['前缀'].value_counts().head(10)}")
    print(f"数据已划分为训练集({len(train_lines)}个样本)和验证集({len(val_lines)}个样本)")