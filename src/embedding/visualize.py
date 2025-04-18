import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns
import pandas as pd
import logging
import os

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_embeddings(embeddings_path):
    """加载保存的词嵌入"""
    logging.info(f"从 {embeddings_path} 加载词嵌入...")
    embeddings = np.load(embeddings_path, allow_pickle=True).item()
    return embeddings

def visualize_embeddings(word_embeddings, output_path, n_components=2, perplexity=30):
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
    vectors = np.array(vectors)
    
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
    plt.figure(figsize=(12, 10))
    
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

if __name__ == "__main__":
    # 配置路径
    embeddings_path = "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/models/ipv6_embeddings.npy"
    output_path = "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/models/embeddings_tsne.png"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载词嵌入并可视化
    word_embeddings = load_embeddings(embeddings_path)
    visualize_embeddings(word_embeddings, output_path)