import ipaddress
import random
import argparse
import os

def standardize_ipv6(address):
    """将IPv6地址转换为标准格式（完全展开，不省略0）"""
    try:
        # 使用ipaddress模块解析地址
        ip = ipaddress.IPv6Address(address)
        # 获取完全展开的格式
        expanded = ip.exploded
        return expanded
    except Exception as e:
        print(f"处理地址 {address} 时出错: {e}")
        return None

def extract_and_standardize(input_file, output_file, count=100, seed=None):
    """从输入文件中提取指定数量的IPv6地址并标准化"""
    # 设置随机种子以便结果可重现
    if seed is not None:
        random.seed(seed)
    
    # 读取所有地址
    with open(input_file, 'r') as f:
        addresses = [line.strip() for line in f if line.strip()]
    
    print(f"从文件中读取了 {len(addresses)} 个IPv6地址")
    
    # 如果请求的数量大于可用地址数量，调整count
    if count > len(addresses):
        print(f"警告: 请求的地址数量 ({count}) 大于可用地址数量 ({len(addresses)})")
        count = len(addresses)
    
    # 随机选择指定数量的地址
    selected_addresses = random.sample(addresses, count)
    
    # 标准化选中的地址
    standardized = []
    for addr in selected_addresses:
        std_addr = standardize_ipv6(addr)
        if std_addr:
            standardized.append(std_addr)
    
    print(f"成功标准化了 {len(standardized)} 个地址")
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        for addr in standardized:
            f.write(f"{addr}\n")
    
    print(f"已将标准化的地址写入 {output_file}")
def main():
    parser = argparse.ArgumentParser(description='从IPv6地址文件中提取并标准化地址')
    parser.add_argument('--input', type=str, 
                        default="D:\\bigchuang\\ipv6地址论文\\10-6VecLM\\6VecLM2\\data\\public_database\\responsive-addresses.txt",
                        help='输入文件路径')
    parser.add_argument('--output', type=str, 
                        default="D:\\bigchuang\\ipv6地址论文\\10-6VecLM\\6VecLM2\\data\\seeds\\test.txt",
                        help='输出文件路径')
    parser.add_argument('--count', type=int, default=100000, help='要提取的地址数量')
    parser.add_argument('--seed', type=int, default=None, help='随机数种子')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    extract_and_standardize(args.input, args.output, args.count, args.seed)

if __name__ == "__main__":
    main()