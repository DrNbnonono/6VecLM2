import random
import argparse
import os
import ipaddress
from tqdm import tqdm
from collections import defaultdict
import re

def count_lines(file_path):
    """计算文件总行数"""
    print(f"正在计算文件总行数...")
    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    return line_count

def classify_ipv6_address(addr):
    """
    将IPv6地址分类为四种类型之一：
    1. 固定IID - 具有固定接口标识符的地址
    2. 子网结构化 - 低64位具有结构化值的地址
    3. EUI-64 SLAAC - 基于EUI-64以太网MAC的SLAAC地址
    4. 隐私SLAAC - 具有伪随机IID的SLAAC隐私地址
    """
    try:
        ip = ipaddress.IPv6Address(addr)
        hex_str = ip.exploded.replace(':', '')
        
        # 提取IID部分（后64位，即后16个十六进制字符）
        iid = hex_str[16:]
        
        # 检查是否为EUI-64 SLAAC地址（包含ff:fe标志）
        if 'fffe' in iid.lower() or 'ff:fe' in addr.lower():
            return 3  # EUI-64 SLAAC
        
        # 检查是否为固定IID（简单的固定值，如全0或特定值）
        if iid.count('0') >= 14 or re.match(r'0*[1-9a-f]?[0-9a-f]{0,3}$', iid.lower()):
            return 1  # 固定IID
        
        # 检查是否为子网结构化（包含明显的子网模式）
        if re.search(r'(00)+[1-9a-f]', iid.lower()) or '::' in addr and re.search(r':[0-9a-f]{1,3}:[0-9a-f]{1,3}$', addr.lower()):
            return 2  # 子网结构化
        
        # 默认为隐私SLAAC（随机IID）
        return 4  # 隐私SLAAC
    except:
        return 0  # 无效地址

def extract_prefix(addr):
    """提取IPv6地址的前缀（前64位）"""
    try:
        ip = ipaddress.IPv6Address(addr)
        parts = ip.exploded.split(':')
        prefix = ':'.join(parts[:4])
        return prefix
    except:
        return None

def process_addresses(input_file, output_file, sample_size=None, seed=None):
    """
    处理IPv6地址文件，按前缀分组并识别四种类型的地址
    """
    if seed is not None:
        random.seed(seed)
    
    # 读取所有地址
    print(f"正在读取IPv6地址...")
    addresses = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取地址"):
            addr = line.strip()
            if addr:
                addresses.append(addr)
    
    # 按前缀分组
    print(f"正在按前缀分组地址...")
    prefix_groups = defaultdict(list)
    for addr in tqdm(addresses, desc="分组进度"):
        prefix = extract_prefix(addr)
        if prefix:
            prefix_groups[prefix].append(addr)
    
    # 按前缀数量排序
    sorted_prefixes = sorted(prefix_groups.keys(), key=lambda x: len(prefix_groups[x]), reverse=True)
    print(f"找到 {len(sorted_prefixes)} 个不同的前缀")
    
    # 为每个前缀分类地址
    print(f"正在对每个前缀内的地址进行分类...")
    classified_addresses = []
    
    for prefix in tqdm(sorted_prefixes, desc="前缀处理"):
        # 对当前前缀下的地址进行分类
        type_groups = defaultdict(list)
        for addr in prefix_groups[prefix]:
            addr_type = classify_ipv6_address(addr)
            if addr_type > 0:  # 有效分类
                type_groups[addr_type].append(addr)
        
        # 从每种类型中选择地址
        for addr_type in sorted(type_groups.keys()):
            classified_addresses.extend(type_groups[addr_type])
    
    # 如果指定了样本大小，随机选择
    if sample_size and sample_size < len(classified_addresses):
        classified_addresses = random.sample(classified_addresses, sample_size)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for addr in classified_addresses:
            f.write(f"{addr}\n")
    
    print(f"已成功处理并保存 {len(classified_addresses)} 个IPv6地址到 {output_file}")
    
    # 统计各类型地址数量
    type_counts = defaultdict(int)
    for addr in classified_addresses:
        addr_type = classify_ipv6_address(addr)
        type_counts[addr_type] += 1
    
    print("\n地址类型统计:")
    print(f"1. 固定IID地址: {type_counts[1]}")
    print(f"2. 子网结构化地址: {type_counts[2]}")
    print(f"3. EUI-64 SLAAC地址: {type_counts[3]}")
    print(f"4. 隐私SLAAC地址: {type_counts[4]}")
    
    return classified_addresses

def extract_balanced_sample(input_file, output_file, sample_size, seed=None):
    """
    提取平衡的样本，尝试从每种类型中获取相等数量的地址
    """
    if seed is not None:
        random.seed(seed)
    
    # 读取所有地址
    print(f"正在读取IPv6地址...")
    addresses = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取地址"):
            addr = line.strip()
            if addr:
                addresses.append(addr)
    
    # 按前缀和类型分组
    print(f"正在分类地址...")
    prefix_type_groups = defaultdict(lambda: defaultdict(list))
    
    for addr in tqdm(addresses, desc="分类进度"):
        prefix = extract_prefix(addr)
        if prefix:
            addr_type = classify_ipv6_address(addr)
            if addr_type > 0:  # 有效分类
                prefix_type_groups[prefix][addr_type].append(addr)
    
    # 选择包含所有四种类型地址的前缀
    complete_prefixes = []
    for prefix, type_dict in prefix_type_groups.items():
        if len(type_dict) >= 3:  # 至少有3种类型
            complete_prefixes.append(prefix)
    
    print(f"找到 {len(complete_prefixes)} 个包含至少3种类型地址的前缀")
    
    # 计算每种类型需要的样本数
    target_per_type = sample_size // 4
    
    # 从每个前缀中选择平衡的样本
    balanced_sample = []
    
    # 首先处理完整前缀
    for prefix in tqdm(complete_prefixes, desc="处理完整前缀"):
        for addr_type in range(1, 5):
            addresses = prefix_type_groups[prefix][addr_type]
            if addresses:
                # 计算从当前类型中需要选择的数量
                to_select = min(len(addresses), target_per_type - len([a for a in balanced_sample if classify_ipv6_address(a) == addr_type]))
                if to_select > 0:
                    balanced_sample.extend(random.sample(addresses, to_select))
    
    # 如果某些类型的地址不足，从其他前缀中补充
    for addr_type in range(1, 5):
        type_count = len([a for a in balanced_sample if classify_ipv6_address(a) == addr_type])
        if type_count < target_per_type:
            needed = target_per_type - type_count
            # 收集所有该类型的地址
            all_of_type = []
            for prefix in prefix_type_groups:
                if prefix not in complete_prefixes:  # 只考虑尚未处理的前缀
                    all_of_type.extend(prefix_type_groups[prefix][addr_type])
            
            # 随机选择需要的数量
            if all_of_type:
                to_select = min(len(all_of_type), needed)
                balanced_sample.extend(random.sample(all_of_type, to_select))
    
    # 如果总数不足，随机添加地址
    if len(balanced_sample) < sample_size:
        remaining = sample_size - len(balanced_sample)
        # 收集所有未选择的地址
        unused_addresses = [addr for addr in addresses if addr not in balanced_sample]
        if unused_addresses:
            to_select = min(len(unused_addresses), remaining)
            balanced_sample.extend(random.sample(unused_addresses, to_select))
    
    # 随机打乱顺序
    random.shuffle(balanced_sample)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for addr in balanced_sample:
            f.write(f"{addr}\n")
    
    print(f"已成功提取 {len(balanced_sample)} 个平衡的IPv6地址样本并保存到 {output_file}")
    
    # 统计各类型地址数量
    type_counts = defaultdict(int)
    for addr in balanced_sample:
        addr_type = classify_ipv6_address(addr)
        type_counts[addr_type] += 1
    
    print("\n地址类型统计:")
    print(f"1. 固定IID地址: {type_counts[1]}")
    print(f"2. 子网结构化地址: {type_counts[2]}")
    print(f"3. EUI-64 SLAAC地址: {type_counts[3]}")
    print(f"4. 隐私SLAAC地址: {type_counts[4]}")
    
    return balanced_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='按照IPv6地址类型分类并提取样本')
    parser.add_argument('--input', type=str, default="D:\\bigchuang\\ipv6地址论文\\10-6VecLM\\6VecLM2\\data\\public_database\\responsive-addresses.txt", 
                        help='输入IPv6地址文件路径')
    parser.add_argument('--output', type=str, default="D:\\bigchuang\\ipv6地址论文\\10-6VecLM\\6VecLM2\\data\\public_database\\classified_addresses.txt", 
                        help='输出分类结果文件路径')
    parser.add_argument('--count', type=int, default=200000, help='需要提取的样本数量')
    parser.add_argument('--balanced', action='store_true', help='是否提取平衡的样本（每种类型数量接近）')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子，用于结果复现')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在!")
        exit(1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 根据选择的方法执行分类和提取
    if args.balanced:
        extract_balanced_sample(args.input, args.output, args.count, args.seed)
    else:
        process_addresses(args.input, args.output, args.count, args.seed)