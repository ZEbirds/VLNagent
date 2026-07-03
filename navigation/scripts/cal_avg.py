import json
import argparse
from collections import defaultdict

def analyze_jsonl_file(file_path):
    """
    读取一个 JSON Lines 文件，计算指定数值指标的平均值。

    Args:
        file_path (str): JSON Lines 文件的路径。
    """
    # 定义需要计算平均值的指标
    metrics_to_average = ["success", "spl", "os", "ne", "steps"]
    
    # 使用 defaultdict 来自动初始化不存在的键
    # sums 用于存储各项指标的总和
    # counts 用于存储各项指标出现的次数
    sums = defaultdict(float)
    counts = defaultdict(int)
    total_lines = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 跳过空行
                line = line.strip()
                if not line:
                    continue
                
                total_lines += 1
                
                try:
                    data = json.loads(line)
                    
                    # 遍历需要计算的指标
                    for metric in metrics_to_average:
                        # 检查指标是否存在于该行数据中
                        if metric in data and isinstance(data[metric], (int, float)):
                            sums[metric] += data[metric]
                            counts[metric] += 1
                        else:
                            print(f"警告: 第 {line_num} 行缺少 '{metric}' 指标或其值非数字，已跳过。")

                except json.JSONDecodeError:
                    print(f"警告: 无法解析第 {line_num} 行的 JSON，已跳过。")

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return

    print(f"\n--- 分析报告 ---")
    print(f"已处理文件: '{file_path}'")
    print(f"总行数: {total_lines}")

    if total_lines == 0 or not sums:
        print("文件中没有找到有效数据，无法计算平均值。")
        return

    print("\n各项指标平均值:")
    for metric in sorted(sums.keys()):
        if counts[metric] > 0:
            average = sums[metric] / counts[metric]
            print(f"  - {metric}: {average:.4f}  (基于 {counts[metric]} 个有效数据点)")
        else:
            print(f"  - {metric}: 未找到有效数据。")

def main():
    """主函数，用于解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从 JSON Lines (.jsonl) 文件中计算各项指标的平均值。"
    )
    parser.add_argument(
        "file_path", 
        type=str, 
        help="要分析的 JSON Lines 文件路径。"
    )
    
    args = parser.parse_args()
    analyze_jsonl_file(args.file_path)

if __name__ == "__main__":
    main()
