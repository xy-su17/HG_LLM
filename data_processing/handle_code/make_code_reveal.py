import json
import csv
import argparse
import os
from pathlib import Path


def json_to_csv(json_file, csv_file=None):
    """
    将 JSON 文件转换为 CSV 文件

    参数:
        json_file: JSON 文件路径
        csv_file: CSV 文件路径（可选，默认为同名 .csv 文件）
    """
    # 如果未指定 CSV 文件名，使用 JSON 文件名
    if csv_file is None:
        csv_file = Path(json_file).with_suffix('.csv')

    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 确保数据是列表格式
    if isinstance(data, dict):
        # 如果是字典，尝试获取第一个列表值
        for key, value in data.items():
            if isinstance(value, list):
                data = value
                break
        else:
            # 如果不是列表，将其包装为列表
            data = [data]

    if not data:
        print("JSON 文件为空")
        return

    # 获取所有字段名
    if isinstance(data[0], dict):
        fieldnames = list(data[0].keys())
    else:
        print("JSON 数据格式不正确，应为对象数组或对象")
        return

    # 写入 CSV 文件
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"转换成功！")
    print(f"JSON 文件: {json_file}")
    print(f"CSV 文件: {csv_file}")
    print(f"记录数: {len(data)}")


def main():
    parser = argparse.ArgumentParser(description='将 JSON 文件转换为 CSV 文件')
    parser.add_argument('-i', '--input',
                        help='输入的 JSON 文件路径',
                        default='/data/AIinspur02/linshi001/dataset/reveal/vulnerables.json')
    parser.add_argument('-o', '--output',
                        help='输出的 CSV 文件路径（可选）',
                        default='/data/AIinspur02/linshi001/dataset/reveal/vulnerables.csv')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误：文件 {args.input} 不存在")
        return

    json_to_csv(args.input, args.output)


if __name__ == '__main__':
    main()
