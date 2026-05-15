import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os


def split_and_filter_csv_to_json(csv_file_path, bad_file_path=None):
    df = pd.read_csv(csv_file_path)

    c_files_directory = "/data/AIlinshi/linshi001/dataset/ALPaper/big_vul/joern_out_files"

    bad_ids = set()
    with open(bad_file_path, 'r') as f:
        bad_ids = json.load(f)
        print(f"从bad_file中读取到 {len(bad_ids)} 个需要过滤的ID")

    valid_rows = []
    for index, row in df.iterrows():
        if f"{row['id']}.c" in bad_ids:
            continue

        c_file_path = os.path.join(c_files_directory, f"{row['id']}.c")
        if os.path.exists(c_file_path):
            valid_rows.append(index)

    df_filtered = df.loc[valid_rows].copy()
    print(f"原始数据总数: {len(df)}")
    print(f"过滤后有效数据数量: {len(df_filtered)}")

    if len(df_filtered) == 0:
        print("警告: 没有找到任何有效的数据")
        return

    train_df, temp_df = train_test_split(df_filtered, test_size=0.2, random_state=42, stratify=df_filtered['target'])

    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['target'])

    train_data = train_df.to_dict('records')
    val_data = val_df.to_dict('records')
    test_data = test_df.to_dict('records')

    output_dir = '/data/AIlinshi/linshi001/dataset/ALPaper/big_vul/split_data'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"训练集数量: {len(train_df)} ({len(train_df) / len(df_filtered) * 100:.1f}%)")
    print(f"验证集数量: {len(val_df)} ({len(val_df) / len(df_filtered) * 100:.1f}%)")
    print(f"测试集数量: {len(test_df)} ({len(test_df) / len(df_filtered) * 100:.1f}%)")

    print("\n训练集类别分布:")
    print(train_df['target'].value_counts().sort_index())
    print("\n验证集类别分布:")
    print(val_df['target'].value_counts().sort_index())
    print("\n测试集类别分布:")
    print(test_df['target'].value_counts().sort_index())


def shuffle_and_split_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)

    print(f"原始数据总数: {len(df)}")

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['vul'])

    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['vul'])

    train_data = train_df.to_dict('records')
    val_data = val_df.to_dict('records')
    test_data = test_df.to_dict('records')

    output_dir = '/data/AIlinshi/linshi001/dataset/ALPaper/big_vul/split_data'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'valid.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"训练集数量: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"验证集数量: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"测试集数量: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")

    print("\n训练集类别分布:")
    print(train_df['vul'].value_counts().sort_index())
    print("\n验证集类别分布:")
    print(val_df['vul'].value_counts().sort_index())
    print("\n测试集类别分布:")
    print(test_df['vul'].value_counts().sort_index())


# 使用示例
if __name__ == "__main__":
    csv_file_path = "/data/AIlinshi/linshi001/dataset/ALPaper/big_vul/extracted_data.csv"
    bad_file_path = "/data/AIlinshi/linshi001/dataset/ALPaper/big_vul/bad_file.json"  # bad_file路径
    # split_and_filter_csv_to_json(csv_file_path, bad_file_path)
    shuffle_and_split_csv(csv_file_path)
