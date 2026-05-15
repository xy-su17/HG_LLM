import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

def shuffle_and_split_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)

    print(f"原始数据总数: {len(df)}")

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['target'])

    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['target'])

    train_data = train_df.to_dict('records')
    val_data = val_df.to_dict('records')
    test_data = test_df.to_dict('records')

    output_dir = '/data/AIlinshi/linshi001/dataset/ALPaper/reveal/split_data_all'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"训练集数量: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"验证集数量: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"测试集数量: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")

    print("\n训练集类别分布:")
    print(train_df['target'].value_counts().sort_index())
    print("\n验证集类别分布:")
    print(val_df['target'].value_counts().sort_index())
    print("\n测试集类别分布:")
    print(test_df['target'].value_counts().sort_index())

if __name__ == "__main__":
    csv_file_path = "/data/AIlinshi/linshi001/dataset/ALPaper/reveal/extracted_data.csv"
    bad_file_path = "/data/AIlinshi/linshi001/dataset/ALPaper/reveal/reveal_1013/bad_file.json"  # bad_file路径
    shuffle_and_split_csv(csv_file_path)
