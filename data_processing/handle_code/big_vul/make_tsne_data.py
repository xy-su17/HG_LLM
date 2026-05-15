import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def extract_and_analyze_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)

    cwe_counts = df['CWE ID'].value_counts()
    null_count = df['CWE ID'].isna().sum()
    total_count = len(df)

    cwe_count_map = cwe_counts.to_dict()

    df['use_cwe_type'] = df['CWE ID'].apply(
        lambda x: 'None Type' if pd.isna(x) else (x if cwe_count_map.get(x, 0) >= 20 else 'Remain class')
    )

    print(f"数据总数: {len(df)}")
    print(f"CWE类型总数: {len(cwe_counts)}")
    print(f"空值数量: {null_count}")
    print(f"空值比例: {(null_count / total_count) * 100:.2f}%")
    print("\n各CWE ID类型数量统计:")
    print("-" * 30)
    for cwe_id, count in cwe_counts.items():
        percentage = (count / total_count) * 100
        print(f"{cwe_id:<15} {count:<10} {percentage:.2f}%")

    return df, cwe_counts

def split_data(data, save_path):
    id_vul_data = []
    for _, row in data.iterrows():
        id_vul_data.append({
            'id': row['id'],
            'cwe_id': row['CWE ID'],
            'target': row['vul'] 
        })
    print(len(id_vul_data))
    train, test = train_test_split(id_vul_data, train_size=0.7,test_size=0.3)
    val, test = train_test_split(test, test_size=0.5)
    print(len(train), len(test), len(val))

    with open(save_path + "train_raw_code.json", "w") as file:
        json.dump(train, file)

    with open(save_path + "test_raw_code.json", "w") as file:
        json.dump(test, file)

    with open(save_path + "valid_raw_code.json", "w") as file:
        json.dump(val, file)

if __name__ == "__main__":
    csv_path = "/data/AIlinshi/linshi001/dataset/ALPaper/big_vul/extracted_data.csv"
    save_path = '/data/AIlinshi/linshi001/dataset/ALPaper/big_vul/tsne_data/split_data/'
    df, cwe_statistics = extract_and_analyze_csv(csv_path)
    # split_data(df, save_path)
