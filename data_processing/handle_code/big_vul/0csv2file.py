# -*- coding: utf-8 -*-

# @Desc    : Fan数据集188636条，漏洞10901条
import csv
import os
import sys
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

base_path = '/data/AIlinshi/linshi001/dataset/ALPaper/big_vul'
csv_path = os.path.join(base_path, 'MSR_data_cleaned.csv')

output_csv_path = os.path.join(base_path, 'extracted_data.csv')
func_before_dir = os.path.join(base_path, 'func_before_files')
os.makedirs(func_before_dir, exist_ok=True)

with open(csv_path, mode='r', encoding='utf-8') as infile:
    csv_reader = csv.DictReader(infile)

    output_fieldnames = ['id', 'CVE ID', 'CWE ID', 'vul']
    with open(output_csv_path, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for row in tqdm(csv_reader):
            extracted_row = {
                'id': row.get(''),
                'CVE ID': row.get('CVE ID', ''),
                'CWE ID': row.get('CWE ID', ''),
                'vul': row.get('vul', '')
            }
            writer.writerow(extracted_row)

            func_before_content = row.get('func_before', '')
            if func_before_content and row.get(''):
                filename = f"{row['']}.c"
                filepath = os.path.join(func_before_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as func_file:
                    func_file.write(func_before_content)

print(f"数据已保存到: {output_csv_path}")
print(f"func_before 文件已保存到: {func_before_dir}")

