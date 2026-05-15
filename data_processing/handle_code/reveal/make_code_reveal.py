import json
import csv
import os
import random

base_dir = '/data/AIlinshi/linshi001/dataset/ALPaper/reveal/'
func_before_dir = os.path.join(base_dir, 'func_before_files')
os.makedirs(func_before_dir, exist_ok=True)
def make_code_reveal():
    with open(base_dir + "extracted_data.csv", mode='w',  encoding='utf-8', newline='') as outfile:
        output_fieldnames = ['id', 'target']
        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        writer.writeheader()
        
        id = 0
        with open(base_dir + "vulnerables.json", 'r') as file:
            json_data = json.load(file)
            for line in json_data:
                extracted_row = {
                    'id': id,
                    'target': 1
                }
                writer.writerow(extracted_row)
                filename = f"{id}.c"
                filepath = os.path.join(func_before_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as func_file:
                    func_file.write(line['code'])
                id += 1
        with open(base_dir + "non-vulnerables.json", 'r') as file:
            json_data = json.load(file)
            for line in json_data:
                extracted_row = {
                    'id': id,
                    'target': 0
                }
                writer.writerow(extracted_row)
                filename = f"{id}.c"
                filepath = os.path.join(func_before_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as func_file:
                    func_file.write(line['code'])
                id += 1
    print(id)

def formart_reveal():
    all_data = []
    id_counter = 0

    with open(base_dir + "vulnerables.json", 'r') as file:
        vulnerables = json.load(file)
        for line in vulnerables:
            all_data.append({
                'id': id_counter,
                'target': 1,
                'code': line['code']
            })
            id_counter += 1

    with open(base_dir + "non-vulnerables.json", 'r') as file:
        non_vulnerables = json.load(file)
        for line in non_vulnerables:
            all_data.append({
                'id': id_counter,
                'target': 0,
                'func': line['code']
            })
            id_counter += 1

    random.shuffle(all_data)

    with open(base_dir + "reveal_data_all.json", 'w', encoding='utf-8') as json_file:
        json.dump(all_data, json_file, ensure_ascii=False, indent=2)

    print(f"已保存 {len(all_data)} 条记录到 all_data.json")

make_code_reveal()