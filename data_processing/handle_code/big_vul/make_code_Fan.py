# -*- coding: utf-8 -*-
# @Desc    :
import pandas as pd
import os
from tqdm import tqdm
out_dir = "/data/AIlinshi/linshi001/dataset/ALPaper/big_vul"
data = pd.read_csv("/data/AIlinshi/linshi001/dataset/ALPaper/big_vul/MSR_data_cleaned.csv")
data_length = data.shape[0]
print(data_length)  # 188636
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
def create_c_file():
    vul_num = 0
    for i in tqdm(range(data_length)):
        func_after = data.at[i, "func_after"]
        func_before = data.at[i,"func_before"]
        vul = data.at[i,"vul"]
        if vul ==1:
            vul_num = vul_num+1
        # data_name = str(i)+"_"+str(vul)+".c"
        data_name = str(i)+".c"
        if func_after != func_before and vul != 1:
            print(data_name)
        filename = data_name
        # 文件有重名现象
        if os.path.exists(out_dir + "/" + filename):
            with open(out_dir + "/" + filename, 'r') as f:
                func = f.read()
                if func == func_after:
                    print(filename)
                    continue
                else:
                    with open(out_dir + "/" +filename, 'w') as f:
                        f.write(func_before)
                i = i + 1
        with open(out_dir + "/" + filename, 'w') as f:
            f.write(func_before)
    print(vul_num)  # 10900

def formart_devign(data):

    import json

    result_data = data.rename(columns={
        'Unnamed: 0': 'id',
        'func_before': 'code',
        'vul': 'target'
    })[['id', 'code', 'target']]

    json_data = result_data.to_dict('records')

    with open('/data/AIlinshi/linshi001/dataset/ALPaper/big_vul/bigvul_data_all.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(json_data)} 条记录到 sampled_data.json")

formart_devign(data)