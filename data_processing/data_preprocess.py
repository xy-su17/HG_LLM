import json
import os

with open('../function.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
new_data = []
for i,d in enumerate(data):
    d['id']=i
    new_data.append(d)

jsObj = json.dumps(new_data)
with open('../function_new.json',"w") as fw:
    fw.write(jsObj)
    fw.close()

func_values = []

def extract_func_values(data):
    if isinstance(data, dict): 
        for key, value in data.items():
            if key == "func":
                func_values.append(value)
            else:
                extract_func_values(value)
    elif isinstance(data, list): 
        for item in data:
            extract_func_values(item) 

extract_func_values(data)

output_folder = 'devign_dataset/devign_raw_code'
os.makedirs(output_folder, exist_ok=True)

for i, func_value in enumerate(func_values):
    filename = os.path.join(output_folder, f"{i}.c")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(func_value)

    print(f"已生成文件: {filename}")