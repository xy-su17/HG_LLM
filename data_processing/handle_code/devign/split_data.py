import os

from sklearn.model_selection import train_test_split
import json


import warnings
warnings.filterwarnings("ignore")
def split_data():
    name_list = []
    for name in os.listdir("/data/AIinspur02/linshi001/dataset/devign_joern/"):
        name_list.append(name)
    lines = []
    with open('/data/AIinspur02/linshi001/dataset/function_new.json', 'r', encoding='utf-8') as file:
        for line in json.load(file):
            if str(line['id']) + '.c' in name_list:
                lines.append(line)
    print(len(lines))
    train, test = train_test_split(lines, train_size=0.8,test_size=0.2)2
    val, test = train_test_split(test, test_size=0.5)
    print(len(train), len(test), len(val))

    with open("/data/AIinspur02/linshi001/dataset/devign_data_split0407/train.json", "w") as file:
        json.dump(train, file)

    with open("/data/AIinspur02/linshi001/dataset/devign_data_split0407/test.json", "w") as file:
        json.dump(test, file)

    with open("/data/AIinspur02/linshi001/dataset/devign_data_split0407/valid.json", "w") as file:
        json.dump(val, file)
split_data()