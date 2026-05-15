
import os

from utils import my_tokenizer
import os
import csv
from gensim.models import Word2Vec
from utils import my_tokenizer 
from tqdm import tqdm

def extract_code_from_csv(csv_path):
    codes = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            headers = next(reader)  

            code_index = headers.index('code') if 'code' in headers else -1

            if code_index == -1:
                print(f"Warning: 'code' column not found in {csv_path}")
                return codes

            for row in reader:
                if len(row) > code_index and row[code_index].strip():
                    codes.append(row[code_index].strip())
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return codes


def tokenize_codes(codes):
    tokenized_codes = []
    for code in codes:
        tokens = my_tokenizer(code)
        if tokens: 
            tokenized_codes.append(tokens)
    return tokenized_codes


def train_word2vec_model(all_tokens, model_save_path):
    model = Word2Vec(
        sentences=all_tokens,
        vector_size=100, 
        window=5, 
        min_count=1,  
        workers=4, 
        sg=1, 
        epochs=10 
    )

    model.save(model_save_path)
    print(f"Word2Vec model saved to {model_save_path}")
    return model

def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data

if __name__ == '__main__':\
    path = '/data/AIinspur02/linshi001/dataset/devign_joern'
    all_tokenized_codes = []

    for file_name in tqdm(os.listdir(path)):
        nodes_path = os.path.join(path, file_name, 'tmp', file_name, 'nodes.csv')

        nodes = read_csv(nodes_path)
        for node in nodes:
            tokenized_codes = my_tokenizer(node['code'].strip())
            all_tokenized_codes.append(tokenized_codes)

    if all_tokenized_codes:
        model_save_path = '/data/AIinspur02/linshi001/dataset/devign_wv_models/devign_train_subtoken_data_myself'
        model = train_word2vec_model(all_tokenized_codes, model_save_path)
        print(f"Training completed. Total sentences: {len(all_tokenized_codes)}")
    else:
        print("No code data found for training")
