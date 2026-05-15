from gensim.models import Word2Vec
import json
import numpy as np
import os
import torch
import torch.nn as nn
import re

device = torch.device(f'cuda:{3}')
print(f"Using device: {device}")

def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def my_tokenizer(code):
    ## Remove code comments
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    ## Remove newlines & tabs
    code = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(/)|(\\r)','',code)
    ## Mix split (characters and words)
    splitter = '\"(.*?)\"| +|(;)|(->)|(&)|(\*)|(\()|(==)|(~)|(!=)|(<=)|(>=)|(!)|(\+\+)|(--)|(\))|(=)|(\+)|(\-)|(\[)|(\])|(<)|(>)|(\.)|({)'
    code = re.split(splitter,code)
    ## Remove None type
    code = list(filter(None, code))
    code = list(filter(str.strip, code))
    # snakecase -> camelcase and split camelcase
    code_1 = []
    for i in code:
        code_1 += convert(i).split('_')
    #filt
    code_2 = []
    for i in code_1:
        if i in ['{', '}', ';', ':']:
            continue
        code_2.append(i)
    return(code_2)


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

def collect_training_data(json_file, csv_base_path):
    all_code_snippets = []
    with open(json_file, 'r') as f:
        data = json.load(f)
        for entry in data:
            file_name = str(entry['id']) + '.c'
            nodes_path = os.path.join(csv_base_path, file_name, 'tmp', file_name, 'nodes.csv')
            if os.path.exists(nodes_path):
                nodes = read_csv(nodes_path)
                for node in nodes:
                    if node['isCFGNode'].strip() == 'True' and node['key'].strip() != 'File':
                        code_content = node['code'].strip()
                        tokens = my_tokenizer(code_content)
                        if tokens:  # 只保留非空的token序列
                            all_code_snippets.append(tokens)
    return all_code_snippets


def build_vocabulary(token_sequences, min_freq=2):
    token_freq = {}
    for tokens in token_sequences:
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1

    vocab = {'<PAD>': 0, '<UNK>': 1}  
    idx = 2
    for token, freq in token_freq.items():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1

    return vocab


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, max_seq_length, word2vec_model, vocab):
        super(BiLSTMModel, self).__init__()
        self.embedding_dim = embedding_dim  # 设为100

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(
            get_embedding_matrix(vocab, word2vec_model, embedding_dim)))
        self.embedding.weight.requires_grad = False

        self.bilstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(lstm_units * 2, embedding_dim)  # 双向LSTM输出维度映射到100维
        self.tanh = nn.Tanh()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.bilstm(embedded)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        concatenated = torch.cat((forward_hidden, backward_hidden), dim=1)

        output = self.fc(concatenated)
        output = self.tanh(output)
        return output


def create_bilstm_model(vocab_size, embedding_dim, lstm_units, max_seq_length, word2vec_model, vocab):
    model = BiLSTMModel(vocab_size, embedding_dim, lstm_units, max_seq_length, word2vec_model, vocab)
    model = model.to(device)
    return model


def get_embedding_matrix(vocab, word2vec_model, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in word2vec_model.wv:
            embedding_matrix[idx] = word2vec_model.wv[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.1, size=(embedding_dim,))
    return embedding_matrix


def train_bilstm_model(model, train_data, epochs=10, batch_size=32, learning_rate=0.001):
    from torch.utils.data import DataLoader, TensorDataset

    device = next(model.parameters()).device

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(train_data[0], train_data[1])  # 输入和目标
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')


def prepare_data_for_pytorch(token_sequences, vocab, max_seq_length):

    X = []
    Y = [] 
    for tokens in token_sequences:
        indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        if len(indices) > max_seq_length:
            indices = indices[:max_seq_length]
        elif len(indices) < max_seq_length:
            indices.extend([vocab['<PAD>']] * (max_seq_length - len(indices)))
        X.append(torch.tensor(indices, dtype=torch.long))

    return torch.stack(X), torch.stack(Y)


def pretrain_bilstm(json_file, csv_path, wv_model_path, output_model_path):
    wv_model = Word2Vec.load(wv_model_path)

    code_snippets = collect_training_data(json_file, csv_path)

    vocab = build_vocabulary(code_snippets)

    vocab_size = len(vocab)
    embedding_dim = 100  
    lstm_units = 128
    max_seq_length = 50

    model = create_bilstm_model(vocab_size, embedding_dim, lstm_units, max_seq_length, wv_model, vocab)

    X, y = prepare_data_for_pytorch(code_snippets, vocab, max_seq_length)

    train_bilstm_model(model, (X, y), epochs=10, batch_size=32, learning_rate=0.001)

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'lstm_units': lstm_units,
        'max_seq_length': max_seq_length,
        'device': str(device) 
    }, output_model_path + '.pth')

    with open(output_model_path + '_vocab.json', 'w') as f:
        json.dump(vocab, f)
    return model, vocab

if __name__ == '__main__':
    json_file = '/data/AIlinshi/linshi001/dataset/function_new.json'
    csv_path = '/data/AIlinshi/linshi001/dataset/devign_joern'
    wv_model_path = '/data/AIlinshi/linshi001/dataset/devign_wv_models/devign_train_subtoken_data'
    output_model_path = '/data/AIlinshi/linshi001/dataset/devign_wv_models/bilstm'
    pretrain_bilstm(json_file, csv_path, wv_model_path, output_model_path)  
