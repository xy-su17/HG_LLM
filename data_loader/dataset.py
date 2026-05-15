import copy
import json
import sys
import os
import numpy as np
os.chdir(sys.path[0])
import torch
from dgl import DGLGraph
from tqdm import tqdm

from data_loader.batch_graph import GGNNBatchGraph
from utils import load_default_identifiers, initialize_batch, debug

class DataEntry:
    def __init__(self, datset, num_nodes, features, edges, target, origin_id,
                 hyper_edges=None, text_feature=None):  # 新增 hyper_edges 参数
        self.dataset = datset
        self.num_nodes = num_nodes
        self.target = target
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)
        self.origin_id = origin_id
        self.text_feature = text_feature
        self.hyper_edges = hyper_edges  # 新增：保存超边
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})
        for s, _type, t in edges:
            etype_number = self.dataset.get_edge_type_number(_type)
            self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})

class DataSet:
    def __init__(self, train_src, valid_src, test_src, batch_size, n_ident=None,
                 g_ident=None, l_ident=None, input_dir=None, text_features_path=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        # 加载文本特征
        self.text_features = None
        if text_features_path:
            self.load_text_features(text_features_path)
        self.n_ident, self.g_ident, self.l_ident= load_default_identifiers(n_ident, g_ident, l_ident)
        self.hyper_edge_tag = 'hyper_edges'  # 新增：超边标签
        self.read_dataset(train_src, valid_src, test_src, input_dir)
        self.initialize_dataset()

    def load_text_features(self, path):
        """加载预提取的文本特征"""
        try:
            self.text_features = np.load(path, allow_pickle=True).item()
            debug(f'Loaded text features for {len(self.text_features)} functions')
        except:
            debug(f'Failed to load text features from {path}')
            self.text_features = {}

    def get_text_feature(self, origin_id):
        """获取指定函数的文本特征"""
        if self.text_features and origin_id in self.text_features:
            return torch.FloatTensor(self.text_features[origin_id])
        return None

    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, train_src, valid_src, test_src, input_dir):
        debug('Reading Train File!')
        id_key = 'origin_id'
        train_data_ids,valid_data_ids,test_data_ids = [],[],[]
        with open(train_src,"r") as fp:
            train_data = []
            train_data = json.load(fp)
            for entry in tqdm(train_data):
                # 获取文本特征
                text_feature = self.get_text_feature(entry[id_key]) if self.text_features else None
                # 获取超边数据
                hyper_edges = entry.get(self.hyper_edge_tag, [])
                example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                    edges=entry[self.g_ident], target=entry[self.l_ident][0][0], origin_id=entry[id_key],
                                    hyper_edges=hyper_edges, text_feature=text_feature)
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)
                    debug('Feature Size %d' % self.feature_size)
                self.train_examples.append(example)
                train_data_ids.append(entry[id_key])
        if valid_src is not None:
            debug('Reading Validation File!')
            
            with open(valid_src,"r") as fp:
                valid_data = []
                valid_data = json.load(fp) 
                for entry in tqdm(valid_data):
                    # 获取文本特征
                    text_feature = self.get_text_feature(entry[id_key]) if self.text_features else None
                    # 获取超边数据
                    hyper_edges = entry.get(self.hyper_edge_tag, [])
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0],
                                        origin_id=entry[id_key],
                                        hyper_edges=hyper_edges, text_feature=text_feature)
                    self.valid_examples.append(example)
                    valid_data_ids.append(entry[id_key])

        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                test_data = []
                test_data = json.load(fp)
                for entry in tqdm(test_data):
                    # 获取文本特征
                    text_feature = self.get_text_feature(entry[id_key]) if self.text_features else None
                    # 获取超边数据
                    hyper_edges = entry.get(self.hyper_edge_tag, [])
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0],
                                        origin_id=entry[id_key],
                                        hyper_edges=hyper_edges, text_feature=text_feature)
                    self.test_examples.append(example)
                    test_data_ids.append(entry[id_key])

        data_ids = {"train_data_ids":train_data_ids,"valid_data_ids":valid_data_ids,"test_data_ids":test_data_ids}
        with open(os.path.join(input_dir, 'data_ids.json'),"w") as f:
            json.dump(data_ids,f)
            print("data_ids 已保存")
    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=False)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size, shuffle=False)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size, shuffle=False)
        
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        ids = [e.origin_id for e in taken_entries]
        batch_graph = GGNNBatchGraph()
        for entry in taken_entries:
            batch_graph.add_subgraph(
                copy.deepcopy(entry.graph),
                hyper_edges=entry.hyper_edges,  # 新增
                text_feature=self.text_features[entry.origin_id] if self.text_features else None
            )
        return batch_graph, torch.FloatTensor(labels), ids

    def get_next_train_batch(self):  # 按批次返回图数据

        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()

        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()

        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)
