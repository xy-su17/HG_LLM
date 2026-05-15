import torch
from dgl import DGLGraph

class BatchGraph:
    def __init__(self):
        self.graph = DGLGraph()
        self.number_of_nodes = 0
        self.graphid_to_nodeids = {}
        self.num_of_subgraphs = 0

    def add_subgraph(self, _g):
        assert isinstance(_g, DGLGraph)

        num_new_nodes = _g.number_of_nodes()

        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes))).to(torch.device('cuda:0'))

        self.graph.add_nodes(num_new_nodes, data=_g.ndata)

        sources, dests = _g.all_edges()

        sources = sources+self.number_of_nodes

        dests = dests+self.number_of_nodes

        self.graph.add_edges(sources, dests, data=_g.edata)

        self.number_of_nodes += num_new_nodes

        self.num_of_subgraphs += 1

    def cuda(self, device=None):
        for k in self.graphid_to_nodeids.keys():
            self.graphid_to_nodeids[k] = self.graphid_to_nodeids[k].cuda(device=device)

    def de_batchify_graphs(self, features=None):
        assert isinstance(features, torch.Tensor)

        vectors = [features.index_select(dim=0, index=self.graphid_to_nodeids[gid]) for gid in
                   self.graphid_to_nodeids.keys()]
        lengths = [f.size(0) for f in vectors]
        max_len = max(lengths)
        for i, v in enumerate(vectors):
            vectors[i] = torch.cat((v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad, device=v.device)), dim=0)
        output_vectors = torch.stack(vectors)

        return output_vectors#, lengths

    def get_network_inputs(self, cuda=False):
        raise NotImplementedError('Must be implemented by subclasses.')


class GGNNBatchGraph(BatchGraph):
    def __init__(self):
        super(GGNNBatchGraph, self).__init__()
        self.text_features = [] 
        self.hyper_edges_batch  = []  

    def add_subgraph(self, _g, hyper_edges=None, text_feature=None):
        assert isinstance(_g, DGLGraph)

        num_new_nodes = _g.number_of_nodes()
        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes))).to(torch.device('cuda:0'))

        self.graph.add_nodes(num_new_nodes, data=_g.ndata)

        sources, dests = _g.all_edges()
        sources = sources + self.number_of_nodes
        dests = dests + self.number_of_nodes
        self.graph.add_edges(sources, dests, data=_g.edata)

        if text_feature is not None:
            self.text_features.append(torch.tensor(text_feature))
        else:
            self.text_features.append(torch.zeros(256).to(torch.device('cuda:0')))

        if hyper_edges is not None:
            adjusted_hyper_edges = []
            for he in hyper_edges:
                if isinstance(he, dict):
                    adjusted_nodes = []
                    for node_idx in he.get('nodes', []):
                        adjusted_idx = node_idx + self.number_of_nodes
                        adjusted_nodes.append(adjusted_idx)

                    adjusted_he = {
                        'type': he.get('type', 'HYPER_CALL'),
                        'edge_type_id': he.get('edge_type_id', 6),
                        'nodes': adjusted_nodes
                    }
                    adjusted_hyper_edges.append(adjusted_he)
                elif isinstance(he, list):
                    adjusted_nodes = [idx + self.number_of_nodes for idx in he]
                    adjusted_hyper_edges.append(adjusted_nodes)

            self.hyper_edges_batch.append(adjusted_hyper_edges)
        else:
            self.hyper_edges_batch.append([])

        self.number_of_nodes += num_new_nodes
        self.num_of_subgraphs += 1

    def get_network_inputs(self, cuda=False, device=None):
        features = self.graph.ndata['features']
        edge_types = self.graph.edata['etype']

        if self.text_features:
            text_features = torch.stack(self.text_features) if self.text_features else None
        else:
            text_features = None
        hyper_edges = self.hyper_edges_batch
        if cuda:
            return (
                self.graph,
                features.cuda(device=device),
                edge_types.cuda(device=device),
                text_features.cuda(device=device) if text_features is not None else None,
                hyper_edges  
            )
        else:
            return (
                self.graph,
                features,
                edge_types,
                text_features,
                hyper_edges 
            )
    def get_hyper_edges(self):
        return self.hyper_edges