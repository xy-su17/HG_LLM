import dgl
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from graph_transformer_layers_new import GraphTransformerLayer
from mlp_readout import MLPReadout


class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, small_kernel, large_kernel, stride, groups):
        super().__init__()
        self.large_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=large_kernel, stride=stride,
                                          padding=large_kernel // 2, groups=groups, dilation=1, bias=True)
        self.large_bn = torch.nn.BatchNorm1d(out_channels)
        self.small_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=small_kernel, stride=stride,
                                           padding=small_kernel // 2, groups=groups, dilation=1)
        self.small_bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        large_out = self.large_conv(inputs.clone())
        large_out = self.large_bn(large_out)
        small_out = self.small_conv(inputs.clone())
        small_out = self.small_bn(small_out)
        return large_out + small_out

HYPER_CALL_EDGE_TYPE = 6
HYPER_LOOP_EDGE_TYPE = 7
HYPER_IF_EDGE_TYPE = 8
HYPER_SWITCH_EDGE_TYPE = 9
HYPER_RETURN_EDGE_TYPE = 10
HYPER_PARAM_EDGE_TYPE = 11

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, gnn_dim=200, text_dim=200, hidden_dim=128, num_heads=4):
        super().__init__()
        self.gnn_dim = gnn_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(hidden_dim, gnn_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, gnn_features, text_features):
        original_dim = len(gnn_features.shape)

        gnn_proj = self.gnn_proj(gnn_features)
        text_proj = self.text_proj(text_features).unsqueeze(1)

        if original_dim == 2:
            gnn_proj = gnn_proj.unsqueeze(1)

        if len(gnn_proj.shape) == 3: 
            num_nodes = gnn_proj.size(1)
            text_proj = text_proj.expand(-1, num_nodes, -1)

        attended, _ = self.cross_attn(
            query=gnn_proj,
            key=text_proj,
            value=text_proj
        )


        gate = self.gate_layer(torch.cat([gnn_proj, attended], dim=-1))
        fused = gate * gnn_proj + (1 - gate) * attended

        output = self.output_proj(fused)
        output = self.dropout(output)

        if original_dim == 2:
            output = output.squeeze(1)
        return output


class HierarchicalMultimodalFusion(nn.Module):
    def __init__(self, gnn_dim, text_dim, num_heads=4):
        super().__init__()

        self.node_fusion = CrossModalAttentionFusion(
            gnn_dim=gnn_dim,
            text_dim=text_dim,
            hidden_dim=128,
            num_heads=num_heads
        )

        self.graph_fusion = CrossModalAttentionFusion(
            gnn_dim=gnn_dim,
            text_dim=text_dim,
            hidden_dim=128,
            num_heads=num_heads
        )

        self.weight_learner = nn.Sequential(
            nn.Linear(gnn_dim * 2, gnn_dim),
            nn.ReLU(),
            nn.Linear(gnn_dim, 1),
            nn.Sigmoid()
        )

        self.node_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, gnn_node_features, gnn_graph_features, text_features):
        fused_nodes = self.node_fusion(gnn_node_features, text_features)
        node_pooled = self.node_pool(fused_nodes.transpose(1, 2)).squeeze(-1)
        fused_graph = self.graph_fusion(gnn_graph_features, text_features)

        alpha = self.weight_learner(torch.cat([node_pooled, fused_graph], dim=-1))
        final_features = alpha * node_pooled + (1 - alpha) * fused_graph

        return final_features, fused_nodes


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class HypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(HypergraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.gate_weight = nn.Parameter(torch.Tensor(in_channels, 1))
        self.gate_bias = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.gate_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.gate_bias)

    def forward(self, x, hyper_edges_list, hyper_edge_types=None):
        num_nodes = x.size(0)
        out = torch.matmul(x, self.weight) 

        if hyper_edges_list and len(hyper_edges_list) > 0:
            num_hyperedges = len(hyper_edges_list)
            H = torch.zeros(num_nodes, num_hyperedges, device=x.device)

            for e_idx, nodes in enumerate(hyper_edges_list):
                if isinstance(nodes, list):
                    node_indices = nodes
                else:
                    try:
                        node_indices = nodes.tolist()
                    except:
                        continue

                if len(node_indices) == 0:
                    continue

                weight = 1.0 / len(node_indices)
                for node_idx in node_indices:
                    if 0 <= node_idx < num_nodes:
                        H[node_idx, e_idx] = weight

            D_v_diag = H.sum(dim=1)  
            D_e_diag = H.sum(dim=0) 

            D_v_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_v_diag + 1e-8))
            D_e_inv = torch.diag(1.0 / (D_e_diag + 1e-8))

            term1 = torch.matmul(D_v_inv_sqrt, H) 
            term2 = torch.matmul(term1, D_e_inv)  
            term3 = torch.matmul(term2, H.t()) 
            L = torch.matmul(term3, D_v_inv_sqrt) 

            hyper_out = torch.matmul(L, out)

            x_mean = x.mean(dim=0, keepdim=True) 
            gate_input = torch.matmul(x_mean, self.gate_weight) + self.gate_bias 
            alpha = torch.sigmoid(gate_input) 

            out = alpha * out + (1 - alpha) * hyper_out

        if self.bias is not None:
            out = out + self.bias

        return out

class MultimodalEnhancedHyperModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, text_dim=200):
        super().__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types

        n_layers = 2
        num_head = 5  
        ffn_ratio = 2
        small_kernel = 3
        large_kernel = 11
        drout_out = 0.1
        k = 3
        hyper_interaction_layer = 3

        self.gtn = nn.ModuleList([GraphTransformerLayer(input_dim, output_dim, num_heads=num_head,
                                                        dropout=0.1, max_edge_types=max_edge_types, layer_norm=False,
                                                        batch_norm=True, residual=True)
                                  for _ in range(n_layers - 1)])
        self.MPL_layer = MLPReadout(output_dim, 2)
        self.sigmoid = nn.Sigmoid()
        self.concat_dim = output_dim

        self.RepLK = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.concat_dim),
            torch.nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0, groups=1,
                            dilation=1),
            torch.nn.ReLU(),
            ReparamLargeKernelConv(in_channels=self.concat_dim * ffn_ratio, out_channels=self.concat_dim * ffn_ratio,
                                   small_kernel=small_kernel, large_kernel=large_kernel, stride=1,
                                   groups=self.concat_dim * ffn_ratio),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0, groups=1,
                            dilation=1),
        )
        self.Avgpool1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(k, stride=k),
            torch.nn.Dropout(drout_out)
        )
        self.ConvFFN = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.concat_dim),
            torch.nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0, groups=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0,
                            groups=1),
        )
        self.Avgpool2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(k, stride=k),
            torch.nn.Dropout(drout_out)
        )
        self.hyper_edge_types = [HYPER_CALL_EDGE_TYPE,
                                 HYPER_LOOP_EDGE_TYPE,
                                 HYPER_IF_EDGE_TYPE,
                                 HYPER_SWITCH_EDGE_TYPE,
                                 HYPER_RETURN_EDGE_TYPE,
                                 HYPER_PARAM_EDGE_TYPE
                                 ]
        self.hyper_attn_layers = nn.ModuleList([
            GraphTransformerLayer(
                input_dim=output_dim,
                output_dim=output_dim,
                max_edge_types=max_edge_types,
                num_heads=num_head,
                dropout=drout_out
            ) for _ in range(len(self.hyper_edge_types))
        ])

        self.hyper_interaction = nn.ModuleList([
            GraphTransformerLayer(
                input_dim=output_dim,
                output_dim=output_dim,
                max_edge_types=max_edge_types,
                num_heads=num_head,  # 多头注意力
                dropout=drout_out
            ) for _ in range(hyper_interaction_layer) 
        ])

        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )

        self.text_dim = text_dim

        self.multimodal_fusion = HierarchicalMultimodalFusion(
            gnn_dim=output_dim,
            text_dim=text_dim,
            num_heads=4  
        )

        self.temperature = 0.07

        self.gnn_align = nn.Linear(output_dim, 128)
        self.text_align = nn.Linear(text_dim, 128)

        self.adaptive_pool1 = nn.AdaptiveAvgPool1d(1)
        self.adaptive_pool2 = nn.AdaptiveAvgPool1d(1)

        self.hyper_conv1 = HypergraphConv(
            in_channels=output_dim,
            out_channels=output_dim
        )

        self.hyper_conv2 = HypergraphConv(
            in_channels=output_dim,
            out_channels=output_dim
        )

        self.hyper_edge_type_map = {
            'HYPER_CALL': 0,
            'HYPER_LOOP': 1,
            'HYPER_IF': 2,
            'HYPER_SWITCH': 3,
            'HYPER_RETURN': 4,
            'HYPER_PARAM': 5,
            'HYPER_DATAFLOW': 6,
            'HYPER_CONTROL': 7
        }

        self.hyper_type_embed = nn.Embedding(
            num_embeddings=8,
            embedding_dim=output_dim
        )

        self.hyper_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.hyper_pool = nn.AdaptiveAvgPool1d(1)

        self.hyper_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )
    def contrastive_loss(self, z1, z2):
        batch_size = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits = torch.mm(z1, z2.T) / self.temperature
        labels = torch.arange(batch_size, device=z1.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def process_hyper_edges(self, node_features, hyper_edges_list):
        if not hyper_edges_list or len(hyper_edges_list) == 0:
            return node_features

        all_hyper_nodes = []

        for he in hyper_edges_list:
            if isinstance(he, dict) and 'nodes' in he:
                nodes = he['nodes']
            elif isinstance(he, list):
                nodes = he
            else:
                continue

            valid_nodes = []
            for idx in nodes:
                if isinstance(idx, (int, np.integer)):
                    node_idx = int(idx)
                elif isinstance(idx, torch.Tensor):
                    node_idx = idx.item()
                else:
                    try:
                        node_idx = int(idx)
                    except:
                        continue

                if 0 <= node_idx < node_features.size(0):
                    valid_nodes.append(node_idx)

            if len(valid_nodes) >= 2:
                all_hyper_nodes.append(valid_nodes)

        if not all_hyper_nodes:
            return node_features

        hyper_out = self.hyper_conv1(node_features, all_hyper_nodes)
        hyper_out = F.relu(hyper_out)
        hyper_out = self.hyper_conv2(hyper_out, all_hyper_nodes)

        output = node_features + hyper_out

        return output

    def forward(self, batch, cuda=False, text_features=None, mode='eval', return_features=False):
        graph, features, edge_types, batch_text_features, hyper_edges = batch.get_network_inputs(cuda=cuda)

        if text_features is None and batch_text_features is not None:
            text_features = batch_text_features

        graph = graph.to(torch.device('cuda:0'))

        for conv in self.gtn:
            features = conv(graph, features, edge_types)

        if hyper_edges:
            all_hyper_edges = []
            for he_list in hyper_edges:
                if isinstance(he_list, list):
                    all_hyper_edges.extend(he_list)

            if all_hyper_edges:
                features = self.process_hyper_edges(features, all_hyper_edges)

        hyper_feat_dict = {}
        for edge_type in self.hyper_edge_types:
            edge_mask = (edge_types == edge_type)
            if torch.any(edge_mask):
                subgraph = graph.edge_subgraph(edge_mask, relabel_nodes=False)
                node_ids = subgraph.nodes()
                sub_features = features[node_ids]

                attn_out = sub_features
                for attn_layer in self.hyper_attn_layers:
                    attn_out = attn_layer(subgraph, attn_out, edge_types[edge_mask])

                hyper_feat_dict[edge_type] = (node_ids, attn_out)

        if hyper_feat_dict:
            existing_types = list(hyper_feat_dict.keys())

            hyper_graph = dgl.DGLGraph()
            hyper_graph.add_nodes(len(existing_types))

            src, dst = [], []
            for i in range(len(existing_types)):
                for j in range(i + 1, len(existing_types)):
                    src.append(i)
                    dst.append(j)
                    src.append(j)
                    dst.append(i)
            hyper_graph.add_edges(src, dst)
            hyper_graph = hyper_graph.to(features.device)

            hyper_feat_matrix = torch.stack([
                hyper_feat_dict[et][1].mean(dim=0)
                for et in existing_types
            ])

            enhanced_hyper_feat = hyper_feat_matrix
            for layer in self.hyper_interaction:
                enhanced_hyper_feat = layer(
                    hyper_graph,
                    enhanced_hyper_feat,
                    torch.zeros(hyper_graph.number_of_edges(), device=features.device)
                )

            for i, edge_type in enumerate(existing_types):
                node_ids, orig_feat = hyper_feat_dict[edge_type]
                combined = torch.cat([orig_feat, enhanced_hyper_feat[i].expand_as(orig_feat)], dim=-1)
                gate = self.fusion_gate(combined)
                features[node_ids] = gate * orig_feat + (1 - gate) * enhanced_hyper_feat[i].expand_as(orig_feat)

        if text_features is not None:
            graph_features_batch = batch.de_batchify_graphs(features)
            graph_features = graph_features_batch.sum(dim=1)  # [batch_size, output_dim]

            fused_graph_features, _ = self.multimodal_fusion(
                gnn_node_features=graph_features_batch,
                gnn_graph_features=graph_features,
                text_features=text_features
            )

            outputs = fused_graph_features.unsqueeze(1).transpose(1, 2)
            outputs = self.adaptive_pool1(outputs)
            outputs = self.adaptive_pool2(outputs)
            outputs = outputs.transpose(1, 2) 
        else:
            outputs = batch.de_batchify_graphs(features)
            outputs = outputs.transpose(1, 2)
            outputs = outputs + self.RepLK(outputs)
            outputs = self.Avgpool1(outputs)
            outputs = outputs + self.ConvFFN(outputs)
            outputs = self.Avgpool2(outputs)
            outputs = outputs.transpose(1, 2)


        outputs = self.MPL_layer(outputs.sum(dim=1))
        if mode == 'train' and text_features is not None:
            gnn_aligned = self.gnn_align(fused_graph_features)
            text_aligned = self.text_align(text_features)
            contrast_loss = self.contrastive_loss(gnn_aligned, text_aligned)

            return nn.Softmax(dim=1)(outputs), contrast_loss

        if return_features:
            return nn.Softmax(dim=1)(outputs), fused_graph_features
        else:
            return nn.Softmax(dim=1)(outputs)