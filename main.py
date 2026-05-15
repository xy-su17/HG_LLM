import argparse
import logging
import os
import pickle
import sys

os.chdir(sys.path[0])
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from data_loader.dataset import DataSet
from trainer import train
from utils import tally_param, debug, set_logger

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
import math
from torch.optim.optimizer import Optimizer
from modules.model import  MultimodalEnhancedHyperModel
import torch.optim

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-6,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        beta2_t = None
        ratio = None
        N_sma_max = None
        N_sma = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                if beta2_t is None:
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    beta1_t = 1 - beta1 ** state['step']
                    if N_sma >= 5:
                        ratio = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / beta1_t

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma >= 5:
                    step_size = group['lr'] * ratio
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / beta1_t
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    name = 'devign_all_code_256'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.', default='/data/AIinspur02/linshi001/output/hyper_w2v_RealHyper_model3')  # 在models文件夹下面，保存模型及日志位置，会被覆盖
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser', default='/data/AIinspur02/linshi001/dataset/hyper_w2v_7edges_RealHyper0403')  #  数据所在文件
    parser.add_argument('--log_dir', default=name+'layer1_1.7Multi.log', type=str)
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=64)
    parser.add_argument('--use_multimodal', action='store_true',default=True,
                       help='Use multimodal fusion with RoBERTa')
    parser.add_argument('--text_features_path', type=str,
                        default='/data/AIinspur02/linshi001/dataset/huggingface_libs/' + name + '.npy',
                        help='Path to pre-extracted text features')
    args = parser.parse_args()

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(model_dir, args.log_dir)
    set_logger(log_dir)
    
    logging.info('Check up feature_size: %d', args.feature_size)
    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        logging.info('Warning!!! Graph Embed dimension should be at least equal to the feature dimension')
        args.graph_embed_size = args.feature_size

    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, name+'.bin')
    logging.info('#' * 100)
    if True and os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        logging.info('Reading already processed data from %s!' % processed_data_path)
    else:
        logging.info('Loading the dataset from %s' % input_dir)
        dataset = DataSet(train_src=os.path.join(input_dir, 'devign-train-v2.json'),
                          valid_src=os.path.join(input_dir, 'devign-valid-v2.json'),
                          test_src=os.path.join(input_dir, 'devign-test-v2.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag,input_dir=input_dir, text_features_path=args.text_features_path)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)   
        file.close()
    logging.info('train_dataset: %d; valid_dataset: %d; test_dataset: %d', len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    logging.info("train_batch: %d, valid_batch: %d, test_batch: %d", len(dataset.train_batches), len(dataset.valid_batches), len(dataset.test_batches))
    logging.info('#' * 100)

    model = MultimodalEnhancedHyperModel(
        input_dim=dataset.feature_size,
        output_dim=100,  # 与graph_embed_size对齐
        max_edge_types=dataset.max_edge_type,
        text_dim=256  # 文本特征维度
    )

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    logging.info('Total Parameters : %d' % tally_param(model))
    logging.info('#' * 100)
    model.cuda()

    loss_function = CrossEntropyLoss(
        weight=torch.tensor([1, 1.7]).float(),
        reduction='sum'
    )

    loss_function.cuda()

    LR = 5e-5
    optim = RAdam(model.parameters(),lr=LR,weight_decay=1e-4,betas=(0.9, 0.98))
    train(model=model, dataset=dataset, epoches=100, dev_every=len(dataset.train_batches),
          loss_function=loss_function, optimizer=optim,
          save_path=model_dir + '/MultimodalModel',
          use_multimodal=args.use_multimodal,  
          contrast_weight=0.1, max_patience=30)

