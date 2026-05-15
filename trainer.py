import copy
import logging
import os.path

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils import debug

def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets.long())
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), accuracy_score(all_targets, all_predictions) * 100
    pass

def evaluate_metrics(model, loss_function, num_batches, data_iter, epoch, log_name, save_path):
    model.eval() 
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets, all_ids = [], [], []
        for _ in range(num_batches):
            graph, targets, ids = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets.long())
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2: 
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else: 
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            all_ids.extend(ids)
        model.train() 
        if log_name in ['valid', 'test', 'end_test']:
            if not os.path.exists(os.path.join(save_path)):
                os.makedirs(os.path.join(save_path))
            save_pd = pd.DataFrame({'ids':all_ids,'targets':all_targets,'predictions':all_predictions})
            save_pd.to_csv(os.path.join(save_path, log_name + '_' + str(epoch) +'.csv'),index=None,encoding='utf8')
        return np.mean(_loss).item(), \
               accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


def train(model, dataset, epoches, dev_every, loss_function, optimizer, save_path,
          log_every=5, max_patience=5, use_multimodal=False, contrast_weight=0.1):
    debug('Start Training')
    logging.info('Start training!')
    logging.info(f"Using model: {model.__class__.__name__}")

    if use_multimodal:
        logging.info("Using multimodal fusion with contrastive learning")
        logging.info(f"Contrastive loss weight: {contrast_weight}")
    max_steps = epoches * dev_every
    log_flag = 0
    patience_counter = 0
    best_f1 = 0
    train_losses = []
    all_train_acc = []
    all_train_loss = []
    all_valid_acc = []
    all_valid_loss = []
    best_model = None
    try:
        for step_count in range(max_steps):
            model.train()
            model.zero_grad()

            graph, targets, ids = dataset.get_next_train_batch()
            targets = targets.cuda()

            if use_multimodal:
                predictions, contrast_loss = model(graph, cuda=True, mode='train')
                cls_loss = loss_function(predictions, targets.long())
                total_loss = cls_loss + contrast_weight * contrast_loss
            else:
                predictions = model(graph, cuda=True)
                total_loss = loss_function(predictions, targets.long())

            train_losses.append(total_loss.detach().item())
            total_loss.backward()
            optimizer.step()

            if step_count % dev_every == (dev_every - 1):

                log_flag += 1
                train_loss, train_acc, train_pr, train_rc, train_f1 = evaluate_metrics(model, loss_function, dataset.initialize_train_batch(), dataset.get_next_train_batch, log_flag, 'train', save_path)
                all_train_acc.append(train_acc)
                all_train_loss.append(train_loss)

                logging.info('-' * 100)
                logging.info('Epoch %d\t---Train--- Average Loss: %10.4f\t Patience %d\t Loss: %10.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tf1: %5.3f\t' % (
                    log_flag, np.mean(train_losses).item(), patience_counter, train_loss, train_acc, train_pr, train_rc, train_f1))
                loss, acc, pr, rc, valid_f1 = evaluate_metrics(model, loss_function, dataset.initialize_valid_batch(), dataset.get_next_valid_batch,log_flag, 'valid',save_path)
                logging.info('Epoch %d\t----Valid---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (log_flag, loss, acc, pr, rc, valid_f1))

                test_loss, test_acc, test_pr, test_rc, test_f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(), dataset.get_next_test_batch,log_flag, 'test',save_path)
                logging.info('Epoch %d\t----Test---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (log_flag, test_loss, test_acc, test_pr, test_rc, test_f1))
                all_valid_acc.append(acc)
                all_valid_loss.append(loss)
                if valid_f1 > best_f1:
                    logging.info('Epoch %d----Best---- ' % log_flag)
                    patience_counter = 0
                    best_f1 = valid_f1
                    best_model = copy.deepcopy(model.state_dict())
                    _save_file = open(save_path + str(log_flag) + '-model.bin', 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                else:
                    patience_counter += 1
                train_losses = []
                if patience_counter == max_patience:
                    break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')
        logging.info('Training Interrupted by user!')
    logging.info('Finish training!')

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open(save_path + '-model.bin', 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()

    logging.info('#' * 100)
    logging.info("Test result")
    loss, acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch,log_flag, 'end_test', save_path)
    debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    logging.info('%s\t----Test---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (save_path, loss, acc, pr, rc, f1))



