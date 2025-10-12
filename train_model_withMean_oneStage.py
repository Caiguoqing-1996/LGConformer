#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :train_model.py
# @Time      :2024/1/21 16:24
# @Author    :Guoqing Cai

'''
Usage:


'''



import numpy as np
import torch
import time
import torch.nn.functional as F
from torch import optim
from sample_split import split_into_two_sets, generate_balanced_batch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from torch.utils.data import DataLoader, Dataset, ConcatDataset




CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)
    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]




def train_in_one_fold(train_set_all, test_set, model, train_para):
    batch_size = train_para['batch_size']
    first_epochs = train_para['first_epochs']
    min_train_epoch = train_para['min_train_epoch']
    patience = train_para['patience']
    stop_criteria = train_para['stop_criteria']
    lr = train_para['lr']

    train_set = MyDataset(train_set_all['X'], train_set_all['Y'])
    test_set = MyDataset(test_set['X'], test_set['Y'])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)



    if CUDA:
        model = model.to(device)

    best_model = model
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    classify_loss_fn = F.nll_loss

    best_acc = 0
    best_epoch = 999
    best_loss = 999
    # first training
    for epoch in range(first_epochs):
        epoch_in_time = time.time()



        pred_train, labels_train, loss_train, loss_cont, loss_class = [], [], [], [], []
        model.train()
        for inputs, targets in train_loader:
            if CUDA:
                inputs, targets = inputs.to(device), targets.to(device)


            optimizer.zero_grad()  # 梯度清零
            logits = model(inputs)
            loss_ce = classify_loss_fn(logits, targets)
            # 反向传播和优化
            loss_ce.backward()
            optimizer.step()
            pred_labels_batch = torch.argmax(logits, dim=1)
            pred_train.append(pred_labels_batch.cpu().numpy())
            labels_train.append(targets.cpu().numpy())
            loss_train.append(loss_ce.item())

        pred_train = np.concatenate(pred_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)
        acc_train = accuracy_score(pred_train, labels_train)
        loss_train = np.sum(np.array(loss_train))

        # 在测试集上评估模型性能
        with torch.no_grad():
            model.eval()
            labels_test, pred_test, loss_test= [], [], []
            for inputs, targets in valid_loader:
                if CUDA:
                    inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss_ce = classify_loss_fn(logits, targets)
                # 使用softmax函数转换logits为概率分布
                # probabilities = nn.Softmax(dim=1)(logits)
                # _, pred_val_batch = torch.max(probabilities, dim=1)
                pred_val_batch = torch.argmax(logits, dim=1)
                pred_test.append(pred_val_batch.cpu().numpy())
                labels_test.append(targets.cpu().numpy())
                loss_test.append(loss_ce.item())

            pred_test = np.concatenate(pred_test, axis=0)
            labels_test = np.concatenate(labels_test, axis=0)
            loss_test = np.sum(np.array(loss_test))

            acc_test = accuracy_score(pred_test, labels_test)
            precision = precision_score(pred_test, labels_test, average='macro')
            f1 = f1_score(pred_test, labels_test, average='macro')
            kappa = cohen_kappa_score(pred_test, labels_test)

        if stop_criteria in ['accuracy']:
            if ((acc_test > best_acc) or (acc_test == best_acc and loss_test < best_loss)) and (epoch > min_train_epoch):
                best_acc = acc_test
                best_precision = precision
                best_f1 = f1
                best_kappa = kappa
                best_loss = loss_test
                best_epoch = epoch + 1
                best_model = model.state_dict()

        print('Epoch:-{:03d}'.format(epoch + 1),
              '    loss_train: {:.3f}'.format(loss_train),
              'acc_train: {:.4f}'.format(acc_train),
              '    loss_test: {:.4f}'.format(loss_test),
              'acc_test: {:.4f}'.format(acc_test),
              '     best: test_acc: {:.2f}'.format(best_acc),
              'Epoch: {:03d}'.format(best_epoch),
              '    time: {:.4f}s'.format(time.time() - epoch_in_time), )

        if (epoch - best_epoch == patience) or (best_acc > 0.999):
            break

    print('*' * 100)

    return best_acc, best_model, best_precision, best_f1, best_kappa

















if __name__ == "__main__":
    run_code = 0
