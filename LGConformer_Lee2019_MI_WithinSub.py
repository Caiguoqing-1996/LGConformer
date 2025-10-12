#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :CV_in_BNCI2014_001.py
# @Time      :2024/1/18 15:12
# @Author    :Guoqing Cai

'''
Usage:
    BNCI2014_001:

'''


from DataPrepare import PrepareData
from ModelPrepare import model_prepare
from prepocess_in_model import filter_EEG
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import itertools
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, cohen_kappa_score
from train_model_withMean_oneStage import train_in_one_fold


CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.constant(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)



def CrossValidation_KU(data_sel, model_sel, model_para, train_para):
    data_prepare = PrepareData(data_sel= data_sel, win_classify=train_para['win_classify'])
    alldata = {'X': [],
               'Y': []}

    all_acc_test, all_model, all_precision, all_recall, all_f1, all_kappa = [], [], [], [], [], []

    for subi in data_prepare.sub_name:
        train_test = data_prepare.load_data_sub(target_sub=subi)
        train_data = filter_EEG(model_sel, data_sel, train_test['train'], train_test['fs'],
                                train_test['win_sel'], train_para['win_classify'])
        test_data = filter_EEG(model_sel, data_sel, train_test['test'], train_test['fs'],
                               train_test['win_sel'], train_para['win_classify'])

        fs = train_data['fs']
        n_repeat = train_para['n_repeat']
        acc_test_repeati = []
        precision_repeati = []
        recall_repeati = []
        f1_repeati = []
        kappa_repeati = []

        for repeati in range(n_repeat):
            print('*' * 100)
            print('Sub: ' + str(subi))
            print('model: ' + model_sel)
            print('repeati: ' + str(repeati))
            print('*' * 100)

            model_para['channel'] = 62
            model_para['time'] = (int(train_para['win_classify'][1] * fs) - int(train_para['win_classify'][0] * fs))
            model_para['class'] = 2
            model_para['fs'] = fs

            model = model_prepare(model_sel, model_para)
            model.apply(weights_init)
            acc_test, last_model, precision, f1, kappa \
                = train_in_one_fold(train_set_all=train_data, test_set=test_data,
                                    model=model, train_para=train_para)

            acc_test_repeati.append(acc_test)
            precision_repeati.append(precision)
            f1_repeati.append(f1)
            kappa_repeati.append(kappa)
            all_model.append(last_model)

        all_acc_test.append(np.mean(np.array(acc_test_repeati)))
        all_precision.append(np.mean(np.array(precision_repeati)))
        all_f1.append(np.mean(np.array(f1_repeati)))
        all_kappa.append(np.mean(np.array(kappa_repeati)))
    return all_acc_test, all_model, all_precision, all_f1, all_kappa






if __name__ == "__main__":
    '''
        'FBCNet', 'EEGNet', 'DeepNet', 'EEGConformer', 'ShallowNet', 'LDMANet'
        'IFNet', 'TSception', 'SMT_2a'
    
    '''


    for i in range(100):
        torch.cuda.empty_cache()
    print(torch.cuda.is_available() )

    data_sel = 'Lee2019_MI'
    train_para = {'win_classify': [0,  4],
                  'lr': 0.001,
                  'batch_size': 64,
                  'n_repeat': 15,
                  'train_prop': 0.8,
                  'second_epoch': 100,
                  'stop_criteria': 'accuracy',
                  'min_second_epoch': 70,}

    n_layer_all = [2]
    spa_dim_all = [50]


    excel_path = 'KU_STformer_CrossSession.xlsx'
    model_sel_all = ['MSConformer', 'MSConformer',]

    df = pd.DataFrame(
        columns=['model_sel', 'n_layer', 'spa_dim', 'acc_all', 'mean_acc', 'best_acc_all', 'best_mean_acc'])

    for model_sel, spa_dim, n_layer in itertools.product(model_sel_all, spa_dim_all, n_layer_all):
        ACC_epoch, Precision_epoch, Recall_epoch, F1_epoch, Kappa_epoch = [], [], [], [], []
        for i in range(100):
            torch.cuda.empty_cache()
        model_para = {}
        model_para['n_layer'] = n_layer
        model_para['spa_dim'] = spa_dim

        train_para['lr'] = 0.001
        train_para['first_epochs'] = 1000
        train_para['patience'] = 100
        train_para['min_train_epoch'] = 100

        all_acc_test, best_model, all_precision, all_f1, all_kappa \
            = CrossValidation_KU(data_sel, model_sel, model_para, train_para)

        torch.save(best_model, 'LGConformer_KU_bestmodels.pth')


        results_set = {'acc': all_acc_test,
                       'precision': all_precision,
                       'f1': all_f1,
                       'kappa': all_kappa}

        df = df.append({'model_sel': model_sel,
                        'n_layer': n_layer,
                        'spa_dim': spa_dim,
                        'best_acc_all': np.round(all_acc_test, 5),
                        'best_mean_acc': [np.round(np.mean(all_acc_test), 4), np.round(np.std(all_acc_test), 4)],
                        'best_kappa_all': np.round(all_kappa, 5),
                        'best_mean_kappa': [np.round(np.mean(all_kappa), 4), np.round(np.std(all_kappa), 4)],
                        'best_acc_f1': np.round(all_f1, 5),
                        'best_mean_f1': [np.round(np.mean(all_f1), 4), np.round(np.std(all_f1), 4)],
                        }, ignore_index=True)

        print(df)
        df.to_excel(excel_path, index=False)



