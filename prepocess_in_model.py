#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :prepocess_model.py
# @Time      :2024/1/18 15:21
# @Author    :Guoqing Cai

'''
Usage:


'''
import numpy as np
import torch
from scipy import signal
from scipy.signal import resample
from scipy.signal import hilbert
import matplotlib.pyplot as plt




def filter_EEG(model_Sel, data_Sel, input_data, fs, win_sel, win_classify):
    if data_Sel in ['SEED']:
        win_len = win_classify['win_len']
        win_step = win_len - win_classify['win_overlap']
    else:
        win_start = int(win_classify[0] * fs - win_sel[0] * fs)
        win_end = int(win_classify[1] * fs - win_sel[0] * fs)
        win = np.arange(win_start, win_end, 1)

    output_data = {'fs': fs}
    X_train = input_data['X']
    label_train = input_data['Y']



    if model_Sel in ['EEGNet', 'EEGConformer', 'LDMANet', 'TSception', 'ShallowNet', 'TransNet','ADFCNN',
                     'MSConformer', 'MSConformer_FullSpa', 'MSConformer_NoLG',
                     'MSConformer_OnlyL', 'MSConformer_OnlyG', 'STformer_MorePare',
                     ]:
        if data_Sel in ['BNCI2014_001', 'BCI4_2b', 'Lee2019_MI', 'PhysionetMI', 'InHouse']:
            b, a = signal.butter(5, [4 / (fs / 2), 40 / (fs / 2)], 'bandpass') # motor imagery 数据集
            X_train_filtered = signal.filtfilt(b, a, X_train, axis=2)
            X_train_filtered = X_train_filtered[:, np.newaxis, :, :]
            output_data['X'] = torch.from_numpy(X_train_filtered[:, :, :, win]).float()
            output_data['Y'] = torch.from_numpy(input_data['Y']).to(torch.long)

            for i in range(output_data['X'].shape[0]):
                sample = output_data['X'][i]
                mean = sample.mean(dim=(1, 2), keepdim=True)
                std = sample.std(dim=(1, 2), keepdim=True)
                output_data['X'][i] = (sample - mean) / std


    return output_data



if __name__ == "__main__":
    run_code = 0
