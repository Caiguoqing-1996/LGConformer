#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :ModelPrepare.py
# @Time      :2024/1/18 9:26
# @Author    :Guoqing Cai

'''
Usage:


'''


def model_prepare(model_sel, model_para):

    if model_sel in ['EEGNet']:
        from compared_model.EEGNet import EEGNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = EEGNet(nChan=nChan, nTime=nTime, nClass=nClass)


    elif model_sel in ['MSConformer']:
        from LGConformer import STformer
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = STformer(n_chan=nChan, n_time=nTime, num_classes=nClass, para=model_para)


    return model






if __name__ == "__main__":
    run_code = 0
