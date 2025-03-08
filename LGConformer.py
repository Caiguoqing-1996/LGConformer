#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :STformerCrossSub.py
# @Time      :2024/10/20 21:54
# @Author    :Guoqing Cai

'''
Usage:


'''

'''
Usage:

目标：时空全局信息学习，脑区

思路构建：
1 全局刚性对齐，与局部柔性配准：选择核心节点，并使用多尺度空间信息聚合,

2 局部空间信息聚合方法： 这里讨论4种
    a 直接平均
    b 可学习参数的 a1*X1 + a2*X2 + a3*X3
    c cat方式的
    d 基于注意力方式的：可以使用IFNet中的方法或者  论文: Exploring the Applicability of Transfer Learning 
      and Feature Engineering in Epilepsy Prediction Using Hybrid Transformer Model

3 时间信息聚合

4  基于transformer和卷积的全局-局部信息挖掘


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropkey_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropkey_rate = dropkey_rate

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout_softmax = nn.Dropout(dropkey_rate)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        # attn_scores = torch.softmax(attn_scores, dim=-1)
        # output = torch.matmul(attn_scores, V)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        # attn_probs = self.dropout_softmax(attn_probs)
        output = torch.matmul(attn_probs, V)

        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def dropkey(self, K):
        if self.training:
            mask = torch.bernoulli(torch.full_like(K[:, :, :, 0], 1 - self.dropkey_rate)).unsqueeze(-1)
            K = K * mask * (1.0 / (1 - self.dropkey_rate))
        return K

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Apply DropKey
        K = self.dropkey(K)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_ffd):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ELU()
        self.drop = nn.Dropout(drop_ffd)

    def forward(self, x):
        return self.linear2(self.drop(self.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropkey_rate):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropkey_rate)
        self.feed_forward = FeedForward(d_model, d_ff, drop_ffd=0.3)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropkey_rate):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropkey_rate) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class OnlyTtransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropkey_rate):
        super(OnlyTtransformer, self).__init__()
        self.transformer = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropkey_rate)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        n_batch, n_feature, n_channel, n_time = x.shape
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.reshape(x, (n_batch * n_channel, n_time, n_feature))
        x = self.norm(x)
        x = self.transformer(x)
        x = torch.reshape(x, (n_batch, n_channel, n_time, n_feature))
        x = torch.permute(x, (0, 3, 1, 2))
        return x




class LocalVarLayer(nn.Module):
    def __init__(self, kernel_size, stride):
        super(LocalVarLayer, self).__init__()
        self.kernel_size = kernel_size  # 卷积核大小（窗口大小）
        self.stride = stride  # 步幅
        # 定义均值池化
        self.mean_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)
        # 定义平方和池化
        self.squared_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        # 计算局部均值
        mean = self.mean_pool(x)
        # 计算平方和池化
        squared = self.squared_pool(x ** 2)
        # 使用公式 variance = E(x^2) - (E(x))^2 计算局部窗口方差
        variance = squared - mean ** 2 
        return variance




class STformer(nn.Module):
    def cal_size(self, n_chan, n_time, ):
        data = torch.rand((1, self.spa_dim, len(self.channel_index), n_time))

        data = self.Spe_opt_layer(data)
        out_shape = data.shape
        return out_shape

    def __init__(self, n_chan, n_time, num_classes, para):
        super(STformer, self).__init__()

        self.nChan = n_chan
        self.nTime = n_time
        self.nClass = num_classes


        spa_dim = para['spa_dim']
        self.spa_dim = spa_dim
        self.n_layer = para['n_layer']
        pooling_kernal = 48
        pooling_stride = 48

        if self.nChan == 22:
            self.channel_index = [[1, 3, 4, 5],
                                [2, 3, 7, 8, 9, 14, 15],
                                [3, 4, 5, 9, 10, 11, 15, 16, 17],
                                [5, 6, 11, 12, 13, 17, 18],
                                [15, 16, 17, 19, 20, 21, 22],
                                ]

        if self.nChan == 62:
            self.channel_index = [[1, 2, 57, 58, 59, 60, 3, 4, 5, 6, 7],
                                    [55, 3, 45, 46, 8, 12, 47, 35, 17, 48, 18, 49, 23],
                                    [33, 9, 10, 34, 13, 36, 14, 37, 15, 39, 19, 40, 20, 41],
                                    [7, 56, 11, 51, 50, 38, 52, 16, 21, 53, 22, 27, 54],
                                    [24, 42, 25, 43, 26, 28, 61, 44, 62, 32, 29, 30, 31],
                                    ]


        self.channel_index_spa_opt = self.channel_index_spa_opt = nn.ModuleList(
            [nn.Conv2d(1, spa_dim, (len(self.channel_index[ri]), 1))
             for ri in range(len(self.channel_index))]
        )

        self.Spe_opt_layer = nn.Sequential(
            nn.BatchNorm2d(spa_dim),
            nn.Conv2d(spa_dim, spa_dim, (1, 24), stride=(1, 1), groups=spa_dim),
            nn.Conv2d(spa_dim, spa_dim, (len(self.channel_index), 1), stride=(1, 1)),
            nn.BatchNorm2d(spa_dim),
            nn.ELU(),
            LocalVarLayer((1, pooling_kernal), (1, pooling_stride)),
            nn.Dropout(0.5),
        )

        
        self.global_att = nn.ModuleList([
            OnlyTtransformer(d_model=spa_dim, num_heads=5, num_layers=1,
                             d_ff=1, dropkey_rate=0.3)
            for _ in range(self.n_layer)
        ])

        self.local_att = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(spa_dim, spa_dim, (1, 4), stride=(1, 1), padding='same', groups=spa_dim),
            nn.Conv2d(spa_dim, spa_dim, (1, 1), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(spa_dim),
            nn.ELU(),
            nn.Dropout(0.5),
            )
            for _ in range(self.n_layer)
        ])


        fea_size = self.cal_size(spa_dim, n_time)
        print('STformer CrossSub Classify feature: {}'.format(fea_size))
        self.Classify_layer = nn.Sequential(
            nn.Linear(spa_dim*fea_size[-1], 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.nClass),
            nn.LogSoftmax(dim = -1)
        )



    def forward(self, x):
        n_batch, _, n_chan, n_time = x.shape

        # 处理索引并提取通道
        x_channels = [
            self.channel_index_spa_opt[i](x[:, :, np.array(self.channel_index[i])-1, :])
            for i in range(len(self.channel_index))
        ]
        # 拼接结果
        x = torch.cat(x_channels, dim=2)

        x = self.Spe_opt_layer(x)
        for layeri in range(self.n_layer):
            g_x = self.global_att[layeri](x)
            l_x = self.local_att[layeri](x)
            x = g_x + l_x

        x = torch.flatten(x, start_dim=1)
        x = self.Classify_layer(x)
        return x



class STformer_FullSpa(nn.Module):
    def cal_size(self):
        data = torch.rand((1, 1, self.nChan, self.nTime))
        data = self.Spe_opt_layer(data)
        # data = self.time_down_sample(data)
        out_shape = data.shape
        return out_shape

    def __init__(self, n_chan, n_time, num_classes, para):
        super(STformer_FullSpa, self).__init__()

        self.nChan = n_chan
        self.nTime = n_time
        self.nClass = num_classes

        spa_dim = para['spa_dim']
        self.Spe_opt_layer = nn.Sequential(
            nn.Conv2d(1,  spa_dim, (self.nChan, 1), stride=(1, 1)),
            nn.BatchNorm2d(spa_dim),
            nn.Conv2d(spa_dim, spa_dim, (1, 24), stride=(1, 1), groups=spa_dim),
            nn.Conv2d(spa_dim, spa_dim, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(spa_dim),
            nn.ELU(),
            LocalVarLayer((1, 64), (1, 32)),
            nn.Dropout(0.5),
        )

        self.n_layer = para['n_layer']
        self.global_att = nn.ModuleList([
            OnlyTtransformer(d_model=spa_dim, num_heads=5, num_layers=1,
                             d_ff=2, dropkey_rate=0.3)
            for _ in range(self.n_layer)
        ])

        self.local_att = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(spa_dim, spa_dim, (1, 4), stride=(1, 1), padding='same', groups=spa_dim),
            nn.BatchNorm2d(spa_dim),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Conv2d(spa_dim, spa_dim, (1, 1), stride=(1, 1), padding='same'),
            )
            for _ in range(self.n_layer)
        ])


        fea_size = self.cal_size()
        print('STformer CrossSub Classify feature: {}'.format(fea_size))
        self.Classify_layer = nn.Sequential(
            nn.Linear(spa_dim*fea_size[-1], 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, self.nClass),
            nn.LogSoftmax(dim=-1)
        )



    def forward(self, x):
        n_batch, _, n_chan, n_time = x.shape
        x = self.Spe_opt_layer(x)
        for layeri in range(self.n_layer):
            g_x = self.global_att[layeri](x)
            l_x = self.local_att[layeri](x)
            x = g_x + l_x

        x = torch.flatten(x, start_dim=1)
        x = self.Classify_layer(x)
        return x










if __name__ == "__main__":
    run_code = 0
