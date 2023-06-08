# -*- coding: utf-8 -*-

"""
@author: 赫凯
@software: PyCharm
@file: model_student.py
@time: 2023/2/26 21:09
"""

import numpy as np
import torch.nn as nn
import torch
import math

from torch.autograd import Variable
import torch.nn.functional as F
import copy

from dataprogress.dataset import Num_Text_dataset


class Embedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class NormLayer(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # 使用两个可以学习的参数来进行 normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # mask掉那些为了padding长度增加的token，让其通过softmax计算后为0
    if mask is not None:
        mask = mask.unsqueeze(-2).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # 根据pos和i创建一个常量pe矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 让 embeddings vector 相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到 embedding 中
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], \
                         requires_grad=False)
        return x


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class student_model(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super(student_model, self).__init__()
        self.N = N
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = NormLayer(d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=64, num_layers=2)

        self.fc_2 = nn.Linear(in_features=64, out_features=32)
        self.ln_2 = nn.LayerNorm(32)
        self.ac_2 = nn.LeakyReLU()
        self.drop_2 = nn.Dropout(0.5)

        self.fc_3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        output, hn = self.gru(x.permute(1, 0, 2))
        h = hn[-1:]  # 用最后一个hidden layer的结果
        x = torch.cat(h.split(1), dim=-1).squeeze(0)
        out1 = self.drop_2(self.ac_2(self.ln_2(self.fc_2(x))))
        out = self.fc_3(out1)
        return out1, out.reshape(-1)

    @staticmethod
    def _metrics(true_data, pred_data):
        # 损失
        loss = nn.MSELoss()(pred_data, true_data)

        # MAE
        mae = nn.L1Loss()(pred_data, true_data)

        # MSE
        mse = nn.MSELoss()(pred_data, true_data)

        # # R2
        # r2 = r2_score(true_data.detach().numpy(), pred_data.detach().numpy())

        return loss, mae, mse

    @staticmethod
    def soft_metrics(y_l, t_l):
        # 损失
        loss = nn.MSELoss(reduction='sum')(y_l, t_l)

        return loss


from util import configer as df
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    m = student_model(11458, 64, 2, 4, 0.5)

    trainDataSet = Num_Text_dataset(state=df.state.eval)
    trainDataLoader = DataLoader(trainDataSet, batch_size=12,
                                 num_workers=3, collate_fn=trainDataSet.collection)
    par = tqdm(trainDataLoader)

    for i, f in enumerate(par):
        print(i, f)
        m(f[0], f[1])

