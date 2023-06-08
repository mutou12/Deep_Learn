# -*- coding: utf-8 -*-

"""
@author: 赫凯
@software: PyCharm
@file: model_teacher.py
@time: 2023/2/21 17:01
"""

import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import r2_score


class teacher_model(nn.Module):
    def __init__(self):
        super(teacher_model, self).__init__()
        self.fc_1 = nn.Linear(in_features=17, out_features=160)
        self.ln_1 = nn.LayerNorm(160)
        self.ac_1 = nn.LeakyReLU()
        self.drop_1 = nn.Dropout(0.5)

        self.fc_2 = nn.Linear(in_features=160, out_features=32)
        self.ln_2 = nn.LayerNorm(32)
        self.ac_2 = nn.LeakyReLU()
        self.drop_2 = nn.Dropout(0.5)

        self.fc_3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, data):
        x = self.drop_1(self.ac_1(self.ln_1(self.fc_1(data))))
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


from util import configer

if __name__ == '__main__':
    m = teacher_model()
    m.load_state_dict(torch.load(fr'{configer.f_path}/checkpoints/Sun_Feb_26_10_17_47_2023/model_873_0.84835_0.84895.pt')['model'])

    a = np.random.random(3 * 17).reshape([3, 17])
    a = torch.tensor(a, dtype=torch.float)
    a = m(a)
    print(a)
