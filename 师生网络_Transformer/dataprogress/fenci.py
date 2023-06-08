# -*- coding: utf-8 -*-

"""
@author: 赫凯
@software: PyCharm
@file: fenci.py
@time: 2023/2/23 16:09
"""

import jieba
import numpy as np

import util.configer as df

from tqdm import tqdm

if __name__ == '__main__':

    print(len(np.load(f'{df.f_path}/data/symbols.npy')))

    fc_arr = []

    t_arr = np.load(f'{df.f_path}/data/text_data.npy', allow_pickle=True)
    for t in tqdm(t_arr):
        fc = jieba.lcut(t, cut_all=True)
        for fc_i in fc:
            if fc_i not in fc_arr:
                fc_arr.append(fc_i)

    np.save(f'{df.f_path}/data/symbols.npy', fc_arr)