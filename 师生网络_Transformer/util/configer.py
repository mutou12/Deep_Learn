# -*- coding: utf-8 -*-

"""
@author: 赫凯
@software: PyCharm
@file: configer.py
@time: 2023/2/23 14:23
"""

from enum import Enum

f_path = 'C:/Project_Code/5G专利/专利代码'
epochs = 1000

batch_size = 20

vocab_size = 11459
d_model = 32
N = 3
heads = 4
dropout = 0.5


class state(Enum):
    train = 'train'
    eval = 'validation'
    test = 'test'
