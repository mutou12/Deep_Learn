# -*- coding: utf-8 -*-

"""
@author: 赫凯
@software: PyCharm
@file: tools.py
@time: 2023/2/23 17:02
"""

import numpy as np


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded