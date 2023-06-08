# -*- coding: utf-8 -*-

"""
@author: 赫凯
@software: PyCharm
@file: data_progress.py
@time: 2023/2/21 17:02
"""

import pandas as pd
import numpy as np

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

from sklearn import preprocessing


def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))


if __name__ == '__main__':
    data_ = pd.read_csv('C:/Project_Code/Jupyter/5gjizahn/ct20220606150645303.csv')
    reduce_mem_usage(data_)

    num_data = data_[
        ['站间距', '所有站点的站间距', '同频段站点的站间距', '同网络站点的站间距', '覆盖面积值', '问题点关联数量值',
         '站型', '网络类型', '场景值', '区域类型值', '建设类型值', '覆盖目标重要程度值', '高度维度网络结构值',
         '距离维度网络结构值', '网络结构值', '需求频段类型值', '问题点类型值']]
    num_col = ['站间距', '所有站点的站间距', '同频段站点的站间距', '同网络站点的站间距', '覆盖面积值']
    label_col = ['站型', '网络类型', '场景值', '区域类型值', '建设类型值', '覆盖目标重要程度值', '高度维度网络结构值',
                 '距离维度网络结构值', '网络结构值', '需求频段类型值', '问题点类型值', '问题点关联数量值']

    num_data1 = num_data[num_col].fillna(0.0)
    num_data2 = num_data[label_col].fillna('0.0')

    num_data1 = num_data1.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

    num_data = num_data1.merge(num_data2, how='left', left_index=True, right_index=True)

    for x in label_col:
        label = preprocessing.LabelEncoder()
        num_data[x] = label.fit_transform(num_data[x])  ##数据标准化
