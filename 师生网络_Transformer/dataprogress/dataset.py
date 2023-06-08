# -*- coding: utf-8 -*-

"""
@author: 赫凯
@software: PyCharm
@file: dataset.py
@time: 2023/2/23 9:34
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np

from tqdm import tqdm
import collections
import torch
from util import configer as df

from util.tools import pad_1D
import jieba


jieba.setLogLevel(jieba.logging.INFO)


class Num_Text_dataset(Dataset):
    def __init__(self, state='train'):

        symbols = np.load(f'{df.f_path}/data/symbols.npy', allow_pickle=True)
        self._symbols_to_id = {s: i+1 for i, s in enumerate(symbols)}

        num_arr = np.load(f'{df.f_path}/data/num_data.npy')
        text_arr = np.load(f'{df.f_path}/data/text_data.npy', allow_pickle=True)
        target_arr = np.load(f'{df.f_path}/data/target.npy')

        len_arr = len(num_arr)

        len_x = int(len_arr * 0.8)
        len_y = int(len_arr * 0.9)

        # 训练
        if state == df.state.train:
            self.num_ = num_arr[:len_x]
            self.text_ = text_arr[:len_x]
            self.target_ = target_arr[:len_x]
        # 验证
        elif state == df.state.eval:
            self.num_ = num_arr[len_x:len_y]
            self.text_ = text_arr[len_x:len_y]
            self.target_ = target_arr[len_x:len_y]
        # 测试
        else:
            self.num_ = num_arr[len_y:]
            self.text_ = text_arr[len_y:]
            self.target_ = target_arr[len_y:]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, index):
        t = self.text_[index]
        tt = jieba.lcut(t, cut_all=True)
        tt_ = np.array([self._symbols_to_id[t] for t in tt])
        nn = self.num_[index]
        ff = self.target_[index]
        sample = {
            'text': tt_,
            'num': nn,
            'score': ff,
            "textlength": len(t)
        }
        return sample

    # 做遮挡
    def make_mask(self, lengths, max_length=None):
        """Makes mask from list of lengths."""
        device = lengths.device if torch.is_tensor(lengths) else 'cpu'
        lengths = lengths if torch.is_tensor(lengths) else torch.tensor(lengths)
        max_length = max_length or torch.max(lengths)
        start = torch.tensor(0).int()
        indices = torch.arange(start=start, end=max_length, device=device)  # noqa
        mask = indices.lt(lengths.view(-1, 1))

        return mask

    # 给数据补0
    def collection(self, batch):
        if isinstance(batch[0], collections.Mapping):
            texts = [d['text'] for d in batch]
            nums = [d['num'] for d in batch]
            score = [d['score'] for d in batch]
            text_length = [d['textlength'] for d in batch]

            texts = [i for i, _ in sorted(zip(texts, text_length), key=lambda x: x[1], reverse=True)]
            nums = [i for i, _ in sorted(zip(nums, text_length), key=lambda x: x[1], reverse=True)]
            score = [i for i, _ in sorted(zip(score, text_length), key=lambda x: x[1], reverse=True)]

            texts_mask = self.make_mask([t.shape[-1] for t in texts])
            texts = pad_1D(texts).astype(np.int32)

            texts = torch.LongTensor(texts)
            nums = torch.FloatTensor(np.array(nums))
            score = torch.FloatTensor(score)
            texts_mask = torch.BoolTensor(texts_mask)

            return (
                texts,
                texts_mask,
                nums,
                score
            )


if __name__ == '__main__':
    trainDataSet = Num_Text_dataset(state=df.state.validation)
    trainDataLoader = DataLoader(trainDataSet, batch_size=10,
                                 num_workers=3, collate_fn=trainDataSet.collection)
    par = tqdm(trainDataLoader)

    for i, f in enumerate(par):
        print(i, f)
