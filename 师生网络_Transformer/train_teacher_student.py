# -*- coding: utf-8 -*-

"""
@author: 赫凯
@software: PyCharm
@file: train_teacher_student.py
@time: 2023/3/1 14:35
"""


import torch
from tensorboardX import SummaryWriter
from dataprogress.dataset import Num_Text_dataset
import util.configer as df

from torch.utils.data import DataLoader

from model.model_student import student_model
from model.model_teacher import teacher_model

from tqdm import tqdm
import time

from glob import glob
import os


def train_student():
    logs_idx = time.asctime(time.localtime(time.time())).replace(":", "_").replace("  ", " ").replace(" ", "_")
    writer = SummaryWriter(log_dir=f'logs/student_{logs_idx}')

    epoch = 0
    train_global_idx = 0
    val_global_idx = 0
    test_global_idx = 0

    trainDataSet = Num_Text_dataset(state=df.state.train)
    evalDataSet = Num_Text_dataset(state=df.state.eval)
    testDataset = Num_Text_dataset(state=df.state.test)

    model = student_model(df.vocab_size, df.d_model, df.N, df.heads, df.dropout)
    model.load_state_dict(
        torch.load(
            f'{df.f_path}/checkpoints/Wed_Mar_1_20_41_40_2023/model_48_5.71542_5.66819.pt'
        )['model']
    )
    t_model = teacher_model()

    t_model.eval()
    t_model.load_state_dict(
        torch.load(
            f'{df.f_path}/checkpoints/Mon_Feb_27_15_15_57_2023/model_884_0.76053_0.77712.pt'
        )['model']
    )
    model.train()
    student_optimizer = torch.optim.NAdam(lr=1e-4, weight_decay=1e-5, eps=1e-8, params=model.parameters())

    e_s = 10000

    for e in range(epoch, df.epochs + 1):
        # 训练
        trainDataLoader = DataLoader(trainDataSet,
                                     batch_size=df.batch_size,
                                     collate_fn=trainDataSet.collection,
                                     shuffle=True,
                                     num_workers=8)

        train_par = tqdm(trainDataLoader)
        lambda_, T = 0.6, 1  # 分别为设置权重参数，T为温度系数
        mae_xx = []
        mse_xx = []
        for i, batch in enumerate(train_par):
            texts, texts_mask, nums, score = batch

            l_32, yred_score = model(texts, texts_mask)

            lt_32, _ = t_model(nums)
            # 分数损失
            score_loss, mae, mse = model._metrics(score, yred_score)

            soft_loss = model.soft_metrics(lt_32, l_32)

            score_loss = (1 - lambda_) * score_loss + lambda_ * T * T * soft_loss

            mae_xx.append(mae.item())
            mse_xx.append(mse.item())

            train_par.set_description(
                f'🌏{e}, score_loss={score_loss:.5f}, mae={sum(mae_xx) / len(mae_xx):.5f}')

            student_optimizer.zero_grad()
            score_loss.backward()
            student_optimizer.step()

            writer.add_scalar('train_student/student_loss',
                              score_loss,
                              train_global_idx)
            writer.add_scalars('train_student/student_metric',
                               {'mae': (sum(mae_xx) / len(mae_xx)), 'mse': (sum(mse_xx) / len(mse_xx))},
                               train_global_idx)

        train_global_idx += 1

        # 验证
        evalDataLoader = DataLoader(evalDataSet,
                                    batch_size=df.batch_size,
                                    collate_fn=evalDataSet.collection,
                                    shuffle=True,
                                    num_workers=1)

        # 测试
        testDataLoader = DataLoader(testDataset,
                                    batch_size=df.batch_size,
                                    collate_fn=testDataset.collection,
                                    shuffle=True,
                                    num_workers=1)

        with torch.no_grad():
            model.eval()
            mae_yy = []
            mse_yy = []
            eval_par = tqdm(evalDataLoader)
            for i, batch in enumerate(eval_par):
                texts, texts_mask, nums, score = batch

                l_32, yred_score = model(texts, texts_mask)
                # 分数损失
                score_loss, e_mae, e_mse = model._metrics(score, yred_score)
                mae_yy.append(e_mae)
                mse_yy.append(e_mse)
                eval_par.set_description(
                    f'💙{e}, score_loss={score_loss:.5f}, mae={sum(mae_yy) / len(mae_yy):.5f}')

                writer.add_scalar('eval_student/student_loss',
                                  score_loss,
                                  val_global_idx)
                writer.add_scalars('eval_student/student_metric',
                                   {'mae': (sum(mae_yy) / len(mae_yy)), 'mse': (sum(mse_yy) / len(mse_yy))},
                                   val_global_idx)

            val_global_idx += 1

            mae_cc = []
            mse_cc = []

            test_par = tqdm(testDataLoader)
            for i, batch in enumerate(test_par):
                texts, texts_mask, nums, score = batch

                l_32, yred_score = model(texts, texts_mask)
                # 分数损失
                score_loss, t_mae, t_mse = model._metrics(score, yred_score)
                mae_cc.append(t_mae)
                mse_cc.append(t_mse)
                test_par.set_description(
                    f'🚀{e}, score_loss={score_loss:.5f}, mae={sum(mae_cc) / len(mae_cc):.5f}')

                writer.add_scalar('test_student/student_loss',
                                  score_loss,
                                  test_global_idx)
                writer.add_scalars('test_student/student_metric',
                                   {'mae': (sum(mae_cc) / len(mae_cc)), 'mse': (sum(mse_cc) / len(mse_cc))},
                                   test_global_idx)

            test_global_idx += 1

        folder_path = f'checkpoints/{logs_idx}'
        if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(folder_path)

        saves = glob(f'checkpoints/{logs_idx}/*.pt')
        if len(saves) == 10:
            saves.sort(key=os.path.getmtime)
            os.remove(saves[0])
        if sum(mae_yy) / len(mae_yy) < e_s:
            torch.save({
                'epoch': e + 1,
                'global_idx': train_global_idx,
                'model': model.state_dict(),
                'optimizer': student_optimizer.state_dict()},
                f'checkpoints/{logs_idx}/model_{e + 1}_{sum(mae_yy) / len(mae_yy):.5f}_{sum(mae_cc) / len(mae_cc):.5f}.pt')
            e_s = sum(mae_yy) / len(mae_yy)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    train_student()