# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zigang Geng (aa397601@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import models
from dataset import get_feature


def rescore_valid(cfg, temp, ori_scores):
    temp = np.array(temp)

    feature = get_feature(temp, cfg.DATASET.DATASET)
    feature = feature.cuda()

    PredictOKSmodel = eval('models.'+'predictOKS'+'.get_pose_net')(
        cfg, feature.shape[1], is_train=False
    )
    pretrained_state_dict = torch.load(cfg.RESCORE.MODEL_FILE)
    need_init_state_dict = {}
    for name, m in pretrained_state_dict.items():
        need_init_state_dict[name] = m
    PredictOKSmodel.load_state_dict(need_init_state_dict, strict=False)
    PredictOKSmodel = torch.nn.DataParallel(
        PredictOKSmodel, device_ids=cfg.GPUS).cuda()
    PredictOKSmodel.eval()

    scores = PredictOKSmodel(feature)
    scores = scores.cpu().numpy()

    scores[np.isnan(scores)] = 0
    mul_scores = scores*np.array(ori_scores).reshape(scores.shape)
    scores = [np.float(i) for i in list(scores)]
    mul_scores = [np.float(i) for i in list(mul_scores)]
    return mul_scores


def rescore_fit(cfg, model, x_data, y_data):
    loss_fn = nn.MSELoss(reduction='mean')
    train_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.RESCORE.LR)

    x_data = Variable(x_data, requires_grad=True)
    y_data = Variable(y_data, requires_grad=True)

    save_final_model_file = cfg.RESCORE.MODEL_FILE
    for epoch in range(cfg.RESCORE.END_EPOCH):
        train_loss = train(x_data, y_data, optimizer, model,
                           loss_fn, cfg.RESCORE.BATCHSIZE)
        train_losses.append(train_loss)

        if epoch % 1 == 0:
            print("step:", epoch+1, "train_loss:", train_loss)

    torch.save(model.state_dict(), save_final_model_file)
    return train_losses


def train(x_data, y_data, optimizer, model, loss_fn, batchsize):
    datasize = len(x_data)
    loss_sum = 0
    index = np.arange(datasize)
    np.random.shuffle(index)
    for i in range(int(datasize/batchsize)):
        x_temp = x_data[index[i*batchsize:(i+1)*(batchsize)]]
        y_temp = y_data[index[i*batchsize:(i+1)*(batchsize)]]
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_temp)

        loss = loss_fn(y_pred, y_temp)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    return loss_sum/int(datasize/batchsize)

