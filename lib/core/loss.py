# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp.cuda()
    return inp


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


class OffsetsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def smooth_l1_loss(self, pred, gt, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5*l1_loss**2/beta, l1_loss-0.5*beta)
        return loss

    def forward(self, pred, gt, weights):
        assert pred.size() == gt.size()
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.smooth_l1_loss(pred, gt) * weights
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss


class MultiLossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._init_check(cfg)

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_stages = cfg.LOSS.NUM_STAGES

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in cfg.LOSS.WITH_HEATMAPS_LOSS
                ]
            )
        self.heatmaps_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR

        self.offsets_loss = \
            nn.ModuleList(
                [
                    OffsetsLoss()
                    if with_offsets_loss else None
                    for with_offsets_loss in cfg.LOSS.WITH_OFFSETS_LOSS
                ]
            )
        self.offsets_loss_factor = cfg.LOSS.OFFSETS_LOSS_FACTOR

    def forward(self, outputs, poffsets, heatmaps,
                masks, offsets, offset_w):
        heatmaps_losses = []
        offsets_losses = []
        for idx in range(len(outputs)):

            with_heatmaps_loss = self.heatmaps_loss[idx]
            with_offsets_loss = self.offsets_loss[idx]

            if with_heatmaps_loss and len(outputs[idx]) > 0:
                num_outputs = len(outputs[idx])
                if num_outputs > 1:
                    heatmaps_pred = torch.cat(outputs[idx], dim=1)
                    c = outputs[idx][0].shape[1]
                    if len(heatmaps[idx]) > 1:
                        heatmaps_gt = [heatmaps[idx][i][:, :c]
                                       for i in range(num_outputs)]
                        heatmaps_gt = torch.cat(heatmaps_gt, dim=1)
                        mask = [masks[idx][i].expand_as(outputs[idx][0])
                                for i in range(num_outputs)]
                        mask = torch.cat(mask, dim=1)
                    else:
                        heatmaps_gt = torch.cat([heatmaps[idx][0][:, :c]
                                                 for i in range(num_outputs)], dim=1)
                        mask = [masks[idx][0].expand_as(outputs[idx][0])
                                for i in range(num_outputs)]
                        mask = torch.cat(mask, dim=1)
                else:
                    heatmaps_pred = outputs[idx][0]
                    c = heatmaps_pred.shape[1]
                    heatmaps_gt = heatmaps[idx][0][:, :c]
                    mask = masks[idx][0].expand_as(heatmaps_pred)

                heatmaps_loss = with_heatmaps_loss(
                    heatmaps_pred, heatmaps_gt, mask
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[0]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)

            if with_offsets_loss and len(poffsets[idx]) > 0:
                num_poffsets = len(poffsets[idx])
                if num_poffsets > 1:
                    offset_pred = torch.cat(poffsets[idx], dim=1)
                    offset_gt = torch.cat([offsets[idx][0]
                                           for i in range(num_poffsets)], dim=1)
                    offset_w = torch.cat([offset_w[idx][0]
                                          for i in range(num_poffsets)], dim=1)
                else:
                    offset_pred = poffsets[idx][0]
                    offset_gt = offsets[idx][0]
                    offset_w = offset_w[idx][0]

                offsets_loss = with_offsets_loss(
                    offset_pred, offset_gt, offset_w
                )
                offsets_loss = offsets_loss * self.offsets_loss_factor[0]
                offsets_losses.append(offsets_loss)
            else:
                offsets_losses.append(None)

        return heatmaps_losses, offsets_losses

    def _init_check(self, cfg):
        assert isinstance(cfg.LOSS.WITH_HEATMAPS_LOSS, (list, tuple)), \
            'LOSS.WITH_HEATMAPS_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.HEATMAPS_LOSS_FACTOR, (list, tuple)), \
            'LOSS.HEATMAPS_LOSS_FACTOR should be a list or tuple'
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
            format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == len(cfg.LOSS.HEATMAPS_LOSS_FACTOR), \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.HEATMAPS_LOSS_FACTOR should have same length, got {} vs {}.'.\
            format(len(cfg.LOSS.WITH_HEATMAPS_LOSS),
                   len(cfg.LOSS.HEATMAPS_LOSS_FACTOR))
