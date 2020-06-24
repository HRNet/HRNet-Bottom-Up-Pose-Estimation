# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (aa397601@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import stat
import pprint

import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapRegParser
from core.rescore import rescore_valid
from dataset import make_test_dataloader
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import up_interpolate

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    #dump_input = torch.rand(
    #    (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    #)
    #logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    data_loader, test_dataset = make_test_dataloader(cfg)

    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    parser = HeatmapRegParser(cfg)

    # for pred kpts and pred heat
    all_preds = []
    all_scores = []

    pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None
    for i, (images, joints, masks, areas) in enumerate(data_loader):
        assert 1 == images.size(0), 'Test batch size should be 1'

        image = images[0].cpu().numpy()
        joints = joints[0].cpu().numpy()
        mask = masks[0].cpu().numpy()
        area = areas[0].cpu().numpy()
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )

        with torch.no_grad():
            heatmap_fuse = 0
            final_heatmaps = None
            final_kpts = None
            input_size = cfg.DATASET.INPUT_SIZE

            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):

                image_resized, joints_resized, _, center, scale = resize_align_multi_scale(
                    image, joints, mask, input_size, s, 1.0
                )

                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, kpts = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST
                )
                final_heatmaps, final_kpts = aggregate_results(
                    cfg, final_heatmaps, final_kpts, heatmaps, kpts
                )

            for heatmap in final_heatmaps:
                heatmap_fuse += up_interpolate(
                    heatmap,
                    size=(base_size[1], base_size[0]),
                    mode='bilinear'
                )
            heatmap_fuse = heatmap_fuse/float(len(final_heatmaps))

            # for pred kpts and pred heatmaps
            grouped, scores = parser.parse(
                final_heatmaps, final_kpts, heatmap_fuse[0], use_heatmap=True
            )
            if len(scores) == 0:
                all_preds.append([])
                all_scores.append([])
                if cfg.TEST.LOG_PROGRESS:
                    pbar.update()
                continue

            final_results = get_final_preds(
                grouped, center, scale,
                [heatmap_fuse.size(-1),heatmap_fuse.size(-2)]
            )

            if cfg.RESCORE.USE:
                scores = rescore_valid(cfg, final_results, scores)

            all_preds.append(final_results)
            all_scores.append(scores)

        if cfg.TEST.LOG_PROGRESS:
            pbar.update()

    sv_all_preds = [all_preds]
    sv_all_scores = [all_scores]
    sv_all_name = ['final']

    if cfg.TEST.LOG_PROGRESS:
        pbar.close()

    for i in range(len(sv_all_preds)):
        print('Testing '+sv_all_name[i])
        preds = sv_all_preds[i]
        scores = sv_all_scores[i]
        test_dataset.evaluate(
            cfg, preds, scores, final_output_dir, sv_all_name[i]
        )


if __name__ == '__main__':
    main()
