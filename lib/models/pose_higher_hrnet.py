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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_module import BasicBlock, Bottleneck, STNBLOCK, HighResolutionModule

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
    'STNBLOCK': STNBLOCK
}


class PoseHigherResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.dim_heat = cfg.DATASET.NUM_JOINTS-1 if cfg.DATASET.WITH_CENTER else cfg.DATASET.NUM_JOINTS
        self.dim_reg = self.dim_heat * 2 + 1
        super(PoseHigherResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        inp_channels = np.int(np.sum(pre_stage_channels))
        multi_output_config_heatmap = cfg['MODEL']['EXTRA']['MULTI_LEVEL_OUTPUT_HEATMAP']
        multi_output_config_regression = cfg['MODEL']['EXTRA']['MULTI_LEVEL_OUTPUT_REGRESSION']
        self.transition_cls = nn.Sequential(
                                nn.Conv2d(inp_channels,
                                          multi_output_config_heatmap['NUM_CHANNELS'][0],
                                          1, 1, 0, bias=False),
                                nn.BatchNorm2d(multi_output_config_heatmap['NUM_CHANNELS'][0]),
                                nn.ReLU(True))
        self.transition_reg = nn.Sequential(
                                nn.Conv2d(inp_channels + self.dim_heat,
                                          multi_output_config_regression['NUM_CHANNELS'][0],
                                          1, 1, 0, bias=False),
                                nn.BatchNorm2d(multi_output_config_regression['NUM_CHANNELS'][0]),
                                nn.ReLU(True))

        self.multi_level_layers_4x_heatmap = self._make_multi_level_layer(
                    multi_output_config_heatmap)
        self.multi_level_layers_4x_regression = self._make_multi_level_layer(
                    multi_output_config_regression)
        
        self.deconv_layers = self._make_deconv_layers(
            cfg, multi_output_config_heatmap['NUM_CHANNELS'][0])
        
        self.final_layers = self._make_final_layers(  \
            cfg, multi_output_config_heatmap, multi_output_config_regression)

        self.num_deconvs = extra.DECONV.NUM_DECONVS
        self.deconv_config = cfg.MODEL.EXTRA.DECONV
        self.loss_config = cfg.LOSS

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def _make_final_layers(self, cfg, multi_output_config_heatmap, multi_output_config_regression):
        extra = cfg.MODEL.EXTRA

        final_layers = []
        final_layers.append(nn.Conv2d(
            in_channels=multi_output_config_heatmap['NUM_CHANNELS'][0],
            out_channels=self.dim_heat,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        ))

        # for regression
        if cfg.DATASET.OFFSET_REG:
            final_layers.append(nn.Conv2d(
                in_channels=multi_output_config_regression['NUM_CHANNELS'][0],
                out_channels=self.dim_reg,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
                ))

        deconv_cfg = extra.DECONV
        if deconv_cfg.NUM_DECONVS > 0:
            for i in range(deconv_cfg.NUM_DECONVS):
                input_channels = deconv_cfg.NUM_CHANNELS[i]
                output_channels = self.dim_heat
                final_layers.append(nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=extra.FINAL_CONV_KERNEL,
                    stride=1,
                    padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
                ))

        return nn.ModuleList(final_layers)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, 
            dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_multi_level_layer(
            self, layer_config):
        multi_level_layers = []

        first_branch = self._make_layer(
                blocks_dict[layer_config['BLOCK'][0]],
                layer_config['NUM_CHANNELS'][0],
                layer_config['NUM_CHANNELS'][0],
                layer_config['NUM_BLOCKS'][0],
                dilation=layer_config['DILATION_RATE'][0]
        )
        multi_level_layers.append(first_branch)
        
        for i, d in enumerate(layer_config['DILATION_RATE'][1:]):
            branch = self._make_layer(
                blocks_dict[layer_config['BLOCK'][i+1]],
                layer_config['NUM_CHANNELS'][i+1],
                layer_config['NUM_CHANNELS'][i+1],
                layer_config['NUM_BLOCKS'][i+1],
                dilation=d
            )

            for module in zip(first_branch.named_modules(), branch.named_modules()):
                if 'conv' in module[0][0] and 'conv' in module[1][0]:
                    module[1][1].weight = module[0][1].weight

            multi_level_layers.append(branch)

        return nn.ModuleList(multi_level_layers)

    def _make_deconv_layers(self, cfg, input_channels):
        extra = cfg.MODEL.EXTRA
        deconv_cfg = extra.DECONV

        deconv_layers = []
        for i in range(deconv_cfg.NUM_DECONVS):
            if deconv_cfg.CAT_OUTPUT[i]:
                final_output_channels = self.dim_heat
                input_channels += final_output_channels
            output_channels = deconv_cfg.NUM_CHANNELS[i]
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(deconv_cfg.KERNEL_SIZE[i])

            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
            for _ in range(cfg.MODEL.EXTRA.DECONV.NUM_BASIC_BLOCKS):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)

        x = torch.cat([x[0], \
            F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear'), \
            F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear'), \
            F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')], 1)

        final_outputs = []
        final_offsets = []

        final_output = []
        final_offset = []

        x_cls = self.transition_cls(x)

        for j in range(len(self.multi_level_layers_4x_heatmap)):
            final_output.append(self.final_layers[0](  \
                self.multi_level_layers_4x_heatmap[j](x_cls)))

        for i in range(self.num_deconvs):
            if self.deconv_config.CAT_OUTPUT[i]:
                x_cls = torch.cat((x_cls, torch.mean(torch.stack(final_output), 0)), 1)

            x_cls = self.deconv_layers[i](x_cls)
            heatmap_2x = self.final_layers[i+2](x_cls)
        
        x_reg = self.transition_reg(torch.cat([x, (torch.mean(torch.stack(final_output), 0))], 1))

        for j in range(len(self.multi_level_layers_4x_regression)):
            final_offset.append(self.final_layers[1]( \
                self.multi_level_layers_4x_regression[j](x_reg)))

        for i in range(len(final_output)):
            final_output[i] = torch.cat([final_output[i], final_offset[0][:,-1:,:,:]], 1)
        
        final_outputs.append(final_output)
        final_outputs.append([heatmap_2x])
        final_offsets.append([final_offset[0][:,:-1,:,:]])

        return final_outputs, final_offsets

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'conv_mask_offset_1'):
                nn.init.constant_(m.conv_mask_offset_1.weight, 0)
                nn.init.constant_(m.conv_mask_offset_1.bias, 0)
            if hasattr(m, 'conv_mask_offset_2'):
                nn.init.constant_(m.conv_mask_offset_2.weight, 0)
                nn.init.constant_(m.conv_mask_offset_2.bias, 0)
            if hasattr(m, 'transform_matrix_conv1'):
                nn.init.constant_(m.transform_matrix_conv1.weight, 0)
                nn.init.constant_(m.transform_matrix_conv1.bias, 0)
            if hasattr(m, 'transform_matrix_conv2'):
                nn.init.constant_(m.transform_matrix_conv2.weight, 0)
                nn.init.constant_(m.transform_matrix_conv2.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model
