# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zigang Geng (aa397601@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import pickle
import cv2

JOINT_COCO_LINK_1 = [0, 0, 1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 7, 8, 11, 11, 12, 13, 14]
JOINT_COCO_LINK_2 = [1, 2, 2, 3, 4, 5, 6, 6, 7, 11, 8, 12, 9, 10, 12, 13, 14, 15, 16]

JOINT_CROWDPOSE_LINK_1 = [12, 13, 13, 0, 1, 2, 3, 0, 1, 6, 7,  8,  9, 6, 0]
JOINT_CROWDPOSE_LINK_2 = [13,  0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 7, 1]


def get_feature(x, dataset):
    joint_abs = x[:, :, :2]
    vis = x[:, :, 2]

    if 'coco' in dataset:
        joint_1, joint_2 = JOINT_COCO_LINK_1, JOINT_COCO_LINK_2
    elif 'crowd_pose' in dataset:
        joint_1, joint_2 = JOINT_CROWDPOSE_LINK_1, JOINT_CROWDPOSE_LINK_2
    else:
        raise ValueError(
            'Please implement flip_index for new dataset: %s.' % dataset)

    #To get the Delta x Delta y
    joint_relate = joint_abs[:, joint_1] - joint_abs[:, joint_2]
    joint_length = ((joint_relate**2)[:, :, 0] +
                    (joint_relate**2)[:, :, 1])**(0.5)

    #To use the torso distance to normalize
    normalize = (joint_length[:, 9]+joint_length[:, 11])/2
    normalize = np.tile(normalize, (len(joint_1), 2, 1)).transpose(2, 0, 1)
    normalize[normalize < 1] = 1

    joint_length = joint_length/normalize[:, :, 0]
    joint_relate = joint_relate/normalize
    joint_relate = joint_relate.reshape((-1, len(joint_1)*2))

    feature = [joint_relate, joint_length, vis]
    feature = np.concatenate(feature, axis=1)
    feature = torch.tensor(feature, dtype=torch.float)
    return feature


def get_joint(filename, num_joints):
    obj = pickle.load(open(filename, "rb"))

    posx, posy = [], []
    for i in range(1, len(obj)):
        pose = list(np.concatenate(
            (obj[i][0], obj[i][1]), axis=1).reshape(3*num_joints))
        posx.append(pose)
        if obj[i][2] == 1:
            obj[i][2] = 0
        posy.append(obj[i][2])

    x = np.array(posx)
    y = np.array(posy)

    x = x.reshape((-1, num_joints, 3))
    y = torch.tensor(y.reshape((-1, 1)), dtype=torch.float)
    return x, y


def read_rescore_data(cfg):
    train_file = cfg.RESCORE.DATA_FILE
    num_joints = cfg.DATASET.NUM_JOINTS - \
        1 if cfg.DATASET.WITH_CENTER else cfg.DATASET.NUM_JOINTS
    x_train, y_train = get_joint(train_file, num_joints)
    feature_train = get_feature(x_train, cfg.DATASET.DATASET)
    return feature_train, y_train
