# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation and CrowdPose.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# (https://github.com/Jeff-sjtu/CrowdPose)
# Modified by Zigang Geng (aa397601@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import logging
import os
import os.path

import cv2
import json_tricks as json
import numpy as np
import pickle
from torch.utils.data import Dataset

from crowdposetools.cocoeval import COCOeval
from utils import zipreader
from .CrowdPoseDataset import CrowdPoseDataset

logger = logging.getLogger(__name__)


class COCOeval_Rescore_Data(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        COCOeval.__init__(self, cocoGt, cocoDt, iouType)
        self.summary = [['pose', 'pose_heatval', 'oks']]
    
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        get predicted pose and oks score for single category and image
        change self.summary
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None
        
        for g in gt:
            tmp_area = g['bbox'][2] * g['bbox'][3] * 0.53
            if g['ignore'] or (tmp_area < aRng[0] or tmp_area > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        gtIg = np.array([g['_ignore'] for g in gt])
        if not len(ious)==0:
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = 0
                m   = -1
                for gind, g in enumerate(gt):
                    #if not iscrowd[gind]:
                    #    continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                        break
                    # continue to next gt unless better match made
                    if ious[dind,gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou=ious[dind,gind]
                    m=gind
                
                dtkeypoint = np.array(d['keypoints']).reshape((14,3))
                self.summary.append([dtkeypoint[:,:2], dtkeypoint[:,2:], iou])

    def dumpdataset(self, data_file):
        pickle.dump(self.summary, open(data_file, 'wb'))


class CrowdPoseDatasetGetScoreData(CrowdPoseDataset):
    def __init__(self, root, dataset, data_format, num_joints, get_rescore_data, transform=None,
                 target_transform=None, bbox_file=None):
        CrowdPoseDataset.__init__(self, root, dataset, data_format, num_joints, get_rescore_data, transform=None,
                 target_transform=None, bbox_file=None)

    def evaluate(self, cfg, preds, scores, output_dir,
                 *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args: 
        :param kwargs: 
        :return: 
        '''
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.dataset)

        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * \
                    (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)

                kpts[int(file_name.split('.')[0])].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'image': int(file_name.split('.')[0]),
                        'area': area
                    }
                )

        # rescoring and oks nms
        oks_nmsed_kpts = []
        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
            img_kpts = kpts[img]
            # person x (keypoints)
            # do not use nms, keep all detections
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        self._do_python_keypoint_eval(
            cfg.RESCORE.DATA_FILE, res_file, res_folder
        )

    def _do_python_keypoint_eval(self, data_file, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval_Rescore_Data(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.dumpdataset(data_file)