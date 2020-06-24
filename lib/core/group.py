# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# Modified by Zigang Geng (aa397601@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import cv2
import copy
from scipy.optimize import curve_fit


def unnormalized_gaussian2d(data_tuple, A, y0, x0, sigma):
    (y, x) = data_tuple
    g = A * np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g


def fit_gaussian_heatmap(func, heatmap, maxval, init_y, init_x, sigma):
    """
    Find the precise float joint coordinates of coarse int coordinate (init_y, init_x) 
    by fitting guassian on heatmap near (init_y, init_x).

    Args:
        func: gaussian2d function
        heatmap: heatmap near (init_x, init_y)
        maxval: the heatmap value at (init_x, init_y)
        sigma: guassian sigma
    Returns:
        fitted guassian's parameter: center_x, center_y, peak value, sigma
    """
    heatmap_y_length = heatmap.shape[0]
    heatmap_x_length = heatmap.shape[1]
    y = np.linspace(0, heatmap_y_length - 1, heatmap_y_length)
    x = np.linspace(0, heatmap_x_length - 1, heatmap_x_length)
    Y, X = np.meshgrid(y, x)
    x_data = np.vstack((X.ravel(), Y.ravel()))

    init_guess = (maxval, init_y, init_x, sigma)
    popt, _ = curve_fit(func, x_data, heatmap.ravel(),
                        p0=init_guess, maxfev=300)
    return popt[1], popt[2], popt[0], popt[3]


class HeatmapRegParser(object):
    def __init__(self, cfg):
        self.input_size = cfg.DATASET.INPUT_SIZE
        self.output_size = cfg.DATASET.OUTPUT_SIZE[-1]
        self.ratio = self.input_size/self.output_size
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.max_num_people = cfg.DATASET.MAX_NUM_PEOPLE
        self.guassian_kernel = cfg.TEST.GUASSIAN_KERNEL
        self.sigma = cfg.TEST.GUASSIAN_SIGMA
        self.reg_thres = cfg.TEST.REG_THRESHOLD
        self.dist_thres = cfg.TEST.DIST_THRESHOLD
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2
        self.overlap_thres = cfg.TEST.OVERLAP_THRESHOLD

        self.scale_decrease = cfg.TEST.SCALE_DECREASE
        self.use_decrease_score = cfg.TEST.USE_DECREASE_SCORE

        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.adjust_thre = cfg.TEST.ADJUST_THRESHOLD
        self.max_distance = cfg.TEST.MAX_ABSORB_DISTANCE

        if cfg.DATASET.WITH_CENTER:
            self.num_joints -= 1

        self.pool1 = torch.nn.MaxPool2d(
            3, 1, 1
        )
        self.pool2 = torch.nn.MaxPool2d(
            5, 1, 2
        )
        self.pool3 = torch.nn.MaxPool2d(
            7, 1, 3
        )

    def hierarchical_pool(self, heatmap):
        map_size = (heatmap.shape[1]+heatmap.shape[2])/2.0
        if map_size > self.pool_thre1:
            maxm = self.pool3(heatmap[None, :, :, :])
        elif map_size > self.pool_thre2:
            maxm = self.pool2(heatmap[None, :, :, :])
        else:
            maxm = self.pool1(heatmap[None, :, :, :])
        return maxm

    def get_keypoints_from_heatmap(self, heatmap, factor):
        maxm = self.hierarchical_pool(heatmap)
        maxm = torch.eq(maxm, heatmap).float()
        heatmap = heatmap * maxm
        scores = heatmap.view(-1)
        scores, pos_ind = scores.topk(self.max_num_people)

        select_ind = (scores > (self.keypoint_thre*factor)).nonzero()
        scores = scores[select_ind][:, 0]
        pos_ind = pos_ind[select_ind][:, 0]

        return pos_ind, scores

    def kpts_nms(self, final_heatmaps, kpts, heatmap, use_heatmap):
        """
        Find the people's center by finding the local maximum coordinates on center heatmap.
        Get the regression output by using the offset at the people's center.
        NMS for the regression result.

        Args:
            final_heatmaps (List): heatmaps (all scales, 4x)
            kpts (List): offsetmaps (all scales, 4x)
            heatmap (Tensor): multiscale fused center heatmap (scale=1, 1x)
            use_heatmap (bool): 
                if not use heatmap, the scores of results is center score * keypoint heatvalue
                if use heatmap, the scores of results is center score
        Returns:
            keep_kpts (Tensor): regression results (shape: N(people number)*17(joint number)*3[x,y,score])
            center_scores (Tensor): people center heatvalue (shape: N(people number))
        """
        final_heatmaps_new = copy.deepcopy(final_heatmaps)

        # Find the maximum scores on heatmap (scale=1)
        # to normalize the heatmap at other scales
        for final_heatmap in final_heatmaps:
            if final_heatmap.shape[-1] == heatmap.shape[-1]/self.ratio:
                center_map = final_heatmap[0, -1:]
                max_ori = center_map.max()

        # Get person center location from all scales centermap 
        for i in range(len(final_heatmaps_new)):
            ratio = heatmap.shape[-1]/final_heatmaps_new[i].shape[-1]
            factor = 1 if ratio == self.ratio else self.scale_decrease
            center_map = final_heatmaps_new[i][0, -1:]
            max_now = center_map.max()
            center_map = factor*center_map/max_now*max_ori

            pos_ind, scores_this_scale = self.get_keypoints_from_heatmap(center_map, factor)

            if i == 0:
                scores = scores_this_scale
            else:
                scores = torch.cat([scores, scores_this_scale], dim=0)

            kpt_temp = kpts[i][0].permute(1, 2, 0).reshape(-1, self.num_joints, 2)
            if i == 0:
                kpts = ratio*kpt_temp[pos_ind]
            else:
                kpts = torch.cat([kpts, ratio*kpt_temp[pos_ind]], dim=0)

        if kpts.shape[0] == 0:
            return [], []
        
        # Regression pose nms:
        # If two poses have more than overlap_thres similar keypoints, 
        # we will keep the pose with high scores.
        # Similar means that the distance of two keypoints is less than dist_thres.
        kpts_diff = kpts[:, None, :, :] - kpts
        kpts_diff.pow_(2)
        kpts_dist = kpts_diff.sum(3)
        kpts_dist.sqrt_()
        kpts_dist = (kpts_dist < self.dist_thres).sum(2)
        nms_kpts = kpts_dist > self.overlap_thres

        ignored_kpts_inds = []
        keep_kpts_inds = []
        for i in range(nms_kpts.shape[0]):
            if i in ignored_kpts_inds:
                continue
            keep_inds = nms_kpts[i].nonzero().cpu().numpy()
            keep_inds = [list(kind)[0] for kind in keep_inds]
            keep_scores = scores[keep_inds]
            ind = torch.argmax(keep_scores)
            keep_ind = keep_inds[ind]
            if keep_ind in ignored_kpts_inds:
                continue
            keep_kpts_inds += [keep_ind]
            ignored_kpts_inds += list(set(keep_inds)-set(ignored_kpts_inds))

        keep_kpts = kpts[keep_kpts_inds]
        keep_scores = scores[keep_kpts_inds]

        #limit the max people number in an image
        if len(keep_kpts_inds) > self.max_num_people:
            keep_scores, topk_inds = torch.topk(keep_scores,
                                                self.max_num_people)
            keep_kpts = keep_kpts[topk_inds]
            
        kpt_now = keep_kpts.shape[0]
        heatval = np.zeros((kpt_now, self.num_joints, 1))

        for i in range(kpt_now):
            for j in range(self.num_joints):
                k1, k2 = int(keep_kpts[i,j,0]), int(keep_kpts[i,j,0])+1
                k3, k4 = int(keep_kpts[i,j,1]), int(keep_kpts[i,j,1])+1
                u = keep_kpts[i,j,0]-int(keep_kpts[i,j,0])
                v = keep_kpts[i,j,1]-int(keep_kpts[i,j,1])
                if k2 < heatmap.shape[2] and k1 >= 0 and k4 < heatmap.shape[1] and k3 >= 0:
                    heatval[i,j,0] = \
                        heatmap[j,k3,k1]*(1-v)*(1-u) + heatmap[j,k4,k1]*(1-u)*v+ \
                        heatmap[j,k3,k2]*u*(1-v) + heatmap[j,k4,k2]*u*v

        center_scores = keep_scores.clone()
        if use_heatmap == False:
            keep_scores = torch.tensor(keep_scores[:,None].expand(-1,self.num_joints)[:,:,None].cpu().numpy()*heatval).float()
        else:
            keep_scores = keep_scores[:,None].expand(-1,self.num_joints)[:,:,None]

        keep_kpts = torch.cat([keep_kpts.cpu(), keep_scores.cpu()], dim=2)
        return keep_kpts, center_scores

    def absorb_heat(self, det, kpts):
        """
        Use heatmap to refine regression result (grouping).

        Args:
            det (Tensor): multiscale fused keypoint heatmap (scale=1, 1x)
            kpts (Tensor): regression results (shape: N(people number)*17(joint number)*3[x,y,score])
        Returns:
            ans (List): grouped results
        """
        new_kpts = kpts.clone()
        det_ori = det.clone()
        det = det.cuda()
        maxm = self.pool2(det[None, :, :, :])
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        det = det[0]
        num_joints = det.size(0)
        w = det.size(2)
        det = det.view(num_joints, -1)
        val_k, ind = det.topk(self.max_num_people, dim=1)

        select_ind = (val_k > self.keypoint_thre).float()
        val_k = val_k*select_ind

        val_k = val_k.cpu()
        ind = ind.cpu()

        x = ind % w
        y = (ind / w).long()
        heats_ind = torch.stack((x, y), dim=2)

        for i in range(num_joints):

            heat_ind = heats_ind[i].float()
            kpt_ind = kpts[:, i, :-1]
            kpts_heat_diff = kpt_ind[:, None, :] - heat_ind
            kpts_heat_diff.pow_(2)
            kpts_heat_diff = kpts_heat_diff.sum(2)
            kpts_heat_diff.sqrt_()
            keep_ind = torch.argmin(kpts_heat_diff, dim=1)

            for p in range(keep_ind.shape[0]):
                if kpts_heat_diff[p, keep_ind[p]] < self.max_distance:
                    new_kpts[p, i, :-1] = heat_ind[keep_ind[p]]
                    new_kpts[p, i, -1] *= val_k[i][keep_ind[p]]
                else:
                    a = max(min(int(new_kpts[p, i, 0]), det_ori.shape[1]-1), 0)
                    b = max(min(int(new_kpts[p, i, 1]), det_ori.shape[2]-1), 0)
                    new_kpts[p, i, -1] *= det_ori[i, a, b]

        return [new_kpts.cpu().numpy()]

    def adjust(self, ans, det, center_scores):
        """
        Use guassian fit to refine final results.
        """
        N = self.guassian_kernel
        local_hm = np.zeros((N, N))
        for batch_id in range(len(ans)):
            for joint_id in range(ans[0].shape[1]):
                dist_xy = {}
                for people_id in range(ans[0].shape[0]):
                    if ans[batch_id][people_id, joint_id, 2] > self.adjust_thre:
                        y, x = ans[batch_id][people_id, joint_id, 0:2]
                        xx, yy = int(x+0.5), int(y+0.5)
                        dist_index = str([xx, yy])
                        tmp = det[batch_id, joint_id, :, :]
                        if xx == 0 or xx == tmp.shape[0]-1 or yy == 0 or yy == tmp.shape[1]-1:
                            continue
                        if dist_index in dist_xy:
                            ans[batch_id][people_id, joint_id, 1] = dist_xy[dist_index][0]
                            ans[batch_id][people_id, joint_id, 0] = dist_xy[dist_index][1]
                            ans[batch_id][people_id, joint_id, 2] = dist_xy[dist_index][2]
                            ans[batch_id][people_id, joint_id, 2] *= center_scores[people_id]
                            continue

                        safe_y_lower_bound = max(0, yy - N)
                        safe_y_upper_bound = min(tmp.shape[1]-1, yy + N)

                        safe_x_lower_bound = max(0, xx - N)
                        safe_x_upper_bound = min(tmp.shape[0]-1, xx + N)

                        local_hm = tmp[safe_x_lower_bound:safe_x_upper_bound + 1,
                                        safe_y_lower_bound:safe_y_upper_bound + 1]
                        
                        try:
                        #If neighborhood around (xx, yy) on heatmap is not a guassian distribution, 
                        #optimal parameters can not be found after max iteration
                        #and the curve_fit function in scipy will raise error. 
                        #This keypoint coordinates will not be adjusted
                        #and this parts do not influence the results. 
                            mean_x, mean_y, value, _ = fit_gaussian_heatmap(
                                unnormalized_gaussian2d, local_hm.cpu(
                                ).numpy(), tmp[xx][yy].cpu().numpy(),
                                xx - safe_x_lower_bound, yy - safe_y_lower_bound, self.sigma
                            )
                            ans[batch_id][people_id, joint_id, 1] = safe_x_lower_bound + mean_x
                            ans[batch_id][people_id, joint_id, 0] = safe_y_lower_bound + mean_y
                            ans[batch_id][people_id, joint_id, 2] = value*center_scores[people_id]
                            dist_xy[dist_index] = [safe_x_lower_bound + mean_x, safe_y_lower_bound + mean_y, value]
                        except:
                            continue

        return ans

    def parse(self, heatmaps, kpts, heatmap_fuse, use_heatmap=True):
        
        ans, center_scores = self.kpts_nms(heatmaps, kpts, heatmap_fuse[:-1, :, :], use_heatmap)

        if len(center_scores) == 0:
            return [], []

        if use_heatmap:
            heatmap_fuse = heatmap_fuse.cpu()
            ans = self.absorb_heat(heatmap_fuse[:-1], ans)
            ans = self.adjust(ans, heatmap_fuse[None, :-1, :, :], center_scores)
        else:
            ans = [ans.numpy()]

        scores = [i[:, 2].mean() for i in ans[0]]

        return ans, scores
