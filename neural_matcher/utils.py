"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""

import os

import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from neural_matcher.eval import infer_nn, viz_heatmap, viz_match_vectors, score_model


def pxys_to_match_vectors_mask(pxys):
    match_vectors_mask = torch.zeros([len(pxys), 2, utils.SIDE, utils.SIDE])
    for bi in range(len(pxys)):
        v = (pxys[bi][:, 2:] - pxys[bi][:, :2]) / (utils.SIDE - 1)
        match_vectors_mask[bi, :, pxys[bi][:, 1], pxys[bi][:, 0]] = torch.Tensor(v.T)
        match_vectors_mask[bi, :, pxys[bi][:, 3], pxys[bi][:, 2]] = -torch.Tensor(v.T)
    return match_vectors_mask


def collater(data):
    ims = []
    match_xys_gt = []
    heatmaps = []
    for d in data:
        x = d[0] / 255.
        ims.append(utils.INPUT_TRANSFORMS(torch.Tensor(x)))
        match_xys_gt.append(torch.Tensor(np.round(d[1]).astype(int)).int())
        heatmaps.append(d[2])

    hm_gt = torch.Tensor(np.stack(heatmaps))
    ims = torch.stack(ims)
    match_vectors_gt = pxys_to_match_vectors_mask(match_xys_gt)

    conf_masks_gt = []
    confs_gt = []
    p_xy = np.vstack(np.meshgrid(np.arange(utils.SIDE), np.arange(utils.SIDE), indexing='xy')).reshape(2, -1)
    for bi in range(len(data)):
        targ_xy_2d = np.clip(np.round(p_xy + (match_vectors_gt[bi].reshape(2, -1)
                                              * (utils.SIDE - 1)).detach().cpu().numpy()),
                             0, utils.SIDE - 1).astype(int)
        targ_xy_1d = targ_xy_2d[0] + targ_xy_2d[1] * utils.SIDE
        conf_targ = hm_gt[bi][1].flatten()[targ_xy_1d]
        conf_mask_gt_1d = (hm_gt[bi][0].flatten() + conf_targ) / 2.
        f = targ_xy_1d[match_xys_gt[bi][:, 3] * utils.SIDE + match_xys_gt[bi][:, 2]]
        # f_ = p_xy[:, match_xys_gt[bi][:, 1] * utils.SIDE + match_xys_gt[bi][:, 0]]  # verified equivalent to above
        # f = f_[0] + f_[1] * utils.SIDE
        conf_masks_gt.append(conf_mask_gt_1d)
        confs = conf_mask_gt_1d[f]
        confs_gt.append(confs)

        # f = conf_mask_gt_1d > 0.5
        # src_xy_2d = torch.Tensor(p_xy[:, f]).int()
        # targ_xy_2d_ = torch.clamp(src_xy_2d + (match_vectors_gt[bi].reshape(2, -1)[:, f]
        #                                        * (utils.SIDE - 1)).round().int(), 0, utils.SIDE - 1)
        # match_xys_gt_ = torch.vstack([src_xy_2d, targ_xy_2d_]).T.numpy()
        # ima = utils.drawMatches(np.rollaxis(data[0][0][0], 0, 3), match_xys_gt[bi][:, :2],
        #                         np.rollaxis(data[0][0][1], 0, 3), match_xys_gt[bi][:, 2:])
        # ima_ = utils.drawMatches(np.rollaxis(data[0][0][0], 0, 3), match_xys_gt_[:, :2],
        #                          np.rollaxis(data[0][0][1], 0, 3), match_xys_gt_[:, 2:])
        # cv2.imwrite('a.png', ima)
        # cv2.imwrite('a_.png', ima_)
        # k = 0

    conf_masks_gt = torch.stack(conf_masks_gt).reshape(-1, utils.SIDE, utils.SIDE)
    return ims, ((hm_gt, match_vectors_gt, conf_masks_gt), (match_xys_gt, confs_gt))


def matplotlib_imshow(img):
    fig = plt.figure()
    plt.imshow(img[:, :, [2, 1, 0]])
    return fig


class SIFTMatcher:

    def __init__(self, sift_tolerance=.4, min_n_matches=5):
        self.side = utils.SIDE
        self.sift_tolerance = sift_tolerance
        self.min_n_matches = int(min_n_matches)
        self.tag = '_'.join([str(sift_tolerance) + '.sift-tolerance',
                             str(min_n_matches) + '.min-n-matches',
                             str(self.side) + '.side'])
        # self.keypoint_detector = cv2.xfeatures2d.SIFT_create()
        self.keypoint_detector = cv2.SIFT_create()
        self.keypoint_matcher = cv2.BFMatcher()

    def sift_match(self, im_a, im_b):
        good = []
        y = []
        im_a_cv2 = cv2.cvtColor(im_a, cv2.COLOR_BGR2GRAY)
        im_b_cv2 = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
        kp_a, desc_a = self.keypoint_detector.detectAndCompute(im_a_cv2, None)
        kp_b, desc_b = self.keypoint_detector.detectAndCompute(im_b_cv2, None)
        if len(kp_a) > self.min_n_matches and len(kp_b) > self.min_n_matches:
            matches = self.keypoint_matcher.knnMatch(desc_a, desc_b, k=2)
            for m, n in matches:
                if m.distance < self.sift_tolerance * n.distance:
                    kp_a_sel = kp_a[m.queryIdx]
                    kp_b_sel = kp_b[m.trainIdx]
                    if not np.array_equal(kp_a_sel.pt, kp_b_sel.pt):
                        y.append(np.hstack([kp_a_sel.pt, kp_b_sel.pt]))
                        good.append([m])
        y = np.array(y)
        return y, (good, kp_a, kp_b)


def checkpoint_model(nmatch, train_measures, device, data_loader_val, ima, imb, model_dir, loss_fn, ei, bi, sess_id,
                     log_fname, val_df_dict, viz_dir, writer, g_idx, ksize, radius_scale, blend_coeff):
    nmatch.eval()
    suffix = '-'.join([str(ei) + 'e', str(bi) + 'b'])
    sess_id_ = sess_id + '_' + suffix

    match_viz, heatmap_a, heatmap_b, match_vectors_viz, conf_mask_viz, inference_outs, \
        (ima_pp, imb_pp) = infer_nn(nmatch, ima, imb, device)
    _, (matches_xy, confs) = inference_outs

    fn_prefix = '_'.join(['viz', sess_id])
    suffix = '-'.join([str(ei) + 'e', str(bi) + 'b', str(matches_xy[0].shape[0]) + 'kp'])
    print('VIZ:', suffix)

    gt_viz_dir = viz_dir + '/gt'
    # if not os.path.isdir(gt_viz_dir):
    os.makedirs(gt_viz_dir, exist_ok=True)
    sm = SIFTMatcher()
    s = heatmap_a.shape[0]
    matches_xy_gt, _ = sm.sift_match(ima_pp, imb_pp)
    matches_xy_gt = np.round(np.clip(matches_xy_gt, 0, s - 1)).astype(int)

    hma = utils.create_heatmap(matches_xy_gt[:, :2], s, ksize=ksize,  radius_scale=radius_scale,
                               blend_coeff=blend_coeff)
    hmb = utils.create_heatmap(matches_xy_gt[:, 2:], s, ksize=ksize, radius_scale=radius_scale,
                               blend_coeff=blend_coeff)
    heatmap_a_gt = viz_heatmap(hma)
    heatmap_b_gt = viz_heatmap(hmb)
    blended_viz_a = blend_coeff * heatmap_a_gt + (1. - blend_coeff) * ima_pp
    blended_viz_b = blend_coeff * heatmap_b_gt + (1. - blend_coeff) * imb_pp

    match_viz_gt = utils.drawMatches(blended_viz_a, matches_xy_gt[:, :2], blended_viz_b, matches_xy_gt[:, 2:])

    match_vectors_gt = pxys_to_match_vectors_mask([matches_xy_gt])
    match_vectors_gt_viz = viz_match_vectors(match_vectors_gt[0])

    targ_xy_2d = np.clip(np.round(nmatch.p_xy[0].detach().cpu().numpy()
                                  + (match_vectors_gt[0].reshape(2, -1) * (s - 1)).detach().cpu().numpy()),
                         0, s - 1).astype(int)
    targ_xy_1d = targ_xy_2d[0] + targ_xy_2d[1] * s
    conf_targ = hmb.flatten()[targ_xy_1d]
    conf_mask_gt = (hma.flatten() + conf_targ) / 2.

    f = conf_mask_gt > nmatch.heatmap_thresh.item()
    txy = targ_xy_2d[:, f]
    confs_gt = conf_mask_gt[f]
    matches_xy_gt_ = np.vstack([nmatch.p_xy[0].detach().cpu().numpy()[:, f], txy]).T
    match_viz_gt_ = utils.drawMatches(blended_viz_a, matches_xy_gt_[:, :2], blended_viz_b, matches_xy_gt_[:, 2:],
                                      confs=[confs_gt])

    conf_mask_gt_viz = viz_heatmap(conf_mask_gt.reshape(s, s))

    suffix_ = sess_id + '_gt'
    mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'matches', suffix_ + '.jpg'])])
    cv2.imwrite(mn, match_viz_gt)
    mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'matches-reconstructed', suffix_ + '.jpg'])])
    cv2.imwrite(mn, match_viz_gt_)
    mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'heatmap-a', suffix_ + '.jpg'])])
    cv2.imwrite(mn, heatmap_a_gt)
    mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'heatmap-b', suffix_ + '.jpg'])])
    cv2.imwrite(mn, heatmap_b_gt)
    mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'vectors', suffix_ + '.jpg'])])
    cv2.imwrite(mn, match_vectors_gt_viz)
    mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'match-confidence', suffix_ + '.jpg'])])
    cv2.imwrite(mn, conf_mask_gt_viz)

    hm_gt = torch.Tensor(np.expand_dims(np.array([hma, hmb]), 0))
    gt_outs = (hm_gt, match_vectors_gt, torch.Tensor(np.expand_dims(conf_mask_gt, 0)).reshape(-1, s, s)), \
        ([torch.Tensor(matches_xy_gt)], [torch.Tensor(confs_gt)])

    writer.add_figure('match_viz-gt', matplotlib_imshow(match_viz_gt), global_step=g_idx)
    writer.add_figure('heatmap-a-gt', matplotlib_imshow(heatmap_a_gt), global_step=g_idx)
    writer.add_figure('heatmap-b-gt', matplotlib_imshow(heatmap_b_gt), global_step=g_idx)
    writer.add_figure('match_vectors_viz-gt', matplotlib_imshow(match_vectors_gt_viz), global_step=g_idx)
    writer.add_figure('match_confidence-gt', matplotlib_imshow(conf_mask_gt_viz), global_step=g_idx)

    with torch.no_grad():
        (example_loss, example_vector_loss, example_conf_loss, vector_loss_map), (tp, fp, fn) = loss_fn(inference_outs,
                                                                                                        gt_outs)
    vector_loss_viz = viz_heatmap(vector_loss_map[0].detach().cpu().numpy())
    suffix_ = suffix + '-pred'
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'matches', suffix_ + '.jpg'])])
    cv2.imwrite(mn, match_viz)
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'heatmap-a', suffix_ + '.jpg'])])
    cv2.imwrite(mn, heatmap_a)
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'heatmap-b', suffix_ + '.jpg'])])
    cv2.imwrite(mn, heatmap_b)
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'vectors', suffix_ + '.jpg'])])
    cv2.imwrite(mn, match_vectors_viz)
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'match-confidence', suffix_ + '.jpg'])])
    cv2.imwrite(mn, conf_mask_viz)
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, '-vector-lossmap', suffix_ + '.jpg'])])
    cv2.imwrite(mn, vector_loss_viz)

    writer.add_figure('match_viz-pred', matplotlib_imshow(match_viz), global_step=g_idx)
    writer.add_figure('heatmap-a-pred', matplotlib_imshow(heatmap_a), global_step=g_idx)
    writer.add_figure('heatmap-b-pred', matplotlib_imshow(heatmap_b), global_step=g_idx)
    writer.add_figure('match_vectors_viz-pred', matplotlib_imshow(match_vectors_viz), global_step=g_idx)
    writer.add_figure('match_confidence-pred', matplotlib_imshow(conf_mask_viz), global_step=g_idx)
    writer.add_figure('vector_loss_map-pred', matplotlib_imshow(vector_loss_viz), global_step=g_idx)

    score_dict = score_model(nmatch, data_loader_val, loss_fn)

    for k in score_dict.keys():
        writer.add_scalar(k, score_dict[k], g_idx)
    writer.add_scalar('val_example_loss', example_loss, g_idx)
    writer.add_scalar('val_example_vector_loss', example_vector_loss, g_idx)
    writer.add_scalar('val_example_conf_loss', example_conf_loss, g_idx)

    if train_measures is not None:
        train_losses, grad_measures = train_measures
        running_loss, running_vector_loss, running_conf_loss = train_losses
        running_g_mean, running_g_std, running_g_max, running_g_min = grad_measures
    else:
        running_loss = 0.
        running_vector_loss = 0.
        running_conf_loss = 0.
        running_g_mean = 0.
        running_g_std = 0.
        running_g_max = 0.
        running_g_min = 0.

    score_dict['epoch'] = ei
    score_dict['epoch_batch_iteration'] = bi
    score_dict['train_loss'] = running_loss
    score_dict['train_vector_loss'] = running_vector_loss
    score_dict['train_conf_loss'] = running_conf_loss
    score_dict['train_grad_mean'] = running_g_mean
    score_dict['train_grad_std'] = running_g_std
    score_dict['train_grad_max'] = running_g_max
    score_dict['train_grad_min'] = running_g_min

    for k in score_dict.keys():
        if k not in val_df_dict.keys():
            val_df_dict[k] = [score_dict[k]]
        else:
            val_df_dict[k].append(score_dict[k])

    score_tag = '_' + str(score_dict['val_loss']) + '_val-loss'
    val_df = pd.DataFrame(val_df_dict)
    val_df.to_csv(log_fname, index=False)

    fn = 'neuramatch_' + sess_id_ + score_tag + '.pt'
    out_fp = model_dir + '/' + fn
    print('Saving to', out_fp)
    torch.save(nmatch.state_dict(), out_fp)

    nmatch.train()
