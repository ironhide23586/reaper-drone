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
import json

import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

import utils

tensor_transform = transforms.ToTensor()
input_transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def viz_heatmap(heatmap):
    hm = np.tile(np.expand_dims(heatmap, -1), [1, 1, 3])
    lc = np.tile([[[0, 0, 255]]], [hm.shape[0], hm.shape[1], 1])
    hc = np.tile([[[255, 140, 0]]], [hm.shape[0], hm.shape[1], 1])
    hm_ = (hc * hm) + (lc * (1. - hm))
    return (hm_ * 255).astype(np.uint8)


def viz_match_vectors(match_vectors):
    mv = (np.tile(match_vectors.detach().cpu().numpy(), [3, 1, 1]) + 1.) / 2.
    v = np.zeros([mv.shape[1], mv.shape[2], 3]).astype(np.uint8)
    lc = np.tile([[[0, 0, 255]]], [v.shape[0], v.shape[1], 1])
    hc = np.tile([[[255, 140, 0]]], [v.shape[0], v.shape[1], 1])
    m0 = np.tile(np.expand_dims(mv[0], -1), (1, 1, 3)) * lc
    m1 = np.tile(np.expand_dims(mv[1], -1), (1, 1, 3)) * hc
    viz = ((m0 + m1) / 2.).astype(np.uint8)
    return viz


def viz_matches(a, b, out_masks, out_matches, blend_coeff=.65):
    (heatmap, match_vectors_pred, conf_mask), (match_xy_pairs_, confs) = out_masks, out_matches
    match_xy_pairs = match_xy_pairs_[0]
    hm_a = viz_heatmap(heatmap[0][0].detach().cpu().numpy())
    hm_b = viz_heatmap(heatmap[0][1].detach().cpu().numpy())

    blended_viz_a = blend_coeff * hm_a + (1. - blend_coeff) * a
    blended_viz_b = blend_coeff * hm_b + (1. - blend_coeff) * b

    img = utils.drawMatches(blended_viz_a, match_xy_pairs[:, :2].detach().cpu().numpy(),
                            blended_viz_b, match_xy_pairs[:, 2:].detach().cpu().numpy())
    match_vectors_pred_viz = viz_match_vectors(match_vectors_pred[0])
    conf_mask_viz = (conf_mask[0].detach().cpu().numpy() * 255).astype(np.uint8)

    return img, hm_a, hm_b, match_vectors_pred_viz, conf_mask_viz


def pxys_to_match_vectors_mask(pxys):
    match_vectors_mask = torch.zeros([len(pxys), 2, utils.SIDE, utils.SIDE])
    for bi in range(len(pxys)):
        v = (pxys[bi][:, 2:] - pxys[bi][:, :2]) / (utils.SIDE - 1)
        match_vectors_mask[bi, :, pxys[bi][:, 1], pxys[bi][:, 0]] = torch.Tensor(v.T)
    return match_vectors_mask


def collater(data):
    ims = []
    pxys = []
    heatmaps = []
    for d in data:
        x = d[0] / 255.
        ims.append(input_transforms(torch.Tensor(x)))
        pxys.append(torch.Tensor(d[1].astype(int)).int())
        heatmaps.append(d[2])
    heatmaps = torch.Tensor(np.stack(heatmaps))
    ims = torch.stack(ims)
    match_vectors_gt = pxys_to_match_vectors_mask(pxys)
    return ims, pxys, heatmaps, match_vectors_gt


def input_preprocess(im):
    min_s = min(im.width, im.height)
    x_offset = (im.width - min_s) // 2
    y_offset = (im.height - min_s) // 2
    frame_square = np.array(im)[y_offset: y_offset + min_s, x_offset: x_offset + min_s]
    frame_square = cv2.resize(frame_square, (utils.SIDE, utils.SIDE))[:, :, [2, 1, 0]]
    return frame_square


def infer_nn(nmatch, ima, imb, device):
    ima_pp = input_preprocess(ima)
    imb_pp = input_preprocess(imb)
    im_a = tensor_transform(ima_pp).to(device)
    im_b = tensor_transform(imb_pp).to(device)
    t = torch.unsqueeze(torch.stack([input_transforms(im_a), input_transforms(im_b)]), 0)
    nmatch.eval()
    with torch.no_grad():
        out_masks, out_matches = nmatch(t)
    match_xy_pairs = out_matches[0][0]
    match_viz, heatmap_a, heatmap_b, match_vectors_viz, conf_mask_viz = viz_matches(ima_pp, imb_pp, out_masks,
                                                                                    out_matches)
    nmatch.train()
    return match_viz, heatmap_a, heatmap_b, match_vectors_viz, conf_mask_viz, match_xy_pairs, (ima_pp, imb_pp)


def score_model(nmatch, data_loader, loss_fn, device):
    print('Scoring model...')
    score_dict = {'final_score': -1.,
                  'precision': -1.,
                  'recall': -1.,
                  'val_loss': -1.}
    val_loss = 0.
    tps = 0
    fps = 0
    fns = 0
    ni = 0
    with torch.no_grad():
        for bi, (ims, pxys, heatmaps_gt) in enumerate(tqdm(data_loader)):
            heatmaps_pred, y_out, (conf_match, match_xy_pairs, confs) = nmatch(ims, pxys)
            loss, (tp, fp, fn) = loss_fn(y_out, heatmaps_pred, heatmaps_gt.to(device))
            tps += tp
            fps += fp
            fns += fn
            val_loss += float(loss.cpu().numpy())
            ni += 1

            val_loss_ = val_loss / ni
            prec = float((tps / (tps + fps)).cpu().numpy())
            rec = float((tps / (tps + fns)).cpu().numpy())
            fsc = (2 * prec * rec) / (prec + rec)

            score_dict['val_loss'] = val_loss_
            score_dict['final_score'] = fsc
            score_dict['precision'] = prec
            score_dict['recall'] = rec
            score_dict['num_samples'] = int(ni * data_loader.batch_size)

            if bi % 5 == 0:
                print(json.dumps(score_dict, indent=4, sort_keys=True))
    return score_dict


def matplotlib_imshow(img):
    fig = plt.figure()
    plt.imshow(img)
    return fig


class SIFTMatcher:

    def __init__(self, sift_tolerance=.8, min_n_matches=5):
        self.side = utils.SIDE
        self.sift_tolerance = sift_tolerance
        self.min_n_matches = int(min_n_matches)
        self.tag = '_'.join([str(sift_tolerance) + '.sift-tolerance',
                             str(min_n_matches) + '.min-n-matches',
                             str(self.side) + '.side'])
        self.keypoint_detector = cv2.xfeatures2d.SIFT_create()
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


def checkpoint_model(nmatch, train_loss, device, data_loader_val, ima, imb, model_dir, loss_fn, ei, bi, sess_id,
                     log_fname, val_df_dict, viz_dir, writer, g_idx, ksize, radius_scale, blend_coeff):
    nmatch.eval()
    suffix = '-'.join([str(ei) + 'e', str(bi) + 'b'])
    sess_id_ = sess_id + '_' + suffix

    match_viz, heatmap_a, heatmap_b, match_vectors_viz, conf_mask_viz, matches_xy, (ima_pp, imb_pp) = infer_nn(nmatch,
                                                                                                               ima,
                                                                                                               imb,
                                                                                                               device)
    fn_prefix = '_'.join(['viz', sess_id])
    suffix = '-'.join([str(ei) + 'e', str(bi) + 'b', str(matches_xy.shape[0]) + 'kp'])
    print('VIZ:', suffix)

    gt_viz_dir = viz_dir + '/gt'
    if not os.path.isdir(gt_viz_dir):
        os.makedirs(gt_viz_dir)
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
        # conf_mask_gt = conf_mask_gt.reshape(s, s)
        txy = targ_xy_2d[:, f]
        matches_xy_gt_ = np.vstack([nmatch.p_xy[0].detach().cpu().numpy()[:, f], txy]).T

        match_viz_gt_ = utils.drawMatches(blended_viz_a, matches_xy_gt_[:, :2], blended_viz_b, matches_xy_gt_[:, 2:])
        cv2.imwrite('a.png', match_viz_gt)
        cv2.imwrite('a_.png', match_viz_gt_)
        # TODO: [bug] make sure match_viz_gt_ == match_viz_gt

        suffix += '-gt'
        mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'matches', suffix + '.jpg'])])
        cv2.imwrite(mn, match_viz_gt)
        mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'heatmap-a', suffix + '.jpg'])])
        cv2.imwrite(mn, heatmap_a_gt)
        mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'heatmap-b', suffix + '.jpg'])])
        cv2.imwrite(mn, heatmap_b_gt)
        mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'vectors', suffix + '.jpg'])])
        cv2.imwrite(mn, match_vectors_gt_viz)
        # mn = os.sep.join([gt_viz_dir, '_'.join([fn_prefix, 'match-confidence', suffix + '.jpg'])])
        # cv2.imwrite(mn, conf_mask_viz)

        writer.add_figure('match_viz-gt', matplotlib_imshow(match_viz), global_step=g_idx)
        writer.add_figure('heatmap-a-gt', matplotlib_imshow(heatmap_a), global_step=g_idx)
        writer.add_figure('heatmap-b-gt', matplotlib_imshow(heatmap_b), global_step=g_idx)
        writer.add_figure('match_vectors_viz-gt', matplotlib_imshow(match_vectors_viz), global_step=g_idx)
        # writer.add_figure('match_confidence', matplotlib_imshow(conf_mask_viz), global_step=g_idx)


    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'matches', suffix + '.jpg'])])
    cv2.imwrite(mn, match_viz)
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'heatmap-a', suffix + '.jpg'])])
    cv2.imwrite(mn, heatmap_a)
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'heatmap-b', suffix + '.jpg'])])
    cv2.imwrite(mn, heatmap_b)
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'vectors', suffix + '.jpg'])])
    cv2.imwrite(mn, match_vectors_viz)
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'match-confidence', suffix + '.jpg'])])
    cv2.imwrite(mn, conf_mask_viz)

    writer.add_figure('match_viz', matplotlib_imshow(match_viz), global_step=g_idx)
    writer.add_figure('heatmap-a', matplotlib_imshow(heatmap_a), global_step=g_idx)
    writer.add_figure('heatmap-b', matplotlib_imshow(heatmap_b), global_step=g_idx)
    writer.add_figure('match_vectors_viz', matplotlib_imshow(match_vectors_viz), global_step=g_idx)
    writer.add_figure('match_confidence', matplotlib_imshow(conf_mask_viz), global_step=g_idx)

    score_dict = score_model(nmatch, data_loader_val, loss_fn, device)
    val_df_dict['epoch'].append(ei)
    val_df_dict['epoch_batch_iteration'].append(bi)
    val_df_dict['final_score'].append(score_dict['final_score'])
    val_df_dict['precision'].append(score_dict['precision'])
    val_df_dict['recall'].append(score_dict['recall'])
    val_df_dict['val_loss'].append(score_dict['val_loss'])

    writer.add_scalar('final_score', score_dict['final_score'], g_idx)
    writer.add_scalar('precision', score_dict['precision'], g_idx)
    writer.add_scalar('recall', score_dict['recall'], g_idx)
    writer.add_scalar('val_loss', score_dict['val_loss'], g_idx)

    if train_loss is not None:
        val_df_dict['train_loss'].append(train_loss)
    else:
        val_df_dict['train_loss'].append(-1.)
    val_df_dict['num_samples'].append(score_dict['num_samples'])

    score_tag = '_' + str(score_dict['final_score']) + '-fsc'
    val_df = pd.DataFrame(val_df_dict)
    val_df.to_csv(log_fname, index=False)

    fn = 'neuramatch-' + sess_id_ + score_tag + '.pt'
    out_fp = model_dir + '/' + fn
    print('Saving to', out_fp)
    torch.save(nmatch.state_dict(), out_fp)

    nmatch.train()
