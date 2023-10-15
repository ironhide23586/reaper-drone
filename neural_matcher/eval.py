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

import json

import torch
import cv2
import numpy as np
from tqdm import tqdm

import utils


def viz_heatmap(heatmap):
    hm = np.tile(np.expand_dims(heatmap, -1), [1, 1, 3])
    lc = np.tile([[[0, 0, 255]]], [hm.shape[0], hm.shape[1], 1])
    hc = np.tile([[[255, 255, 255]]], [hm.shape[0], hm.shape[1], 1])
    hm_ = (hc * hm) + (lc * (1. - hm))
    f = np.tile(np.expand_dims(heatmap == 0, -1), [1, 1, 3])
    hm_[f] = 0.
    return hm_.astype(np.uint8)


def viz_match_vectors(match_vectors):
    mv = (np.tile(match_vectors.detach().cpu().numpy(), [3, 1, 1]) + 1.) / 2.
    v = np.zeros([mv.shape[1], mv.shape[2], 3]).astype(np.uint8)
    lc = np.tile([[[0, 0, 255]]], [v.shape[0], v.shape[1], 1])
    hc = np.tile([[[255, 140, 0]]], [v.shape[0], v.shape[1], 1])
    m0 = np.tile(np.expand_dims(mv[0], -1), (1, 1, 3)) * lc
    m1 = np.tile(np.expand_dims(mv[1], -1), (1, 1, 3)) * hc
    viz = ((m0 + m1) / 2.).astype(np.uint8)
    return viz


def viz_matches(a, b, inference_outs, blend_coeff=.65):
    (heatmap, match_vectors_pred, conf_mask), (match_xy_pairs_, confs) = inference_outs
    match_xy_pairs = match_xy_pairs_[0]
    hm_a = viz_heatmap(heatmap[0][0].detach().cpu().numpy())
    hm_b = viz_heatmap(heatmap[0][1].detach().cpu().numpy())

    blended_viz_a = blend_coeff * hm_a + (1. - blend_coeff) * a
    blended_viz_b = blend_coeff * hm_b + (1. - blend_coeff) * b

    img = utils.drawMatches(blended_viz_a, match_xy_pairs[:, :2].detach().cpu().numpy(),
                            blended_viz_b, match_xy_pairs[:, 2:].detach().cpu().numpy(),
                            confs=confs)
    match_vectors_pred_viz = viz_match_vectors(match_vectors_pred[0])
    conf_mask_viz = viz_heatmap(conf_mask[0].detach().cpu().numpy())

    return img, hm_a, hm_b, match_vectors_pred_viz, conf_mask_viz


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
    im_a = utils.TENSOR_TRANSFORM(ima_pp).to(device)
    im_b = utils.TENSOR_TRANSFORM(imb_pp).to(device)
    t = torch.unsqueeze(torch.stack([utils.INPUT_TRANSFORMS(im_a), utils.INPUT_TRANSFORMS(im_b)]), 0)
    nmatch.eval()
    with torch.no_grad():
        inference_outs = nmatch(t)
    match_viz, heatmap_a, heatmap_b, match_vectors_viz, conf_mask_viz = viz_matches(ima_pp, imb_pp, inference_outs)
    nmatch.train()
    return match_viz, heatmap_a, heatmap_b, match_vectors_viz, conf_mask_viz, inference_outs, (ima_pp, imb_pp)


def score_model(nmatch, data_loader, loss_fn):
    print('Scoring model...')
    score_dict = {}
    val_loss = 0.
    val_vector_loss = 0.
    val_conf_loss = 0.
    tps = 0
    fps = 0
    fns = 0
    ni = 0
    with torch.no_grad():
        for bi, (ims, gt_outs) in enumerate(tqdm(data_loader)):
            nn_outs = nmatch(ims)
            (loss, vector_loss, conf_loss, vector_loss_map), (tp, fp, fn) = loss_fn(nn_outs, gt_outs)

            tps += tp
            fps += fp
            fns += fn
            val_loss += float(loss.cpu().numpy())
            val_vector_loss += float(vector_loss.cpu().numpy())
            val_conf_loss += float(conf_loss.cpu().numpy())
            ni += 1

            val_loss_ = val_loss / ni
            val_vector_loss_ = val_vector_loss / ni
            val_conf_loss_ = val_conf_loss / ni
            prec = float((tps / (tps + fps)).cpu().numpy())
            rec = float((tps / (tps + fns)).cpu().numpy())
            fsc = (2 * prec * rec) / (prec + rec)

            score_dict['val_loss'] = val_loss_
            score_dict['val_vector_loss'] = val_vector_loss_
            score_dict['val_conf_loss'] = val_conf_loss_
            score_dict['fscore'] = fsc
            score_dict['precision'] = prec
            score_dict['recall'] = rec
            score_dict['num_samples'] = int(ni * data_loader.batch_size)

            if bi % 5 == 0:
                print(json.dumps(score_dict, indent=4, sort_keys=True))
    return score_dict
