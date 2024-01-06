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

import torch
from torch import nn

import utils


class KeypointLoss(nn.Module):

    def __init__(self, train_module, device, smooth=1., alpha=.6, gamma=.75, vector_loss_weight=.96,
                 vector_loss_h_weight=.5):
        super(KeypointLoss, self).__init__()
        self.device = device
        self.train_module = train_module
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.vector_loss_weight = torch.scalar_tensor(vector_loss_weight).to(self.device)
        self.vector_loss_h_weight = torch.scalar_tensor(vector_loss_h_weight).to(self.device)

    def loss_compute(self, y_pred, y_true, smooth, alpha, gamma):
        tp = torch.sum(y_true * y_pred)
        fp = torch.sum((1. - y_true) * y_pred)
        fn = torch.sum(y_true * (1. - y_pred))
        l = (tp + smooth) / torch.max((tp + (alpha * fn) + ((1 - alpha) * fp + smooth)),
                                      torch.scalar_tensor(1.).to(self.device))
        tversky_loss = 1. - l
        focal_tversky_loss = torch.float_power(tversky_loss, gamma)
        return focal_tversky_loss, (tp, fp, fn)

    def make_eq(self, x_pred, x_gt, thresh=0.):
        f_gt = x_gt > thresh
        x_gt_nz = x_gt[f_gt]
        x_gt_z = x_gt[~f_gt]
        x_pred_nz = x_pred[f_gt]
        x_pred_z = x_pred[~f_gt]

        x_lz = torch.min(torch.Tensor([x_gt_z.shape[0], x_gt_nz.shape[0]])).to(self.device).int()
        x_pred_z_idx = torch.argsort(x_pred_z, descending=True)[:x_lz]
        x_pred_z = x_pred_z[x_pred_z_idx]
        x_gt_z = x_gt_z[x_pred_z_idx]

        x_pred_eq = torch.hstack([x_pred_z, x_pred_nz])
        x_gt_eq = torch.hstack([x_gt_z, x_gt_nz])
        return x_pred_eq, x_gt_eq

    def matcher_loss_compute(self, conf_pxy_pos_gt, conf_pxy_neg_gt_, un_conf_pxy_neg_gt):
        pos_outs = torch.hstack(conf_pxy_pos_gt)
        neg_y_outs_ = torch.hstack(conf_pxy_neg_gt_)
        neg_un_y_outs = torch.hstack(un_conf_pxy_neg_gt)

        neg_outs = torch.hstack([neg_y_outs_, neg_un_y_outs])
        pos_outs_sorted, _ = torch.sort(pos_outs, descending=False)
        neg_outs_sorted, _ = torch.sort(neg_outs, descending=True)

        ms = torch.min(torch.Tensor([pos_outs.shape[0], neg_outs.shape[0]])).int()
        yp = pos_outs_sorted[:ms]
        yn = neg_outs_sorted[:ms]

        y_pred = torch.hstack([yp, yn])
        y_true = torch.zeros_like(y_pred)
        y_true[:yp.shape[0]] = 1.
        loss_matcher, (tp, fp, fn) = self.loss_compute(y_true, y_pred, self.smooth, self.alpha, self.gamma)
        return loss_matcher, (tp, fp, fn)

    def mask_loss(self, m_pred, m_gt):
        loss, (tp, fp, fn) = self.loss_compute(m_pred, m_gt, self.smooth, self.alpha, self.gamma)
        m_pred_eq, m_gt_eq = self.make_eq(m_pred, m_gt)
        loss_eq, (tp_eq, fp_eq, fn_eq) = self.loss_compute(m_pred_eq, m_gt_eq, self.smooth, self.alpha, self.gamma)
        losses = (loss, loss_eq)
        cnts = ((tp, fp, fn), (tp_eq, fp_eq, fn_eq))
        return losses, cnts

    def forward(self, nn_outs, gt_outs):
        (hm_gt, match_vectors_gt, conf_masks_gt), (match_xys_gt, confs_gt) = gt_outs
        (hm_pred, match_vectors_pred, conf_masks_pred), (match_xys_pred, confs_pred) = nn_outs
        hm_gt = hm_gt.to(self.device)
        match_vectors_gt = match_vectors_gt.to(self.device)
        conf_masks_gt = conf_masks_gt.to(self.device)
        # match_xys_gt = [t.to(self.device) for t in match_xys_gt]
        # confs_gt = [t.to(self.device) for t in confs_gt]
        hm_pred = hm_pred.to(self.device)
        match_vectors_pred = match_vectors_pred.to(self.device)
        conf_masks_pred = conf_masks_pred.to(self.device)
        # match_xys_pred = [t.to(self.device) for t in match_xys_pred]
        # confs_pred = [t.to(self.device) for t in confs_pred]
        if self.train_module == 'heatmap':
            loss_heatmap, (tp, fp, fn) = self.loss_compute(hm_pred, hm_gt, self.smooth, self.alpha, self.gamma)
            loss = loss_heatmap
        elif self.train_module == 'matcher':
            vector_diffs = match_vectors_gt - match_vectors_pred
            vector_loss_map = torch.norm(vector_diffs, dim=1)**2
            vector_loss = torch.mean(vector_loss_map)
            conf_loss, (tp, fp, fn) = self.loss_compute(conf_masks_pred, conf_masks_gt, self.smooth, self.alpha,
                                                        self.gamma)
            loss = .9 * vector_loss + .1 * conf_loss
        elif self.train_module == 'all':
            vector_diffs = match_vectors_gt - match_vectors_pred
            vector_loss_map = torch.norm(vector_diffs, dim=1) / 2.8284271247461903  # root of 8
            # vector_loss_map = (vector_diffs[:, 0] ** 2) + (vector_diffs[:, 1] ** 2)
            residual_weight = (1. - self.vector_loss_weight) #/ 2.

            f = conf_masks_gt == 1.
            v_map = vector_loss_map.reshape(-1)
            vector_loss_nz = vector_loss_map[f]

            hmi = torch.argsort(conf_masks_pred[~f], descending=True)

            vector_loss_z_all = v_map[hmi]
            lz = torch.min(torch.Tensor([vector_loss_nz.shape[0], vector_loss_z_all.shape[0]])).to(self.device).int()

            vector_loss_z = torch.sort(vector_loss_z_all, descending=True)[0][:lz]
            vector_loss_all = torch.hstack([vector_loss_nz, vector_loss_z])
            vector_loss = vector_loss_all.mean()

            # vector_loss_h = torch.nan_to_num(torch.mean(vector_loss_map[vector_loss_map > .1]))
            # vector_loss_l = torch.nan_to_num(torch.mean(vector_loss_map[vector_loss_map <= .1]))
            # vector_loss = ((self.vector_loss_h_weight * vector_loss_h)
            #                + ((1. - self.vector_loss_h_weight) * vector_loss_l))

            (conf_loss, conf_loss_eq), ((tp, fp, fn), (tp_eq, fp_eq, fn_eq)) = self.mask_loss(conf_masks_pred,
                                                                                              conf_masks_gt)
            (loss_heatmap, loss_heatmap_eq), _ = self.mask_loss(hm_pred, hm_gt)

            loss_heatmap = .95 * loss_heatmap_eq + .05 * loss_heatmap
            conf_loss = .95 * conf_loss_eq + .05 * conf_loss

            loss_hm = (.4 * conf_loss + .6 * loss_heatmap)
            loss = (self.vector_loss_weight * vector_loss) + (residual_weight * loss_hm)

        return (loss, vector_loss, conf_loss, vector_loss_map), (tp, fp, fn)
