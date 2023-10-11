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


class KeypointLoss(nn.Module):

    def __init__(self, train_module, device, smooth=1., alpha=.6, gamma=.75):
        super(KeypointLoss, self).__init__()
        self.device = device
        self.train_module = train_module
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def loss_compute(self, y_pred, y_true, smooth, alpha, gamma):
        tp = torch.sum(y_true * y_pred)
        fp = torch.sum((1. - y_true) * y_pred)
        fn = torch.sum(y_true * (1. - y_pred))
        l = (tp + smooth) / (tp + (alpha * fn) + ((1 - alpha) * fp + smooth))
        tversky_loss = 1. - l
        focal_tversky_loss = torch.float_power(tversky_loss, gamma)
        return focal_tversky_loss, (tp, fp, fn)

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
            vector_loss_map = torch.pow(torch.norm(vector_diffs, dim=1), .6)
            vector_loss = torch.mean(vector_loss_map)
            conf_loss, (tp, fp, fn) = self.loss_compute(conf_masks_pred, conf_masks_gt, self.smooth, self.alpha,
                                                        self.gamma)
            loss_heatmap, _ = self.loss_compute(hm_pred, hm_gt, self.smooth, self.alpha, self.gamma)
            loss = .6 * vector_loss + .2 * conf_loss + .2 * loss_heatmap

        return (loss, vector_loss, conf_loss, vector_loss_map), (tp, fp, fn)

