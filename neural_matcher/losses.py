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

    def __init__(self, smooth=1., alpha=.6, gamma=.75):
        super(KeypointLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def loss_compute(self, y_pred, y_true):
        smooth = self.smooth
        alpha = self.alpha
        gamma = self.gamma
        tp = torch.sum(y_true * y_pred)
        fp = torch.sum((1. - y_true) * y_pred)
        fn = torch.sum(y_true * (1. - y_pred))
        l = (tp + smooth) / (tp + (alpha * fn) + ((1 - alpha) * fp + smooth))
        tversky_loss = 1. - l
        focal_tversky_loss = torch.float_power(tversky_loss, gamma)
        return focal_tversky_loss, (tp, fp, fn)

    def forward(self, y_out, hm_pred, hm_gt, smooth=1., alpha=.7, gamma=.75):
        (match_pxy_pos_gt, conf_pxy_pos_gt, desc_pxy_pos_gt), \
            (match_pxy_neg_gt_, conf_pxy_neg_gt_, desc_pxy_neg_gt_), \
            (un_match_pxy_neg_gt, un_conf_pxy_neg_gt, un_desc_pxy_neg_gt) = y_out

        loss_hm = self.loss_compute(hm_pred, hm_gt, smooth, alpha, gamma)
        #
        # pos_outs = torch.hstack(conf_pxy_pos_gt)
        # neg_y_outs_ = torch.hstack(conf_pxy_neg_gt_)
        # neg_un_y_outs = torch.hstack(un_conf_pxy_neg_gt)
        #
        # neg_outs = torch.hstack([neg_y_outs_, neg_un_y_outs])
        # pos_outs_sorted, _ = torch.sort(pos_outs, descending=False)
        # neg_outs_sorted, _ = torch.sort(neg_outs, descending=True)
        #
        # ms = torch.min(torch.Tensor([pos_outs.shape[0], neg_outs.shape[0]])).int()
        # yp = pos_outs_sorted[:ms]
        # yn = neg_outs_sorted[:ms]
        #
        # y_pred = torch.hstack([yp, yn])
        # y_true = torch.zeros_like(y_pred)
        # y_true[:yp.shape[0]] = 1.
        #
        # focal_tversky_loss = self.loss_compute(y_true, y_pred, smooth, alpha, gamma,
        #                                        smooth, alpha, gamma)

        return loss_hm

