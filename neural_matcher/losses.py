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

    def __init__(self, train_module, smooth=1., alpha=.6, gamma=.75):
        super(KeypointLoss, self).__init__()
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

    def forward(self, y_out, hm_pred, hm_gt):
        match_vectors_pred, match_vectors_gt = y_out

        if self.train_module == 'heatmap':
            loss_heatmap, (tp, fp, fn) = self.loss_compute(hm_pred, hm_gt, self.smooth, self.alpha, self.gamma)
            loss = loss_heatmap
        elif self.train_module == 'matcher':
            loss_matcher, (tp, fp, fn) = self.loss_compute(match_vectors_pred, match_vectors_gt, self.smooth,
                                                           self.alpha, self.gamma)
            loss = loss_matcher
        else:
            loss_heatmap, (tp_, fp_, fn_) = self.loss_compute(hm_pred, hm_gt, self.smooth, self.alpha, self.gamma)
            loss_matcher, (tp__, fp__, fn__) = self.loss_compute(match_vectors_pred, match_vectors_gt, self.smooth,
                                                                 self.alpha, self.gamma)
            loss = loss_heatmap + loss_matcher
            tp = tp_ + tp__
            fp = fp_ + fp__
            fn = fn_ + fn__

        return loss, (tp, fp, fn)

