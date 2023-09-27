import torch
from torch import nn


class KeypointLoss(nn.Module):

    def __init__(self):
        super(KeypointLoss, self).__init__()

    def forward(self, y_out, smooth=1., alpha=.7, gamma=.75):
        (match_pxy_pos_gt, conf_pxy_pos_gt, desc_pxy_pos_gt), \
            (match_pxy_neg_gt_, conf_pxy_neg_gt_, desc_pxy_neg_gt_), \
            (un_match_pxy_neg_gt, un_conf_pxy_neg_gt, un_desc_pxy_neg_gt) = y_out

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

        tp = torch.sum(y_true * y_pred)
        fp = torch.sum((1. - y_true) * y_pred)
        fn = torch.sum(y_true * (1. - y_pred))

        l = (tp + smooth) / (tp + (alpha * fn) + ((1 - alpha) * fp + smooth))
        tversky_loss = 1. - l
        focal_tvesky_loss = torch.float_power(tversky_loss, gamma)

        return focal_tvesky_loss

