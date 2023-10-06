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
import torchvision.transforms as transforms
from torch import nn

import utils


class NeuraMatch(nn.Module):

    def __init__(self, device, side, cutoff_n_points):
        super().__init__()
        self.device = device
        self.side = side
        self.cutoff_n_points = cutoff_n_points

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.side), torch.arange(0, self.side), indexing='xy')
        grid_x = torch.tile(torch.unsqueeze(torch.unsqueeze(grid_x.to(self.device), 0), 0), (2, 1, 1, 1))
        grid_y = torch.tile(torch.unsqueeze(torch.unsqueeze(grid_y.to(self.device), 0), 0), (2, 1, 1, 1))

        grid_xy = torch.concat([grid_x, grid_y], dim=1)
        self.p_xy = grid_xy.reshape(-1, 2, self.side * self.side)

        self.heatmap_thresh = nn.Parameter(torch.tensor(.5), requires_grad=False)
        self.final_thresh = nn.Parameter(torch.tensor(.5), requires_grad=False)

        self.conv0_block_a = nn.Sequential(nn.Conv2d(3, 64, 6, 1, bias=False),
                                           nn.BatchNorm2d(64), nn.LeakyReLU(),
                                           nn.Conv2d(64, 64, 4, 1, bias=False),
                                           nn.BatchNorm2d(64), nn.LeakyReLU(),
                                           nn.Conv2d(64, 32, 3, 1, bias=False),
                                           nn.BatchNorm2d(32), nn.LeakyReLU())
        self.conv0_block_b = nn.Sequential(nn.Conv2d(3, 64, 6, 1, bias=False),
                                           nn.BatchNorm2d(64), nn.LeakyReLU(),
                                           nn.Conv2d(64, 64, 4, 1, bias=False),
                                           nn.BatchNorm2d(64), nn.LeakyReLU(),
                                           nn.Conv2d(64, 32, 3, 1, bias=False),
                                           nn.BatchNorm2d(32), nn.LeakyReLU())

        self.conv0_block_ab = nn.Sequential(nn.Conv2d(6, 64, 3, 1, bias=False),
                                            nn.BatchNorm2d(64), nn.LeakyReLU(),
                                            nn.Conv2d(64, 64, 3, 1, bias=False),
                                            nn.BatchNorm2d(64), nn.LeakyReLU(),
                                            nn.Conv2d(64, 128, 3, 1, bias=False),
                                            nn.BatchNorm2d(128), nn.LeakyReLU(),
                                            nn.Conv2d(128, 32, 3, 1, bias=False),
                                            nn.BatchNorm2d(32), nn.LeakyReLU(),
                                            nn.Conv2d(32, 2, 3, 1, bias=True),
                                            nn.Sigmoid())

        self.matcher = nn.Sequential(nn.Linear(64, 32, bias=True),
                                     nn.LeakyReLU(),
                                     nn.Linear(32, 16, bias=True),
                                     nn.LeakyReLU(),
                                     nn.Linear(16, 1, bias=True),
                                     nn.Sigmoid())

        self.vector_condenser = nn.Sequential(nn.ConvTranspose2d(66, 32, 6, 1, bias=False),
                                              nn.BatchNorm2d(32), nn.LeakyReLU(),
                                              nn.ConvTranspose2d(32, 32, 1, 1, bias=False),
                                              nn.BatchNorm2d(32), nn.LeakyReLU(),
                                              nn.ConvTranspose2d(32, 2, 6, 1, bias=True))

        self.heatmap_condenser = nn.Sequential(nn.ConvTranspose2d(66, 16, 6, 1, bias=False),
                                               nn.BatchNorm2d(16), nn.LeakyReLU(),
                                               nn.ConvTranspose2d(16, 2, 6, 1, bias=True),
                                               nn.Sigmoid())

    def extract_descriptors(self, p_xy_kp_a, p_xy_kp_b, s, f_a, f_b, heatmap_1d):
        p_xy_kp_a_1d = p_xy_kp_a[:, 0] + p_xy_kp_a[:, 1] * s
        p_xy_kp_b_1d = p_xy_kp_b[:, 0] + p_xy_kp_b[:, 1] * s

        f_a_sel = f_a[p_xy_kp_a_1d]
        f_b_sel = f_b[p_xy_kp_b_1d]
        heatmap_1d_sel_a = heatmap_1d[0][p_xy_kp_a_1d]
        heatmap_1d_sel_b = heatmap_1d[1][p_xy_kp_b_1d]

        f_ab_sel = torch.concat([f_a_sel, f_b_sel], -1)
        y_sel = (torch.squeeze(self.matcher(f_ab_sel)) + heatmap_1d_sel_a + heatmap_1d_sel_b) / 3.
        return y_sel, f_ab_sel

    def forward(self, x_, gt_xy_pairs_=None):
        x = x_.to(self.device)
        x_a = x[:, 0]
        x_b = x[:, 1]
        gt_xy_pairs = None
        if gt_xy_pairs_ is not None:
            gt_xy_pairs = [p.to(self.device) for p in gt_xy_pairs_]

        f_a_raw = self.conv0_block_a(x_a)
        f_b_raw = self.conv0_block_b(x_b)

        heatmap_raw = self.conv0_block_ab(torch.concat([x_a, x_b], dim=1))
        hm_in = torch.concat([f_a_raw, f_b_raw, heatmap_raw], dim=1)
        heatmap = self.heatmap_condenser(hm_in)

        match_vectors_pred = torch.clamp(self.vector_condenser(hm_in), -1., 1.)

        s = self.side
        nb = x_a.shape[0]

        mv = (match_vectors_pred.reshape(-1, 2, s * s) * (s - 1)).int()

        p_xy_tiled = torch.tile(torch.unsqueeze(self.p_xy[0], 0), (nb, 1, 1))

        targ_xy = torch.clamp(p_xy_tiled + mv, 0, s - 1)
        targ_xy_1d = targ_xy[:, 0, :] + targ_xy[:, 1, :] * s
        hm_targ = heatmap.reshape(-1, 2, s * s)[:, 1]
        conf_targ = torch.stack([hm_targ[i][targ_xy_1d[i]].reshape(s, s) for i in range(nb)])

        conf_match = (heatmap[:, 0] + conf_targ) / 2.

        src_xys = []
        targ_xys = []
        confs = []

        cm = conf_match.reshape(-1, s * s)
        for bi in torch.arange(nb).to(self.device):
            f = cm[bi] > self.heatmap_thresh
            confs.append(cm[bi, f])
            src_xys.append(self.p_xy[0, :, f])
            targ_xys.append(targ_xy[bi, :, f])
        src_xys = torch.vstack(src_xys)
        targ_xys = torch.vstack(targ_xys)
        match_xy_pairs = torch.vstack([src_xys, targ_xys]).T

        match_vectors_gt = torch.zeros_like(match_vectors_pred)
        if gt_xy_pairs_ is not None:
            for bi in torch.arange(nb).to(self.device):
                p = gt_xy_pairs[bi]
                v = (p[:, 2:] - p[:, :2]) / s
                match_vectors_gt[bi, :, p[:, 1], p[:, 0]] = v.T

        return heatmap, (match_vectors_pred, match_vectors_gt), (conf_match, match_xy_pairs, confs)
