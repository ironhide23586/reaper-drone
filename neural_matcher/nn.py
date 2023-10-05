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
        self.p_xy = torch.moveaxis(grid_xy.reshape(-1, 2, self.side * self.side), 1, -1)

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

        self.desc_condenser = nn.Sequential(nn.ConvTranspose2d(32, 32, 6, 1, bias=False),
                                            nn.BatchNorm2d(32), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(32, 32, 6, 1, bias=False),
                                            nn.BatchNorm2d(32), nn.LeakyReLU())

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

    def filter_matches(self, p_xy_kp_a, p_xy_kp_b, s, f_a, f_b, heatmap_1d):
        kpi = torch.combinations(torch.arange(0, torch.max(torch.Tensor([p_xy_kp_a.shape[0],
                                                                         p_xy_kp_b.shape[0]]))).to(self.device))
        if p_xy_kp_b.shape[0] < p_xy_kp_a.shape[0]:
            f = kpi[:, 1] < torch.min(torch.Tensor([p_xy_kp_a.shape[0], p_xy_kp_b.shape[0]]))
        else:
            f = kpi[:, 0] < torch.min(torch.Tensor([p_xy_kp_a.shape[0], p_xy_kp_b.shape[0]]))
        kpi = kpi[f].int()
        p_xy_kp_a = p_xy_kp_a[kpi[:, 0]]
        p_xy_kp_b = p_xy_kp_b[kpi[:, 1]]
        p_xy_kp_ab = torch.hstack([p_xy_kp_a, p_xy_kp_b])
        y_sel, f_ab_sel = self.extract_descriptors(p_xy_kp_a, p_xy_kp_b, s, f_a, f_b, heatmap_1d)
        f_sel = y_sel > self.final_thresh
        return p_xy_kp_ab, y_sel, f_ab_sel, f_sel, kpi

    def post_process_matches(self, p_xy_kp_ab, y_sel, f_ab_sel, f_sel):
        p_xy_kp_ab_sel = p_xy_kp_ab[f_sel]
        y_conf_sel = y_sel[f_sel]
        desc_sel = f_ab_sel[f_sel]
        n_p_xy_kp_ab_sel = p_xy_kp_ab[~f_sel]
        n_y_conf_sel = y_sel[~f_sel]
        n_desc_sel = f_ab_sel[~f_sel]
        return p_xy_kp_ab_sel, y_conf_sel, desc_sel, n_p_xy_kp_ab_sel, n_y_conf_sel, n_desc_sel

    def get_matches(self, p_xy_kp_a, p_xy_kp_b, s, f_a, f_b, heatmap_1d):
        p_xy_kp_ab, y_sel, f_ab_sel, f_sel, kpi = self.filter_matches(p_xy_kp_a, p_xy_kp_b, s, f_a, f_b, heatmap_1d)
        p_xy_kp_ab_sel, y_conf_sel, desc_sel, \
            n_p_xy_kp_ab_sel, n_y_conf_sel, n_desc_sel = self.post_process_matches(p_xy_kp_ab, y_sel, f_ab_sel, f_sel)
        kpi_sel = kpi[f_sel]
        kpi_sel_qis = kpi_sel[:, 0]
        qi = torch.unique(kpi_sel_qis)
        pi = []
        pi_n = []
        ni_sel = torch.arange(p_xy_kp_ab_sel.shape[0]).to(self.device)
        for q in qi:
            fq = kpi_sel_qis == q
            y_conf_sel_local = y_conf_sel[fq]
            fi = torch.argmax(y_conf_sel_local)
            pi.append(ni_sel[fq][fi])
            pi_n.append(torch.hstack([ni_sel[fq][:fi], ni_sel[fq][fi + 1:]]))
        pi = torch.Tensor(pi).int()
        if len(pi_n) > 0:
            pi_n = torch.hstack(pi_n)
        else:
            pi_n = torch.Tensor([]).int().to(self.device)
        p_xy_kp_ab_sel_matched, y_conf_sel_matched, desc_sel_matched = p_xy_kp_ab_sel[pi], y_conf_sel[pi], desc_sel[pi]
        p_xy_kp_ab_sel_unmatched, y_conf_sel_unmatched, desc_sel_unmatched = p_xy_kp_ab_sel[pi_n], y_conf_sel[pi_n], \
            desc_sel[pi_n]
        return (p_xy_kp_ab_sel_matched, y_conf_sel_matched, desc_sel_matched), \
            (p_xy_kp_ab_sel_unmatched, y_conf_sel_unmatched, desc_sel_unmatched), \
            (n_p_xy_kp_ab_sel, n_y_conf_sel, n_desc_sel)

    def extract_points(self, bi, f, f_, heatmap_1d):
        p_xy_local_a = self.p_xy[0][f[bi]]
        p_xy_local_b = self.p_xy[1][f_[bi]]

        hm = heatmap_1d[bi][0][f[bi]]
        hmi_a = torch.argsort(hm)[-self.cutoff_n_points:]

        hm = heatmap_1d[bi][1][f_[bi]]
        hmi_b = torch.argsort(hm)[-self.cutoff_n_points:]
        return p_xy_local_a[hmi_a], p_xy_local_b[hmi_b]

    def hash_match(self, p):
        a = utils.cantor_fn(p[:, 0], p[:, 1])
        b = utils.cantor_fn(p[:, 2], p[:, 3])
        c = utils.cantor_fn(a, b)
        return c

    def remove_gt_pairs(self, mp, cp, dp, match_pxy_gt_hashes_bi):
        mp = mp.int()
        mp_hashes = self.hash_match(mp)
        m_fn = lambda h: True in (match_pxy_gt_hashes_bi == h)
        hits_f = torch.Tensor(list(map(m_fn, mp_hashes))) == 0
        mp_ = mp[hits_f]
        cp_ = cp[hits_f]
        dp_ = dp[hits_f]
        return mp_, cp_, dp_

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

        f_a = self.desc_condenser(f_a_raw)
        f_b = self.desc_condenser(f_b_raw)

        s = self.side
        r = x_a.shape[-1] / s
        nb = x_a.shape[0]

        f_a_1d = torch.moveaxis(f_a.reshape(-1, 32, s * s), 1, -1)
        f_b_1d = torch.moveaxis(f_b.reshape(-1, 32, s * s), 1, -1)
        heatmap_1d = heatmap.reshape(-1, 2, s * s)

        f = heatmap[:, 0] > self.heatmap_thresh
        f = f.reshape(-1, s * s)

        f_n = heatmap[:, 0] <= self.heatmap_thresh
        f_n = f_n.reshape(-1, s * s)

        f_ = heatmap[:, 1] > self.heatmap_thresh
        f_ = f_.reshape(-1, s * s)

        f_n_ = heatmap[:, 1] <= self.heatmap_thresh
        f_n_ = f_n_.reshape(-1, s * s)

        match_pxy_pos_gt = []
        conf_pxy_pos_gt = []
        desc_pxy_pos_gt = []
        match_pxy_gt_hashes = []

        match_pxy = []
        conf_pxy = []
        desc_pxy = []

        un_match_pxy = []
        un_conf_pxy = []
        un_desc_pxy = []

        n_match_pxy = []
        n_conf_pxy = []
        n_desc_pxy = []

        match_pxy_ = []
        conf_pxy_ = []
        desc_pxy_ = []

        un_match_pxy_ = []
        un_conf_pxy_ = []
        un_desc_pxy_ = []

        n_match_pxy_ = []
        n_conf_pxy_ = []
        n_desc_pxy_ = []

        for bi in torch.arange(nb).to(self.device):

            if gt_xy_pairs is not None:
                p_a = (gt_xy_pairs[bi][:, :2] / r).int()
                p_b = (gt_xy_pairs[bi][:, 2:] / r).int()

                y_conf_sel_gt, desc_sel_gt = self.extract_descriptors(p_a, p_b, s, f_a_1d[bi], f_b_1d[bi], heatmap_1d[bi])
                match_pxy_pos_gt.append(gt_xy_pairs[bi])
                conf_pxy_pos_gt.append(y_conf_sel_gt)
                desc_pxy_pos_gt.append(desc_sel_gt)
                match_pxy_gt_hashes.append(self.hash_match(gt_xy_pairs[bi]))

            p_a, p_b = self.extract_points(bi, f, f_, heatmap_1d)
            matches, unmatches, nomatches = self.get_matches(p_a, p_b, s, f_a_1d[bi], f_b_1d[bi], heatmap_1d[bi])

            (p_xy_kp_ab_sel_matched, y_conf_sel_matched, desc_sel_matched), \
                (p_xy_kp_ab_sel_unmatched, y_conf_sel_unmatched, desc_sel_unmatched), \
                (n_p_xy_kp_ab_sel, n_y_conf_sel, n_desc_sel) = matches, unmatches, nomatches

            match_pxy.append(p_xy_kp_ab_sel_matched * r)
            conf_pxy.append(y_conf_sel_matched)
            desc_pxy.append(desc_sel_matched)
            un_match_pxy.append(p_xy_kp_ab_sel_unmatched * r)
            un_conf_pxy.append(y_conf_sel_unmatched)
            un_desc_pxy.append(desc_sel_unmatched)
            n_match_pxy.append(n_p_xy_kp_ab_sel * r)
            n_conf_pxy.append(n_y_conf_sel)
            n_desc_pxy.append(n_desc_sel)

            p_a_, p_b_ = self.extract_points(bi, f_n, f_n_, heatmap_1d)
            matches_, unmatches_, nomatches_ = self.get_matches(p_a_, p_b_, s, f_a_1d[bi], f_b_1d[bi], heatmap_1d[bi])

            (p_xy_kp_ab_sel_matched_, y_conf_sel_matched_, desc_sel_matched_), \
                (p_xy_kp_ab_sel_unmatched_, y_conf_sel_unmatched_, desc_sel_unmatched_), \
                (n_p_xy_kp_ab_sel_, n_y_conf_sel_, n_desc_sel_) = matches_, unmatches_, nomatches_

            yi = torch.argsort(y_conf_sel_matched_)[-y_conf_sel_matched.shape[0]:]
            yi_ = torch.argsort(y_conf_sel_unmatched_)[-y_conf_sel_unmatched.shape[0]:]
            yi_n = torch.argsort(n_y_conf_sel_)[-n_y_conf_sel.shape[0]:]

            match_pxy_.append(p_xy_kp_ab_sel_matched_[yi] * r)
            conf_pxy_.append(y_conf_sel_matched_[yi])
            desc_pxy_.append(desc_sel_matched_[yi])
            un_match_pxy_.append(p_xy_kp_ab_sel_unmatched_[yi_] * r)
            un_conf_pxy_.append(y_conf_sel_unmatched_[yi_])
            un_desc_pxy_.append(desc_sel_unmatched_[yi_])
            n_match_pxy_.append(n_p_xy_kp_ab_sel_[yi_n] * r)
            n_conf_pxy_.append(n_y_conf_sel_[yi_n])
            n_desc_pxy_.append(n_desc_sel_[yi_n])

        match_pxy_neg_gt_, conf_pxy_neg_gt_, desc_pxy_neg_gt_ = [], [], []
        un_match_pxy_neg_gt, un_conf_pxy_neg_gt, un_desc_pxy_neg_gt = [], [], []

        y_out = None
        if gt_xy_pairs is not None:
            for bi in torch.arange(nb).to(self.device):
                mp, cp, dp = self.remove_gt_pairs(match_pxy_[bi], conf_pxy_[bi], desc_pxy_[bi],
                                                  match_pxy_gt_hashes[bi])
                match_pxy_neg_gt_.append(mp)
                conf_pxy_neg_gt_.append(cp)
                desc_pxy_neg_gt_.append(dp)
                mp, cp, dp = self.remove_gt_pairs(un_match_pxy[bi], un_conf_pxy[bi], un_desc_pxy[bi],
                                                  match_pxy_gt_hashes[bi])
                un_match_pxy_neg_gt.append(mp)
                un_conf_pxy_neg_gt.append(cp)
                un_desc_pxy_neg_gt.append(dp)
            y_out = ((match_pxy_pos_gt, conf_pxy_pos_gt, desc_pxy_pos_gt),
                     (match_pxy_neg_gt_, conf_pxy_neg_gt_, desc_pxy_neg_gt_),
                     (un_match_pxy_neg_gt, un_conf_pxy_neg_gt, un_desc_pxy_neg_gt))

        return heatmap, ((match_pxy, conf_pxy, desc_pxy), (un_match_pxy, un_conf_pxy, un_desc_pxy),
                         (n_match_pxy, n_conf_pxy, n_desc_pxy)), \
            ((match_pxy_, conf_pxy_, desc_pxy_), (un_match_pxy_, un_conf_pxy_, un_desc_pxy_),
             (n_match_pxy_, n_conf_pxy_, n_desc_pxy_)), y_out
