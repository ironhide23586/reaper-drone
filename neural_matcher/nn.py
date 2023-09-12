import torch
from torch import nn


class NeuraMatch(nn.Module):

    def __init__(self):
        super().__init__()
        self.cutoff_n_points = 20
        self.heatmap_thresh = nn.Parameter(torch.tensor(.5))
        self.final_thresh = nn.Parameter(torch.tensor(.5))
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
                                            nn.Conv2d(32, 1, 3, 1, bias=True),
                                            nn.Sigmoid())

        self.matcher = nn.Sequential(nn.Linear(64, 32, bias=False),
                                     nn.BatchNorm1d(32),
                                     nn.Linear(32, 16, bias=False),
                                     nn.BatchNorm1d(16),
                                     nn.Linear(16, 1, bias=True),
                                     nn.Sigmoid())

    def filter_matches(self, p_xy_local, s, f_a, f_b, heatmap_1d):
        kpi = torch.combinations(torch.arange(0, p_xy_local.shape[0]), with_replacement=False)
        p_xy_kp_a = p_xy_local[kpi[:, 0]]
        p_xy_kp_b = p_xy_local[kpi[:, 1]]
        p_xy_kp_ab = torch.hstack([p_xy_kp_a, p_xy_kp_b])
        p_xy_kp_a_1d = p_xy_kp_a[:, 0] + p_xy_kp_a[:, 1] * s
        p_xy_kp_b_1d = p_xy_kp_b[:, 0] + p_xy_kp_b[:, 1] * s

        f_a_sel = f_a[p_xy_kp_a_1d]
        f_b_sel = f_b[p_xy_kp_b_1d]
        heatmap_1d_sel_a = heatmap_1d[p_xy_kp_a_1d]
        heatmap_1d_sel_b = heatmap_1d[p_xy_kp_b_1d]

        f_ab_sel = torch.concat([f_a_sel, f_b_sel], -1)
        y_sel = (torch.squeeze(self.matcher(f_ab_sel)) + heatmap_1d_sel_a + heatmap_1d_sel_b) / 3.
        f_sel = y_sel > self.final_thresh
        return p_xy_kp_ab, y_sel, f_ab_sel, f_sel

    def post_process_matches(self, p_xy_kp_ab, y_sel, f_ab_sel, f_sel):
        p_xy_kp_ab_sel = p_xy_kp_ab[f_sel]
        y_conf_sel = y_sel[f_sel]
        desc_sel = f_ab_sel[f_sel]
        n_p_xy_kp_ab_sel = p_xy_kp_ab[~f_sel]
        n_y_conf_sel = y_sel[~f_sel]
        n_desc_sel = f_ab_sel[~f_sel]
        return p_xy_kp_ab_sel, y_conf_sel, desc_sel, n_p_xy_kp_ab_sel, n_y_conf_sel, n_desc_sel

    def get_matches(self, p_xy_local, s, f_a, f_b, heatmap_1d):
        p_xy_kp_ab, y_sel, f_ab_sel, f_sel = self.filter_matches(p_xy_local, s, f_a, f_b, heatmap_1d)
        p_xy_kp_ab_sel, y_conf_sel, desc_sel, \
            n_p_xy_kp_ab_sel, n_y_conf_sel, n_desc_sel = self.post_process_matches(p_xy_kp_ab, y_sel, f_ab_sel, f_sel)
        return p_xy_kp_ab_sel, y_conf_sel, desc_sel, n_p_xy_kp_ab_sel, n_y_conf_sel, n_desc_sel

    def forward(self, x_a, x_b):
        f_a = self.conv0_block_a(x_a)
        f_b = self.conv0_block_b(x_b)

        heatmap = self.conv0_block_ab(torch.concat([x_a, x_b], dim=1))
        s = f_a.shape[-1]
        r = x_a.shape[-1] / s
        nb = x_a.shape[0]

        f_a_1d = torch.moveaxis(f_a.reshape(-1, 32, s * s), 1, -1)
        f_b_1d = torch.moveaxis(f_b.reshape(-1, 32, s * s), 1, -1)
        heatmap_1d = heatmap.reshape(-1, s * s)

        grid_x, grid_y = torch.meshgrid(torch.arange(0, s), torch.arange(0, s), indexing='xy')

        grid_x = torch.tile(torch.unsqueeze(torch.unsqueeze(grid_x, 0), 0), (2, 1, 1, 1))
        grid_y = torch.tile(torch.unsqueeze(torch.unsqueeze(grid_y, 0), 0), (2, 1, 1, 1))

        grid_xy = torch.concat([grid_x, grid_y], dim=1)
        p_xy = torch.moveaxis(grid_xy.reshape(-1, 2, s * s), 1, -1)

        f = heatmap > self.heatmap_thresh
        f = torch.squeeze(torch.moveaxis(f.reshape(-1, 1, s * s), 1, -1), -1)

        f_n = heatmap <= self.heatmap_thresh
        f_n = torch.squeeze(torch.moveaxis(f_n.reshape(-1, 1, s * s), 1, -1), -1)

        match_pxy = []
        conf_pxy = []
        desc_pxy = []

        n_match_pxy = []
        n_conf_pxy = []
        n_desc_pxy = []

        match_pxy_ = []
        conf_pxy_ = []
        desc_pxy_ = []

        n_match_pxy_ = []
        n_conf_pxy_ = []
        n_desc_pxy_ = []

        for bi in torch.arange(nb):
            p_xy_local = p_xy[bi][f[bi]]
            hm = heatmap_1d[bi][f[bi]]
            hmi = torch.argsort(hm)[-self.cutoff_n_points:]

            p_xy_kp_ab_sel, y_conf_sel, desc_sel, \
                n_p_xy_kp_ab_sel, n_y_conf_sel, n_desc_sel = self.get_matches(p_xy_local[hmi], s, f_a_1d[bi],
                                                                              f_b_1d[bi], heatmap_1d[bi])
            match_pxy.append(p_xy_kp_ab_sel * r)
            conf_pxy.append(y_conf_sel)
            desc_pxy.append(desc_sel)
            n_match_pxy.append(n_p_xy_kp_ab_sel * r)
            n_conf_pxy.append(n_y_conf_sel)
            n_desc_pxy.append(n_desc_sel)

            p_xy_local_ = p_xy[bi][f_n[bi]]
            hm = heatmap_1d[bi][f_n[bi]]
            hmi = torch.argsort(hm)[-self.cutoff_n_points:]

            p_xy_kp_ab_sel_, y_conf_sel_, desc_sel_, \
                n_p_xy_kp_ab_sel_, n_y_conf_sel_, n_desc_sel_ = self.get_matches(p_xy_local_[hmi], s, f_a_1d[bi],
                                                                                 f_b_1d[bi], heatmap_1d[bi])

            yi = torch.argsort(y_conf_sel_)[-y_conf_sel.shape[0]:]
            yi_n = torch.argsort(n_y_conf_sel_)[-n_y_conf_sel.shape[0]:]

            match_pxy_.append(p_xy_kp_ab_sel_[yi] * r)
            conf_pxy_.append(y_conf_sel_[yi])
            desc_pxy_.append(desc_sel_[yi])
            n_match_pxy_.append(n_p_xy_kp_ab_sel_[yi_n] * r)
            n_conf_pxy_.append(n_y_conf_sel_[yi_n])
            n_desc_pxy_.append(n_desc_sel_[yi_n])

        return heatmap, ((match_pxy, conf_pxy, desc_pxy), (n_match_pxy, n_conf_pxy, n_desc_pxy)), \
            ((match_pxy_, conf_pxy_, desc_pxy_), (n_match_pxy_, n_conf_pxy_, n_desc_pxy_))