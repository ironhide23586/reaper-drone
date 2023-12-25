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

from utils import CACHE_DIR
import clip


class NeuraMatch(nn.Module):

    def __init__(self, device, side):
        super().__init__()
        self.device = device
        self.side = side

        self.clip_model, _ = clip.load(CACHE_DIR + "/ViT-B-32.pt", device=self.device)

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.side), torch.arange(0, self.side), indexing='xy')
        grid_x = torch.tile(torch.unsqueeze(torch.unsqueeze(grid_x.to(self.device), 0), 0), (2, 1, 1, 1))
        grid_y = torch.tile(torch.unsqueeze(torch.unsqueeze(grid_y.to(self.device), 0), 0), (2, 1, 1, 1))

        grid_xy = torch.concat([grid_x, grid_y], dim=1)
        self.p_xy = grid_xy.reshape(-1, 2, self.side * self.side)

        self.heatmap_thresh = nn.Parameter(torch.tensor(.5), requires_grad=False)
        self.final_thresh = nn.Parameter(torch.tensor(.5), requires_grad=False)

        self.matcher = nn.Sequential(nn.Linear(64, 32, bias=True),
                                     nn.LeakyReLU(),
                                     nn.Linear(32, 16, bias=True),
                                     nn.LeakyReLU(),
                                     nn.Linear(16, 1, bias=True),
                                     nn.Sigmoid())

        self.clip_condenser = nn.Sequential(nn.ConvTranspose2d(1, 8, (5, 1),
                                                               (2, 1), bias=False),
                                            nn.BatchNorm2d(8), nn.LeakyReLU(),
                                            nn.Conv2d(8, 16, (1, 99), (1, 3),
                                                      bias=False),
                                            nn.BatchNorm2d(16), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(16, 32, (5, 1),
                                                               (2, 1), bias=False),
                                            nn.BatchNorm2d(32), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(32, 32, (5, 1),
                                                               (1, 1), bias=False),
                                            nn.BatchNorm2d(32), nn.LeakyReLU())

        self.heatmap_decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, (5, 1),
                                                               (1, 1), bias=False),
                                            nn.BatchNorm2d(16), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(16, 8, (5, 1),
                                                               (1, 1), bias=False),
                                            nn.BatchNorm2d(8), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(8, 1, (4, 1),
                                                               (1, 1), bias=False),
                                            nn.BatchNorm2d(1), nn.LeakyReLU())

        self.vector_decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, (5, 1),
                                                               (1, 1), bias=False),
                                            nn.BatchNorm2d(16), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(16, 8, (5, 1),
                                                               (1, 1), bias=False),
                                            nn.BatchNorm2d(8), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(8, 1, (4, 1),
                                                               (1, 1), bias=False),
                                            nn.BatchNorm2d(1), nn.LeakyReLU())

        self.to(self.device)

    def forward(self, x_):
        x = x_.to(self.device)
        x_a = x[:, 0]
        x_b = x[:, 1]

        v_a = self.clip_model.visual(x_a)
        v_b = self.clip_model.visual(x_b)

        va = self.clip_condenser(v_a)
        vb = self.clip_condenser(v_b)

        ha = self.heatmap_decoder(va)
        hb = self.heatmap_decoder(vb)

        va_ = self.vector_decoder(va)
        vb_ = self.vector_decoder(vb)

        heatmap = torch.concat([ha, hb], dim=1)
        match_vectors_pred = torch.concat([va_, vb_], dim=1)

        s = self.side
        nb = x_a.shape[0]

        mv = (match_vectors_pred.reshape(-1, 2, s * s) * (s - 1)).round().int()

        p_xy_tiled = torch.tile(torch.unsqueeze(self.p_xy[0], 0), (nb, 1, 1))

        targ_xy_2d = torch.clamp(p_xy_tiled + mv, 0, s - 1)
        targ_xy_1d = targ_xy_2d[:, 0, :] + targ_xy_2d[:, 1, :] * s
        hm_targ = heatmap.reshape(-1, 2, s * s)[:, 1]
        conf_targ = torch.stack([hm_targ[i][targ_xy_1d[i]].reshape(s, s) for i in range(nb)])

        conf_mask = (heatmap[:, 0] + conf_targ) / 2.

        match_xy_pairs = []
        confs = []

        cm = conf_mask.reshape(-1, s * s)
        for bi in torch.arange(nb).to(self.device):
            f = cm[bi] > self.heatmap_thresh
            confs.append(cm[bi, f])
            match_xy_pairs.append(torch.vstack([self.p_xy[0, :, f], targ_xy_2d[bi, :, f]]).T)

        return (heatmap, match_vectors_pred, conf_mask), (match_xy_pairs, confs)
