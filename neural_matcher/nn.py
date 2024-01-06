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
from torchvision import transforms

from utils import CACHE_DIR
import clip


class NeuraMatch(nn.Module):

    def __init__(self, device, side):
        super().__init__()
        self.device = device
        self.side = side
        self.clip_resize = transforms.Resize(224, antialias=True)

        # self.clip_model, _ = clip.load(CACHE_DIR + "/ViT-B-32.pt", device=self.device)

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.side), torch.arange(0, self.side), indexing='xy')
        grid_x = torch.tile(torch.unsqueeze(torch.unsqueeze(grid_x.to(self.device), 0), 0), (2, 1, 1, 1))
        grid_y = torch.tile(torch.unsqueeze(torch.unsqueeze(grid_y.to(self.device), 0), 0), (2, 1, 1, 1))

        grid_xy = torch.concat([grid_x, grid_y], dim=1)
        self.p_xy = grid_xy.reshape(-1, 2, self.side * self.side)

        self.heatmap_thresh = nn.Parameter(torch.tensor(.5), requires_grad=False)
        self.final_thresh = nn.Parameter(torch.tensor(.5), requires_grad=False)

        self.clip_condenser = nn.Sequential(nn.ConvTranspose2d(768 * 2, 512,
                                                               (5, 5), (1, 1), bias=False),
                                            nn.BatchNorm2d(512), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(512, 256, (4, 4),
                                                               (2, 2), bias=False),
                                            nn.BatchNorm2d(256), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(256, 128, (4, 4),
                                                               (2, 2), bias=False),
                                            nn.BatchNorm2d(128), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(128, 64, (5, 5),
                                                               (2, 2), bias=False),
                                            nn.BatchNorm2d(64))

        self.heatmap_decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, (9, 9),
                                                                (1, 1), bias=False),
                                             nn.BatchNorm2d(64), nn.LeakyReLU(),
                                             nn.ConvTranspose2d(64, 32, (7, 7),
                                                                (1, 1), bias=False),
                                             nn.BatchNorm2d(32), nn.LeakyReLU(),
                                             nn.ConvTranspose2d(32, 16, (6, 6),
                                                                (2, 2), bias=False),
                                             nn.BatchNorm2d(16), nn.LeakyReLU(),
                                             nn.ConvTranspose2d(16, 2, (6, 6),
                                                                (2, 2), bias=True), nn.ReLU())
        # self.heatmap_decoder.to('cuda')
        # ha = self.heatmap_decoder(va)
        self.vector_decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, (9, 9),
                                                               (1, 1), bias=False),
                                            nn.BatchNorm2d(64), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(64, 32, (7, 7),
                                                               (1, 1), bias=False),
                                            nn.BatchNorm2d(32), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(32, 16, (6, 6),
                                                               (2, 2), bias=False),
                                            nn.BatchNorm2d(16), nn.LeakyReLU(),
                                            nn.ConvTranspose2d(16, 2, (6, 6),
                                                               (2, 2), bias=True), nn.Tanh())

        self.residual_encoder = nn.Sequential(nn.Conv2d(3, 32, (5, 5),
                                                        (2, 2), bias=False),
                                              nn.BatchNorm2d(32), nn.LeakyReLU(),
                                              nn.Conv2d(32, 64, (5, 5),
                                                        (2, 2), bias=False),
                                              nn.BatchNorm2d(64), nn.LeakyReLU(),
                                              nn.Conv2d(64, 128, (7, 7),
                                                        (1, 1), bias=False),
                                              nn.BatchNorm2d(128), nn.LeakyReLU(),
                                              nn.Conv2d(128, 32, (9, 9),
                                                        (1, 1), bias=True))
        # self.residual_endcoder.to('cuda')
        # self.residual_endcoder(x_a).shape
        self.model_params = (list(self.heatmap_decoder.parameters()) + list(self.vector_decoder.parameters()) \
                             + list(self.residual_encoder.parameters()))
        self.to(self.device)

    def forward(self, x_):
        x = x_.to(self.device)
        x_a = x[:, 0]
        x_b = x[:, 1]

        # x_a_clip = self.clip_resize(x_a)
        # x_b_clip = self.clip_resize(x_b)
        #
        # v_a = self.clip_model.visual(x_a_clip)
        # v_b = self.clip_model.visual(x_b_clip)
        # x_in = torch.concat([v_a, v_b], dim=1)
        #
        # v_ab_clip = self.clip_condenser(x_in)

        v_a_res = self.residual_encoder(x_a)
        v_b_res = self.residual_encoder(x_b)
        v_ab_res = torch.concat([v_a_res, v_b_res], dim=1)

        # v_ab = v_ab_clip + v_ab_res
        v_ab = v_ab_res

        heatmap = self.heatmap_decoder(v_ab)
        match_vectors_pred = self.vector_decoder(v_ab)

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
