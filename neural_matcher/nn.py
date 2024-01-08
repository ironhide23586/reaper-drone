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

import utils
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

        self.heatmap_thresh = 1.1
        # self.final_thresh = nn.Parameter(torch.tensor(.5), requires_grad=False)

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

        self.heatmap_decoder = nn.Sequential(nn.ConvTranspose2d(32, 32, (9, 9),
                                                                (1, 1), bias=False),
                                             nn.BatchNorm2d(32), nn.LeakyReLU(),
                                             nn.ConvTranspose2d(32, 32, (7, 7),
                                                                (1, 1), bias=False),
                                             nn.BatchNorm2d(32), nn.LeakyReLU(),
                                             nn.ConvTranspose2d(32, 16, (6, 6),
                                                                (2, 2), bias=False),
                                             nn.BatchNorm2d(16), nn.LeakyReLU(),
                                             nn.ConvTranspose2d(16, 8, (4, 4),
                                                                (2, 2), bias=False),
                                             nn.BatchNorm2d(8), nn.LeakyReLU(),
                                             nn.ConvTranspose2d(8, 1, (3, 3),
                                                                (1, 1), bias=True), nn.ReLU())
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
                                                               (2, 2), bias=True))

        self.residual_encoder = nn.Sequential(nn.Conv2d(6, 32, (5, 5),
                                                        (2, 2), bias=False),
                                              nn.BatchNorm2d(32), nn.LeakyReLU(),
                                              nn.Conv2d(32, 64, (5, 5),
                                                        (2, 2), bias=False),
                                              nn.BatchNorm2d(64), nn.LeakyReLU(),
                                              nn.Conv2d(64, 128, (7, 7),
                                                        (1, 1), bias=False),
                                              nn.BatchNorm2d(128), nn.LeakyReLU(),
                                              nn.Conv2d(128, 64, (9, 9),
                                                        (1, 1), bias=False),
                                              nn.BatchNorm2d(64), nn.LeakyReLU())

        self.residual_encoder_vector = nn.Sequential(nn.Conv2d(6, 32, (5, 5),
                                                               (2, 2), bias=False),
                                                     nn.BatchNorm2d(32), nn.LeakyReLU(),
                                                     nn.Conv2d(32, 64, (5, 5),
                                                               (2, 2), bias=False),
                                                     nn.BatchNorm2d(64), nn.LeakyReLU(),
                                                     nn.Conv2d(64, 128, (7, 7),
                                                               (1, 1), bias=False),
                                                     nn.BatchNorm2d(128), nn.LeakyReLU(),
                                                     nn.Conv2d(128, 64, (9, 9),
                                                               (1, 1), bias=False),
                                                     nn.BatchNorm2d(64), nn.LeakyReLU())

        # self.residual_endcoder.to('cuda')
        # self.residual_endcoder(x_a).shape
        # self.model_params = (list(self.heatmap_decoder.parameters()) + list(self.vector_decoder.parameters()) \
        #                      + list(self.residual_encoder.parameters()))

        # self.model_params = (list(self.heatmap_decoder.parameters()) + list(self.residual_encoder.parameters()))
        self.model_params = (list(self.vector_decoder.parameters()) + list(self.residual_encoder_vector.parameters()))

        self.frozen_modules = [self.heatmap_decoder, self.residual_encoder]
        self.to(self.device)

    def viz_fmap(self, v_a_res):
        a = v_a_res[0, 10:13]
        b = a - a.min()
        b = b / b.max()
        c = torch.transpose(torch.transpose(b, 0, 2), 0, 1)
        d = c.detach().cpu().numpy()
        e = d * 255
        return e

    def forward(self, x_):
        x = x_.to(self.device)
        # x_a = x[:, 0]
        # x_b = x[:, 1]

        # x_a_clip = self.clip_resize(x_a)
        # x_b_clip = self.clip_resize(x_b)
        #
        # v_a = self.clip_model.visual(x_a_clip)
        # v_b = self.clip_model.visual(x_b_clip)
        # x_in = torch.concat([v_a, v_b], dim=1)
        #
        # v_ab_clip = self.clip_condenser(x_in)

        x_rs = x.reshape(-1, 6, utils.SIDE, utils.SIDE)

        v_ab_conf = self.residual_encoder(x_rs)
        v_ab_vec_residudal = self.residual_encoder_vector(x_rs)

        v_a_conf = v_ab_conf[:, :32]
        v_b_conf = v_ab_conf[:, 32:]

        heatmap_a = self.heatmap_decoder(v_a_conf)
        heatmap_b = self.heatmap_decoder(v_b_conf)
        heatmap = torch.concat([heatmap_a, heatmap_b], dim=1)

        v_ab_vec = v_ab_conf + v_ab_vec_residudal

        match_vectors_pred = self.vector_decoder(v_ab_vec)
        match_vectors_pred = torch.clip(match_vectors_pred, -2, 2)

        s = self.side
        nb = x.shape[0]

        mv = (match_vectors_pred.reshape(-1, 2, s * s) * (s - 1)).round().int()

        p_xy_tiled = torch.tile(torch.unsqueeze(self.p_xy[0], 0), (nb, 1, 1))

        targ_xy_2d = torch.clamp(p_xy_tiled + mv, 0, s - 1)
        targ_xy_1d = targ_xy_2d[:, 0, :] + targ_xy_2d[:, 1, :] * s

        hm_targ = heatmap.reshape(-1, 2, s * s)[:, 1]
        conf_targ = torch.stack([hm_targ[i][targ_xy_1d[i]].reshape(s, s) for i in range(nb)])

        mv_targ = torch.transpose(match_vectors_pred.reshape(-1, 2, s * s), 2, 1)
        match_vectors_pred_targ = torch.stack([mv_targ[i][targ_xy_1d[i]].T.reshape(2, s, s) for i in range(nb)])

        conf_mask = (heatmap[:, 0] + conf_targ)

        match_xy_pairs = []
        confs = []

        cm = conf_mask.reshape(-1, s * s)
        for bi in torch.arange(nb).to(self.device):
            f = cm[bi] > self.heatmap_thresh
            confs.append(cm[bi, f])
            match_xy_pairs.append(torch.vstack([self.p_xy[0, :, f], targ_xy_2d[bi, :, f]]).T)

        return (heatmap, match_vectors_pred, match_vectors_pred_targ, conf_mask), (match_xy_pairs, confs)
