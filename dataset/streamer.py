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

from glob import glob
import pickle

import cv2
import numpy as np

from torch.utils.data import Dataset

import utils


class ImagePairDataset(Dataset):

    def __init__(self, data_dir, mode, ksize=23, radius_scale=.6, blend_coeff=.55):
        super().__init__()
        self.blend_coeff = blend_coeff
        self.ksize = ksize
        self.radius_scale = radius_scale
        self.data_dir = data_dir
        self.mode = mode
        dirs = glob(data_dir + '/*_' + mode)
        self.fpaths = np.hstack([glob(d + '/*') for d in dirs])
        self.sz = self.fpaths.shape[0]
        self.ni = np.arange(self.sz)
        np.random.shuffle(self.ni)

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        i = self.ni[idx]
        if idx == self.sz - 1 and self.mode == 'train':
            np.random.shuffle(self.ni)
            print('Epoch finished, shuffled indices....')
        with open(self.fpaths[i], 'rb') as f:
            payload = pickle.load(f)
        im_ab, matches_xy, metadata = payload

        _, _, _, im_s = im_ab.shape
        matches_xy = np.clip(matches_xy / im_s, 0, 1.) * (utils.SIDE - 1)
        im_ab_ = []
        for i in range(im_ab.shape[0]):
            im_ = cv2.resize(np.rollaxis(im_ab[i], 0, 3), (utils.SIDE, utils.SIDE))
            im_ab_.append(np.rollaxis(im_, 2, 0))
        im_ab = np.array(im_ab_)

        k = utils.create_heatmap(matches_xy[:, :2], im_ab.shape[-1], ksize=self.ksize, radius_scale=self.radius_scale,
                                 blend_coeff=self.blend_coeff)
        k_ = utils.create_heatmap(matches_xy[:, 2:], im_ab.shape[-1], ksize=self.ksize, radius_scale=self.radius_scale,
                                  blend_coeff=self.blend_coeff)
        heatmaps = np.stack([k, k_])

        # import cv2
        # cv2.imwrite('k.png', k * 255)
        # cv2.imwrite('k_.png', k_ * 255)
        #
        # cv2.imwrite('ki.png', np.rollaxis(im_ab[0], 0, 3))
        # cv2.imwrite('ki_.png', np.rollaxis(im_ab[1], 0, 3))
        #
        # im_ab_viz = utils.drawMatches(np.rollaxis(im_ab[0], 0, 3), matches_xy[:, :2],
        #                               np.rollaxis(im_ab[1], 0, 3), matches_xy[:, 2:])
        # cv2.imwrite('k__.png', im_ab_viz)
        #
        # b = np.rollaxis(im_ab[0], 0, 3) * .3 + np.tile(np.expand_dims(k * 255, -1), [1, 1, 3]) * .7
        # cv2.imwrite('k___.png', b)
        #
        # b_ = np.rollaxis(im_ab[1], 0, 3) * .3 + np.tile(np.expand_dims(k_ * 255, -1), [1, 1, 3]) * .7
        # cv2.imwrite('k____.png', b_)

        return im_ab, matches_xy, heatmaps


