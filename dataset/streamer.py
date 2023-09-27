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

import numpy as np

from torch.utils.data import Dataset

import utils


class ImagePairDataset(Dataset):

    def __init__(self, data_dir, mode):
        super().__init__()
        self.data_dir = data_dir
        dirs = glob(data_dir + '/*_' + mode)
        self.fpaths = np.hstack([glob(d + '/*') for d in dirs])
        self.sz = self.fpaths.shape[0]
        self.ni = np.arange(self.sz)
        np.random.shuffle(self.ni)

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        i = self.ni[idx]
        if idx == self.sz - 1:
            np.random.shuffle(self.ni)
            print('Epoch finished, shuffled indices....')
        with open(self.fpaths[i], 'rb') as f:
            payload = pickle.load(f)
        im_ab, matches_xy, metadata = payload

        k = utils.create_heatmap(matches_xy[:, :2], im_ab.shape[-1])
        k_ = utils.create_heatmap(matches_xy[:, 2:], im_ab.shape[-1])
        heatmaps = np.stack([k, k_])

        # cv2.imwrite('k.png', k * 255)
        # cv2.imwrite('k_.png', k_ * 255)
        # im_ab_viz = utils.drawMatches(np.rollaxis(im_ab[0], 0, 3), matches_xy[:, :2],
        #                               np.rollaxis(im_ab[1], 0, 3), matches_xy[:, 2:])
        # cv2.imwrite('k__.png', im_ab_viz)
        # b = np.rollaxis(im_ab[0], 0, 3) * .3 + np.tile(np.expand_dims(k * 255, -1), [1, 1, 3]) * .7
        # cv2.imwrite('k___.png', b)
        # b = np.rollaxis(im_ab[1], 0, 3) * .3 + np.tile(np.expand_dims(k_ * 255, -1), [1, 1, 3]) * .7
        # cv2.imwrite('k____.png', b)

        return im_ab, matches_xy, heatmaps


