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

LEARN_RATE = 1e-8
SIDE = 480
BATCH_SIZE = 5

import os
import json
from datetime import datetime

import pytz
import torch
import cv2
import numpy as np

from coolname import generate_slug
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader

from neural_matcher.nn import NeuraMatch
from dataset.streamer import ImagePairDataset
from neural_matcher.losses import KeypointLoss


tensor_transform = transforms.ToTensor()

input_transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def viz_matches(masked_matches, ima, imb, heatmap):
    pxy_matches, conf_matches, desc_matches = masked_matches
    a = cv2.resize(np.array(ima), (SIDE, SIDE))[:, :, [2, 1, 0]]
    b = cv2.resize(np.array(imb), (SIDE, SIDE))[:, :, [2, 1, 0]]
    img = drawMatches(a, pxy_matches[0][:, :2].numpy(), b, pxy_matches[0][:, 2:].numpy(), pxy_matches[0].numpy())
    hm_a = (heatmap[0][0].detach().numpy() * 255).astype(np.uint8)
    hm_b = (heatmap[0][1].detach().numpy() * 255).astype(np.uint8)
    return img, hm_a, hm_b


def collater(data):
    ims = []
    pxys = []
    heatmaps = []
    for d in data:
        x = d[0] / 255.
        ims.append(input_transforms(torch.Tensor(x)))
        pxys.append(torch.Tensor(d[1].astype(int)).int())
        heatmaps.append(d[2])
    heatmaps = torch.Tensor(np.stack(heatmaps))
    ims = torch.stack(ims)
    return ims, pxys, heatmaps


if __name__ == '__main__':
    tz = pytz.timezone('Asia/Kolkata')
    curr_time = datetime.now(tz)
    sess_id = generate_slug(2)
    print(sess_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device', device)
    out_dir = 'scratchspace/trained_models/' + sess_id + '.' + curr_time.strftime("%d-%m-%Y.%H_%M_%S")
    model_dir = out_dir + '/model_files'
    os.makedirs(model_dir, exist_ok=True)
    ds = ImagePairDataset('scratchspace/gt_data', 'train')
    data_loader = DataLoader(ds, 5, collate_fn=collater)
    loss_fn = KeypointLoss()

    train_config = {'learn_rate': LEARN_RATE,
                    'side': SIDE,
                    'batch_size': BATCH_SIZE,
                    'session_id': sess_id}
    config_fpath = out_dir + '/train_config.json'
    with open(config_fpath, 'w') as f:
        json.dump(train_config, f, indent=4, sort_keys=True)

    nmatch = NeuraMatch(device)
    nmatch.to(device)
    opt = torch.optim.Adam(nmatch.parameters(), lr=LEARN_RATE)

    for bi, (ims, pxys, heatmaps) in enumerate(tqdm(data_loader)):
        heatmap, ((match_pxy, conf_pxy, desc_pxy), (un_match_pxy, un_conf_pxy, un_desc_pxy),
                  (n_match_pxy, n_conf_pxy, n_desc_pxy)), \
            ((match_pxy_, conf_pxy_, desc_pxy_), (un_match_pxy_, un_conf_pxy_, un_desc_pxy_),
             (n_match_pxy_, n_conf_pxy_, n_desc_pxy_)), y_out = nmatch(ims, pxys)
        nmatch.zero_grad()
        loss = loss_fn(y_out)
        loss.backward()
        opt.step()

        if bi % 50 == 0:
            print(sess_id, '-', loss)

        if bi % 100 == 0:
            fn = 'neuramatch-' + sess_id + '-' + str(bi) + '.pt'
            out_fp = model_dir + '/' + fn
            print('Saving to', out_fp)
            torch.save(nmatch.state_dict(), out_fp)

    #
    # matches_xy, heatmaps = ds[56]
    # matches_xy_, heatmaps_ = ds[58]
    #
    # device = torch.device("cpu")
    #
    # ima = Image.open('scratchspace/IMG_3806.HEIC')
    # imb = Image.open('scratchspace/IMG_3807.HEIC')
    #
    # im_a = tensor_transform(ima).to(device)
    # im_b = tensor_transform(imb).to(device)
    #
    # t_a = torch.stack([input_transforms(im_a), input_transforms(im_a)])
    # t_b = torch.stack([input_transforms(im_b), input_transforms(im_b)])
    #
    # nmatch = NeuraMatch()
    #
    # heatmap, masked_outs, unmasked_outs = nmatch(t_a, t_b)
    #
    #
    # masked_matches, masked_unmatches, n_masked_matches = masked_outs
    #
    # match_viz, heatmap_a, heatmap_b = viz_matches(masked_matches, ima, imb, heatmap)
    # cv2.imwrite('match_viz_match.png', match_viz)
    # cv2.imwrite('heatmap_a_match.png', heatmap_a)
    # cv2.imwrite('heatmap_b_match.png', heatmap_b)
    #
    # match_viz, heatmap_a, heatmap_b = viz_matches(masked_unmatches, ima, imb, heatmap)
    # cv2.imwrite('match_viz_unmatch.png', match_viz)
    # cv2.imwrite('heatmap_a_unmatch.png', heatmap_a)
    # cv2.imwrite('heatmap_b_unmatch.png', heatmap_b)
    #
    # match_viz, heatmap_a, heatmap_b = viz_matches(n_masked_matches, ima, imb, heatmap)
    # cv2.imwrite('match_viz_nomatch.png', match_viz)
    # cv2.imwrite('heatmap_a_nomatch.png', heatmap_a)
    # cv2.imwrite('heatmap_b_nomatch.png', heatmap_b)
    #
    # k = 0
