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

LEARN_RATE = 8e-5
SIDE = 480
BATCH_SIZE = 8
NUM_EPOCHS = 100000
SAVE_EVERY_N_BATCHES = 600
BLEND_COEFF = .55

import os
import json
from datetime import datetime
from multiprocessing import cpu_count
from PIL import Image

from pillow_heif import register_heif_opener
register_heif_opener()

import pytz
import torch
import cv2
import numpy as np
import pandas as pd

from coolname import generate_slug
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader

from neural_matcher.nn import NeuraMatch
from dataset.streamer import ImagePairDataset
from neural_matcher.losses import KeypointLoss
import utils

tensor_transform = transforms.ToTensor()

input_transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def viz_matches(masked_matches, a, b, heatmap):
    pxy_matches, conf_matches, desc_matches = masked_matches
    img = utils.drawMatches(a, pxy_matches[0][:, :2].detach().cpu().numpy(),
                            b, pxy_matches[0][:, 2:].detach().cpu().numpy())
    hm_a = (heatmap[0][0].detach().cpu().numpy() * 255).astype(np.uint8)
    hm_b = (heatmap[0][1].detach().cpu().numpy() * 255).astype(np.uint8)
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


def input_preprocess(im):
    min_s = min(im.width, im.height)
    x_offset = (im.width - min_s) // 2
    y_offset = (im.height - min_s) // 2
    frame_square = np.array(im)[y_offset: y_offset + min_s, x_offset: x_offset + min_s]
    frame_square = cv2.resize(frame_square, (SIDE, SIDE))[:, :, [2, 1, 0]]
    return frame_square


def infer_nn(nmatch, ima, imb):
    ima_pp = input_preprocess(ima)
    imb_pp = input_preprocess(imb)
    im_a = tensor_transform(ima_pp).to(device)
    im_b = tensor_transform(imb_pp).to(device)
    t = torch.unsqueeze(torch.stack([input_transforms(im_a), input_transforms(im_b)]), 0)
    nmatch.eval()
    with torch.no_grad():
        heatmap, masked_outs, _, _ = nmatch(t)
    masked_matches, masked_unmatches, n_masked_matches = masked_outs
    match_viz, heatmap_a, heatmap_b = viz_matches(masked_matches, ima_pp, imb_pp, heatmap)
    nmatch.train()
    return match_viz, heatmap_a, heatmap_b, masked_matches


def score_model(nmatch, data_loader, loss_fn):
    print('Scoring model...')
    nmatch.eval()
    score_dict = {'final_score': -1.,
                  'precision': -1.,
                  'recall': -1.,
                  'val_loss': -1.}
    val_loss = 0.
    tps = 0
    fps = 0
    fns = 0
    ni = 0
    with torch.no_grad():
        for bi, (ims, pxys, heatmaps_gt) in enumerate(tqdm(data_loader)):
            heatmaps_pred, ((match_pxy, conf_pxy, desc_pxy), (un_match_pxy, un_conf_pxy, un_desc_pxy),
                            (n_match_pxy, n_conf_pxy, n_desc_pxy)), \
                ((match_pxy_, conf_pxy_, desc_pxy_), (un_match_pxy_, un_conf_pxy_, un_desc_pxy_),
                 (n_match_pxy_, n_conf_pxy_, n_desc_pxy_)), y_out = nmatch(ims, pxys)
            loss, (tp, fp, fn) = loss_fn(y_out, heatmaps_pred, heatmaps_gt.to(device))
            tps += tp
            fps += fp
            fns += fn
            val_loss += float(loss.cpu().numpy())
            ni += 1

            val_loss_ = val_loss / ni
            prec = float((tps / (tps + fps)).cpu().numpy())
            rec = float((tps / (tps + fns)).cpu().numpy())
            fsc = (2 * prec * rec) / (prec + rec)

            score_dict['val_loss'] = val_loss_
            score_dict['final_score'] = fsc
            score_dict['precision'] = prec
            score_dict['recall'] = rec
            score_dict['num_samples'] = int(ni * data_loader.batch_size)

            if bi % 5 == 0:
                print(json.dumps(score_dict, indent=4, sort_keys=True))
    return score_dict


if __name__ == '__main__':
    ima = Image.open('IMG_3806.HEIC')
    imb = Image.open('IMG_3807.HEIC')

    tz = pytz.timezone('Asia/Kolkata')
    curr_time = datetime.now(tz)
    sess_id = generate_slug(2)
    print(sess_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device', device)

    out_dir = 'scratchspace/trained_models/' + sess_id + '.' + curr_time.strftime("%d-%m-%Y.%H_%M_%S")
    model_dir = out_dir + '/model_files'
    log_fname = out_dir + '/' + sess_id + '.' + 'log.csv'
    os.makedirs(model_dir, exist_ok=True)
    viz_dir = out_dir + '/viz'
    os.makedirs(viz_dir, exist_ok=True)

    ds = ImagePairDataset('scratchspace/gt_data', 'train', blend_coeff=BLEND_COEFF)
    data_loader = DataLoader(ds, BATCH_SIZE, collate_fn=collater, num_workers=cpu_count())

    ds_val = ImagePairDataset('scratchspace/gt_data', 'val', blend_coeff=BLEND_COEFF)
    data_loader_val = DataLoader(ds_val, BATCH_SIZE, collate_fn=collater, num_workers=cpu_count())

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

    nmatch.train()
    opt = torch.optim.Adam(nmatch.parameters(), lr=LEARN_RATE)

    bi = 0
    ei = 0

    match_viz, heatmap_a, heatmap_b, masked_outs = infer_nn(nmatch, ima, imb)
    fn_prefix = '_'.join(['viz', sess_id])
    suffix = '-'.join([str(ei) + 'e', str(bi) + 'b', str(masked_outs[0][0].shape[0]) + 'kp'])
    print('VIZ:', suffix)
    mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'matches', suffix + '.jpg'])])
    ha = os.sep.join([viz_dir, '_'.join([fn_prefix, 'heatmap-a', suffix + '.png'])])
    hb = os.sep.join([viz_dir, '_'.join([fn_prefix, 'heatmap-b', suffix + '.png'])])
    cv2.imwrite(mn, match_viz)
    cv2.imwrite(ha, heatmap_a)
    cv2.imwrite(hb, heatmap_b)

    val_df_dict = {'epoch': [],
                   'epoch_batch_iteration': [],
                   'final_score': [],
                   'precision': [],
                   'recall': [],
                   'val_loss': [],
                   'train_loss': [],
                   'num_samples': []}

    for ei in range(NUM_EPOCHS):
        for bi, (ims, pxys, heatmaps_gt) in enumerate(tqdm(data_loader)):
            heatmaps_pred, ((match_pxy, conf_pxy, desc_pxy), (un_match_pxy, un_conf_pxy, un_desc_pxy),
                      (n_match_pxy, n_conf_pxy, n_desc_pxy)), \
                ((match_pxy_, conf_pxy_, desc_pxy_), (un_match_pxy_, un_conf_pxy_, un_desc_pxy_),
                 (n_match_pxy_, n_conf_pxy_, n_desc_pxy_)), y_out = nmatch(ims, pxys)
            nmatch.zero_grad()
            loss, _ = loss_fn(y_out, heatmaps_pred, heatmaps_gt.to(device))
            loss.backward()
            opt.step()

            suffix = '-'.join([str(ei) + 'e', str(bi) + 'b'])
            sess_id_ = sess_id + '_' + suffix
            if bi % 50 == 0:
                print('Loss:', sess_id_, '-', loss)

            if bi % SAVE_EVERY_N_BATCHES == 0:
                score_dict = score_model(nmatch, data_loader_val, loss_fn)

                val_df_dict['epoch'].append(ei)
                val_df_dict['epoch_batch_iteration'].append(bi)
                val_df_dict['final_score'].append(score_dict['final_score'])
                val_df_dict['precision'].append(score_dict['precision'])
                val_df_dict['recall'].append(score_dict['recall'])
                val_df_dict['val_loss'].append(score_dict['val_loss'])
                val_df_dict['train_loss'].append(float(loss.detach().cpu().numpy()))
                val_df_dict['num_samples'].append(score_dict['num_samples'])

                score_tag = '_' + str(score_dict['final_score']) + '-fsc'

                val_df = pd.DataFrame(val_df_dict)
                val_df.to_csv(log_fname, index=False)
                fn = 'neuramatch-' + sess_id_ + score_tag + '.pt'
                out_fp = model_dir + '/' + fn
                print('Saving to', out_fp)
                torch.save(nmatch.state_dict(), out_fp)

                match_viz, heatmap_a, heatmap_b, masked_outs = infer_nn(nmatch, ima, imb)
                fn_prefix = '_'.join(['viz', sess_id])
                suffix = '-'.join([str(ei) + 'e', str(bi) + 'b', str(masked_outs[0][0].shape[0]) + 'kp'])
                print('VIZ:', suffix)
                mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'matches', suffix + score_tag + '.jpg'])])
                ha = os.sep.join([viz_dir, '_'.join([fn_prefix, 'heatmap-a', suffix + score_tag + '.png'])])
                hb = os.sep.join([viz_dir, '_'.join([fn_prefix, 'heatmap-b', suffix + score_tag + '.png'])])
                cv2.imwrite(mn, match_viz)
                cv2.imwrite(ha, heatmap_a)
                cv2.imwrite(hb, heatmap_b)

                nmatch.train()

        fn = 'neuramatch-' + sess_id_ + '-EPOCH.pt'
        out_fp = model_dir + '/' + fn
        print('Saving to', out_fp)
        torch.save(nmatch.state_dict(), out_fp)

        match_viz, heatmap_a, heatmap_b, masked_outs = infer_nn(nmatch, ima, imb)
        fn_prefix = '_'.join(['viz', sess_id])
        suffix = '-'.join([str(ei) + 'e', str(bi) + 'b', str(masked_outs[0][0].shape[0]) + 'kp'])
        print('VIZ:', suffix)
        mn = os.sep.join([viz_dir, '_'.join([fn_prefix, 'matches', suffix + '.jpg'])])
        ha = os.sep.join([viz_dir, '_'.join([fn_prefix, 'heatmap-a', suffix + '.png'])])
        hb = os.sep.join([viz_dir, '_'.join([fn_prefix, 'heatmap-b', suffix + '.png'])])
        cv2.imwrite(mn, match_viz)
        cv2.imwrite(ha, heatmap_a)
        cv2.imwrite(hb, heatmap_b)

        nmatch.train()

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
