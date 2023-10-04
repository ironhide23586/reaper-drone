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

RESUME_MODEL_FPATH = 'scratchspace/trained_models/little-mamba.04-10-2023.08_39_11/model_files/neuramatch-little-mamba_50e-600b_0.43906313267958846-fsc.pt'

LEARN_RATE = 8e-5
SIDE = 480
BATCH_SIZE = 8
NUM_EPOCHS = 100000
SAVE_EVERY_N_BATCHES = 600
BLEND_COEFF = .55
KSIZE = 13
RADIUS_SCALE = .3
BLEND_COEFF = .55
TVERSKY_SMOOTH = 1.
TVERSKY_ALPHA = .7
TVERSKY_GAMMA = .75


from datetime import datetime
from multiprocessing import cpu_count
from PIL import Image

from pillow_heif import register_heif_opener
register_heif_opener()

import pytz
from tqdm import tqdm
from coolname import generate_slug
from torch.utils.data import DataLoader

from neural_matcher.nn import NeuraMatch
from dataset.streamer import ImagePairDataset
from neural_matcher.losses import KeypointLoss
from neural_matcher.utils import *


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

    ds = ImagePairDataset('scratchspace/gt_data', 'train', ksize=KSIZE, radius_scale=RADIUS_SCALE,
                          blend_coeff=BLEND_COEFF)
    data_loader = DataLoader(ds, BATCH_SIZE, collate_fn=collater, num_workers=cpu_count())

    ds_val = ImagePairDataset('scratchspace/gt_data', 'val', ksize=KSIZE, radius_scale=RADIUS_SCALE,
                              blend_coeff=BLEND_COEFF)
    data_loader_val = DataLoader(ds_val, BATCH_SIZE, collate_fn=collater, num_workers=cpu_count())

    loss_fn = KeypointLoss(smooth=TVERSKY_SMOOTH, alpha=TVERSKY_ALPHA, gamma=TVERSKY_GAMMA)

    train_config = {'start_learn_rate': LEARN_RATE,
                    'side': SIDE,
                    'batch_size': BATCH_SIZE,
                    'blend_coeff': BLEND_COEFF,
                    'ksize': KSIZE,
                    'radius_scale': RADIUS_SCALE,
                    'tversky_smooth': TVERSKY_SMOOTH,
                    'tversky_alpha': TVERSKY_ALPHA,
                    'tversky_gamma': TVERSKY_GAMMA,
                    'session_id': sess_id}
    if RESUME_MODEL_FPATH is not None:
        train_config['resume_model_fpath'] = RESUME_MODEL_FPATH
    config_fpath = out_dir + '/train_config.json'
    with open(config_fpath, 'w') as f:
        json.dump(train_config, f, indent=4, sort_keys=True)

    nmatch = NeuraMatch(device)
    nmatch.to(device)

    if RESUME_MODEL_FPATH is not None:
        print('Resuming from', RESUME_MODEL_FPATH)
        nmatch.load_state_dict(torch.load(RESUME_MODEL_FPATH, map_location=device), strict=True)
        print('Loaded!')
    else:
        print('Training from scratch...')
    opt = torch.optim.Adam(nmatch.parameters(), lr=LEARN_RATE)

    bi = 0
    ei = 0

    val_df_dict = {'epoch': [],
                   'epoch_batch_iteration': [],
                   'final_score': [],
                   'precision': [],
                   'recall': [],
                   'val_loss': [],
                   'train_loss': [],
                   'num_samples': []}

    checkpoint_model(nmatch, None, device, data_loader_val, ima, imb, model_dir, loss_fn, ei, bi, sess_id, log_fname,
                     val_df_dict, SIDE, viz_dir)

    for ei in range(NUM_EPOCHS):
        nmatch.train()
        for bi, (ims, pxys, heatmaps_gt) in enumerate(tqdm(data_loader)):
            heatmaps_pred, ((match_pxy, conf_pxy, desc_pxy), (un_match_pxy, un_conf_pxy, un_desc_pxy),
                      (n_match_pxy, n_conf_pxy, n_desc_pxy)), \
                ((match_pxy_, conf_pxy_, desc_pxy_), (un_match_pxy_, un_conf_pxy_, un_desc_pxy_),
                 (n_match_pxy_, n_conf_pxy_, n_desc_pxy_)), y_out = nmatch(ims, pxys)
            nmatch.zero_grad()
            loss, _ = loss_fn(y_out, heatmaps_pred, heatmaps_gt.to(device))
            loss.backward()
            opt.step()

            if bi % 50 == 0:
                print('Loss:', sess_id + '_' + '-'.join([str(ei) + 'e', str(bi) + 'b']), '-', loss)

            if bi % SAVE_EVERY_N_BATCHES == 0:
                checkpoint_model(nmatch, loss, device, data_loader_val, ima, imb, model_dir, loss_fn, ei, bi, sess_id,
                                 log_fname, val_df_dict, SIDE, viz_dir)
            nmatch.train()
        checkpoint_model(nmatch, loss, device, data_loader_val, ima, imb, model_dir, loss_fn, ei, bi, sess_id,
                         log_fname, val_df_dict, SIDE, viz_dir)
