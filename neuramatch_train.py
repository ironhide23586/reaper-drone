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

RESUME_MODEL_FPATH = 'scratchspace/trained_models/ambrosial-skink-matcher.05-10-2023.17_37_47/model_files/neuramatch-ambrosial-skink-matcher_4e-974b_0.4935259740624083-fsc.pt'
TRAIN_MODULE = 'matcher'  # 'heatmap' or 'matcher'
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
CUTOFF_N_POINTS = 20


from datetime import datetime
from multiprocessing import cpu_count
from PIL import Image

from pillow_heif import register_heif_opener
register_heif_opener()

import pytz
from tqdm import tqdm
from coolname import generate_slug
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from neural_matcher.nn import NeuraMatch
from dataset.streamer import ImagePairDataset
from neural_matcher.losses import KeypointLoss
from neural_matcher.utils import *


if __name__ == '__main__':
    ima = Image.open('IMG_3806.HEIC')
    imb = Image.open('IMG_3807.HEIC')

    tz = pytz.timezone('Asia/Kolkata')
    curr_time = datetime.now(tz)
    sess_id = generate_slug(2) + '-' + TRAIN_MODULE
    print(sess_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device', device)

    root_dir = 'scratchspace/trained_models'
    out_dir = root_dir + '/' + sess_id + '.' + curr_time.strftime("%d-%m-%Y.%H_%M_%S")
    model_dir = out_dir + '/model_files'
    log_fname = out_dir + '/' + sess_id + '.' + 'log.csv'
    tensorboard_log_dir = root_dir + '/tensorboard_logs'
    os.makedirs(model_dir, exist_ok=True)
    viz_dir = out_dir + '/viz'
    os.makedirs(viz_dir, exist_ok=True)

    writer = SummaryWriter(tensorboard_log_dir, filename_suffix='.' + sess_id + '.tb')

    ds_train = ImagePairDataset('scratchspace/gt_data', 'train', ksize=KSIZE, radius_scale=RADIUS_SCALE,
                                blend_coeff=BLEND_COEFF)
    data_loader_train = DataLoader(ds_train, BATCH_SIZE, collate_fn=collater, num_workers=cpu_count())

    ds_val = ImagePairDataset('scratchspace/gt_data', 'val', ksize=KSIZE, radius_scale=RADIUS_SCALE,
                              blend_coeff=BLEND_COEFF)
    data_loader_val = DataLoader(ds_val, BATCH_SIZE, collate_fn=collater, num_workers=cpu_count())

    loss_fn = KeypointLoss(smooth=TVERSKY_SMOOTH, alpha=TVERSKY_ALPHA, gamma=TVERSKY_GAMMA,
                           train_module=TRAIN_MODULE)

    train_config = {'start_learn_rate': LEARN_RATE,
                    'side': SIDE,
                    'batch_size': BATCH_SIZE,
                    'blend_coeff': BLEND_COEFF,
                    'ksize': KSIZE,
                    'radius_scale': RADIUS_SCALE,
                    'tversky_smooth': TVERSKY_SMOOTH,
                    'tversky_alpha': TVERSKY_ALPHA,
                    'tversky_gamma': TVERSKY_GAMMA,
                    'cutoff_n_points': CUTOFF_N_POINTS,
                    'train_module': TRAIN_MODULE,
                    'session_id': sess_id}
    if RESUME_MODEL_FPATH is not None:
        train_config['resume_model_fpath'] = RESUME_MODEL_FPATH
    config_fpath = out_dir + '/train_config.json'
    with open(config_fpath, 'w') as f:
        json.dump(train_config, f, indent=4, sort_keys=True)

    nmatch = NeuraMatch(device, SIDE, CUTOFF_N_POINTS)
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
                     val_df_dict, SIDE, viz_dir, writer, 0)
    running_loss = 0.
    prev_running_loss = running_loss
    for ei in range(NUM_EPOCHS):
        nmatch.train()
        for bi, (ims, pxys, heatmaps_gt) in enumerate(tqdm(data_loader_train)):
            heatmaps_pred, ((match_pxy, conf_pxy, desc_pxy), (un_match_pxy, un_conf_pxy, un_desc_pxy),
                      (n_match_pxy, n_conf_pxy, n_desc_pxy)), \
                ((match_pxy_, conf_pxy_, desc_pxy_), (un_match_pxy_, un_conf_pxy_, un_desc_pxy_),
                 (n_match_pxy_, n_conf_pxy_, n_desc_pxy_)), y_out = nmatch(ims, pxys)
            nmatch.zero_grad()
            loss, _ = loss_fn(y_out, heatmaps_pred, heatmaps_gt.to(device))
            loss.backward()
            running_loss += loss.item()

            if TRAIN_MODULE == 'matcher':
                nmatch.conv0_block_a.zero_grad()
                nmatch.conv0_block_b.zero_grad()
                nmatch.conv0_block_ab.zero_grad()
                nmatch.heatmap_condenser.zero_grad()

            opt.step()

            if bi % 50 == 0:
                running_loss /= 50
                prev_running_loss = running_loss
                print('Loss:', sess_id + '_' + '-'.join([str(ei) + 'e', str(bi) + 'b']), '-', running_loss)
                writer.add_scalar('train_loss', running_loss, ei * len(data_loader_train) + bi)
                running_loss = 0.

            if bi % SAVE_EVERY_N_BATCHES == 0 and bi > 0:
                checkpoint_model(nmatch, prev_running_loss, device, data_loader_val, ima, imb, model_dir, loss_fn, ei,
                                 bi, sess_id, log_fname, val_df_dict, SIDE, viz_dir, writer,
                                 ei * len(data_loader_train) + bi)
            nmatch.train()
        checkpoint_model(nmatch, prev_running_loss, device, data_loader_val, ima, imb, model_dir, loss_fn, ei,
                         bi, sess_id, log_fname, val_df_dict, SIDE, viz_dir, writer,
                         ei * len(data_loader_train) + bi)
