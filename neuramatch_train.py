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
LEARN_RATE = 1e-4
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
RUNNING_LOSS_WINDOW = 50


from datetime import datetime
from multiprocessing import cpu_count
from PIL import Image
import json

from pillow_heif import register_heif_opener
register_heif_opener()

import pytz
from tqdm import tqdm
from coolname import generate_slug
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2

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

    root_dir = 'scratchspace/trained_models_0'
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

    loss_fn = KeypointLoss(device=device, smooth=TVERSKY_SMOOTH, alpha=TVERSKY_ALPHA, gamma=TVERSKY_GAMMA,
                           train_module=TRAIN_MODULE)

    train_config = {'start_learn_rate': LEARN_RATE,
                    'side': utils.SIDE,
                    'num_epochs': NUM_EPOCHS,
                    'im_means': utils.IM_MEANS,
                    'im_stds': utils.IM_STDS,
                    'batch_size': BATCH_SIZE,
                    'blend_coeff': BLEND_COEFF,
                    'ksize': KSIZE,
                    'running_train_loss_window': RUNNING_LOSS_WINDOW,
                    'radius_scale': RADIUS_SCALE,
                    'tversky_smooth': TVERSKY_SMOOTH,
                    'tversky_alpha': TVERSKY_ALPHA,
                    'tversky_gamma': TVERSKY_GAMMA,
                    'train_module': TRAIN_MODULE,
                    'session_id': sess_id}
    if RESUME_MODEL_FPATH is not None:
        train_config['resume_model_fpath'] = RESUME_MODEL_FPATH
    config_fpath = out_dir + '/train_config.json'
    with open(config_fpath, 'w') as f:
        json.dump(train_config, f, indent=4, sort_keys=True)

    nmatch = NeuraMatch(device, utils.SIDE)

    if RESUME_MODEL_FPATH is not None:
        print('Resuming from', RESUME_MODEL_FPATH)
        nmatch.load_state_dict(torch.load(RESUME_MODEL_FPATH, map_location=device), strict=False)
        print('Loaded!')
    else:
        print('Training from scratch...')
    opt = torch.optim.Adam(nmatch.parameters(), lr=LEARN_RATE)

    bi = 0
    ei = 0

    val_df_dict = {'epoch': [],
                   'epoch_batch_iteration': []}

    checkpoint_model(nmatch, None, device, data_loader_val, ima, imb, model_dir, loss_fn, ei, bi, sess_id, log_fname,
                     val_df_dict, viz_dir, writer, 0, KSIZE, RADIUS_SCALE, BLEND_COEFF)
    running_loss = 0.
    running_vector_loss = 0.
    running_conf_loss = 0.
    running_g_mean = 0.
    running_g_std = 0.
    running_g_max = 0.
    running_g_min = 0.

    prev_running_loss = running_loss
    for ei in range(NUM_EPOCHS):
        nmatch.train()
        for bi, (ims, gt_outs) in enumerate(tqdm(data_loader_train)):
            pred_outs = nmatch(ims)
            nmatch.zero_grad()
            (loss, vector_loss, conf_loss, _), _ = loss_fn(pred_outs, gt_outs)
            loss.backward()
            running_loss += loss.item()
            running_vector_loss += vector_loss.item()
            running_conf_loss += conf_loss.item()

            if TRAIN_MODULE == 'matcher':
                nmatch.conv0_block_a.zero_grad()
                nmatch.conv0_block_b.zero_grad()
                nmatch.conv0_block_ab.zero_grad()
                nmatch.heatmap_condenser.zero_grad()

            p = list(nmatch.parameters())
            grad_extract = lambda fn: torch.mean(torch.stack([fn(t.grad) for t in p if t.grad is not None])).item()
            g_mean = grad_extract(torch.mean)
            g_std = grad_extract(torch.std)
            g_max = grad_extract(torch.max)
            g_min = grad_extract(torch.min)

            opt.step()

            if bi % RUNNING_LOSS_WINDOW == 0:
                running_loss /= RUNNING_LOSS_WINDOW
                running_vector_loss /= RUNNING_LOSS_WINDOW
                running_conf_loss /= RUNNING_LOSS_WINDOW
                running_g_mean /= RUNNING_LOSS_WINDOW
                running_g_std /= RUNNING_LOSS_WINDOW
                running_g_max /= RUNNING_LOSS_WINDOW
                running_g_min /= RUNNING_LOSS_WINDOW
                prev_running_losses = [running_loss, running_vector_loss, running_conf_loss]
                prev_running_grad_measures = [running_g_mean, running_g_std, running_g_max, running_g_min]
                print('Loss:', sess_id + '_' + '-'.join([str(ei) + 'e', str(bi) + 'b']), '-', running_loss)
                print('Vector Loss:', sess_id + '_' + '-'.join([str(ei) + 'e', str(bi) + 'b']), '-',
                      running_vector_loss)
                print('Conf Loss:', sess_id + '_' + '-'.join([str(ei) + 'e', str(bi) + 'b']), '-', running_conf_loss)
                print('Grad Mean:', sess_id + '_' + '-'.join([str(ei) + 'e', str(bi) + 'b']), '-', g_mean)
                print('Grad Std:', sess_id + '_' + '-'.join([str(ei) + 'e', str(bi) + 'b']), '-', g_std)
                print('Grad Max:', sess_id + '_' + '-'.join([str(ei) + 'e', str(bi) + 'b']), '-', g_max)
                print('Grad Min:', sess_id + '_' + '-'.join([str(ei) + 'e', str(bi) + 'b']), '-', g_min)
                writer.add_scalar('train_loss', running_loss, ei * len(data_loader_train) + bi)
                writer.add_scalar('train_vector_loss', running_vector_loss, ei * len(data_loader_train) + bi)
                writer.add_scalar('train_conf_loss', running_conf_loss, ei * len(data_loader_train) + bi)
                writer.add_scalar('grad_mean', g_mean, ei * len(data_loader_train) + bi)
                writer.add_scalar('grad_std', g_std, ei * len(data_loader_train) + bi)
                writer.add_scalar('grad_max', g_max, ei * len(data_loader_train) + bi)
                writer.add_scalar('grad_min', g_min, ei * len(data_loader_train) + bi)
                running_loss = 0.
                running_vector_loss = 0.
                running_conf_loss = 0.
                running_g_mean = 0.
                running_g_std = 0.
                running_g_max = 0.
                running_g_min = 0.

            if bi % SAVE_EVERY_N_BATCHES == 0 and bi > 0:
                checkpoint_model(nmatch, (prev_running_losses, prev_running_grad_measures), device, data_loader_val,
                                 ima, imb, model_dir, loss_fn, ei,
                                 bi, sess_id, log_fname, val_df_dict, viz_dir, writer,
                                 ei * len(data_loader_train) + bi)
            nmatch.train()
        checkpoint_model(nmatch, (prev_running_losses, prev_running_grad_measures), device, data_loader_val,
                         ima, imb, model_dir, loss_fn, ei,
                         bi, sess_id, log_fname, val_df_dict, viz_dir, writer,
                         ei * len(data_loader_train) + bi)
