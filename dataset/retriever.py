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
import fnmatch
import os

import cv2
import numpy as np

import utils


class VideoCap:

    def __init__(self, fp):
        self.fp = fp
        self.fn = fp.split(os.sep)[-1]
        self.cap = cv2.VideoCapture(fp)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.min_s = min(self.width, self.height)
        self.x_offset = (self.width - self.min_s) // 2
        self.y_offset = (self.height - self.min_s) // 2

    def center_crop(self, frame):
        im = frame[self.y_offset: self.y_offset + self.min_s, self.x_offset: self.x_offset + self.min_s]
        return im

    def get_frame(self, frame_idx, side):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        res, frame = self.cap.read()
        frame_square = self.center_crop(frame)
        frame_square = cv2.resize(frame_square, (side, side))
        return frame_square


class HandCurated:

    def __init__(self, dataset_dir, mode='val', side=480, sift_tolerance=.4, min_n_matches=5, frame_interval=150,
                 video_dir='videos'):
        self.mode = mode
        self.side = side
        self.video_caps = []
        self.sift_tolerance = sift_tolerance
        self.min_n_matches = int(min_n_matches)
        self.frame_interval = int(frame_interval)
        self.tag = '_'.join([str(sift_tolerance) + '.sift-tolerance',
                             str(min_n_matches) + '.min-n-matches',
                             str(frame_interval) + '.frame-interval',
                             str(self.side) + '.side',
                             self.mode])
        self.data_dirs = [d + os.sep + self.mode for d in glob(dataset_dir + os.sep + '*')]
        self.video_dir = video_dir
        # self.image_dir = fnmatch.filter(self.data_dirs, '*' + os.sep + 'images' + os.sep + '*')[0]
        self.n_videos = self.ingest_videos(self.video_dir)
        self.keypoint_detector = cv2.xfeatures2d.SIFT_create()
        self.keypoint_matcher = cv2.BFMatcher()

    def ingest_videos(self, dir):
        fps = glob(dir + os.sep + '*.mp4')
        for fp in fps:
            video_obj = VideoCap(fp)
            self.video_caps.append(video_obj)
        return len(fps)

    def sift_match(self, im_a, im_b):
        good = []
        y = []
        im_a_cv2 = cv2.cvtColor(im_a, cv2.COLOR_BGR2GRAY)
        im_b_cv2 = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
        kp_a, desc_a = self.keypoint_detector.detectAndCompute(im_a_cv2, None)
        kp_b, desc_b = self.keypoint_detector.detectAndCompute(im_b_cv2, None)
        if len(kp_a) > self.min_n_matches and len(kp_b) > self.min_n_matches:
            matches = self.keypoint_matcher.knnMatch(desc_a, desc_b, k=2)
            for m, n in matches:
                if m.distance < self.sift_tolerance * n.distance:
                    kp_a_sel = kp_a[m.queryIdx]
                    kp_b_sel = kp_b[m.trainIdx]
                    y.append(np.hstack([kp_a_sel.pt, kp_b_sel.pt]))
                    good.append([m])
        y = np.array(y)
        return y, (good, kp_a, kp_b)

    def get_matches_worker(self, window_size):
        # if self.mode == 'train':
        window_size = np.random.randint(1, self.frame_interval)
        ni = np.random.randint(0, self.n_videos)
        fi = np.random.randint(3 * self.video_caps[ni].fps, self.video_caps[ni].n_frames - window_size - 1 - 100)
        im_a = self.video_caps[ni].get_frame(fi, self.side)
        im_b = self.video_caps[ni].get_frame(fi + window_size, self.side)
        matches_xy, match_data = self.sift_match(im_a, im_b)
        frame_meta_data_a = [self.video_caps[ni].fn, fi]
        frame_meta_data_b = [self.video_caps[ni].fn, fi + window_size]
        frame_meta_data = [frame_meta_data_a, frame_meta_data_b]
        return im_a, im_b, (matches_xy, match_data), frame_meta_data

    def viz_matches(self, im_a, im_b, matches):
        matches_xy, match_data = matches
        m, kp_a, kp_b = match_data
        img_gt_match_viz = cv2.drawMatchesKnn(im_a, kp_a, im_b, kp_b, matches1to2=m, outImg=None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_gt_match_viz

    def sample_image_pair(self, viz=True, window_size=None):
        if window_size is None:
            window_size = self.frame_interval
        im_a, im_b, matches, frame_metadata = self.get_matches_worker(window_size)
        while matches[0].shape[0] < self.min_n_matches:
            im_a, im_b, matches, frame_metadata = self.get_matches_worker(window_size)
        im_viz = None
        if viz:
            im_viz = self.viz_matches(im_a, im_b, matches)
        im_ab = np.rollaxis(np.stack([im_a, im_b]), 3, 1)
        return im_ab, matches, frame_metadata, im_viz
