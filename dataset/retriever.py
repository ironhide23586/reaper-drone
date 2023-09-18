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

import os

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np

import utils


class VideoCap:

    def __init__(self, fp):
        self.fp = fp
        self.fn = fp.split(os.sep)[-1]
        self.cap ,self.fps, self.n_frames, self.width, self.height = self.read_video_file(fp)
        self.min_s = min(self.width, self.height)
        self.x_offset = (self.width - self.min_s) // 2
        self.y_offset = (self.height - self.min_s) // 2

    def read_video_file(self, fp):
        cap = cv2.VideoCapture(fp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap, fps, n_frames, width, height

    # def read_video_file(self, fp):
    #     cap = VideoFileClip(fp)
    #     fps = self.cap.get(cv2.CAP_PROP_FPS)
    #     n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     return cap, fps, n_frames, width, height

    def read_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        res, frame = self.cap.read()
        return frame

    def center_crop(self, frame):
        im = frame[self.y_offset: self.y_offset + self.min_s, self.x_offset: self.x_offset + self.min_s]
        return im

    def random_crop(self, frame):
        yo = np.random.randint(0, self.y_offset + 1)
        xo = np.random.randint(0, self.x_offset + 1)
        im = frame[yo: yo + self.min_s, xo: xo + self.min_s]
        return im

    def get_frame(self, frame_idx, side, random_crop=False, random_rotate=False):
        frame = self.read_frame(frame_idx)
        if random_crop:
            frame_square = self.random_crop(frame)
        else:
            frame_square = self.center_crop(frame)
        if random_rotate:
            frame_square = utils.rotate_image(frame_square, np.random.randint(-180, 180))
        frame_square = cv2.resize(frame_square, (side, side))
        return frame_square


class HandCurated:

    def __init__(self, video_fps, mode='val', side=480, sift_tolerance=.4, min_n_matches=5, frame_interval=150):
        self.mode = mode
        self.side = side
        self.video_caps = []
        self.sift_tolerance = sift_tolerance
        self.min_n_matches = int(min_n_matches)
        self.frame_interval = int(frame_interval)
        self.tag = '_'.join([str(sift_tolerance) + '.sift-tolerance',
                             str(min_n_matches) + '.min-n-matches',
                             str(frame_interval) + '.max-frame-interval',
                             str(self.side) + '.side',
                             self.mode])
        self.n_videos = self.ingest_videos(video_fps)
        self.keypoint_detector = cv2.xfeatures2d.SIFT_create()
        self.keypoint_matcher = cv2.BFMatcher()

    def ingest_videos(self, fps):
        # fps = glob(dir + os.sep + '*.mp4')
        for fp in fps:
            video_obj = VideoCap(fp)
            if video_obj.n_frames > 25:
                self.video_caps.append(video_obj)
            else:
                print('Skipping', fp, 'as number of frames is too low (', video_obj.n_frames, ')')
        return len(self.video_caps)

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
                    if not np.array_equal(kp_a_sel.pt, kp_b_sel.pt):
                        y.append(np.hstack([kp_a_sel.pt, kp_b_sel.pt]))
                        good.append([m])
        y = np.array(y)
        return y, (good, kp_a, kp_b)

    def get_matches_worker(self):
        random_flag = self.mode == 'train'
        window_size = np.random.randint(1, self.frame_interval)
        ni = np.random.randint(0, self.n_videos)
        si = min(3 * self.video_caps[ni].fps, max(0, self.video_caps[ni].n_frames - 2 * window_size))
        ei = max(self.video_caps[ni].n_frames - window_size - 1 - 100, si + 1)
        fi = int(np.clip(np.random.randint(si, ei), 0, self.video_caps[ni].n_frames - 2))
        im_a = self.video_caps[ni].get_frame(fi, self.side,
                                             random_crop=random_flag, random_rotate=random_flag)
        im_b = self.video_caps[ni].get_frame(min(fi + window_size, fi + 1), self.side,
                                             random_crop=random_flag, random_rotate=random_flag)
        matches_xy, match_data = self.sift_match(im_a, im_b)
        match_angle_std = 0.

        if matches_xy.shape[0] > 0:
            match_vecs = matches_xy[:, 2:] - matches_xy[:, :2]
            match_vecs = match_vecs / np.linalg.norm(match_vecs)
            match_vec_angles = np.rad2deg(np.arctan2(match_vecs[:, 1], match_vecs[:, 0]))
            match_angle_std = np.std(match_vec_angles)

        frame_meta_data_a = [self.video_caps[ni].fn, fi]
        frame_meta_data_b = [self.video_caps[ni].fn, fi + window_size]
        frame_meta_data = [frame_meta_data_a, frame_meta_data_b]
        return im_a, im_b, (matches_xy, match_data), frame_meta_data, match_angle_std

    def viz_matches(self, im_a, im_b, matches):
        matches_xy, match_data = matches
        m, kp_a, kp_b = match_data
        img_gt_match_viz = cv2.drawMatchesKnn(im_a, kp_a, im_b, kp_b, matches1to2=m, outImg=None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_gt_match_viz

    def sample_image_pair(self, viz=True):
        im_a, im_b, matches, frame_metadata, match_angle_std = self.get_matches_worker()
        while matches[0].shape[0] < self.min_n_matches:
            im_a, im_b, matches, frame_metadata, match_angle_std = self.get_matches_worker()
        im_viz = None
        if viz:
            im_viz = self.viz_matches(im_a, im_b, matches)
        im_ab = np.rollaxis(np.stack([im_a, im_b]), 3, 1)
        return im_ab, matches, frame_metadata, im_viz
