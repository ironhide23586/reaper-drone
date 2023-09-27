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

import math
import os

import numpy as np
import cv2


DATASET_DIR = 'scratchspace/datasets'
OPENAI_API_KEY = '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'

HAND_CURATED_DATASET_DIR = DATASET_DIR + os.sep + 'hand_curated'

DOWNLOADED_DATASET_DIR = DATASET_DIR + os.sep + 'downloaded'


def create_heatmap(pxy, s):
  m = np.zeros(s * s, dtype=np.uint8)
  pxy_1d = (pxy[:, 0].astype(int) + (pxy[:, 1].astype(int) * s))
  m[pxy_1d] = 1
  m = m.reshape([s, s])
  return m


def cantor_fn(x, y):  # maps 2 ints to a unique integer
  return (((x ** 2) + x + (2 * x * y) + (3 * y) + (y ** 2)) / 2).long()


def rotate_image(image, angle, remove_black_patches=True):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  if remove_black_patches:
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0], np.deg2rad(angle))
    h, w, _ = image.shape
    offset_w = (w - int(wr)) // 2
    offset_h = (h - int(hr)) // 2
    im_cropped = result[offset_h: offset_h + int(hr), offset_w: offset_w + int(wr)]
    result = cv2.resize(im_cropped, (w, h))
  return result

def drawMatches(img1, kp1, img2, kp2):
  rows1 = img1.shape[0]
  cols1 = img1.shape[1]
  rows2 = img2.shape[0]
  cols2 = img2.shape[1]
  out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
  out[:rows1, :cols1] = img1
  out[:rows2, cols1:cols1 + cols2] = img2
  for mi in range(kp1.shape[0]):
    (x1, y1) = kp1[mi]
    (x2, y2) = kp2[mi]
    cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
    cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)
    cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)
  return out


def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr

