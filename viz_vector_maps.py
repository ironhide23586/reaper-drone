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

import cv2
import numpy as np


DIR = 'scratchspace/trained_models_1/lurking-nuthatch-all.25-10-2023.20_10_21/viz'


if __name__ == '__main__':
    fpaths = glob(DIR + '/*all_vectors*')

    for fpath in fpaths:
        img = cv2.imread(fpath)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(28, 28))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Stacking the original image with the enhanced image
        result = np.hstack((img, enhanced_img))
        cv2.imshow('Result', result)
        cv2.waitKey(0)

    k = 0
