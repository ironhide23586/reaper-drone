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

import time

import cv2
from picamera2 import Picamera2



if __name__ == '__main__':
    cam = Picamera2()
    time.sleep(2)
    cam.start()

    im = cam.capture_image()

    k = 0

    cam.stop()
