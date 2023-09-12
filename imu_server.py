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
from queue import Queue

from mpu_i2c import *
time.sleep(1)  # delay necessary to allow mpu9250 to settle
from threading import Thread

from flask import Flask
import numpy as np

app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


all_res = Queue(10)


def acc_to_pitch_roll(ax, ay, az):
    yaw = 90 - np.rad2deg(np.arctan2(ay, ax))
    pitch = 90 - np.rad2deg(np.arctan2(az, -ax))
    roll = 90 - np.rad2deg(np.arctan2(az, ay))
    return yaw, pitch, roll


def gyro_worker(all_res):
    t_prev = 0
    g_prev = np.zeros(3)
    thetas = np.zeros(3)
    ki = 0
    buff_size = 3
    theta_buff = np.zeros([buff_size, 3])
    while True:
        try:
            ax, ay, az, wx, wy, wz = mpu6050_conv()  # read and convert mpu6050 data=
            acc_yaw, acc_pitch, acc_roll = acc_to_pitch_roll(ax, ay, az)
        except:
            print('Error reading value...')
        data_val = '\t'.join(map(str, [ax, ay, az])) + ' ' + '\t'.join(map(str, thetas))
        if ki % 100 == 0:
            print(data_val)
        gyro = np.array([wz, wy, wx])
        if all_res.full():
            all_res.queue.clear()
        if t_prev == 0:
            t_prev = time.time()
            g_prev = gyro
        else:
            tc = time.time()
            t_d = tc - t_prev
            t_prev = tc
            g_d = gyro - g_prev
            d_theta = g_d * t_d
            thetas += d_theta
            # theta_buff[ki % buff_size] = thetas
            # if ki > buff_size:
            #     thetas = theta_buff.mean(axis=0)
            if ki % 500 == 0:
                thetas[1:] = .9 * np.array([acc_pitch, acc_roll]) + .1 * thetas[1:]
            else:
                res = '_'.join(map(str, list(thetas) + [ax, ay, az] + [wx, wy, wz] + [acc_yaw, acc_pitch, acc_roll]))
                all_res.put(res)
        ki += 1


@app.route("/imu")
def imu():
    return all_res.get()


if __name__ == "__main__":
    t = Thread(target=gyro_worker, args=(all_res, ))
    t.start()
    app.run(debug=False, host='0.0.0.0')
