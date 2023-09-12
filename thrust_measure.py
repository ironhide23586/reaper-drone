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
import sys

import pandas as pd
from hx711 import HX711
import RPi.GPIO as GPIO
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt


# importing time library to make Rpi wait because its too impatient

# os.system("sudo pigpiod")  # Launching GPIO library
# time.sleep(1)  # As i said it is too impatient and so if this delay is removed you will get an error
import pigpio  # importing GPIO library

ESC = 4  # Connect the ESC in this GPIO pin

pi = pigpio.pi()
if not pi.connected:
    sys.exit(1)

max_value = 2000
min_value = 1315
window = max_value - min_value


def init(pin_id=ESC):
    print('Initializing BLDC...')
    pi.set_servo_pulsewidth(pin_id, 0)
    time.sleep(1)
    pi.set_servo_pulsewidth(pin_id, max_value)
    time.sleep(7)
    pi.set_servo_pulsewidth(pin_id, min_value)
    time.sleep(1)


def arm(pin_id=ESC):
    print('Arming... :D')
    pi.set_servo_pulsewidth(pin_id, 0)
    time.sleep(1)
    pi.set_servo_pulsewidth(pin_id, max_value)
    time.sleep(1)
    pi.set_servo_pulsewidth(pin_id, min_value)
    time.sleep(3)
    # pi.set_servo_pulsewidth(pin_id, max_value)
    # time.sleep(1)
    # pi.set_servo_pulsewidth(pin_id, min_value)
    # time.sleep(3)




def drive(s, pin_id=ESC, delay=0.1):
    if s == 0:
        v = 0
    else:
        v = int(window * s + min_value)
    # print('Driving motor at', 100 * s, '% speed with PWM width', v)
    pi.set_servo_pulsewidth(pin_id, v)
    # time.sleep(delay)
    return v


def brake(pin_id=ESC):
    drive(-0.1)


def get_thrust(s):
    pwm_val = drive(s)
    time.sleep(3)
    # ramp_thrust = []
    # for i in range(20):  # measuring thrust change characeristics
    #     ramp_thrust.append(hx711.get_weight(3))
    #     # time.sleep(0.45)
    m = hx711.get_weight(3)
    if m != m:
        m = 0.
    print('Generating thrust of', m, 'grams at', 100. * s, '% power')
    # decay_thrust = []
    # for i in range(8):  # measuring thrust change characeristics
    #     decay_thrust.append(hx711.get_weight(3))
        # time.sleep(0.25)
    return m, pwm_val #, [ramp_thrust, decay_thrust]


if __name__ == '__main__':

    # print('Done!', hx711.min_z_err, hx711.max_z_err)
    # ws = np.hstack([wp, wn])
    # f = np.logical_and(ws > -60, ws < 60)
    # if True in f:
    #     hx711.mea_mean = np.mean(ws[f])
    #     hx711.mea_std = np.std(ws[f])
    #     print('Initial measurement error mean and std', hx711.mea_mean, hx711.mea_std)
    #
    # print('Finding Error spread...')
    # ws = []
    # for i in range(10):
    #     w = hx711.get_weight(times=3, filter=True)
    #     ws.append(w)
    # hx711.kf = None
    # ws = np.array(ws)
    # f = np.logical_and(ws > -60, ws < 60)
    # hx711.mea_mean = np.mean(ws[f])
    # hx711.mea_std = np.std(ws[f])
    # print('Done!', hx711.mea_mean, hx711.mea_std)

    # ws = []
    # while True:
    #     w = hx711.get_weight(times=3)
    #     ws.append(w)
    #     print(w)
    #
    #     if len(ws) % 100 == 0:
    #         fig, ax = plt.subplots(figsize=(10, 7))
    #         ax.hist(ws, bins=50)
    #         plt.show()

    c = {'power': [],
         'thrust': [],
         'pwm_val': []}

    # init()
    # arm()

    hx711 = HX711(5, 6)

    hx711.set_reading_format("MSB", "MSB")

    ref_unit = 660
    hx711.set_reference_unit(ref_unit)
    # time.sleep(3)

    hx711.reset()
    hx711.tare()
    # ws = []
    #
    wp = []
    wn = []
    print('Calibrating weight sensor...')
    for i in range(10):
        w = hx711.get_weight(times=3)
        if w >= 0:
            wp.append(w)
        else:
            wn.append(abs(w))
    hx711.min_z_err = np.mean(wn)
    hx711.max_z_err = np.mean(wp)

    if float(sys.argv[1]) > .3:
        print('HIGH POWER TEST WARNING: Waiting 20 seconds before test starts, VACATE THE AREA!!!!!')
        time.sleep(20)

    for i in tqdm(range(int(sys.argv[2]))):
        pval = np.random.uniform(0, float(sys.argv[1]))
        th, pwm_val = get_thrust(pval)

        c['power'].append(pval)
        c['thrust'].append(th)

        c['pwm_val'].append(pwm_val)
        brake()
        df = pd.DataFrame(c)
        df.to_csv('motor_characteristics_withbrake_2450kv_.csv', index=False)
        time.sleep(1)
    df = pd.DataFrame(c)
    df.to_csv('motor_characteristics_withbrake_2450kv_.csv', index=False)

    brake()
    pi.stop()
    GPIO.cleanup()

