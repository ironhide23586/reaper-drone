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


# importing time library to make Rpi wait because its too impatient

# os.system("sudo pigpiod")  # Launching GPIO library
# time.sleep(1)  # As i said it is too impatient and so if this delay is removed you will get an error
import pigpio  # importing GPIO library

ESC = 4  # Connect the ESC in this GPIO pin

pi = pigpio.pi()
if not pi.connected:
    sys.exit(1)

max_value = 2000
min_value = 1225
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
    drive(-0.0001)


def get_thrust(s):
    pwm_val = drive(s)
    time.sleep(9)
    m = hx711.get_weight(9)
    print('Generating thrust of', m, 'grams at', 100. * s, '% power')
    time.sleep(2)
    return m, pwm_val


if __name__ == '__main__':
    hx711 = HX711(5, 6)

    hx711.reset()
    hx711.tare()

    ref_unit = 660
    hx711.set_reference_unit(ref_unit)

    # for i in range(50):
    #     m = hx711.get_weight(9)
    #     print(m)

    c = {'power': [],
         'thrust': [],
         'pwm_val': []}

    # init()
    # arm()

    for i in tqdm(range(50)):
        pval = np.random.random()
        th, pwm_val = get_thrust(pval)

        c['power'].append(pval)
        c['thrust'].append(th)
        c['pwm_val'].append(pwm_val)
        brake()
        time.sleep(1)
    df = pd.DataFrame(c)
    df.to_csv('motor_characteristics_.csv', index=False)

    brake()
    pi.stop()
    GPIO.cleanup()

