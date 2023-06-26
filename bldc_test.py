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

import os  # importing os library so as to communicate with the system
import time
import sys

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
min_value = 1190
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


def drive(s, pin_id=ESC, delay=0.1):
    if s == 0:
        v = 0
    else:
        v = int(window * s + min_value)
    print('Driving motor at', 100 * s, '% speed with PWM width', v)
    pi.set_servo_pulsewidth(pin_id, v)
    # time.sleep(delay)


def brake(pin_id=ESC):
    drive(-0.0001)


if __name__ == '__main__':
    init()
    arm()

    drive(.1)
    brake()
    for i in range(0, 100000):
        drive(np.random.random())
        # time.sleep(1)
        k = 0
    brake()
    pi.stop()

