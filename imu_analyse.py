
import os
import datetime
from glob import glob
import sys
# import fnmatch

import serial
# import pandas as pd
# from matplotlib import pyplot as plt


out_dir = 'sensor_logs'
os.makedirs(out_dir, exist_ok=True)

fid = datetime.datetime.now().strftime("%x.%X").replace('/', '_').replace(':', '-')
out_fpath = out_dir + '/' + fid + '_mpu6050.csv'


def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

# existing_df_fpaths = glob(out_dir + '/*')
#
# for fp in existing_df_fpaths:
#     df = pd.read_csv(fp)[:50]
#     yaw_cols = list(map(lambda x: x.replace(':', ''), fnmatch.filter(df.columns, '*yaw*')))
#     pitch_cols = list(map(lambda x: x.replace(':', ''), fnmatch.filter(df.columns, '*pitch*')))
#     roll_cols = list(map(lambda x: x.replace(':', ''), fnmatch.filter(df.columns, '*roll*')))
#
#     tags = ['yaw', 'pitch', 'roll']
#     all_cols = [yaw_cols, pitch_cols, roll_cols]
#
#     for i in range(len(tags)):
#         c = [p for p in all_cols[i] if 'unfiltered_kf' in p or '-filtered_kf' in p]
#         all_cols[i] = c
#
#     for i in range(len(tags)):
#         for j in range(len(all_cols[i])):
#             plt.plot(df[all_cols[i][j] + ':'], label=all_cols[i][j])
#
#         plt.legend()
#         plt.title(tags[i])
#         plt.show()
#
#         k=0
#
#
#     k = 0

import pygame

# from OpenGL.GL import *
# from OpenGL.GLU import *
from pygame.constants import OPENGL, DOUBLEBUF, QUIT, KEYDOWN, K_ESCAPE
from imu_client import viz_init, draw, resizewin

video_flags = OPENGL | DOUBLEBUF
screen = pygame.display.set_mode((400, 400), video_flags)
pygame.display.set_caption("IMU orientation visualization")
resizewin(400, 400)
viz_init()
pygame.init()
ticks = pygame.time.get_ticks()

sps = serial_ports()
print(sps[-1])
s = serial.Serial(sps[-1], 250000)
r = s.readline()
cnt = 0
res = {}
while True:
    r = s.readline()
    a = r.decode('utf-8', errors='ignore').strip()
    chunks = a.split('\t')
    if '<->' in a:
        y, p, r = list(map(float, chunks[:3]))
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
        draw(y, p, r)
        pygame.display.flip()
    elif len(a) > 0:
        print(a)


    # if len(chunks) == 16:
    #     for i in range(4):
    #         si = (i * 4) + 1
    #         ei = si + 3
    #         k = chunks[si - 1].split('-')[-1]
    #         v = list(map(float, chunks[si:ei]))
    #         ks = [b + '-' + k for b in ['yaw', 'pitch', 'roll']]
    #         for ki in range(len(ks)):
    #             if ks[ki] not in res:
    #                 res[ks[ki]] = [v[ki]]
    #             else:
    #                 res[ks[ki]].append(v[ki])
    #     cnt += 1
    # print(a)

    # if cnt >= 100:
    #     res_df = pd.DataFrame(res)
    #     res_df.to_csv(out_fpath, index=False)
    #     cnt = 0
