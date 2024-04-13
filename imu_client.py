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

import requests
import time

import pygame

from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *


REMOTE_IP = '192.168.68.58'



def resizewin(width, height):
    """
    For resizing window
    """
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0*width/height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def viz_init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)


def draw(nx, ny, nz):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0)

    # drawText((-2.6, 1.8, 2), "PyTeapot", 18)
    # drawText((-2.6, 1.6, 2), "Module to visualize quaternion or Euler angles data", 4)
    # drawText((-2.6, -2, 2), "Press Escape to exit.", 16)

    yaw = nx
    pitch = ny
    roll = -nz
    # drawText((-2.6, -1.8, 2), "Yaw: %f, Pitch: %f, Roll: %f" %(yaw, pitch, roll), 4)

    glRotatef(roll, 0.00, 0.00, 1.00)
    glRotatef(pitch, 1.00, 0.00, 0.00)
    glRotatef(yaw, 0.00, 1.00, 0.00)

    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(1.0, 0.2, 1.0)

    glColor3f(1.0, 0.5, 0.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(1.0, -0.2, -1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, -1.0)
    glEnd()


def drawText(position, textString, size):
    font = pygame.font.SysFont("Courier", size, True)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


if __name__ == '__main__':
    url = 'http://' + REMOTE_IP + ':5000/imu'
    video_flags = OPENGL | DOUBLEBUF
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), video_flags)
    pygame.display.set_caption("PyTeapot IMU orientation visualization")
    resizewin(1920, 1080)
    viz_init()
    frames = 0
    ticks = pygame.time.get_ticks()
    while True:
        try:
            res = requests.get(url)
            if res.status_code == 200:
                event = pygame.event.poll()
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    break
                gyro = res.content.decode('utf-8')
                # acc, gyro = y.split(' ')
                gyro = list(map(float, gyro.split('_')))
                gyro_yaw, gyro_pitch, gyro_roll, ax, ay, az, wx, wy, wz, acc_yaw, acc_pitch, acc_roll = gyro
                draw(1, gyro_yaw, gyro_pitch, gyro_roll, ax, ay, az, wx, wy, wz)
                # draw(1, gyro_yaw, acc_pitch, acc_roll, ax, ay, az, wx, wy, wz)
                pygame.display.flip()
                frames += 1
            else:
                print('Invalid response...')
        except:
            print('Server down, waiting....')
            time.sleep(2)

    print("fps: %d" % ((frames * 1000) / (pygame.time.get_ticks() - ticks)))

    # while True:
    #     url = 'http://' + REMOTE_IP + ':5000/imu'
    #
    #     if res.status_code == 200:
    #         y = res.content.decode('utf-8')
    #         acc, gyro = y.split(' ')
    #         acc = list(map(float, acc.split('_')))
    #         gyro = list(map(float, gyro.split('_')))
    #         print(gyro)
    # k = 0


