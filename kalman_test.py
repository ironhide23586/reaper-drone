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


import numpy as np
from matplotlib import pyplot as plt

from kalman_filter.algorithm import KalmanFilter


N = 10
N_SAMPLES = 100
NOISE_FACTOR = .5


def gen_signal(noise=.35):
    x = np.linspace(1, N, N_SAMPLES)
    n = np.random.normal(size=x.shape[0])
    f = lambda x_, noise_factor: np.log(2 * x_) * (np.exp(np.sin(2 * x_) + noise_factor * n) + .4 * np.exp(.2 * x_))
    y_gt = f(x, 0.)
    y_noisy = f(x, noise)
    return x, y_gt, y_noisy


if __name__ == '__main__':
    x, y, y_n = gen_signal(NOISE_FACTOR)
    # plt.plot(x, y, color='blue')
    plt.plot(x, y_n, color='green')

    kf = KalmanFilter(y_n[0])
    y_pred = [y_n[0]]
    for i in range(1, x.shape[0]):
        y_p = kf.predict(y_n[i])
        y_pred.append(y_p)
    y_pred = np.array(y_pred)
    plt.plot(x, y_pred, 'black')
    plt.show()

    k = 0
