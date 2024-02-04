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



class KalmanFilter:

    def __init__(self, init_est, measured_error_mean=0., measured_error_std=1.,
                 estimated_error_mean=.0, estimated_error_std=1.):
        self.mea_err_mean = measured_error_mean
        self.mea_err_std = measured_error_std
        self.est_err_mean = estimated_error_mean
        self.est_err_std = estimated_error_std
        self.x_est = init_est
        self.est_err = np.abs(np.random.normal(self.est_err_mean, self.est_err_std))
        self.mea_err = None
        # self.est_err = 5.

    def get_kalman_gain(self, mea_err=None):
        if mea_err is None:
            mea_err = np.random.normal(self.mea_err_mean, self.mea_err_std)
            # mea_err = np.random.uniform(-3, 3)
        kgain = self.est_err / (self.est_err + np.abs(mea_err))
        return kgain, np.abs(mea_err)

    def predict(self, x_mea_, n_passes=5, mea_err=None):
        x_mea = x_mea_
        if mea_err is None:
            mea_err = self.mea_err
        for _ in range(n_passes):
            k_gain, self.mea_err = self.get_kalman_gain(mea_err)

            y = self.x_est + k_gain * (x_mea - self.x_est)
            self.est_err = (1. - k_gain) * self.mea_err
            self.x_est = y
            # y = np.clip(y, -10, 20)
            x_mea = y
        return y


