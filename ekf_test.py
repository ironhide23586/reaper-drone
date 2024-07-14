import numpy as np

from ekf import ExtendedKalmanFilter


if __name__ == '__main__':

    # Generate some 16 dim data to test the EKF with (which is the state vector)

    ekf = ExtendedKalmanFilter()
    accel = np.array([0.2, 0.1, 0.8])
    gyro = np.array([-.03, .01, -.02])
    delta_t = 0.1
    sensor_readings = accel, gyro
    ekf.predict(delta_t, sensor_readings)




