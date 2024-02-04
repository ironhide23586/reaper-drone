#ifndef KALMAN_H
#define KALMAN_H

#include <math.h>
#include <Arduino.h>


class KalmanFilter {

  public:

    KalmanFilter(float init_est=0.0f, float measured_error_mean=0.0f, float measured_error_std=1.0f,
                 float estimated_error_mean=0.0f, float estimated_error_std=1.0f);

    float Predict(float x_mea_, int n_passes=5, float mea_err=0.1f);

  private:

    void get_kalman_gain(float mea_err);

    float mea_err_mean, mea_err_std, est_err_mean, est_err_std, x_est;
    float est_err = 0.1f, mea_err_pv = 0.1f;
    float k_gain;

};

#endif