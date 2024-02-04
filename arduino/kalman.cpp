#include "kalman.h"


float randf(float l=-1.0f, float h=1.0f, long scale=1000) {
  float offset = ((float)random(scale) / scale) * (h - l);
  float res = l + offset;
  // Serial.println(res);
  return res;
}


KalmanFilter::KalmanFilter(float init_est, float measured_error_mean, float measured_error_std, 
                           float estimated_error_mean, float estimated_error_std) {
  KalmanFilter::mea_err_mean = measured_error_mean;
  KalmanFilter::mea_err_std = measured_error_std;
  KalmanFilter::est_err_mean = estimated_error_mean;
  KalmanFilter::est_err_std = estimated_error_std;
  KalmanFilter::x_est = init_est;
  KalmanFilter::est_err = randf();
  
  // Serial.println("========");
  // Serial.println(KalmanFilter::mea_err_mean);
  // Serial.println("====+====");

}

void KalmanFilter::get_kalman_gain(float mea_err) {
  if (mea_err == 0.1f) {
    mea_err = randf();
  }
  KalmanFilter::mea_err_pv = abs(mea_err);
  KalmanFilter::k_gain = KalmanFilter::est_err / (KalmanFilter::est_err + KalmanFilter::mea_err_pv);
}


float KalmanFilter::Predict(float x_mea_, int n_passes, float mea_err) {
  float x_mea = x_mea_;
  float y;
  if (mea_err == 0.1f) {
    mea_err = KalmanFilter::mea_err_pv;
  }
  if (KalmanFilter::x_est == 0.0f) KalmanFilter::x_est = x_mea_;
  for (int i = 0; i < n_passes; i++) {
    // Serial.println(i);
    get_kalman_gain(mea_err);
    
    y = KalmanFilter::x_est + (KalmanFilter::k_gain * (x_mea - KalmanFilter::x_est));
    // Serial.println("fl__");
    KalmanFilter::est_err = (1. - KalmanFilter::k_gain) * KalmanFilter::mea_err_pv;
    KalmanFilter::x_est = y;
    x_mea = y;
    // Serial.println("fl_");
  }
  return y;
}

