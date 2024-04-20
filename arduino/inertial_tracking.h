#ifndef IMU_H
#define IMU_H

// #include "mpu6050.h"
#include "lsm_imu_module.h"
#include "kalman.h"

// #include <Arduino_LSM9DS1.h>


namespace InertialTracking {

    class IMUDevice {
    private:

        bool kf_init = false;
        uint8_t dev_id;
        uint8_t dev_addr;
        // MPU6050* dev;
        LSM9DS1* dev;

    public:

        bool calibrated = false;

        float gyro_xyz[3];
        float acc_xyz[3];
        float mag_xyz[3];
        double read_timestamp;

        float gyro_xyz_offset[3] = {0};

        float temp;
        
        // IMUDevice(MPU6050* dev);
        IMUDevice(LSM9DS1* dev);

        void init();
        void read_data();
        void compute_calibration_constants();
    };


    class Pose {
    private:

        int calibration_counter = 1;
        float yaw_tmp = 0, pitch_tmp, roll_tmp, heading_tmp;
        float num_tmp = 0, den_tmp = 0;
        float yaw_offset;

        float gyro_xyz[3], acc_xyz[3], mag_xyz[3];
        float gyro_drift[3] = {0};
        float acc_drift[3] = {0};
        // float ypr_offset[3] = {0};
        bool ypr_offset_recorded = false;
        float gyro_w[2] = {GYRO_WEIGHT, GYRO_WEIGHT};

        float accAngleX, accAngleY;
        float gyroAngleX = 0, gyroAngleY = 0;//, gyroAngleZ = 0;
        double previousTime, elapsedTime;
        
        float yaw_raw, yaw_kf;
        float pitch_raw, pitch_kf;
        float roll_raw, roll_kf;

        IMUDevice *imu_device;

        KalmanFilter* kf_yaw = new KalmanFilter();
        KalmanFilter* kf_pitch = new KalmanFilter();
        KalmanFilter* kf_roll = new KalmanFilter();

        KalmanFilter* kf_gx = new KalmanFilter();
        KalmanFilter* kf_gy = new KalmanFilter();
        KalmanFilter* kf_gz = new KalmanFilter();

        KalmanFilter* kf_ax = new KalmanFilter();
        KalmanFilter* kf_ay = new KalmanFilter();
        KalmanFilter* kf_az = new KalmanFilter();

        void filter_yaw_pitch_roll(float yaw, float pitch, float roll, 
                                   float *yaw_out, float *pitch_out, float *roll_out);
        void filter_imu_readings();
        void get_attitude(float *yaw_arg, float *pitch_arg, float *roll_arg, float *heading_arg, float *imu_raw_vals, bool filtered);
        float get_heading();

    public:
    
        double currentTime;

        Pose(IMUDevice* dev);

        void get_pose(float *yaw_arg, float *pitch_arg, float *roll_arg, float *heading_arg, float *imu_raw_vals, bool filter_ypr=false);
    };

    class MotionTracking {
    private:

        // MPU6050* mpu;
        LSM9DS1* mpu;
        bool first_sampled = false;
        IMUDevice* dev;
        Pose* pose;
        
    public:

        MotionTracking();
        void init();
        void get_pose(float *yaw_arg, float *pitch_arg, float *roll_arg, float *heading_arg, float *imu_raw_vals, bool filter_ypr);
    };
}

#endif