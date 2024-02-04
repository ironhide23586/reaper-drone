#ifndef IMU_H
#define IMU_H

#include "mpu6050.h"
#include "kalman.h"


namespace IMU {

    class IMUDevice {
    private:

        bool kf_init = false;
        uint8_t dev_id;
        uint8_t dev_addr;
        MPU6050* dev_mpu6050;

    public:

        bool calibrated = false;

        float gyro_xyz[3];
        float acc_xyz[3];
        double read_timestamp;

        float gyro_xyz_offset[3] = {0};

        float temp;
        
        IMUDevice(MPU6050* dev);

        void init();
        void read_data();
        void compute_calibration_constants();
    };


    class Pose {
    private:

        int calibration_counter = 0;
        float yaw_tmp, pitch_tmp, roll_tmp;
        float yaw_scale;

        float gyro_xyz[3], acc_xyz[3];
        float gyro_drift[3] = {0};
        float gyro_w[2] = {0};

        float accAngleX, accAngleY;
        float gyroAngleX = 0, gyroAngleY = 0, gyroAngleZ = 0;
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
        void get_yaw_pitch_roll(float *yaw_arg, float *pitch_arg, float *roll_arg, bool filtered);


    public:
    
        double currentTime;

        Pose(IMUDevice* dev);

        void get_pose(float *yaw_arg, float *pitch_arg, float *roll_arg, bool filter_ypr=false, bool filter_imu=false);
    };

    class MotionTracking {
    private:

        MPU6050* mpu;
        bool first_sampled = false;
        IMUDevice* dev;
        Pose* pose;
        
    public:

        MotionTracking();
        void init();
        void get_pose(float *yaw_arg, float *pitch_arg, float *roll_arg);
    };
}

#endif