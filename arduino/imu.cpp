#include "imu.h"


namespace IMU {

    MotionTracking::MotionTracking() {
        MotionTracking::mpu = new IMU::MPU6050();
        MotionTracking::dev = new IMU::IMUDevice(mpu);
        MotionTracking::pose = new IMU::Pose(dev);
    }

    void MotionTracking::init() {
        MotionTracking::dev->init();
    }

    void MotionTracking::get_pose(float *yaw_arg, float *pitch_arg, float *roll_arg) {
        if (!MotionTracking::first_sampled) {
            // MotionTracking::warmup();
            MotionTracking::pose->currentTime = micros();
            Serial.println("Measuring Gyro Drift.");
            for (int i = 0; i < NUM_CALIBRATION_SAMPLES; i++) {
                MotionTracking::pose->get_pose(yaw_arg, pitch_arg, roll_arg);
            }
            MotionTracking::first_sampled = true;
        }
        // stall();
        MotionTracking::pose->get_pose(yaw_arg, pitch_arg, roll_arg);
    }

    IMUDevice::IMUDevice(MPU6050* dev) {
        IMUDevice::dev_mpu6050 = dev;
        IMUDevice::dev_id = dev->get_device_id();
        IMUDevice::dev_addr = dev->get_device_address();
    }

    void IMUDevice::init() {
        if (IMUDevice::dev_id == 0x68) {
            IMUDevice::dev_mpu6050->init_mpu();
        }
        IMUDevice::compute_calibration_constants();
    }

    void IMUDevice::compute_calibration_constants() {
        Serial.print("Computing Calibration constants over ");
        IMUDevice::calibrated = false;
        Serial.print(NUM_CALIBRATION_SAMPLES);
        Serial.println(" samples...");
        Serial.println(IMUDevice::calibrated);

        for (int i = 0; i < NUM_CALIBRATION_SAMPLES; i++) {
            IMUDevice::read_data();
            for (int j = 0; j < 3; j++) {
                IMUDevice::gyro_xyz_offset[j] += IMUDevice::gyro_xyz[j];
            }
        }
        Serial.print("Gyroscope Offset XYZ:\t");
        for (int j = 0; j < 3; j++) {
            IMUDevice::gyro_xyz_offset[j] /= NUM_CALIBRATION_SAMPLES;
            Serial.print(IMUDevice::gyro_xyz_offset[j]);
            Serial.print("\t");
        }
        Serial.print("\n");
        IMUDevice::calibrated = true;
        Serial.println(IMUDevice::calibrated);
        // stall();
    }

    void IMUDevice::read_data() {
        if (IMUDevice::dev_id == 0x68) {
            IMUDevice::temp = IMUDevice::dev_mpu6050->mpu_read_data(&IMUDevice::gyro_xyz[0], &IMUDevice::acc_xyz[0], &read_timestamp);
        }
        if (IMUDevice::calibrated) {
            for (int i = 0; i < 3; i++) {
                IMUDevice::gyro_xyz[i] -= IMUDevice::gyro_xyz_offset[i];
            }
        }
    }

    Pose::Pose(IMUDevice* dev) {
        Pose::imu_device = dev;
        // Pose::currentTime = micros();
    }

    void Pose::get_pose(float *yaw_arg, float *pitch_arg, float *roll_arg, bool filter_ypr, bool filter_imu) {
        Pose::imu_device->read_data();

        Pose::previousTime = Pose::currentTime;        // Previous time is stored before the actual time read
        Pose::currentTime = Pose::imu_device->read_timestamp;           // Current time actual time read
        Pose::elapsedTime = (Pose::currentTime - Pose::previousTime) / 1000000; // Divide by 1000000 to get seconds

        get_yaw_pitch_roll(&yaw_raw, &pitch_raw, &roll_raw, filter_imu);
        if (filter_ypr)
            filter_yaw_pitch_roll(yaw_raw, pitch_raw, roll_raw, yaw_arg, pitch_arg, roll_arg);
        else {
            *yaw_arg = yaw_raw;
            *pitch_arg = pitch_raw;
            *roll_arg = roll_raw;
        }
    }

    void Pose::filter_yaw_pitch_roll(float yaw, float pitch, float roll, 
                                     float *yaw_out, float *pitch_out, float *roll_out) {
        // Serial.println("Filter");
        *yaw_out = Pose::kf_yaw->Predict(yaw);
        *pitch_out = Pose::kf_pitch->Predict(pitch);
        *roll_out = Pose::kf_roll->Predict(roll);
    }

    void Pose::filter_imu_readings() {
        // Serial.println("Filter");
        for (int i = 0; i < 3; i++) {
            Pose::gyro_xyz[i] = Pose::kf_gz->Predict(Pose::imu_device->gyro_xyz[i]);
            Pose::acc_xyz[i] = Pose::kf_ax->Predict(Pose::imu_device->acc_xyz[i]);
        }
    }

    void Pose::get_yaw_pitch_roll(float *yaw_arg, float *pitch_arg, float *roll_arg, bool filtered=false) {
        if (filtered) {
            Pose::filter_imu_readings();
        } else {
            for (int i = 0; i < 3; i++) {
                Pose::gyro_xyz[i] = Pose::imu_device->gyro_xyz[i];
                Pose::acc_xyz[i] = Pose::imu_device->acc_xyz[i];
            }
        }
        Pose::accAngleX = (atan(Pose::acc_xyz[1] / sqrt(pow(Pose::acc_xyz[0], 2) + pow(Pose::acc_xyz[2], 2))) * 180 / PI);
        Pose::accAngleY = (atan(-1 * Pose::acc_xyz[0] / sqrt(pow(Pose::acc_xyz[1], 2) + pow(Pose::acc_xyz[2], 2))) * 180 / PI);

        Pose::gyroAngleX = Pose::gyroAngleX + Pose::gyro_xyz[0] * Pose::elapsedTime; // deg/s * s = deg
        Pose::gyroAngleY = Pose::gyroAngleY + Pose::gyro_xyz[1] * Pose::elapsedTime;

        Pose::yaw_tmp = Pose::gyroAngleZ + Pose::gyro_xyz[2] * Pose::elapsedTime;

        if (Pose::calibration_counter < NUM_CALIBRATION_SAMPLES) {
            Pose::pitch_tmp = GYRO_WEIGHT * Pose::gyroAngleY + (1.0f - GYRO_WEIGHT) * Pose::accAngleY;
            Pose::roll_tmp = GYRO_WEIGHT * Pose::gyroAngleX + (1.0f - GYRO_WEIGHT) * Pose::accAngleX;

            Pose::gyro_drift[0] += (Pose::yaw_tmp - Pose::gyroAngleZ);
            Pose::gyro_drift[1] += (Pose::pitch_tmp - Pose::gyroAngleY);
            Pose::gyro_drift[2] += (Pose::roll_tmp - Pose::gyroAngleX);
            Pose::gyro_w[0] += Pose::accAngleX;
            Pose::gyro_w[1] += Pose::accAngleY;
            Pose::calibration_counter++;
            if (Pose::calibration_counter >= NUM_CALIBRATION_SAMPLES) {
                Pose::gyro_drift[0] /= NUM_CALIBRATION_SAMPLES;
                Pose::gyro_drift[1] /= NUM_CALIBRATION_SAMPLES;
                Pose::gyro_drift[2] /= NUM_CALIBRATION_SAMPLES;
                Serial.print("Gyro drift (YPR):\t");
                
                Serial.print(Pose::gyro_drift[0], 9);
                Serial.print("\t");
                Serial.print(Pose::gyro_drift[1], 9);
                Serial.print("\t");
                Serial.print(Pose::gyro_drift[2], 9);
                Serial.print("\n");

                Pose::gyro_w[0] /= NUM_CALIBRATION_SAMPLES;
                Pose::gyro_w[1] /= NUM_CALIBRATION_SAMPLES;

                Pose::gyro_w[0] = 1.0f - (Pose::gyro_drift[2] / Pose::gyro_w[0]);
                Pose::gyro_w[1] = 1.0f - (Pose::gyro_drift[1] / Pose::gyro_w[1]);
                Serial.print("Gyro weight (PR):\t");
                Serial.print(Pose::gyro_w[0]);
                Serial.print("\t");
                Serial.print(Pose::gyro_w[1]);
                Serial.print("\n");

                Pose::yaw_tmp = 0;
                Pose::pitch_tmp = 0;
                Pose::roll_tmp = 0;
            }
        } else {
            Pose::pitch_tmp = Pose::gyro_w[0] * Pose::gyroAngleY + (1.0f - Pose::gyro_w[0]) * Pose::accAngleY;
            Pose::roll_tmp = Pose::gyro_w[1] * Pose::gyroAngleX + (1.0f - Pose::gyro_w[1]) * Pose::accAngleX;

            Pose::yaw_tmp -= Pose::gyro_drift[0];
            Pose::pitch_tmp -= Pose::gyro_drift[1];
            Pose::roll_tmp -= Pose::gyro_drift[2];
        }
        
        Pose::gyroAngleZ = Pose::yaw_tmp;

        Pose::yaw_scale = sinf(Pose::yaw_tmp * 0.0174556f);
        Pose::roll_tmp -= Pose::pitch_tmp * Pose::yaw_scale;
        Pose::pitch_tmp += Pose::roll_tmp * Pose::yaw_scale;

        *yaw_arg = Pose::yaw_tmp;
        *pitch_arg = Pose::pitch_tmp;
        *roll_arg = Pose::roll_tmp;
    }
}