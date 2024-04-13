#include "inertial_tracking.h"


namespace InertialTracking {

    MotionTracking::MotionTracking() {
        // MotionTracking::mpu = new InertialTracking::MPU6050();
        MotionTracking::mpu = new InertialTracking::LSM9DS1();
        MotionTracking::dev = new InertialTracking::IMUDevice(mpu);
        MotionTracking::pose = new InertialTracking::Pose(dev);
    }

    void MotionTracking::init() {
        MotionTracking::dev->init();
    }

    void MotionTracking::get_pose(float *yaw_arg, float *pitch_arg, float *roll_arg, bool filter_ypr) {
        if (!MotionTracking::first_sampled) {
            // MotionTracking::warmup();
            MotionTracking::pose->currentTime = micros();
            Serial.println("Measuring Gyro Drift.");
            for (int i = 0; i < NUM_CALIBRATION_SAMPLES; i++) {
                MotionTracking::pose->get_pose(yaw_arg, pitch_arg, roll_arg, filter_ypr);
            }
            MotionTracking::first_sampled = true;
        }
        // stall();
        MotionTracking::pose->get_pose(yaw_arg, pitch_arg, roll_arg, filter_ypr);
    }

    // IMUDevice::IMUDevice(MPU6050* dev) {
    //     IMUDevice::dev = dev;
    //     IMUDevice::dev_id = dev->get_device_id();
    //     IMUDevice::dev_addr = dev->get_device_address();
    // }

    IMUDevice::IMUDevice(LSM9DS1* dev) {
        IMUDevice::dev = dev;
        IMUDevice::dev_id = dev->get_device_id();
        IMUDevice::dev_addr = dev->get_device_address();
    }

    void IMUDevice::init() {
        IMUDevice::dev->init_mpu();
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
    }

    void IMUDevice::read_data() {
        IMUDevice::temp = IMUDevice::dev->mpu_read_data(&IMUDevice::gyro_xyz[0], &IMUDevice::acc_xyz[0], &IMUDevice::mag_xyz[0], &read_timestamp);
        if (IMUDevice::calibrated) {
            for (int i = 0; i < 3; i++) {
                IMUDevice::gyro_xyz[i] -= IMUDevice::gyro_xyz_offset[i];
            }
        }
    }

    Pose::Pose(IMUDevice* dev) {
        Pose::imu_device = dev;
    }

    void Pose::get_pose(float *yaw_arg, float *pitch_arg, float *roll_arg, bool filter_ypr) {
        Pose::imu_device->read_data();

        Pose::previousTime = Pose::currentTime;        // Previous time is stored before the actual time read
        Pose::currentTime = Pose::imu_device->read_timestamp;           // Current time actual time read
        Pose::elapsedTime = (Pose::currentTime - Pose::previousTime) / 1000000; // Divide by 1000000 to get seconds

        get_yaw_pitch_roll(&yaw_raw, &pitch_raw, &roll_raw, filter_ypr);
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
        float dx = pow(Pose::acc_xyz[0], 2) + pow(Pose::acc_xyz[2], 2);
        if (dx > 0)
            Pose::accAngleX = -(atan((.1 * Pose::acc_xyz[1]) / sqrt(dx)) * 180 / PI);
        else Pose::accAngleX = 90;

        dx = pow(Pose::acc_xyz[1], 2) + pow(Pose::acc_xyz[2], 2);
        if (dx > 0)
            Pose::accAngleY = (atan(-1 * (.2 * Pose::acc_xyz[0]) / sqrt(dx)) * 180 / PI);
        else Pose::accAngleY = 90;

        Pose::gyroAngleX = ((Pose::gyro_xyz[0] * Pose::elapsedTime) - (Pose::gyro_drift[2] / Pose::calibration_counter));
        Pose::gyroAngleY = ((Pose::gyro_xyz[1] * Pose::elapsedTime) - (Pose::gyro_drift[1] / Pose::calibration_counter));

        Pose::yaw_tmp += (Pose::gyro_xyz[2] * Pose::elapsedTime) - (Pose::gyro_drift[0] / Pose::calibration_counter);

        if (Pose::calibration_counter <= NUM_CALIBRATION_SAMPLES) {
            Pose::gyro_drift[0] += Pose::yaw_tmp;
            Pose::gyro_drift[1] += Pose::gyroAngleY;
            Pose::gyro_drift[2] += Pose::gyroAngleX;

            Pose::acc_drift[1] += Pose::accAngleY;
            Pose::acc_drift[2] += Pose::accAngleX;
            
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

                Pose::acc_drift[1] /= NUM_CALIBRATION_SAMPLES;
                Pose::acc_drift[2] /= NUM_CALIBRATION_SAMPLES;

                Pose::gyroAngleX = Pose::acc_drift[2];
                Pose::gyroAngleY = Pose::acc_drift[1];

                Pose::pitch_tmp = Pose::gyroAngleY;
                Pose::roll_tmp = Pose::gyroAngleX;

                Serial.print("Accelerometer drift (PR):\t");

                Serial.print(Pose::acc_drift[1], 9);
                Serial.print("\t");
                Serial.print(Pose::acc_drift[2], 9);
                Serial.print("\n");
            }
            // delay(1);
            Pose::calibration_counter++;
        } else {
            // Pose::pitch_tmp = GYRO_WEIGHT * Pose::gyroAngleY + (1.0f - GYRO_WEIGHT) * Pose::accAngleY;
            // Pose::roll_tmp = GYRO_WEIGHT * Pose::gyroAngleX + (1.0f - GYRO_WEIGHT) * Pose::accAngleX;

            Pose::pitch_tmp = GYRO_WEIGHT * (Pose::pitch_tmp + Pose::gyroAngleY) + (1. - GYRO_WEIGHT) * Pose::accAngleY;
            Pose::roll_tmp = GYRO_WEIGHT * (Pose::roll_tmp + Pose::gyroAngleX) + (1. - GYRO_WEIGHT) * Pose::accAngleX;

        }

        // Pose::yaw_scale = sinf(Pose::yaw_tmp * 0.0174556f);
        // Pose::roll_tmp -= Pose::pitch_tmp * Pose::yaw_scale;
        // Pose::pitch_tmp += Pose::roll_tmp * Pose::yaw_scale;
        
        // if (Pose::calibration_counter > NUM_CALIBRATION_SAMPLES) {
        //     Serial.print(yaw_tmp);
        //     Serial.print("\t");

        //     Serial.print(Pose::gyroAngleY - Pose::gyro_drift[1]);
        //     Serial.print(" - ");
        //     Serial.print(Pose::accAngleY);
        //     Serial.print(" => ");
        //     Serial.print(Pose::gyroAngleY - Pose::gyro_drift[1] - (Pose::accAngleY));
        //     Serial.print("\t");

        //     Serial.print(Pose::gyroAngleX - Pose::gyro_drift[2]);
        //     Serial.print(" - ");
        //     Serial.print(Pose::accAngleX);
        //     Serial.print(" => ");
        //     Serial.print(Pose::gyroAngleX - Pose::gyro_drift[2] - (Pose::accAngleX));
        //     Serial.print("\n");
        // }

        *yaw_arg = Pose::yaw_tmp;
        *pitch_arg = Pose::pitch_tmp;
        *roll_arg = Pose::roll_tmp;
    }
}