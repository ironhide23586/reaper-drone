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

    void MotionTracking::get_pose(float *yaw_arg, float *pitch_arg, float *roll_arg, float *heading_arg, float *imu_raw_vals, bool filter_ypr) {
        if (!MotionTracking::first_sampled) {
            // MotionTracking::warmup();
            MotionTracking::pose->currentTime = micros();
            Serial.println("Measuring Gyro Drift.");
            for (int i = 0; i < NUM_CALIBRATION_SAMPLES; i++) {
                MotionTracking::pose->get_pose(yaw_arg, pitch_arg, roll_arg, heading_arg, imu_raw_vals, filter_ypr);
            }
            MotionTracking::first_sampled = true;
        }
        // stall();
        MotionTracking::pose->get_pose(yaw_arg, pitch_arg, roll_arg, heading_arg, imu_raw_vals, filter_ypr);
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

    void Pose::get_pose(float *yaw_arg, float *pitch_arg, float *roll_arg, float *heading_arg, float *imu_raw_vals, bool filter_ypr) {
        Pose::imu_device->read_data();

        Pose::previousTime = Pose::currentTime;        // Previous time is stored before the actual time read
        Pose::currentTime = Pose::imu_device->read_timestamp;           // Current time actual time read
        Pose::elapsedTime = (Pose::currentTime - Pose::previousTime) / 1000000; // Divide by 1000000 to get seconds

        get_attitude(&yaw_raw, &pitch_raw, &roll_raw, heading_arg, imu_raw_vals, filter_ypr);
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

    float Pose::get_heading() {
        float heading_final_ = atan2f(Pose::mag_xyz[0] * (1 + sinf(abs(Pose::pitch_tmp) * PI / 180)), Pose::mag_xyz[1] * (1 + sinf(abs(Pose::roll_tmp) * PI / 180))) * 180 / PI;
        if (heading_final_ > 0) {
            heading_final_ = (heading_final_ / 180.) * 270;
        } else {
            heading_final_ = 360 + ((heading_final_ / 180) * 90);
        }

        heading_final_ = (360 - heading_final_);
        if (heading_final_ >= 0 && heading_final_ < 90) {
            heading_final_ = 270 + heading_final_;
        } else {
            heading_final_ -= 90;
        }
        heading_final_ += 20;
        if (heading_final_ > 360) heading_final_ -= 360; 
        return heading_final_;
    }

    void Pose::get_attitude(float *yaw_arg, float *pitch_arg, float *roll_arg, float *heading_arg,
                            float *imu_raw_vals, bool filtered=false) {
        for (int i = 0; i < 3; i++) {
            Pose::gyro_xyz[i] = Pose::imu_device->gyro_xyz[i];
            Pose::acc_xyz[i] = Pose::imu_device->acc_xyz[i];
            Pose::mag_xyz[i] = Pose::imu_device->mag_xyz[i]; //- (Pose::mag_drift[i] / Pose::calibration_counter);

            imu_raw_vals[i] = Pose::gyro_xyz[i];
            imu_raw_vals[3 + i] = Pose::acc_xyz[i];
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
        float yaw_final = Pose::yaw_tmp;
        if (Pose::calibration_counter <= NUM_CALIBRATION_SAMPLES) {
            Pose::gyro_drift[0] += Pose::yaw_tmp;
            Pose::gyro_drift[1] += Pose::gyroAngleY;
            Pose::gyro_drift[2] += Pose::gyroAngleX;

            Pose::acc_drift[1] += Pose::accAngleY;
            Pose::acc_drift[2] += Pose::accAngleX;
            
            if (Pose::calibration_counter >= NUM_CALIBRATION_SAMPLES) {
                Serial.print("Gyro drift (YPR):\t");

                Serial.print(Pose::gyro_drift[0] / Pose::calibration_counter, 9);
                Serial.print("\t");
                Serial.print(Pose::gyro_drift[1] / Pose::calibration_counter, 9);
                Serial.print("\t");
                Serial.print(Pose::gyro_drift[2] / Pose::calibration_counter, 9);
                Serial.print("\n");

                Pose::acc_drift[1] /= Pose::calibration_counter;
                Pose::acc_drift[2] /= Pose::calibration_counter;

                Pose::gyroAngleX = Pose::acc_drift[2];
                Pose::gyroAngleY = Pose::acc_drift[1];

                Pose::pitch_tmp = Pose::gyroAngleY;
                Pose::roll_tmp = Pose::gyroAngleX;
                yaw_offset = get_heading();
                Serial.print("Yaw Offset in Degrees: ");
                Serial.println(yaw_offset);
                // stall();

                Serial.print("Accelerometer drift (PR):\t");

                Serial.print(Pose::acc_drift[1], 9);
                Serial.print("\t");
                Serial.print(Pose::acc_drift[2], 9);
                Serial.print("\n");
                digitalWrite(INIT_COMPLETE_LED, HIGH);
                digitalWrite(INIT_ONGOING_LED, LOW);
            }
            // delay(1);
            Pose::calibration_counter++;
        } else {
            Pose::pitch_tmp = GYRO_WEIGHT * (Pose::pitch_tmp + Pose::gyroAngleY) + (1. - GYRO_WEIGHT) * Pose::accAngleY;
            Pose::roll_tmp = GYRO_WEIGHT * (Pose::roll_tmp + Pose::gyroAngleX) + (1. - GYRO_WEIGHT) * Pose::accAngleX;

            Pose::heading_tmp = get_heading();

            float d = yaw_offset - Pose::heading_tmp;
            if ((yaw_offset - Pose::heading_tmp) > 180)
                d = -(Pose::heading_tmp + (360 - yaw_offset));

            Pose::yaw_tmp = GYRO_WEIGHT * (Pose::yaw_tmp) + (1. - GYRO_WEIGHT) * d;

            // if (Pose::yaw_tmp < 0) yaw_final = Pose::yaw_tmp + 360;
            // if (Pose::yaw_tmp > 360) yaw_final = Pose::yaw_tmp - 360;

            for (int i = 0; i < 3; i++) {
                imu_raw_vals[i + 6] = Pose::mag_xyz[i];
            }
        }
        *yaw_arg = yaw_final;
        *pitch_arg = Pose::pitch_tmp * .68 + 8;
        *roll_arg = Pose::roll_tmp + 4;   

        *heading_arg = Pose::heading_tmp;

    }
}