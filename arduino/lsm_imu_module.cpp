#include "lsm_imu_module.h"


namespace InertialTracking {

    LSM9DS1::LSM9DS1() {}

    void LSM9DS1::init_mpu() {
        Serial.println('Initializing IMU Device.');
        I2CwriteByte(MPU_ADDRESS, CTRL_REG8, 0x05);
        I2CwriteByte(MAG_ADDRESS, CTRL_REG2_M, 0x0C);

        delay(10);

        uint8_t mpu_whomai;
        I2Cread(MPU_ADDRESS, MPU_WHO_AM_I_REG, 1, &mpu_whomai);
        if (mpu_whomai != MPU_WHOMAI) {
            Serial.println("MPU not found");
            Serial.println(mpu_whomai, HEX);
            stall();
        }

        I2Cread(MAG_ADDRESS, MPU_WHO_AM_I_REG, 1, &mpu_whomai);
        if (mpu_whomai != MAG_WHOMAI) {
            Serial.println("Magnetometer not found");
            Serial.println(mpu_whomai, HEX);
            stall();
        }

        I2CwriteByte(MPU_ADDRESS, CTRL_REG1_G, 0x78); // 119 Hz, 2000 dps, 16 Hz BW TODO: Try reducing frequency
        I2CwriteByte(MPU_ADDRESS, CTRL_REG6_XL, 0x78); // 119 Hz, 8g, 408 Hz antialiasing filter bandwidth
        // I2CwriteByte(MPU_ADDRESS, CTRL_REG7_XL, 0x44); // 0 1 0 0 0 1 0 0

        I2CwriteByte(MAG_ADDRESS, CTRL_REG1_M, 0xF4); // 1 1 1 1 0 1 0 0
        I2CwriteByte(MAG_ADDRESS, CTRL_REG2_M, 0x00); // 12 gauss 0 0 0 0 0 0 0 0 
        I2CwriteByte(MAG_ADDRESS, CTRL_REG3_M, 0x00); // Continuous conversion mode

    }

    bool LSM9DS1::magnetic_data_available() {
        uint8_t status_data;
        I2Cread(MAG_ADDRESS, STATUS_REG_M, 1, &status_data);
        if (status_data & 0x08 | status_data & 0x0F) {
            return true;
        }
        return false;
    }


    float LSM9DS1::mpu_read_data(float* mpu_gyro_xyz, float* mpu_acc_xyz, float* mpu_mag_xyz, double* read_timestamp) {
        uint8_t gyro_data[6], acc_data[6], mag_data[6];
        I2Cread(MPU_ADDRESS, OUT_X_G, 6, gyro_data);
        I2Cread(MPU_ADDRESS, OUT_X_XL, 6, acc_data);
        // while (!magnetic_data_available()); // causing slowdown
        I2Cread(MAG_ADDRESS, OUT_X_L_M, 6, mag_data);
        *read_timestamp = micros();
        for (int i = 0; i < 3; i++){
            mpu_gyro_xyz[i] = (gyro_data[(i * 2) + 1] << 8 | gyro_data[i * 2]) * 0.06103515625f;
            mpu_acc_xyz[i] = (acc_data[(i * 2) + 1] << 8 | acc_data[i * 2]) * 0.000244140625f;
            mpu_mag_xyz[i] = (mag_data[(i * 2) + 1] << 8 | mag_data[i * 2]) * 0.01220703125f;
        }
        return 0;
    }

    uint8_t LSM9DS1::get_device_id() {
        return LSM9DS1::MPU_WHOMAI;
    }

    uint8_t LSM9DS1::get_device_address() {
        return LSM9DS1::MPU_ADDRESS;
    }

}