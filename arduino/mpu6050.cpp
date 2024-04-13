#include "mpu6050.h"


namespace InertialTracking {

    MPU6050::MPU6050() {}

    void MPU6050::init_mpu() {
        Serial.println("Reset all registers on MPU...");
        I2CwriteByte(MPU_ADDRESS, PWR_MGMT_1_REG, 0x00);

        uint8_t mpu_whomai;
        I2Cread(MPU_ADDRESS, MPU_WHO_AM_I_REG, 1, &mpu_whomai);
        if (mpu_whomai != MPU_WHOMAI) {
            Serial.println("MPU not found");
            Serial.println(mpu_whomai, HEX);
            stall();
        }

        Serial.println("Setting internal clock source of MPU to gyroscope.");
        I2CwriteByte(MPU_ADDRESS, PWR_MGMT_1_REG, 0x01);

        Serial.println("Resetting signal paths (WRITE-ONLY REGISTER)");
        I2CwriteByte(MPU_ADDRESS, SIGNAL_PATH_RESET_REG, 0x06);

        MPU6050::mpu_enable_self_test_mode();
        if (mpu_self_test())  {
            Serial.println("MPU self test passed!");
            MPU6050::mpu_disable_self_test_mode();
        } else {
            Serial.println("MPU self test failed!");
            stall();
        }
    }

    void MPU6050::mpu_enable_self_test_mode() {
        Serial.println("Enabling gyroscope self-test mode 250 dps.");
        I2CwriteByte(MPU_ADDRESS, GYRO_CONFIG_REG, 0xE0);
        Serial.println("Enabling accelerometer self-test mode +-8g.");
        I2CwriteByte(MPU_ADDRESS, ACCEL_CONFIG_REG, 0xF0);
    }

    void MPU6050::mpu_disable_self_test_mode() {
        Serial.println("Disabling gyroscope self-test mode 250 dps.");
        I2CwriteByte(MPU_ADDRESS, GYRO_CONFIG_REG, 0x00);
        Serial.println("Disabling accelerometer self-test mode +-8g.");
        I2CwriteByte(MPU_ADDRESS, ACCEL_CONFIG_REG, 0x10);
    }

    bool MPU6050::mpu_self_test() {
        Serial.println("Setting gyroscope self-test mode 250 dps.");
        I2CwriteByte(MPU_ADDRESS, GYRO_CONFIG_REG, 0xE0);
        Serial.println("Setting accelerometer self-test mode +-8g.");
        I2CwriteByte(MPU_ADDRESS, ACCEL_CONFIG_REG, 0xF0);

        uint8_t self_test_data[4];
        I2Cread(MPU_ADDRESS, SELF_TEST_X_REG, 4, self_test_data);

        uint8_t gyro_test_vals[3] = {0};
        uint8_t acc_test_vals[3] = {0};

        for (int i = 0; i < 3; i++) {
            uint8_t low_gyro_bits = self_test_data[i] & 0x1F;
            uint8_t high_acc_bits = (self_test_data[i] & 0xE0) >> 5;
            
            gyro_test_vals[i] = low_gyro_bits;
            acc_test_vals[i] |= high_acc_bits << 2;
        }

        for (int i = 0; i < 3; i++) {
            acc_test_vals[i] |= (self_test_data[3] & (0x03 << (2 * (2 - i)))) >> (2 * (2 - i));
        }

        float ft_gyro[3] = {0.0f};
        float ft_acc[3] = {0.0f};

        Serial.println("Gyroscope Test Vals / Factory Trim in decimal format (X, Y, Z) -");
        for (int i = 0; i < 3; i++) {
            if (gyro_test_vals[i] != 0)
                ft_gyro[i] = 3275.0f * powf(1.046f, ((float)gyro_test_vals[i] - 1.0f));
            if (i == 1)
                ft_gyro[i] *= -1;

            float err = 100.0f + 100.0f * (((float)gyro_test_vals[i] - ft_gyro[i]) / ft_gyro[i]);

            Serial.print((float)gyro_test_vals[i], DEC);
            Serial.print(" | ");
            Serial.print(ft_gyro[i]);
            Serial.print(" |-> ");
            Serial.print(err, 1);
            Serial.print("\t");

            if (err  >= 1) return false;
        }
        Serial.print("\n");

        Serial.println("Accelerometer Test Vals / Factory Trim / Self-Test-Score (should be below 1) in decimal format (X, Y, Z) -");
        for (int i = 0; i < 3; i++) {
            if (acc_test_vals[i] != 0)
                ft_acc[i] = 1392.64f * powf(2.705882352941176f, (((float)acc_test_vals[i] - 1.0f) / 30.0f));

            float err = 100.0f + 100.0f * (((float)acc_test_vals[i] - ft_acc[i]) / ft_acc[i]);

            Serial.print((float)acc_test_vals[i], DEC);
            Serial.print(" | ");
            Serial.print(ft_acc[i]);
            Serial.print(" |-> ");
            Serial.print(err, 1);
            Serial.print("\t");

            if (err  >= 1) return false;
        }
        Serial.print("\n");
        return true;
    }

    float MPU6050::mpu_read_data(float* mpu_gyro_xyz, float* mpu_acc_xyz, double* read_timestamp) {
        return mpu_read_data_worker(mpu_gyro_xyz, mpu_acc_xyz, read_timestamp);
    }

    float MPU6050::mpu_read_data_worker(float* gyro_xyz, float* acc_xyz, double* read_timestamp) {
        I2Cread(MPU_ADDRESS, ACCEL_XOUT_H_REG, 14, MPU_DATA_BYTE_BUFFER);
        *read_timestamp = micros();
        float mpu_temp = -1;
        for (int i = 0; i < 7; i++) {
            MPU6050::mpu_data = MPU_DATA_BYTE_BUFFER[i * 2] << 8 | MPU_DATA_BYTE_BUFFER[(i * 2) + 1];
            if (i < 3) {  // accelerometer sensor
                MPU6050::mpu_data /= 4096.0f;
                acc_xyz[i] = MPU6050::mpu_data;
            } else if (i == 3) {  // temperature sensor
                MPU6050::mpu_data /= 340.0f;
                MPU6050::mpu_data += 35.53f;
                mpu_temp = MPU6050::mpu_data;
            } else { // gyroscope sensor
                MPU6050::mpu_data /= 131.0f;
                gyro_xyz[i - 4] = mpu_data;
            }
        }
        return mpu_temp;
    }

    uint8_t MPU6050::get_device_id() {
        return MPU6050::MPU_WHOMAI;
    }

    uint8_t MPU6050::get_device_address() {
        return MPU6050::MPU_ADDRESS;
    }
}