#ifndef MPU_H
#define MPU_H

#include "utils.h"


namespace IMU {

    class MPU6050 {
    private:

        uint8_t MPU_ADDRESS = 0x68;
        uint8_t MPU_DATA_BYTE_BUFFER[14];

        const uint8_t MAG_ADDRESS = 0x0C;
        const uint8_t MAG_WHOAMI = 0x48;
        const uint8_t MPU_WHOMAI = 0x68;

        const uint8_t PWR_MGMT_1_REG = 0x6B;

        const uint8_t INT_PIN_CFG_REG = 0x37;
        const uint8_t INT_ENABLE_REG = 0x38;
        const uint8_t USER_CTRL_REG = 0x6A;
        const uint8_t MPU_WHO_AM_I_REG = 0x75;
        const uint8_t SIGNAL_PATH_RESET_REG = 0x68;  // not a typo

        const uint8_t SELF_TEST_X_REG = 0x0D;
        const uint8_t SELF_TEST_Y_REG = 0x0E;
        const uint8_t SELF_TEST_Z_REG = 0x0F;
        const uint8_t SELF_TEST_A_REG = 0x10;

        const uint8_t GYRO_CONFIG_REG = 0x1B;
        const uint8_t ACCEL_CONFIG_REG = 0x1C;
        const uint8_t FIFO_EN_REG = 0x23;

        const uint8_t ACCEL_XOUT_H_REG = 0x3B;
        const uint8_t ACCEL_XOUT_L_REG = 0x3C;
        const uint8_t ACCEL_YOUT_H_REG = 0x3D;
        const uint8_t ACCEL_YOUT_L_REG = 0x3E;
        const uint8_t ACCEL_ZOUT_H_REG = 0x3F;
        const uint8_t ACCEL_ZOUT_L_REG = 0x40;

        const uint8_t TEMP_OUT_H_REG = 0x41;
        const uint8_t TEMP_OUT_L_REG = 0x42;

        const uint8_t GYRO_XOUT_H_REG = 0x43;
        const uint8_t GYRO_XOUT_L_REG = 0x44;
        const uint8_t GYRO_YOUT_H_REG = 0x45;
        const uint8_t GYRO_YOUT_L_REG = 0x46;
        const uint8_t GYRO_ZOUT_H_REG = 0x47;
        const uint8_t GYRO_ZOUT_L_REG = 0x48;

        float mpu_data;
        float mpu_data_new, mpu_temp;

        float mpu_read_data_worker(float* mpu_gyro_xyz, float* mpu_acc_xyz, double* read_timestamp);
        void mpu_enable_self_test_mode();
        void mpu_disable_self_test_mode();
        bool mpu_self_test();

    public:
        MPU6050();
        MPU6050(uint8_t device_address);

        void init_mpu();
        uint8_t get_device_id();
        uint8_t get_device_address();
        float mpu_read_data(float* mpu_gyro_xyz, float* mpu_acc_xyz, double* read_timestamp);
    };
}

#endif