#ifndef LSM_H
#define LSM_H

// #include <Arduino_LSM9DS1.h>

#include "utils.h"


namespace InertialTracking {

    class LSM9DS1 {
    private:
        const uint8_t MPU_ADDRESS = 0x6B;

        const uint8_t MAG_ADDRESS = 0x1E;
        const uint8_t MAG_WHOMAI = 0x3D;
        const uint8_t MPU_WHOMAI = 0x68;

        const uint8_t MPU_WHO_AM_I_REG = 0x0F;

        const uint8_t CTRL_REG1_G = 0x10;
        const uint8_t CTRL_REG6_XL = 0x20;
        const uint8_t CTRL_REG7_XL = 0x21;

        const uint8_t OUT_TEMP = 0x15;

        const uint8_t STATUS_REG = 0x17;
        const uint8_t OUT_X_G = 0x18;
        const uint8_t CTRL_REG8 = 0x22;
        const uint8_t OUT_X_XL = 0x28;

        const uint8_t CTRL_REG1_M = 0x20;
        const uint8_t CTRL_REG2_M = 0x21;
        const uint8_t CTRL_REG3_M = 0x22;
        const uint8_t STATUS_REG_M = 0x27;
        const uint8_t OUT_X_L_M = 0x28;
 
    public:
        LSM9DS1();

        void init_mpu();
        uint8_t get_device_id();
        uint8_t get_device_address();
        bool magnetic_data_available();
        float mpu_read_data(float* mpu_gyro_xyz, float* mpu_acc_xyz, float* mag_acc_xyz, double* read_timestamp);
    };
    
} 

#endif
