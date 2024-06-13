#ifndef ACTUATOR_H
#define ACTUATOR_H

#include "utils.h"

#include <Arduino.h>
#include <Servo.h>



namespace Actuator {

    
    void print_propeller_name(int pin_num);

    class Propeller {
    private:
        // float pwm_range = HIGH_THROTTLE_PWM_VAL - LOW_THROTTLE_PWM_VAL;
        
        // int cap_pwm_val = SAFETY_CAP_PWM_VAL;
        void calibrate_esc();
    
    public:
        Propeller(int pin_num);
        Servo esc;
        int pwm_val = 0;
        void drive_throttle(float throttle_power);
    };

    class PropellerSet {
    // 6 -> Rear-Right
    // 5 -> Front-Right
    // 10 -> Rear-Left
    // 9 -> Front-Left
    private:

        float torque_x, torque_y;
        float f_xyz_norm;
        float t_rear_right, t_front_right;
        float t_rear_left, t_front_left;
        // float throttle_rear_right, throttle_front_right;
        // float throttle_rear_left, throttle_front_left;

        float cos_alpha, cos_beta, cos_gamma;
        float sin_alpha, sin_beta, sin_gamma;
        float yaw_rad, pitch_rad, roll_rad;
        float nx_proj, ny_proj, nz_proj;
        uint8_t num_props = 4;

        float radius_front = .12f, radius_rear = .14f;
        float angle_front_deg = 120, angle_rear_deg = 90;
        // float thrust_torque_coeffs[4] = {.3, .3, .3, .3};
        // float t_mat_inv[4][4] = {{-2.08333333,  1.92307692, -6.94444444,  0.26923077},
        //                          {-1.78571429, -1.92307692,  5.95238095,  0.23076923},
        //                          {1.78571429, -1.92307692, -5.95238095,  0.23076923},
        //                          {2.08333333,  1.92307692,  6.94444444,  0.26923077}};
        float vehicle_width = 0.20291799782024927f;
        float vehicle_length = 0.1589949493661167f;

        Servo *escs[4];
        void init();
        void calibrate_multiple_escs();

    public:

        PropellerSet();
        float mass = .8f;
        Propeller *prop_set[4];

        volatile bool drive_enabled = true;

        void drive_throttle_rear_right(float throttle_power);
        void drive_throttle_front_right(float throttle_power);
        void drive_throttle_rear_left(float throttle_power);
        void drive_throttle_front_left(float throttle_power);

        void brake();
        bool safety_check();

        // void apply_throttles(float throttle_rear_right, float throttle_front_right, float throttle_rear_left, float throttle_front_left);

        void actuate_force_torques(float fx, float fy, float fz, float yaw_torque);

        void actuate(float nx, float ny, float nz, float yaw, float pitch, float roll);
    };
}


#endif