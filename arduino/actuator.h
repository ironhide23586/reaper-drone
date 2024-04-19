#ifndef ACTUATOR_H
#define ACTUATOR_H

#include <Arduino.h>
#include <Servo.h>


namespace Actuator {

    
    void print_propeller_name(int pin_num);

    class Propeller {
    private:

        int min_pwm_val = 1100;
        int max_pwm_val = 2000;
        float pwm_range;
        
        int cap_pwm_val = 1300;
        void calibrate_esc();
    
    public:
        Propeller(int pin_num);
        Servo esc;
        int pwm_val = 0;
        void drive_throttle(float throttle_power);
    };

    class PropellerSet {
    // 9 -> Rear-Right
    // 10 -> Front-Right
    // 6 -> Rear-Left
    // 5 -> Front-Left
    private:

        uint8_t num_props = 4;
        int prop_pin_map[4] = {9, 10, 6, 5};
        Propeller *prop_set[4];
        Servo *escs[4];
        void init();
        void PropellerSet::calibrate_multiple_escs();

    public:

        PropellerSet();

        void drive_throttle_rear_right(float throttle_power);
        void drive_throttle_front_right(float throttle_power);
        void drive_throttle_rear_left(float throttle_power);
        void drive_throttle_front_left(float throttle_power);
    };
}


#endif