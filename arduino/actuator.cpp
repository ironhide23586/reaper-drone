#include "actuator.h"

namespace Actuator {

    void print_propeller_name(int pin_num) {
        switch (pin_num) {
            case 9:
                Serial.print("Rear-Right");
                break;
            case 10:
                Serial.print("Front-Right");
                break;
            case 6:
                Serial.print("Rear-Left");
                break;
            case 5:
                Serial.print("Front-Left");
                break;        
            default:
                break;
        }
    }

    Propeller::Propeller(int pin_num) {
        esc.attach(pin_num);
        delay(20);
        Serial.print("> Powered up ");
        print_propeller_name(pin_num);
        Serial.println(" propeller");
        pwm_range = max_pwm_val - min_pwm_val;
        // calibrate_esc();
    }

    void Propeller::calibrate_esc() {
        Serial.println("Calibrating...");
        Serial.println("Sending full-throttle signal.");
        // Send a high throttle signal to start the calibration process
        Propeller::esc.writeMicroseconds(2000);
        delay(5000); // Wait for 5 seconds for calibration to complete
        Serial.println("Detected High point.");
        
        Serial.println("Sending no-throttle signal.");
        // Send a low throttle signal to finish calibration
        Propeller::esc.writeMicroseconds(1100);
        delay(5000); // Wait for 2 seconds
        Serial.println("Detected Low point.");

        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
        Serial.println("-----------------Calibrated-----------------\n");
    }

    void Propeller::drive_throttle(float throttle_power) {
        float pwm_val = (throttle_power * pwm_range) + min_pwm_val;
        if (pwm_val > cap_pwm_val) pwm_val = cap_pwm_val;
        // throttle_power = (pwm_val - min_pwm_val) / pwm_range;
        // Serial.print("Driving motor at ");
        // Serial.print(throttle_power * 100.);
        // Serial.print("% power with pwm val ");
        // Serial.println(pwm_val);
        Propeller::esc.writeMicroseconds(pwm_val);
    }

    PropellerSet::PropellerSet() {
        init();
    }

    void PropellerSet::init() {
        for (uint8_t i = 0; i < num_props; i++) {
            prop_set[i] = new Propeller(prop_pin_map[i]);
            escs[i] = &prop_set[i]->esc;
        }
        calibrate_multiple_escs();
    }

    void PropellerSet::calibrate_multiple_escs() {
        int i;
        Serial.println("\n-----------------Calibrating-----------------");
        Serial.println("Sending full-throttle signal.");
        
        for (i = 0; i < 4; i++) {
            // Send a high throttle signal to start the calibration process
            escs[i]->writeMicroseconds(2000);
            Serial.print("Triggering Propeller ");
            print_propeller_name(prop_pin_map[i]);
            Serial.print("\n");
        }
        delay(5000); // Wait for 5 seconds for calibration to complete
        Serial.println("Detected High point.");
            
        Serial.println("Sending no-throttle signal.");
        for (i = 0; i < 4; i++) {
            // Send a low throttle signal to finish calibration
            escs[i]->writeMicroseconds(1100);
            Serial.print("Triggering Propeller ");
            print_propeller_name(prop_pin_map[i]);
            Serial.print("\n");
        }
        delay(5000); // Wait for 5 seconds
        Serial.println("Detected Low point.");

        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
        Serial.println("-----------------Calibrated-----------------\n");
    }

    void PropellerSet::drive_throttle_rear_right(float throttle_power) {
        // print_propeller_name(prop_pin_map[0]);
        // Serial.print(": ");
        prop_set[0]->drive_throttle(throttle_power);
    }

    void PropellerSet::drive_throttle_front_right(float throttle_power) {
        // print_propeller_name(prop_pin_map[1]);
        // Serial.print(": ");
        prop_set[1]->drive_throttle(throttle_power);
    }

    void PropellerSet::drive_throttle_rear_left(float throttle_power) {
        // print_propeller_name(prop_pin_map[2]);
        // Serial.print(": ");
        prop_set[2]->drive_throttle(throttle_power);
    }

    void PropellerSet::drive_throttle_front_left(float throttle_power) {
        // print_propeller_name(prop_pin_map[3]);
        // Serial.print(": ");
        prop_set[3]->drive_throttle(throttle_power);
    }
}