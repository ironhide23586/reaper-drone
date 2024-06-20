#include "actuator.h"

namespace Actuator {

    void print_propeller_name(int pin_num) {
        Serial.print(pin_num);
        if (pin_num == 6) Serial.print("-Rear-Right");
        if (pin_num == 5) Serial.print("-Front-Right");
        if (pin_num == 10) Serial.print("-Rear-Left");
        if (pin_num == 9) Serial.print("-Front-Left");
    }

    Propeller::Propeller(int pin_num) {
        esc.attach(pin_num);
        delay(50);
        Serial.print("> Powered up ");
        print_propeller_name(pin_num);
        Serial.println(" propeller");
        // pwm_range = max_pwm_val - min_pwm_val;
        // calibrate_esc();
    }

    void Propeller::calibrate_esc() {
        Serial.println("Calibrating...");
        Serial.println("Sending full-throttle signal.");
        // Send a high throttle signal to start the calibration process
        Propeller::esc.writeMicroseconds(HIGH_THROTTLE_PWM_VAL);
        delay(5000); // Wait for 5 seconds for calibration to complete
        Serial.println("Detected High point.");
        
        Serial.println("Sending no-throttle signal.");
        // Send a low throttle signal to finish calibration
        Propeller::esc.writeMicroseconds(LOW_THROTTLE_PWM_VAL);
        delay(5000); // Wait for 2 seconds
        Serial.println("Detected Low point.");

        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
        Serial.println("-----------------Calibrated-----------------\n");
    }

    void Propeller::drive_throttle(float throttle_power) {
        int pwm_val = (throttle_power * (HIGH_THROTTLE_PWM_VAL - LOW_THROTTLE_PWM_VAL)) + LOW_THROTTLE_PWM_VAL;
        if (pwm_val > SAFETY_CAP_PWM_VAL) pwm_val = SAFETY_CAP_PWM_VAL;
        throttle_power = (float) (pwm_val - LOW_THROTTLE_PWM_VAL) / (HIGH_THROTTLE_PWM_VAL - LOW_THROTTLE_PWM_VAL);
        Serial.print("Driving motor at ");
        Serial.print(throttle_power * 100., 6);
        Serial.print("% power with pwm val ");
        Serial.println(pwm_val);
        Propeller::esc.writeMicroseconds(pwm_val);
    }

    PropellerSet::PropellerSet() {
        init();
    }

    void PropellerSet::init() {
        prop_set[0] = new Propeller(6);
        escs[0] = &prop_set[0]->esc;
        prop_set[1] = new Propeller(5);
        escs[1] = &prop_set[0]->esc;
        prop_set[2] = new Propeller(10);
        escs[2] = &prop_set[0]->esc;
        prop_set[3] = new Propeller(9);
        escs[3] = &prop_set[0]->esc;
        calibrate_multiple_escs();
    }

    void PropellerSet::calibrate_multiple_escs() {
        if (!safety_check()) return;
        Serial.println("\n-----------------Calibrating-----------------");
        Serial.println("Sending full-throttle signal.");

        prop_set[0]->esc.writeMicroseconds(HIGH_THROTTLE_PWM_VAL);
        prop_set[1]->esc.writeMicroseconds(HIGH_THROTTLE_PWM_VAL);
        prop_set[2]->esc.writeMicroseconds(HIGH_THROTTLE_PWM_VAL);
        prop_set[3]->esc.writeMicroseconds(HIGH_THROTTLE_PWM_VAL);

        delay(5000); // Wait for 5 seconds for calibration to complete
        Serial.println("Detected High point.");
            
        Serial.println("Sending no-throttle signal.");

        prop_set[0]->esc.writeMicroseconds(LOW_THROTTLE_PWM_VAL);
        prop_set[1]->esc.writeMicroseconds(LOW_THROTTLE_PWM_VAL);
        prop_set[2]->esc.writeMicroseconds(LOW_THROTTLE_PWM_VAL);
        prop_set[3]->esc.writeMicroseconds(LOW_THROTTLE_PWM_VAL);

        delay(5000); // Wait for 5 seconds
        Serial.println("Detected Low point.");

        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
        delay(2000);
        Serial.println("-----------------Calibrated-----------------\n");
        // stall();
    }

    void PropellerSet::drive_throttle_rear_right(float throttle_power) {
        if (!safety_check()) return;
        prop_set[0]->drive_throttle(throttle_power);
    }

    void PropellerSet::drive_throttle_front_right(float throttle_power) {
        if (!safety_check()) return;
        prop_set[1]->drive_throttle(throttle_power);
    }

    void PropellerSet::drive_throttle_rear_left(float throttle_power) {
        if (!safety_check()) return;
        prop_set[2]->drive_throttle(throttle_power);
    }

    void PropellerSet::drive_throttle_front_left(float throttle_power) {
        if (!safety_check()) return;
        prop_set[3]->drive_throttle(throttle_power);
    }

    void PropellerSet::brake() {
        Serial.println("Braking.");
        prop_set[0]->esc.writeMicroseconds(0);
        prop_set[1]->esc.writeMicroseconds(0);
        prop_set[2]->esc.writeMicroseconds(0);
        prop_set[3]->esc.writeMicroseconds(0);
    }

    bool PropellerSet::safety_check() {
        if (digitalRead(STATUS_PIN) == 1) PropellerSet::drive_enabled = false;
        if (!PropellerSet::drive_enabled) {
            PropellerSet::brake();
            stall();
        }
        return PropellerSet::drive_enabled;
    }

    void PropellerSet::actuate_force_torques(float fx, float fy, float fz, float yaw_torque) {
        torque_x = fy * vehicle_length;
        torque_y = -fx * vehicle_width;
        f_xyz_norm = sqrtf((fx * fx) + (fy * fy) + (fz * fz));
        t_front_right = max(((-2.08333333f * torque_x) + (1.92307692f * torque_y) + (-6.94444444f * yaw_torque) + (0.26923077f * f_xyz_norm)) / 5.0f, 0);
        t_rear_right = max(((-1.78571429f * torque_x) + (-1.92307692f * torque_y) + (5.95238095f * yaw_torque) + (0.23076923f * f_xyz_norm)) / 5.0f, 0);
        t_rear_left = max(((1.78571429f * torque_x) + (-1.92307692f * torque_y) + (-5.95238095f * yaw_torque) + (0.23076923f * f_xyz_norm)) / 5.0f, 0);
        t_front_left = max(((2.08333333f * torque_x) + (1.92307692f * torque_y) + (6.94444444f * yaw_torque) + (0.26923077f * f_xyz_norm)) / 5.0f, 0);

        Serial.print(f_xyz_norm, 6);
        Serial.print("\t");
        Serial.print(torque_x, 6);
        Serial.print("\t");
        Serial.print(torque_y, 6);
        Serial.print("\n");

        Serial.print(t_rear_right, 6);
        Serial.print("\t");
        Serial.print(t_front_right, 6);
        Serial.print("\t");
        Serial.print(t_rear_left, 6);
        Serial.print("\t");
        Serial.print(t_front_left, 6);
        Serial.print("\n");

        if (!safety_check()) return;
        prop_set[0]->drive_throttle(t_rear_right);
        prop_set[1]->drive_throttle(t_front_right);
        prop_set[2]->drive_throttle(t_rear_left);
        prop_set[3]->drive_throttle(t_front_left);
    }

    void PropellerSet::actuate(float nx, float ny, float nz, 
                               float yaw, float pitch, float roll) {
        if (!safety_check()) return;                   
        yaw_rad = yaw * PI / 180;
        pitch_rad = pitch * PI / 180;
        roll_rad = roll * PI / 100;

        cos_alpha = cosf(yaw_rad);
        cos_beta = cosf(pitch_rad);
        cos_gamma = cosf(roll_rad);

        sin_alpha = sinf(yaw_rad);
        sin_beta = sinf(pitch_rad);
        sin_gamma = sinf(roll_rad);

        nx_proj = (nx * (cos_alpha * cos_beta))
                + (ny * ((cos_alpha * sin_beta * sin_gamma) + (sin_alpha * cos_gamma)))
                + (nz * (-(cos_alpha * sin_beta * cos_gamma) + (sin_alpha * sin_gamma)));
        ny_proj = (nx * (-(sin_alpha * cos_beta)))
                + (ny * (-(sin_alpha * sin_beta * sin_gamma) + (cos_alpha * cos_gamma)))
                + (nz * ((sin_alpha * sin_beta * cos_gamma) + (cos_alpha * sin_gamma)));
        nz_proj = (nx * sin_beta)
                + (ny * (cos_beta * -sin_gamma))
                + (nz * (cos_beta * cos_gamma));

        
    }
}