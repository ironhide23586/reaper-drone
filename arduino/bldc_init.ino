#define MASTER_NANO
// #define SLAVE_NANO_0

#ifdef MASTER_NANO
#include <Servo.h>
#include "actuator.h"
#include "inertial_tracking.h"
using namespace InertialTracking;
using namespace Actuator;

#define TEST_MS 4000


void wait_loop(int num_ticks) {
  for (int i = num_ticks; i > 0; i--) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(300);
    digitalWrite(LED_BUILTIN, LOW);
    delay(300);
    Serial.print("Tick tick ");
    Serial.println(i);
  }
}

// Serial.begin(BAUD_RATE);
InertialTracking::MotionTracking* motion_tracker = new InertialTracking::MotionTracking();
Actuator::PropellerSet* props;


int pitch_led = 11;
int roll_led = 3;
int init_complete_led = 12;
int init_ongoing_led = 4;

volatile bool dummy_trigger = false;

// 9 -> Rear-Right
// 10 -> Front-Right
// 6 -> Rear-Left
// 5 -> Front-Left

#endif

#ifdef SLAVE_NANO_0
#include "perception.h"
using namespace Perception;
Perception::Lidar* alt_sensor;
#endif


void safety_stall() {
  if (!dummy_trigger) dummy_trigger = true;
  else {
    props->brake();
    digitalWrite(INIT_ONGOING_LED, HIGH);
    digitalWrite(INIT_COMPLETE_LED, HIGH);
    analogWrite(roll_led, 255);
    analogWrite(pitch_led, 255);
    Serial.println("Killswitch triggered.");
    props->drive_enabled = false;
  }
}



void setup() {
  pinMode(PITCH_LED, OUTPUT);
  pinMode(INIT_COMPLETE_LED, OUTPUT);
  pinMode(INIT_ONGOING_LED, OUTPUT);
  pinMode(STATUS_PIN, INPUT);

  digitalWrite(INIT_COMPLETE_LED, LOW);
  digitalWrite(INIT_ONGOING_LED, HIGH);
  initialize_mcu();

  Serial.println("WAITING TO POWER ON THE ESCs NOW!");
  while(digitalRead(STATUS_PIN) == 0) {
    digitalWrite(INIT_ONGOING_LED, HIGH);
    delay(20);
    digitalWrite(INIT_COMPLETE_LED, LOW);
    delay(10);
    analogWrite(roll_led, 255);
    delay(5);
    analogWrite(pitch_led, 0);
    delay(1);

    digitalWrite(INIT_ONGOING_LED, LOW);
    delay(20);
    digitalWrite(INIT_COMPLETE_LED, HIGH);
    delay(10);
    analogWrite(roll_led, 0);
    delay(5);
    analogWrite(pitch_led, 255);
    delay(1);
  }
  analogWrite(roll_led, 0);
  analogWrite(pitch_led, 0);
  digitalWrite(INIT_COMPLETE_LED, LOW);
  digitalWrite(INIT_ONGOING_LED, HIGH);
  delay(500);
  pinMode(STATUS_PIN, INPUT);

  Serial.println("ARMING NOW...");

#ifdef MASTER_NANO
  props = new Actuator::PropellerSet();

  Wire.begin(IMU_MASTER_I2C_ADDRESS);
  // Wire.onReceive(receiveEvent);
  Wire.setClock(I2C_FREQUENCY_HZ);
  motion_tracker->init();

  // attachInterrupt(digitalPinToInterrupt(STATUS_PIN), safety_stall, RISING);
  // int cnt_idx = 0;

  // analogWrite(pitch_led, 255);
  // props->actuate_force_torques(0, 0, 100, 0);
  // for (cnt_idx = 0; cnt_idx < 50; cnt_idx++) {
  //   delay(1);
  //   props->safety_check();
  // }
  // props->brake();
  // analogWrite(pitch_led, 0);

  // for (cnt_idx = 0; cnt_idx < 2000; cnt_idx++) {
  //   analogWrite(roll_led, cnt_idx % 255);
  //   props->safety_check();
  //   delay(2);
  //   analogWrite(roll_led, 0);
  // }

  // analogWrite(pitch_led, 80);
  // props->actuate_force_torques(0, 0, .5, 0);
  // for (cnt_idx = 0; cnt_idx < TEST_MS; cnt_idx++) {
  //   delay(1);
  //   props->safety_check();
  // }
  // props->brake();
  // analogWrite(pitch_led, 0);

  // for (cnt_idx = 0; cnt_idx < 2000; cnt_idx++) {
  //   analogWrite(roll_led, cnt_idx % 255);
  //   props->safety_check();
  //   delay(2);
  //   analogWrite(roll_led, 0);
  // }

  // analogWrite(pitch_led, 140);
  // props->actuate_force_torques(0, 0, 1., 0);
  // for (cnt_idx = 0; cnt_idx < TEST_MS; cnt_idx++) {
  //   delay(1);
  //   props->safety_check();
  // }
  // props->brake();
  // analogWrite(pitch_led, 0);

  // for (cnt_idx = 0; cnt_idx < 2000; cnt_idx++) {
  //   analogWrite(roll_led, cnt_idx % 255);
  //   props->safety_check();
  //   delay(2);
  //   analogWrite(roll_led, 0);
  // }

  // analogWrite(pitch_led, 200);
  // props->actuate_force_torques(0, 0, 1.5, 0);
  // for (cnt_idx = 0; cnt_idx < TEST_MS; cnt_idx++) {
  //   delay(1);
  //   props->safety_check();
  // }
  // props->brake();
  // analogWrite(pitch_led, 0);

  // for (cnt_idx = 0; cnt_idx < 2000; cnt_idx++) {
  //   analogWrite(roll_led, cnt_idx % 255);
  //   props->safety_check();
  //   delay(2);
  //   analogWrite(roll_led, 0);
  // }

  // analogWrite(pitch_led, 255);
  // props->actuate_force_torques(0, 0, 2., 0);
  // for (cnt_idx = 0; cnt_idx < TEST_MS; cnt_idx++) {
  //   delay(1);
  //   props->safety_check();
  // }
  // props->brake();
  // analogWrite(pitch_led, 0);

  // for (cnt_idx = 0; cnt_idx < 2000; cnt_idx++) {
  //   analogWrite(roll_led, cnt_idx % 255);
  //   props->safety_check();
  //   delay(2);
  //   analogWrite(roll_led, 0);
  // }

  // analogWrite(pitch_led, 255);
  // props->actuate_force_torques(0, 0, 2.5, 0);
  // for (cnt_idx = 0; cnt_idx < TEST_MS; cnt_idx++) {
  //   delay(1);
  //   props->safety_check();
  // }
  // props->brake();
  // analogWrite(pitch_led, 0);

  // for (cnt_idx = 0; cnt_idx < 2000; cnt_idx++) {
  //   analogWrite(roll_led, cnt_idx % 255);
  //   props->safety_check();
  //   delay(2);
  //   analogWrite(roll_led, 0);
  // }

  // analogWrite(pitch_led, 255);
  // props->actuate_force_torques(0, 0, 3, 0);
  // for (cnt_idx = 0; cnt_idx < TEST_MS; cnt_idx++) {
  //   delay(1);
  //   props->safety_check();
  // }
  // props->brake();
  // analogWrite(pitch_led, 0);


  // stall();
#endif

#ifdef SLAVE_NANO_0
  alt_sensor = new Perception::Lidar();
  Wire.begin(LIDAR_SLAVE_I2C_ADDRESS);
#endif
}

#ifdef MASTER_NANO
float y, p, r, h;
float t_rr = 0., t_fr = 0., t_rl = 0., t_fl = 0.;
float imu_raw_vals[9];
#endif

#ifdef SLAVE_NANO_0
uint16_t dist, strength;
uint8_t tmp;
static int cnt = 1;
#endif

void loop() {
#ifdef SLAVE_NANO_0
  int res = alt_sensor->get_reading(&dist, &strength);
  // while (res != MEASUREMENT_OK) {
  //   Serial.print("Reinitializing lidar");
  //   alt_sensor = new Perception::Lidar();
  //   res = alt_sensor->get_reading(&dist, &strength);
  // }
  digitalWrite(LED_BUILTIN, HIGH);
  Wire.beginTransmission(IMU_MASTER_I2C_ADDRESS);
  Wire.write(dist);
  Wire.endTransmission(true);
  // Serial.println("Sent data through I2C to master.");
  // Serial.println(dist);
  // Serial.println(strength);
  // Serial.println(res);
  digitalWrite(LED_BUILTIN, LOW);
  // Serial.println(cnt);
  // cnt++;
  delay(100);
#endif


// Driving motor at 2.300000% power with pwm val 1023
// Driving motor at 2.600000% power with pwm val 1026
// Driving motor at 2.300000% power with pwm val 1023
// Driving motor at 2.600000% power with pwm val 1026
// Braking.
// Driving motor at 4.600000% power with pwm val 1046
// Driving motor at 5.300000% power with pwm val 1053
// Driving motor at 4.600000% power with pwm val 1046
// Driving motor at 5.300000% power with pwm val 1053
// Braking.
// Driving motor at 6.900000% power with pwm val 1069
// Driving motor at 8.000000% power with pwm val 1080
// Driving motor at 6.900000% power with pwm val 1069
// Driving motor at 8.000000% power with pwm val 1080
// Braking.
// Driving motor at 9.200000% power with pwm val 1092
// Driving motor at 10.700000% power with pwm val 1107
// Driving motor at 9.200000% power with pwm val 1092


#ifdef MASTER_NANO
  // float t_val = .1;

  // props->drive_throttle_rear_right(t_val);
  // props->drive_throttle_front_right(t_val);
  // props->drive_throttle_rear_left(t_val);
  // props->drive_throttle_front_left(t_val);
  // // stall();
  motion_tracker->get_pose(&y, &p, &r, &h, &imu_raw_vals[0], false);
  
  
  // float delta_t_rr_pitch = max(p / 50, 0);
  // float delta_t_fr_pitch = max(-p / 50, 0);
  // float delta_t_rl_pitch = max(p / 50, 0);
  // float delta_t_fl_pitch = max(-p / 50, 0);
 
  // float delta_t_rr_roll = max(r / 50, 0);
  // float delta_t_fr_roll = max(r / 50, 0);
  // float delta_t_rl_roll = max(-r / 50, 0);
  // float delta_t_fl_roll = max(-r / 50, 0);

  // float blend_coeff = 1.;
  
  // t_rr = (1. - blend_coeff) * t_rr + blend_coeff * ((delta_t_rr_pitch + delta_t_rr_roll) / 2.);
  // t_fr = (1. - blend_coeff) * t_fr + blend_coeff * ((delta_t_fr_pitch + delta_t_fr_roll) / 2.);
  // t_rl = (1. - blend_coeff) * t_rl + blend_coeff * ((delta_t_rl_pitch + delta_t_rl_roll) / 2.);
  // t_fl = (1. - blend_coeff) * t_fl + blend_coeff * ((delta_t_fl_pitch + delta_t_fl_roll) / 2.);


  // props->drive_throttle_rear_right(t_rr);
  // props->drive_throttle_front_right(t_fr);
  // props->drive_throttle_rear_left(t_rl);
  // props->drive_throttle_front_left(t_fl);

  analogWrite(pitch_led, abs(min(p / 50., 1.) * 255));
  analogWrite(roll_led, abs(min(r / 50., 1.) * 255));

  Serial.print(y);
  Serial.print("\t");
  Serial.print(p);
  Serial.print("\t");
  Serial.print(r);
  Serial.print("\t");
  Serial.print(h);
  Serial.print("\t<->");

  // Serial.print(t_rr);
  // Serial.print("\t");
  // Serial.print(t_fr);
  // Serial.print("\t");
  // Serial.print(t_rl);
  // Serial.print("\t");
  // Serial.print(t_fl);
  // Serial.print("\t<->");

  for (int i = 0; i < 9; i++) {
    Serial.print("\t");
    Serial.print(imu_raw_vals[i]);
  }
  Serial.print("\n");
  // stall();
#endif
}

#ifdef MASTER_NANO
void receiveEvent(int howMany) {
  uint8_t c = Wire.read(); // receive a character
  digitalWrite(LED_BUILTIN, HIGH);
  // Serial.println("\nReceived data through I2C");
  // Serial.println(c, HEX);
  delay(100);
  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
  // stall();
}
#endif