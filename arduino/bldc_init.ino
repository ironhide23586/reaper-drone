#define MASTER_NANO
// #define SLAVE_NANO_0

#ifdef MASTER_NANO
#include "actuator.h"
#include "inertial_tracking.h"
using namespace InertialTracking;
using namespace Actuator;


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




void setup() {
  initialize_mcu();
  Serial.println("POWER ON THE ESCs NOW!");
  wait_loop(10);
  Serial.println("ARMING NOW...");

#ifdef MASTER_NANO
  props = new Actuator::PropellerSet();
  Wire.begin(IMU_MASTER_I2C_ADDRESS);
  // Wire.onReceive(receiveEvent);
  Wire.setClock(I2C_FREQUENCY_HZ);
  motion_tracker->init();
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

#ifdef MASTER_NANO
  // float t_val = .1;

  // props->drive_throttle_rear_right(t_val);
  // props->drive_throttle_front_right(t_val);
  // props->drive_throttle_rear_left(t_val);
  // props->drive_throttle_front_left(t_val);
  // // stall();
  motion_tracker->get_pose(&y, &p, &r, &h, &imu_raw_vals[0], false);

  
  float delta_t_rr_pitch = max(p / 60, 0);
  float delta_t_fr_pitch = max(-p / 60, 0);
  float delta_t_rl_pitch = max(p / 60, 0);
  float delta_t_fl_pitch = max(-p / 60, 0);
 
  float delta_t_rr_roll = max(r / 60, 0);
  float delta_t_fr_roll = max(r / 35, 0);
  float delta_t_rl_roll = max(-r / 35, 0);
  float delta_t_fl_roll = max(-r / 35, 0);

  float blend_coeff = .6;
  
  t_rr = (1. - blend_coeff) * t_rr + blend_coeff * ((delta_t_rr_pitch + delta_t_rr_roll) / 2.);
  t_fr = (1. - blend_coeff) * t_fr + blend_coeff * ((delta_t_fr_pitch + delta_t_fr_roll) / 2.);
  t_rl = (1. - blend_coeff) * t_rl + blend_coeff * ((delta_t_rl_pitch + delta_t_rl_roll) / 2.);
  t_fl = (1. - blend_coeff) * t_fl + blend_coeff * ((delta_t_fl_pitch + delta_t_fl_roll) / 2.);


  props->drive_throttle_rear_right(t_rr);
  props->drive_throttle_front_right(t_fr);
  props->drive_throttle_rear_left(t_rl);
  props->drive_throttle_front_left(t_fl);


  Serial.print(y);
  Serial.print("\t");
  Serial.print(p);
  Serial.print("\t");
  Serial.print(r);
  Serial.print("\t");
  Serial.print(h);
  Serial.print("\t<->");

  Serial.print(t_rr);
  Serial.print("\t");
  Serial.print(t_fr);
  Serial.print("\t");
  Serial.print(t_rl);
  Serial.print("\t");
  Serial.print(t_fl);
  Serial.print("\t<->");

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