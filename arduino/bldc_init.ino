#define MASTER_NANO
// #define SLAVE_NANO_0

#ifdef MASTER_NANO
#include "inertial_tracking.h"
using namespace InertialTracking;
InertialTracking::MotionTracking* motion_tracker = new InertialTracking::MotionTracking();
#endif

#ifdef SLAVE_NANO_0
#include "perception.h"
using namespace Perception;
Perception::Lidar* alt_sensor;
#endif

void setup() {
initialize_mcu();

#ifdef MASTER_NANO
  Wire.begin(IMU_MASTER_I2C_ADDRESS);
  // Wire.onReceive(receiveEvent);
  pinMode(LED_BUILTIN, OUTPUT);
  Wire.setClock(I2C_FREQUENCY_HZ);
  motion_tracker->init();
  // stall();
#endif

#ifdef SLAVE_NANO_0
  alt_sensor = new Perception::Lidar();
  Wire.begin(LIDAR_SLAVE_I2C_ADDRESS);
#endif


}
#ifdef MASTER_NANO
float y, p, r, h;
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
  motion_tracker->get_pose(&y, &p, &r, &h, &imu_raw_vals[0], false);
  Serial.print(y);
  Serial.print("\t");
  Serial.print(p);
  Serial.print("\t");
  Serial.print(r);
  Serial.print("\t");
  Serial.print(h);
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