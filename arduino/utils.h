#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <Wire.h>
#include <math.h>
#include <Arduino.h>

#define IMU_MASTER_I2C_ADDRESS 0x00
#define LIDAR_SLAVE_I2C_ADDRESS 0x01

#define HIGH_THROTTLE_PWM_VAL 2000
#define LOW_THROTTLE_PWM_VAL 1000
#define SAFETY_CAP_PWM_VAL 1800

#define STATUS_PIN 2

#define G_VAL 9.8f
#define TFMINI_BAUDRATE   
#define BAUD_RATE 19200
#define I2C_FREQUENCY_HZ 400000
#define GYRO_WEIGHT 0.95f
#define NUM_CALIBRATION_SAMPLES 2000
#define PI 3.1415926535897932384626433832795f

#define PITCH_LED 11
#define ROLL_LED 3
#define INIT_COMPLETE_LED 12
#define INIT_ONGOING_LED 4

#define NEW_OFFSET_WEIGHT 0.7f

#define NUM_CALIBRATION_SAMPLES_MULTIPLIER 4


inline void stall() {
  while(1) {
    delay(5000);
  }
}


inline void initialize_mcu() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
#ifdef MASTER_NANO
  Serial.begin(BAUD_RATE);
#endif
#ifdef SLAVE_NANO_0
  Serial.begin(TFMINI_BAUDRATE);
#endif

  while (!Serial);
  
#ifdef MASTER_NANO
  Serial.print("Serial Bus Initialized at ");
  Serial.print(BAUD_RATE);
  Serial.println(" baud rate.");
#endif
#ifdef SLAVE_NANO_0
  // Serial.print(TFMINI_BAUDRATE);
#endif
  digitalWrite(LED_BUILTIN, LOW);
}


inline void I2CsendToMaster(uint8_t d) {
  Wire.beginTransmission(IMU_MASTER_I2C_ADDRESS);
  Wire.write(d);
  Wire.endTransmission(true);
}


inline void I2CsendToMaster(uint16_t d) {
  Wire.beginTransmission(IMU_MASTER_I2C_ADDRESS);
  Wire.write(d);
  Wire.endTransmission(true);
}


// inline void cross_product(float a, float b, float c, float x, float y, float z, float *res_x, float *res_y) {
//   *res_x = (b * z) - (c * y);
//   *res_y = (c * x) - (a * z);
//   // *res_z = (a * y) - (b * x);
// }


inline void I2Cread(uint8_t Address, uint8_t Register, uint8_t Nbytes, uint8_t* Data)
{
  Wire.beginTransmission(Address);
  Wire.write(Register);
  Wire.endTransmission();
  Wire.requestFrom(Address, Nbytes);
  uint8_t index = 0;
  while (Wire.available())
  Data[index++] = Wire.read();
  delayMicroseconds(100);
}

inline void I2CwriteByte(uint8_t Address, uint8_t Register, uint8_t Data)
{
  uint8_t Data_, Data__;
  I2Cread(Address, Register, 1, &Data__);

  Wire.beginTransmission(Address);
  Wire.write(Register);
  Wire.write(Data);
  Wire.endTransmission(true);
  delayMicroseconds(100);

  I2Cread(Address, Register, 1, &Data_);
  if (Data != Data_) {
    Serial.println("I2C Write Failed ");
    Serial.print("0x");
    Serial.print(Data, HEX);
    Serial.print("!=");
    Serial.print("0x");
    Serial.println(Data_, HEX);
  } else {
    Serial.print("0x");
    Serial.print(Data, HEX);
    Serial.print("==");
    Serial.print("0x");
    Serial.println(Data_, HEX);
    if (Data_ != Data__) {
      Serial.print("0x");
      Serial.print(Data__, HEX);
      Serial.print(" changed to ");
      Serial.print("0x");
      Serial.println(Data_, HEX);
    }
  }
}

#endif