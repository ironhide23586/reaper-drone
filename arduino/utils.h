#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <Wire.h>
#include <math.h>
#include <Arduino.h>

#define IMU_MASTER_I2C_ADDRESS 0x00
#define LIDAR_SLAVE_I2C_ADDRESS 0x01

#define TFMINI_BAUDRATE   115200
#define BAUD_RATE 250000
#define I2C_FREQUENCY_HZ 400000
#define GYRO_WEIGHT 0.97f
#define NUM_CALIBRATION_SAMPLES 2000
#define PI 3.1415926535897932384626433832795f


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