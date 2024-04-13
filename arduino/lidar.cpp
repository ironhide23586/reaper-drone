#include "perception.h"



namespace Perception {

    Lidar::Lidar() {
        // Lidar::mySerial = new SoftwareSerial(2, 3); 
        // Lidar::mySerial = new HardwareSerial(2); 
        Lidar::tfmini = new TFMini();
        // Lidar::mySerial->begin(TFMINI_BAUDRATE);
        // Lidar::mySerial->begin(TFMINI_BAUDRATE, SERIAL_8N1, 27, 26);
        // Serial.println("Initializing Lidar Altitude Ranger.");
        Lidar::tfmini->begin();
        
        uint16_t dist, strength;
        // Serial.println("Reading dumy values to warm-up sensor.");
        for (int i = 0; i < 20; i++) {
            Lidar::get_reading(&dist, &strength);
            // Serial.println(dist);
            // Serial.println(strength);
            // delay(100);
        }
        // Serial.println("Lidar Altitude Ranger initialized successfully.");
    }

    int Lidar::get_reading(uint16_t* dist, uint16_t* strength) {
        read_status = tfmini->getReading(dist, strength);
        // if (read_status != MEASUREMENT_OK) {
        //     // init();
        //     // stall();
        // }
        return read_status;
    }

}
