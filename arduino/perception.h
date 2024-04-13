#include "utils.h"
#include "TFMini.h"

// #include <HardwareSerial.h>
// #include <SoftwareSerial.h>



namespace Perception {

    class Lidar {
    private:

        // SoftwareSerial* mySerial;     // Uno RX (TFMINI TX), Uno TX (TFMINI RX)
        // HardwareSerial mySerial(2);
        TFMini* tfmini;
        int read_status;

    public:
        
        Lidar();
        int get_reading(uint16_t* dist, uint16_t* strength);
    };

}