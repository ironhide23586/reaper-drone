import serial
from time import sleep

ser = serial.Serial("/dev/ttyUSB0", 9600)    # Open port with baud rate
ser.flush()
while True:
    line = ser.readline().decode('utf-8').rstrip()
    print(line)
    # k = 0

    # print('lol')
    # b = ser.read()
    # received_data = int.from_bytes(b, 'little')     #read serial port
    # k = 0
    # print(received_data)
    # # sleep(0.03)
    # ser.write(str.encode('k'))

    # data_left = ser.inWaiting()             #check for remaining byte
    # received_data += ser.read(data_left)
    # print(received_data)                   #print received data
    # ser.write(received_data)
