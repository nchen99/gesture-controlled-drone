// This example shows how to read from all three channels on
// the OPT3101 and store the results in arrays.  It also shows
// how to use the sensor in a non-blocking way: instead of
// waiting for a sample to complete, the sensor code runs
// quickly so that the loop() function can take care of other
// tasks at the same time.

#include <OPT3101.h>
#include <Wire.h>



const uint8_t TRANSISTOR_PIN_1 = 12;
const uint8_t TRANSISTOR_PIN_2 = 8;


OPT3101 sensor1;
OPT3101 sensor2;


uint16_t amplitudes[6];
int16_t distances[6];

void setup()
{
  //pinMode(5, INPUT);
  pinMode(TRANSISTOR_PIN_1, OUTPUT);
  pinMode(TRANSISTOR_PIN_2, OUTPUT);

  Serial.begin(9600);
  Wire.begin();

  // Wait for the serial port to be opened before printing
  // messages (only applies to boards with native USB).

  while (!Serial) {}

  Serial.println("Init Sensor 1 start");

  initSensor1();

  Serial.println("Init Sensor 2 start");
  
  initSensor2();

}

void loop()

/*
{
  if (sensor.isSampleDone())
  {
    sensor.readOutputRegs();

    Serial.println(sensor.channelUsed);
    amplitudes[sensor.channelUsed] = sensor.amplitude;
    distances[sensor.channelUsed] = sensor.distanceMillimeters;

    if (sensor.channelUsed == 2)
    {
      for (uint8_t i = 0; i < 1; i++)
      {
        //Serial.print(amplitudes[i]);
        //Serial.print(',');
        //Serial.print(distances[i]);
        //Serial.print(", ");
      }
      Serial.println();
    }
    sensor.nextChannel();
    sensor.startSample();
  }
}
*/


{
  startSensor1();
  startSensor2();
  finSensor1(0);
  startSensor1();
  finSensor2(0);
  startSensor2();
  finSensor1(1);
  startSensor1();
  finSensor2(1);
  startSensor2();
  finSensor1(2);
  finSensor2(2);
  

  for (uint8_t i = 0; i < 6; i++)
  {
    //Serial.print(amplitudes[i]);
    //Serial.print(',');
    Serial.print(distances[i]);
    Serial.print(", ");
  }
  Serial.println();

  
 
}



void startSensor1(){
  setSensor1();
  sensor1.startSample();
  return;
}

void startSensor2(){
  setSensor2();
  sensor2.startSample();
  return;
}

void finSensor1(uint8_t channel){
  setSensor1();
  
  while(!sensor1.isSampleDone()){
  }
  
  sensor1.readOutputRegs();
  distances[channel] = sensor1.distanceMillimeters;
  sensor1.nextChannel();
  return;
}

void finSensor2(uint8_t channel){
  setSensor2();
  
  while(!sensor2.isSampleDone()){
  }
  
  sensor2.readOutputRegs();
  distances[channel+3] =  sensor2.distanceMillimeters;
  sensor2.nextChannel();
  return;
}


void initSensor1(){
  setSensor1();

  sensor1.init();
  if (sensor1.getLastError())
  {
    Serial.print(F("Failed to initialize OPT3101 1: error "));
    Serial.println(sensor1.getLastError());
    while (1) {}
  }

  sensor1.setFrameTiming(64);
  sensor1.setChannel(0);
  sensor1.setBrightness(OPT3101Brightness::High);
  return;
}


void initSensor2(){
  setSensor2();

  sensor2.init();
  if (sensor2.getLastError())
  {
    Serial.print(F("Failed to initialize OPT3101 2: error "));
    Serial.println(sensor2.getLastError());
    while (1) {}
  }

  sensor2.setFrameTiming(64);
  sensor2.setChannel(0);
  sensor2.setBrightness(OPT3101Brightness::High);
  return;
}



void setSensor1(){
  delayMicroseconds(1);
  digitalWrite(TRANSISTOR_PIN_2, LOW);
  digitalWrite(TRANSISTOR_PIN_1, HIGH);
  delayMicroseconds(1);
  //setSensor2();
  return;
}

void setSensor2(){
  
  delayMicroseconds(1);
  digitalWrite(TRANSISTOR_PIN_1, LOW);
  digitalWrite(TRANSISTOR_PIN_2, HIGH);
  delayMicroseconds(1);
  return;
}
