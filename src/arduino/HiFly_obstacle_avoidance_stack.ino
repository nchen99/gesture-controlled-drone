 /*
 * --- HiFly obstacle avoidance sensor stack for Tello ---
 * Description:
 * This program utlizies 2 OPT3101 sensors and one VL53L0X ToF sensor.
 * All the sensor readings are sent through wifi to a static IP and port.
 * An LED matrix (8 LEDs) is used to provide direct feedback on the sensor readings.
 * The LED matrix is also used to implement user-feedback functionality such as a green blink.
 * 
 * Author: Benedek Hegedus, Huawei Canada
 */


// ------ WIFI START ------
#include <SPI.h>
#include <WiFiNINA.h>
#include <WiFiUdp.h>



#define SECRET_SSID "TP-Link_A5A3_tello"
#define SECRET_PASS ""

IPAddress local_ip(192, 168, 0, 152); // for static ip   

#define ATLASDK_IP "192.168.0.101"
const int16_t ATLASDK_PORT = 9121;

int status = WL_IDLE_STATUS;
char ssid[] = SECRET_SSID;        // your network SSID (name)
char pass[] = SECRET_PASS;    // your network password (use for WPA, or use as key for WEP)
int keyIndex = 0;            // your network key index number (needed only for WEP)

unsigned int localPort = 2390;      // local port to listen on

char packetBuffer[8]; //buffer to hold incoming packet
char  ReplyBuffer[] = "acknowledged";       // a string to send back

bool send_flag = false;

WiFiUDP Udp;
// ------ WIFI END ------



// ------ VL53L0X ToF SENSOR START ------
#include "Arduino.h"
#include "DFRobot_VL53L0X.h"
DFRobot_VL53L0X tofSensor;
// ------ VL53L0X ToF SENSOR END ------



// ------ OPT3101 SENSOR START ------
#include <OPT3101.h>
#include <Wire.h>

const uint8_t TRANSISTOR_PIN_1 = 8;
const uint8_t TRANSISTOR_PIN_2 = 12;

const int16_t MAX_DISTANCE = 3000;
const int16_t MIN_DISTANCE = 1;

OPT3101 sensor1;
OPT3101 sensor2;
uint16_t amplitudes[6];
int16_t distances[7];
// ------ OPT3101 SENSOR END ------



// ------ 8 LED RGB START ------

#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
 #include <avr/power.h> // Required for 16 MHz Adafruit Trinket
#endif

// Which pin on the Arduino is connected to the NeoPixels?
#define PIN        10 // On Trinket or Gemma, suggest changing this to 1

// How many NeoPixels are attached to the Arduino?
#define NUMPIXELS 8 // Popular NeoPixel ring size

#define DELAYVAL 500 // Time (in milliseconds) to pause between pixels

// When setting up the NeoPixel library, we tell it how many pixels,
// and which pin to use to send signals. Note that for older NeoPixel
// strips you might need to change the third parameter -- see the
// strandtest example for more information on possible values.
Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

uint8_t RGB_values[3];

// ------ 8 LED RGB END ------

    


void setup()
{
 



  // Init pins to switch between OPT sensors
  pinMode(TRANSISTOR_PIN_1, OUTPUT);
  pinMode(TRANSISTOR_PIN_2, OUTPUT);

  //Serial.begin(9600);
  Wire.begin();

  //while (!Serial) {}

  //Serial.println("Init Sensor 1 start");

  initSensor1();

  //Serial.println("Init Sensor 2 start");
  
  initSensor2();

  //Serial.println("Init ToF start");

  init_tof_sensor();

  initLEDs();

  //Serial.println("Init WIFI start");

  initWIFI();


}

void loop()
{
  readOPTs();  
  readTofSensor();
  setLEDs();
  if(send_flag){
    sendSensorData();
  }
  handleRequest();

/*

  for (uint8_t i = 0; i < 7; i++)
  {
    //Serial.print(amplitudes[i]);
    //Serial.print(RGB_values[i]);
    //Serial.print(',');
    Serial.print(distances[i]);
    Serial.print(", ");
  }
  
  Serial.println();
*/
  
}


// ------ OPT3101 SENSOR FUNCTIONS START ------
void readOPTs(){
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
  int timeout_count = 0;
  while(!sensor1.isSampleDone() && timeout_count < 100000){
    timeout_count++;
  }
  if(timeout_count >= 100000){
    Serial.println("TIMEOUT!!!");
  }
  
  sensor1.readOutputRegs();
  distances[channel] = sensor1.distanceMillimeters;
  sensor1.nextChannel();
  return;
}

void finSensor2(uint8_t channel){
  setSensor2();
  int timeout_count = 0;
  while(!sensor2.isSampleDone() && timeout_count < 100000){
    timeout_count++;
  }
  if(timeout_count >= 100000){
    Serial.println("TIMEOUT!!!");
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
    //Serial.print(F("Failed to initialize OPT3101 1: error "));
    //Serial.println(sensor1.getLastError());
    while (1) {}
  }

  sensor1.setFrameTiming(128);
  sensor1.setChannel(0);
  sensor1.setBrightness(OPT3101Brightness::High);
  return;
}


void initSensor2(){
  setSensor2();

  sensor2.init();
  if (sensor2.getLastError())
  {
    //Serial.print(F("Failed to initialize OPT3101 2: error "));
    //Serial.println(sensor2.getLastError());
    while (1) {}
  }

  sensor2.setFrameTiming(128);
  sensor2.setChannel(0);
  sensor2.setBrightness(OPT3101Brightness::High);
  return;
}



void setSensor1(){
  delayMicroseconds(10);
  digitalWrite(TRANSISTOR_PIN_2, LOW);
  digitalWrite(TRANSISTOR_PIN_1, HIGH);
  delayMicroseconds(10);
  return;
}

void setSensor2(){
  delayMicroseconds(10);
  digitalWrite(TRANSISTOR_PIN_1, LOW);
  digitalWrite(TRANSISTOR_PIN_2, HIGH);
  delayMicroseconds(10);
  //setSensor1();
  return;
}
// ------ OPT3101 SENSOR FUNCTIONS END ------



// ------ RGB LED FUNCTIONS START ------

void setLEDs (){ 
  distanceToRGB(distances[0]);
  pixels.setPixelColor(1, pixels.Color(RGB_values[0], RGB_values[1], RGB_values[1]));

  distanceToRGB(distances[1]);
  pixels.setPixelColor(2, pixels.Color(RGB_values[0], RGB_values[1], RGB_values[1]));

  distanceToRGB(distances[2]);
  pixels.setPixelColor(3, pixels.Color(RGB_values[0], RGB_values[1], RGB_values[1]));

  distanceToRGB(distances[3]);
  pixels.setPixelColor(5, pixels.Color(RGB_values[0], RGB_values[1], RGB_values[1]));

  distanceToRGB(distances[4]);
  pixels.setPixelColor(6, pixels.Color(RGB_values[0], RGB_values[1], RGB_values[1]));

  distanceToRGB(distances[5]);
  pixels.setPixelColor(7, pixels.Color(RGB_values[0], RGB_values[1], RGB_values[1]));  
  

  distanceToRGB(distances[6]);
  pixels.setPixelColor(0, pixels.Color(RGB_values[0], RGB_values[1], RGB_values[1]));
  pixels.setPixelColor(4, pixels.Color(RGB_values[0], RGB_values[1], RGB_values[1]));
  
  pixels.show();   // Send the updated pixel colors to the hardware.
  return;
}

// Uses RGB_values global array
void distanceToRGB(int16_t distance){
  int16_t bounded_distance = distance;
  
  if(distance >= MAX_DISTANCE){
    bounded_distance = MAX_DISTANCE;
  }
  if(distance < MIN_DISTANCE){
    bounded_distance = MIN_DISTANCE;
  }
  float distance_percent = (float)(bounded_distance - MIN_DISTANCE) / (float)(MAX_DISTANCE - 1500 - MIN_DISTANCE);
  RGB_values[0] = (int8_t)((1.0 - distance_percent) * 50.0); // scale RED inverse with distance
  RGB_values[1] = (int8_t)(distance_percent * 50.0); // scale GREEN directly with distance
  RGB_values[2] = 0;

  return;
}

void initLEDs(){
  // These lines are specifically to support the Adafruit Trinket 5V 16 MHz.
  // Any other board, you can remove this part (but no harm leaving it):
  
  #if defined(__AVR_ATtiny85__) && (F_CPU == 16000000)
  clock_prescale_set(clock_div_1);
  #endif
  // END of Trinket-specific code.

  pixels.begin(); // INITIALIZE NeoPixel strip object (REQUIRED)

  pixels.clear(); // Set all pixel colors to 'off'
  return;
}
// ------ RGB LED FUNCTIONS END ------



// ------ ToF SENSOR FUNCTIONS START ------
void init_tof_sensor(){
  // default address is 0x50
  tofSensor.begin(0x50);
  //Set to Back-to-back mode and high precision mode
  tofSensor.setMode(tofSensor.eContinuous,tofSensor.eHigh);
  //Laser rangefinder begins to work
  tofSensor.start();
  return;
}

void readTofSensor(){
  distances[6] = tofSensor.getDistance();
  return;
}
// ------ ToF SENSOR FUNCTIONS END ------



// ------ WIFI FUNCTIONS START ------
void initWIFI(){
  // check for the WiFi module:
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    // don't continue
    while (true);
  }

 
  // attempt to connect to WiFi network:
  WiFi.config(local_ip);  // use static ip
  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
    status = WiFi.begin(ssid, pass);

    // wait 10 seconds for connection:
    delay(10000);
  }
  //Serial.println("Connected to WiFi");
  printWifiStatus();

  //Serial.println("\nStarting connection to server...");
  // if you get a connection, report back via serial:
  Udp.begin(localPort);

  return;
}

void printWifiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your board's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
  return;
}

void sendSensorData(){

   int16_t bounded_distances[7];

   for(int i=0; i < 7; i++){
    if (distances[i] <= MIN_DISTANCE){
      bounded_distances[i] = MIN_DISTANCE;
    }
    else if (distances[i] > MAX_DISTANCE){
      bounded_distances[i] = MAX_DISTANCE;
    }
    else {
      bounded_distances[i] = distances[i];
    }
   }
  
   String semicolon = String(";");
   String sensorData = (String(bounded_distances[0]) + semicolon
                        + String(bounded_distances[1]) + semicolon  
                        + String(bounded_distances[2]) + semicolon  
                        + String(bounded_distances[3]) + semicolon  
                        + String(bounded_distances[4]) + semicolon  
                        + String(bounded_distances[5]) + semicolon 
                        + String(bounded_distances[6]));
                        

   
   Udp.beginPacket(ATLASDK_IP, ATLASDK_PORT);
   Udp.write(sensorData.c_str());
   //Serial.println(sensorData);
   Udp.endPacket();
   return;
}


void handleRequest(){
  // if there's data available, read a packet
  int packetSize = Udp.parsePacket();
  
  if (packetSize) {
    // read the packet into packetBufffer
    int len = Udp.read(packetBuffer, 7);
    if (len > 0) {
      packetBuffer[len] = 0;
    }
    
    Serial.println("Contents:");
    Serial.println(packetBuffer);

    if(packetBuffer[0] == '1'){
      //Serial.println("one");
      userFeedback1();
    }
    if(packetBuffer[0] == '2'){
      //Serial.println("two");
      userFeedback2();
    }
    if(packetBuffer[0] == '5'){
      //Serial.println("two");
      startSending();
    }    
  }
  return;
}

void userFeedback1(){
  // Blink LED GREEN
  pixels.clear(); // Set all pixel colors to 'off'
  for(int i=0; i<8; i++) { 

    // pixels.Color() takes RGB values, from 0,0,0 up to 255,255,255
    pixels.setPixelColor(i, pixels.Color(0, 100, 0));
    pixels.show();   // Send the updated pixel colors to the hardware.

  }
  delay(200); // Delay so that user can notice blink
  return;
}

void userFeedback2(){
  // Blink LED BLUE
  pixels.clear(); // Set all pixel colors to 'off'
  for(int i=0; i<8; i++) { 

    // pixels.Color() takes RGB values, from 0,0,0 up to 255,255,255
    pixels.setPixelColor(i, pixels.Color(0, 0, 100));
    pixels.show();   // Send the updated pixel colors to the hardware.

  }
  delay(200); // Delay so that user can notice blink
  return;
}

void startSending(){
  send_flag = true;
  return;
}


// ------ WIFI FUNCTIONS END ------
