#include <Keypad.h>

const byte ROWS = 4; //four rows
const byte COLS = 4; //four columns

char keys[ROWS][COLS] = {
  {'1','4','7','*'},
  {'2','5','8','0'},
  {'3','6','9','#'},
  {'A','B','C','D'}
};

byte rowPins[COLS] = {13, 12, 14, 26}; //connect to the row pinouts of the keypad
byte colPins[ROWS] = {27, 25, 33, 32}; //connect to the column pinouts of the keypad

//Create an object of keypad
Keypad keypad = Keypad( makeKeymap(keys), rowPins, colPins, ROWS, COLS );

void setup() {
  Serial.begin(9600); // Initialize serial communication
}

void loop() {
  char key = keypad.getKey();// Read the key
  
  // Print if key pressed
  if (key){
    Serial.println(key);
  }
}
