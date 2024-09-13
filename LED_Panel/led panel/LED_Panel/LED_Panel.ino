#include <MD_Parola.h>
#include <MD_MAX72xx.h>
#include <SPI.h>

#define HARDWARE_TYPE MD_MAX72XX::FC16_HW
#define MAX_DEVICES 16
#define NUM_ZONES 16

#define ZONE_SIZE (MAX_DEVICES / NUM_ZONES)

#define CLK_PIN 13
#define DATA_PIN 11
#define CS_PIN 10

MD_Parola P = MD_Parola(HARDWARE_TYPE, CS_PIN, MAX_DEVICES);

#define SPEED_TIME 0
#define PAUSE_TIME 500
#define DEBUG 0

#if DEBUG
#define PRINT(s, x) { Serial.print(F(s)); Serial.print(x); }
#define PRINTS(x) Serial.print(F(x))
#define PRINTX(x) Serial.println(x, HEX)
#else
#define PRINT(s, x)
#define PRINTS(x)
#define PRINTX(x)
#endif

// Global Variables
uint8_t curText;
uint8_t curZone;
uint8_t curFX = 0;
int displayArray[16] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

const char *pc[] = {"0", "#", "*", "+", "X"}; // Character 0, 1, 2, 3, 4

textEffect_t effect[] =
{
  PA_PRINT,
  PA_SCAN_HORIZ,
  PA_SCROLL_LEFT,
  PA_WIPE,
  PA_RANDOM,
  PA_SCROLL_UP_LEFT,
  PA_SCROLL_UP,
  PA_FADE,
  PA_OPENING_CURSOR,
  PA_GROW_UP,
  PA_SCROLL_UP_RIGHT,
  PA_BLINDS,
  PA_MESH,
  PA_CLOSING,
  PA_GROW_DOWN,
  PA_SCAN_VERT,
  PA_SCROLL_DOWN_LEFT,
  PA_WIPE_CURSOR,
  PA_DISSOLVE,
  PA_OPENING,
  PA_CLOSING_CURSOR,
  PA_SCROLL_DOWN_RIGHT,
  PA_SCROLL_RIGHT,
  PA_SLICE,
  PA_SCROLL_DOWN,
};

void setup(void)
{
  Serial.begin(9600);

  P.begin(NUM_ZONES);
  for (uint8_t i = 0; i < NUM_ZONES; i++)
  {
    P.setZone(i, ZONE_SIZE * i, (ZONE_SIZE * (i + 1)) - 1);
    PRINT("\nZ", i);
    PRINT(" from ", ZONE_SIZE * i);
    PRINT(" to ", (ZONE_SIZE * (i + 1)) - 1);
  }
  P.setInvert(false);
}

void loop(void)
{
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    Serial.print("Received data on COM12: ");
    Serial.println(data);

    // Parse the received data
    int startIndex = 0;
    int endIndex = data.indexOf(',');
    for (int i = 0; i < 16; i++) {
      if (endIndex == -1) {
        endIndex = data.length();
      }
      String numStr = data.substring(startIndex, endIndex);
      displayArray[i] = numStr.toInt();
      startIndex = endIndex + 1;
      endIndex = data.indexOf(',', startIndex);
    }
  }

  for (uint8_t i = 0; i < 16; i++) {
    curText = displayArray[i];
    curZone = i;
    uint8_t inFX = (curFX + 1) % ARRAY_SIZE(effect);
    uint8_t outFX = (curFX + 1) % ARRAY_SIZE(effect);

    PRINT("\nNew Z", curZone);
    PRINT(": ", pc[curText]);
    PRINT(" @ ", millis());
    P.displayZoneText(curZone, pc[curText], PA_CENTER, SPEED_TIME, PAUSE_TIME, effect[inFX], effect[outFX]);
  }
  while (!P.getZoneStatus(curZone))
    P.displayAnimate();
}
