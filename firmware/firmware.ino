#include <Servo.h>
#include "DHT.h"

#define DHTPIN 2
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

Servo sUpper, sLower;
const int PIN_UPPER = 3;
const int PIN_LOWER = 5;

const int UPPER_OPEN  = 90;   // <- was 180 (grinds). Use largest angle that clears w/o buzz
const int UPPER_CLOSE = 0;    // <- your calibrated value, not 90
const int LOWER_OPEN  = 90;   // <- was 180 (grinds)
const int LOWER_CLOSE = 0;   // <- your calibrated value, not 90


const unsigned long SETTLE_MS = 350;

unsigned long lastHumidity = 0;
const unsigned long HUM_INTERVAL = 2000;   // DHT22 minimum is 2 s

// Dispense state machine
enum State { IDLE, U_OPEN, U_CLOSE, L_OPEN, L_CLOSE };
State state = IDLE;
unsigned long stageStart = 0;

void closeBoth() {
  sUpper.write(UPPER_CLOSE);
  sLower.write(LOWER_CLOSE);
}

void startDispense() {
  if (state == IDLE) {           // ignore if already mid-dispense
    sUpper.write(UPPER_OPEN);
    stageStart = millis();
    state = U_OPEN;
  }
}

void updateDispense() {
  if (state == IDLE) return;
  if (millis() - stageStart < SETTLE_MS) return;   // current stage still settling
  stageStart = millis();
  switch (state) {
    case U_OPEN:  sUpper.write(UPPER_CLOSE); state = U_CLOSE; break;  // seal chamber
    case U_CLOSE: sLower.write(LOWER_OPEN);  state = L_OPEN;  break;  // drop to bay
    case L_OPEN:  sLower.write(LOWER_CLOSE); state = L_CLOSE; break;  // reseal
    case L_CLOSE: state = IDLE;                               break;  // done
    default:      state = IDLE;                               break;
  }
}

void setup() {
  Serial.begin(9600);
  dht.begin();
  sUpper.attach(PIN_UPPER);
  sLower.attach(PIN_LOWER);
  closeBoth();                   // boot into safe (closed) state
}

void loop() {
  // 1. Commands from LabVIEW
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if      (cmd == 'D' || cmd == 'd' || cmd == '1') startDispense();
    else if (cmd == 'T' || cmd == 't')               startDispense();   // same motion
    else if (cmd == 'S' || cmd == 's') { closeBoth(); state = IDLE; }    // abort now
  }

  // 2. Advance the dispense without blocking
  updateDispense();

  // 3. Humidity on its own clock — keeps flowing even mid-dispense
  unsigned long now = millis();
  if (now - lastHumidity >= HUM_INTERVAL) {
    lastHumidity = now;
    float h = dht.readHumidity();
    Serial.println(isnan(h) ? 0.0 : h);
  }
}