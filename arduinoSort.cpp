/*
  ================================================================
  ECOSORT - FULL ARDUINO UNO R3 FIRMWARE
  ================================================================
  - Motor runs continuously (sorted items flow past).
  - IR break-beam triggers camera capture ('CAPTURE' over Serial).
  - Receives commands: 'O' (Organic), 'T' (Trash), 'R' (Recycle), 'U' (Idle).
  - Sends bin full alerts: "FULL:ORGANIC", "FULL:TRASH", "FULL:RECYCLING".
  - Servos use a "nudge/bounce" motion to ensure items are pushed off.
  - Designed to work with ecosort_pc_standalone.py (PC backend).

  PIN MAP (Uno R3):
  -----------------
  IN1 (Pin 8)  : L298N Motor (always HIGH)
  IN2 (Pin 9)  : L298N Motor (always LOW)

  SERVO1 (Pin 5) : Organic (45°) and Trash (110°) flipper
  SERVO2 (Pin 6) : Recycling (45°) flipper

  IR_CAMERA (Pin 4) : Break-beam sensor (LOW = beam broken)

  Ultrasonics (Trig, Echo):
    Organic  : (Pin 2, Pin 3)
    Trash    : (Pin 10, Pin 11)
    Recycle  : (Pin 12, Pin 13)

  Serial: 9600 baud
  ================================================================
*/

#include <Servo.h>

// ================================================================
// 1. TUNING CONSTANTS (ADJUST THESE IF MECHANICAL ANGLES ARE OFF)
// ================================================================
/*
   🔧 HOW TO TWEAK:
   - If a servo doesn't push the item far enough, INCREASE the target angle.
   - If a servo pushes the item off the belt too early or hits the belt,
     DECREASE the target angle.
   - The "Closed" angle should be 0 (fully retracted). If your servo
     jitters at 0, try 5 or 10.
*/
const int S1_CLOSED = 0;       // Servo 1 fully retracted (idle)
const int S1_ORGANIC = 45;     // Servo 1 push for Organic (adjust 30–60)
const int S1_TRASH = 110;      // Servo 1 push for Trash (adjust 90–130)

const int S2_CLOSED = 0;       // Servo 2 fully retracted (idle)
const int S2_RECYCLING = 45;   // Servo 2 push for Recycling (adjust 30–60)

// Nudge / Bounce settings
const int NUDGE_CYCLES = 3;    // How many times the servo bounces
const int NUDGE_PULLBACK = 15; // Degrees to pull back during each nudge

// Full bin distance threshold (cm)
const int FULL_DISTANCE_CM = 8;

// ================================================================
// 2. PIN DEFINITIONS
// ================================================================
// Motor (always running)
const int IN1 = 8;
const int IN2 = 9;

// Servos
Servo servo1;          // Organic & Trash (flips left/centre)
Servo servo2;          // Recycling (flips right)
const int SERVO1_PIN = 5;
const int SERVO2_PIN = 6;

// IR Break-Beam (Camera Trigger)
const int IR_CAMERA_PIN = 4;   // LOW when beam is broken

// Ultrasonic Sensors
const int TRIG_ORGANIC = 2, ECHO_ORGANIC = 3;
const int TRIG_TRASH   = 10, ECHO_TRASH   = 11;
const int TRIG_RECYCLE = 12, ECHO_RECYCLE = 13;

// ================================================================
// 3. GLOBAL VARIABLES
// ================================================================
bool wasPartPresent = false;          // IR state tracker (one‑shot trigger)
unsigned long lastSensorCheck = 0;
const unsigned long CHECK_INTERVAL = 1500; // Check bin levels every 1.5s

// ================================================================
// 4. HELPER FUNCTIONS
// ================================================================

// Read distance from HC-SR04 (cm)
int getDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 25000);  // 25ms timeout
  if (duration == 0) return 999;                  // No object
  return duration * 0.0343 / 2;                   // Convert to cm
}

// Check all 3 bins and send "FULL:..." over Serial if full
void checkBinLevels() {
  int distOrganic = getDistance(TRIG_ORGANIC, ECHO_ORGANIC);
  int distTrash   = getDistance(TRIG_TRASH, ECHO_TRASH);
  int distRecycle = getDistance(TRIG_RECYCLE, ECHO_RECYCLE);

  if (distOrganic <= FULL_DISTANCE_CM) Serial.println("FULL:ORGANIC");
  if (distTrash   <= FULL_DISTANCE_CM) Serial.println("FULL:TRASH");
  if (distRecycle <= FULL_DISTANCE_CM) Serial.println("FULL:RECYCLING");
}

/*
   🔧 BOUNCE / NUDGE FUNCTION
   This moves the servo to the target angle, pulls back slightly,
   and repeats NUDGE_CYCLES times. This helps dislodge stubborn items.
*/
void bounceServo(Servo &s, int closedAngle, int targetAngle) {
  for (int i = 0; i < NUDGE_CYCLES; i++) {
    // Push to target
    s.write(targetAngle);
    delay(250);
    
    // Pull back slightly to "nudge" the item
    int pullBack = max(0, targetAngle - NUDGE_PULLBACK);
    s.write(pullBack);
    delay(200);
  }
  // Final push and hold, then retract
  s.write(targetAngle);
  delay(500);
  s.write(closedAngle);
  delay(300);
}

// ================================================================
// 5. SETUP
// ================================================================
void setup() {
  Serial.begin(9600);

  // Motor – runs continuously
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);   // Runs forward

  // IR break-beam
  pinMode(IR_CAMERA_PIN, INPUT_PULLUP);

  // Servos
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo1.write(S1_CLOSED);
  servo2.write(S2_CLOSED);

  // Ultrasonics
  pinMode(TRIG_ORGANIC, OUTPUT); pinMode(ECHO_ORGANIC, INPUT);
  pinMode(TRIG_TRASH,   OUTPUT); pinMode(ECHO_TRASH,   INPUT);
  pinMode(TRIG_RECYCLE, OUTPUT); pinMode(ECHO_RECYCLE, INPUT);

  // Quick servo test on startup (shows they're working)
  delay(500);
  servo1.write(S1_ORGANIC);
  servo2.write(S2_RECYCLING);
  delay(300);
  servo1.write(S1_CLOSED);
  servo2.write(S2_CLOSED);

  Serial.println("EcoSort Arduino Ready.");
}

// ================================================================
// 6. MAIN LOOP
// ================================================================
void loop() {
  // ----- A) Bin Fullness Check (periodic) -----
  if (millis() - lastSensorCheck >= CHECK_INTERVAL) {
    lastSensorCheck = millis();
    checkBinLevels();
  }

  // ----- B) IR Camera Trigger (one-shot on item arrival) -----
  bool isPartPresent = (digitalRead(IR_CAMERA_PIN) == LOW);   // LOW = beam broken
  
  if (isPartPresent && !wasPartPresent) {
    // Item just arrived at camera position
    Serial.println("CAPTURE");
    
    // Brief pause to let item stabilise (optional)
    delay(100);
    
    // Wait for Pi/PC response (blocking)
    while (!Serial.available()) {
      // Wait for command from PC
    }
    
    char command = Serial.read();
    
    // Execute sorting command
    if (command == 'O' || command == 'o') {
      // Organic: Servo 1 pushes to Organic angle
      bounceServo(servo1, S1_CLOSED, S1_ORGANIC);
      Serial.println("Organic sorted.");
    }
    else if (command == 'T' || command == 't') {
      // Trash: Servo 1 pushes to Trash angle
      bounceServo(servo1, S1_CLOSED, S1_TRASH);
      Serial.println("Trash sorted.");
    }
    else if (command == 'R' || command == 'r') {
      // Recycling: Servo 2 pushes (S1 stays closed)
      servo1.write(S1_CLOSED);   // Ensure S1 doesn't interfere
      bounceServo(servo2, S2_CLOSED, S2_RECYCLING);
      Serial.println("Recycling sorted.");
    }
    else {
      // command == 'U' (Manual/Unsure) – close both paddles
      servo1.write(S1_CLOSED);
      servo2.write(S2_CLOSED);
      Serial.println("Manual review - no push.");
    }
  }
  
  // Update state for next loop
  wasPartPresent = isPartPresent;
  
  // ----- C) Handle any extra serial data (just in case) -----
  // (Main logic is handled inside the trigger block)
  
  // Small delay to prevent CPU overrun
  delay(10);
}