#include <DHT.h>

#define DHTPIN 15
#define DHTTYPE DHT22

#define POT_PIN 34

#define GREEN_LED 25
#define YELLOW_LED 26
#define RED_LED 27

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();

  pinMode(GREEN_LED, OUTPUT);
  pinMode(YELLOW_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
}

// 🔥 SIMULATED EDGE MODEL
float predictRUL(float temp, int vibration) {
  float temp_norm = temp / 100.0;
  float vib_norm = vibration / 4095.0;

  float rul = 120 - (temp_norm * 60 + vib_norm * 80);

  return rul;
}

String classify(float rul) {
  if (rul > 80) return "SAFE";
  else if (rul > 30) return "WARNING";
  else return "CRITICAL";
}

void loop() {
  float temp = dht.readTemperature();
  int vibration = analogRead(POT_PIN);

  float rul = predictRUL(temp, vibration);
  String state = classify(rul);

  digitalWrite(GREEN_LED, LOW);
  digitalWrite(YELLOW_LED, LOW);
  digitalWrite(RED_LED, LOW);

  if (state == "SAFE") digitalWrite(GREEN_LED, HIGH);
  else if (state == "WARNING") digitalWrite(YELLOW_LED, HIGH);
  else digitalWrite(RED_LED, HIGH);

  Serial.println("------");
  Serial.print("Temp: "); Serial.println(temp);
  Serial.print("Vibration: "); Serial.println(vibration);
  Serial.print("Predicted RUL: "); Serial.println(rul);
  Serial.print("State: "); Serial.println(state);

  delay(2000);
}