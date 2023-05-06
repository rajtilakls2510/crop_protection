#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#define ALARM_LED 26
#define PIR_OUT 27
#define TRIGGER_PIN 5
#define ECHO_PIN 18
#define SOUND_SPEED 0.034
#define CM_TO_INCH 0.393701
#define MODULE_ID 1
const char* ssid = "RAJTILAK-DELL 8168";  
const char* password = "25102015";  

const char* serverName = "http://192.168.137.41:5000/module";
StaticJsonDocument<200> doc;


long ultrason_duration;
float distance_cm;
int count= 0;
// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pin LED_BUILTIN as an output.

  pinMode(ALARM_LED, OUTPUT);
  pinMode(PIR_OUT, INPUT);
  digitalWrite(ALARM_LED, LOW);
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  Serial.begin(9600);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) 
  {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected..!");
  Serial.print("Got IP: ");  Serial.println(WiFi.localIP());

}

// the loop function runs over and over again forever
void loop() {

  // PIR Logic
  bool pir_out = digitalRead(PIR_OUT);

  // Ultrasonic Logic
  digitalWrite(TRIGGER_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGGER_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER_PIN, LOW);
  ultrason_duration = pulseIn(ECHO_PIN, HIGH);
  distance_cm = ultrason_duration * SOUND_SPEED/2.0;
  Serial.printf("%d Distance (cm): %f\n", count, distance_cm);


  if(WiFi.status()== WL_CONNECTED)
  {
    WiFiClient client;
    HTTPClient http;
    http.begin(client, serverName);

    http.addHeader("Content-Type", "application/x-www-form-urlencoded");
    String httpRequestData = "module="+String(MODULE_ID)+"&motion="+ String(pir_out) + "&distance="+ String(distance_cm)+"&";
    Serial.println(httpRequestData);
    int httpResponseCode = http.POST(httpRequestData);

    if (httpResponseCode==200) 
    {
      Serial.print("HTTP Response code: ");
      Serial.println(httpResponseCode);
      String payload = http.getString();
      DeserializationError error = deserializeJson(doc, payload);
      if (error) 
      {
        Serial.print(F("deserializeJson() failed: "));
        Serial.println(error.f_str());
        return;
      }
      int sev = doc["severity"];
      if(sev)
        digitalWrite(ALARM_LED, HIGH);
      else
       digitalWrite(ALARM_LED, LOW);
      Serial.println(sev);
    }
    else 
    {
      Serial.print("Error code: ");
      Serial.println(httpResponseCode);
    }

    http.end(); 
  }

  delay(1000);


}

