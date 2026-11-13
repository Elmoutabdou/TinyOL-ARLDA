// ============================================================
// mbits_record.ino
// Streams raw microphone samples from the Mbits (ESP32-WROVER-B)
// analog mic (GPIO35) to the PC over Serial, on command.
//
// The mic is ANALOG (not I2S) on this board -- confirmed from the
// Elecrow wiki: quiescent value ~1800-1900, drops when sound is
// detected. Read via analogRead(35).
//
// Protocol (one recording per 'R' command received over Serial):
//   PC  -> ESP32 : single byte 'R'
//   ESP32 -> PC  : "WAVSTART\n"
//                  <NUM_SAMPLES * 2 bytes, raw uint16_t, little-endian>
//                  "WAVEND\n"
//
// Sample rate is intentionally 8000 Hz (not 16000): reliably
// achievable with simple micros()-paced busy-wait timing on this
// board. The Python-side dataset pipeline already resamples any
// input file to its working rate (SR = 16000 in the notebook), so
// recording at 8000 Hz is transparent to the rest of the pipeline.
// ============================================================

#define MIC_PIN         35      // ADC1_CH7, matches Elecrow's own example
#define SAMPLE_RATE_HZ  8000
#define RECORD_SECONDS  5
#define NUM_SAMPLES     (SAMPLE_RATE_HZ * RECORD_SECONDS)
#define PERIOD_US       (1000000UL / SAMPLE_RATE_HZ)

uint16_t *buf;   // allocated on the regular internal heap (no PSRAM required)
                 // NUM_SAMPLES * 2 bytes = 80000 bytes for the 8000Hz/5s default,
                 // which comfortably fits in the ESP32's internal SRAM.

void setup() {
  Serial.begin(921600);
  delay(200);
  analogReadResolution(12);          // 0-4095, matches the 1800-1900 quiescent range
  analogSetAttenuation(ADC_11db);    // full 0-3.3V range

  Serial.print("Free heap before alloc: ");
  Serial.println(ESP.getFreeHeap());

  buf = (uint16_t *) malloc(NUM_SAMPLES * sizeof(uint16_t));
  if (buf == nullptr) {
    // Should not happen at this buffer size, but fail loudly and visibly
    // (over Serial) instead of hanging silently if it ever does.
    while (1) {
      Serial.println("ERROR: heap allocation failed");
      delay(1000);
    }
  }

  Serial.println("READY");   // PC waits for this line before sending 'R' commands
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'R') {
      recordAndSend();
    }
    // any other byte is silently ignored (e.g. stray newline/whitespace)
  }
}

void recordAndSend() {
  // --- Timed sampling: busy-wait paced by micros() for even spacing ---
  unsigned long t0 = micros();
  for (uint32_t i = 0; i < NUM_SAMPLES; i++) {
    unsigned long target = t0 + (unsigned long)i * PERIOD_US;
    while ((long)(micros() - target) < 0) {
      // spin-wait until the exact sample instant
    }
    buf[i] = analogRead(MIC_PIN);
  }

  // --- Send the buffer back to the PC ---
  Serial.println("WAVSTART");
  Serial.write((uint8_t *)buf, NUM_SAMPLES * sizeof(uint16_t));
  Serial.println("WAVEND");
}
