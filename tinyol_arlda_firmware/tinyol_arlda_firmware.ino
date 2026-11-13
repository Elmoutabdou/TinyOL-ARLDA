// ============================================================================
// TinyOL-ARLDA firmware — Elecrow Mbits (ESP32-WROVER-B)
// ----------------------------------------------------------------------------
// Supervised online Regularized LDA with:
//   * EMA updates of class means, priors and a SHARED DIAGONAL covariance
//   * corrected shrinkage toward the scaled identity: sigma <- (1-l)*sigma + l*mean(sigma)
//   * adaptive lambda from the condition number of the diagonal covariance:
//       kappa = max(sigma)/min(sigma)
//       lambda_t = L_MIN + (L_MAX - L_MIN) * sigmoid(ALPHA * (kappa - KAPPA_REF))
//   * the 4 features retained by the manuscript's selection study:
//       rms, zcr, skewness, kurtosis   (no FFT, no MFCC)
//
// This matches the Python evaluation notebook (TinyOL_ARLDA class, 4-feature
// mode, supervised test-then-train). Every number the paper reports for the
// embedded platform must come from THIS code.
//
// Microphone: analog MEMS on GPIO35 (12-bit ADC), quiescent ~1850. NOT I2S.
//
// Serial protocol (921600 baud), one command per line:
//   R            reset model to initial state
//   S            print status: hyperparameters, model params, free heap
//   M            print memory report (static buffers + free heap)
//   L            live mode: capture 1 s from mic, predict, print
//   T <0|1>      live capture + supervised update with the given label
//   F <0|1>      frame replay: PC then sends NUM_SAMPLES int16 (little-endian);
//                device extracts features, predicts, updates with label,
//                replies  "PRED <yhat> <margin> <t_feat_us> <t_pred_us> <t_upd_us>"
//   P <0|1>      same as F but predict only (no update) — for a frozen model
// ============================================================================

#include <Arduino.h>
#include <math.h>

// ---------------------------- configuration --------------------------------
#define MIC_PIN          35
#define SAMPLE_RATE_HZ   16000          // must match the desktop pipeline (SR)
#define FRAME_SEC        1
#define NUM_SAMPLES      (SAMPLE_RATE_HZ * FRAME_SEC)
#define PERIOD_US        (1000000UL / SAMPLE_RATE_HZ)
#define ADC_MID          1850.0f        // quiescent ADC value (Elecrow: 1800-1900)
#define ADC_FULL         4095.0f

#define D                4              // rms, zcr, skewness, kurtosis
#define K                2              // normal (0), abnormal (1)

// ---- hyperparameters: UPDATE from the notebook's tuning output (4-feature run) ----
#define BETA        0.15f
#define L_MIN       0.001f
#define L_MAX       0.05f
#define ALPHA       0.01f
#define KAPPA_REF   100.0f
#define SIGMA_INIT  0.05f
#define SIGMA_EPS   1e-8f

// ---------------------------- static memory --------------------------------
// The paper's memory table must be reconstructed from these + the heap report.
static int16_t audio[NUM_SAMPLES];     // 32,000 bytes at 16 kHz / 1 s
static float   feat[D];                //     16 bytes

struct Model {
  float mu[K][D];                      // 2 * 4 * 4 =  32 bytes
  float pi[K];                         //          8 bytes
  float sigma[D];                      //         16 bytes
  float lambda;                        //          4 bytes
  uint32_t n_updates;
};
static Model model;                    // ~64 bytes total

static unsigned long t_acq_us = 0, t_feat_us = 0, t_pred_us = 0, t_upd_us = 0;

// ---------------------------- model --------------------------------------
void modelReset() {
  for (int k = 0; k < K; k++) {
    for (int d = 0; d < D; d++) model.mu[k][d] = 0.0f;
    model.pi[k] = 1.0f / K;
  }
  for (int d = 0; d < D; d++) model.sigma[d] = SIGMA_INIT;
  model.lambda = L_MIN;
  model.n_updates = 0;
}

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// discriminant scores; returns argmax and writes margin (top1 - top2)
int predict(const float *x, float *margin_out) {
  float g[K];
  for (int k = 0; k < K; k++) {
    float s = 0.0f;
    for (int d = 0; d < D; d++) {
      float inv = 1.0f / (model.sigma[d] > SIGMA_EPS ? model.sigma[d] : SIGMA_EPS);
      s += x[d] * inv * model.mu[k][d] - 0.5f * model.mu[k][d] * inv * model.mu[k][d];
    }
    s += logf(model.pi[k] > SIGMA_EPS ? model.pi[k] : SIGMA_EPS);
    g[k] = s;
  }
  int best = (g[1] > g[0]) ? 1 : 0;
  if (margin_out) *margin_out = fabsf(g[1] - g[0]);
  return best;
}

// supervised EMA update with corrected shrinkage + sigmoid lambda
void update(const float *x, int y) {
  // --- adaptive lambda from condition number of the diagonal covariance (O(D)) ---
  float smax = model.sigma[0], smin = model.sigma[0];
  for (int d = 1; d < D; d++) {
    if (model.sigma[d] > smax) smax = model.sigma[d];
    if (model.sigma[d] < smin) smin = model.sigma[d];
  }
  float kappa = smax / (smin > SIGMA_EPS ? smin : SIGMA_EPS);
  float z = ALPHA * (kappa - KAPPA_REF);
  z = clampf(z, -50.0f, 50.0f);
  float sig = 1.0f / (1.0f + expf(-z));
  model.lambda = L_MIN + (L_MAX - L_MIN) * sig;

  // --- EMA of mean, prior, diagonal covariance (residual w.r.t. old mean) ---
  for (int d = 0; d < D; d++) {
    float r = x[d] - model.mu[y][d];              // residual with OLD mean
    model.mu[y][d] = (1.0f - BETA) * model.mu[y][d] + BETA * x[d];
    model.sigma[d]  = (1.0f - BETA) * model.sigma[d] + BETA * r * r;
  }
  model.pi[y] = (1.0f - BETA) * model.pi[y] + BETA;
  float psum = model.pi[0] + model.pi[1];
  model.pi[0] /= psum; model.pi[1] /= psum;

  // --- corrected shrinkage toward scaled identity: tau = mean(sigma) ---
  float tau = 0.0f;
  for (int d = 0; d < D; d++) tau += model.sigma[d];
  tau /= D;
  for (int d = 0; d < D; d++) {
    model.sigma[d] = (1.0f - model.lambda) * model.sigma[d] + model.lambda * tau;
    if (model.sigma[d] < SIGMA_EPS) model.sigma[d] = SIGMA_EPS;
  }
  model.n_updates++;
}

// ---------------------------- features -----------------------------------
// Exact counterparts of the desktop extract_features() for the 4 retained
// descriptors, including the same normalization constants.
void extractFeatures(const int16_t *s, int n, float *out) {
  const float inv = 1.0f / 32767.0f;
  // pass 1: mean, sum of squares, zero crossings
  double mean = 0.0, ss = 0.0;
  uint32_t zc = 0;
  float prev = s[0] * inv;
  for (int i = 0; i < n; i++) {
    float v = s[i] * inv;
    mean += v; ss += (double)v * v;
    if (i > 0 && (prev * v) < 0.0f) zc++;
    prev = v;
  }
  mean /= n;
  float rms = sqrtf((float)(ss / n));
  float zcr = (float)zc / (float)(n - 1);
  // pass 2: central moments
  double m2 = 0.0, m3 = 0.0, m4 = 0.0;
  for (int i = 0; i < n; i++) {
    double dv = s[i] * inv - mean;
    double d2 = dv * dv;
    m2 += d2; m3 += d2 * dv; m4 += d2 * d2;
  }
  m2 /= n; m3 /= n; m4 /= n;
  float sd = sqrtf((float)m2);
  float skew = (float)(m3 / (sd * sd * sd + 1e-12));
  float kurt = (float)(m4 / (sd * sd * sd * sd + 1e-12));
  // same normalization as the notebook
  out[0] = clampf(rms * 10.0f, 0.0f, 2.0f) / 2.0f;     // rms_n
  out[1] = clampf(zcr * 10.0f, 0.0f, 1.0f);            // zcr_n
  out[2] = clampf((skew + 3.0f) / 6.0f, 0.0f, 1.0f);   // skewness_n
  out[3] = clampf((kurt + 3.0f) / 10.0f, 0.0f, 1.0f);  // kurtosis_n
}

// ---------------------------- acquisition --------------------------------
// Busy-wait paced sampling on the analog mic. Converts ADC counts to the
// same amplitude scale the desktop pipeline sees: (adc - mid)/full in [-1,1],
// stored as int16 (x32767) so the feature code has a single input format.
void captureFrame() {
  unsigned long t0 = micros();
  for (uint32_t i = 0; i < NUM_SAMPLES; i++) {
    unsigned long target = t0 + (unsigned long)i * PERIOD_US;
    while ((long)(micros() - target) < 0) { }
    float v = ((float)analogRead(MIC_PIN) - ADC_MID) / ADC_FULL;
    v = clampf(v, -1.0f, 1.0f);
    audio[i] = (int16_t)(v * 32767.0f);
  }
  t_acq_us = micros() - t0;
}

// read NUM_SAMPLES int16 (little-endian) from serial into audio[]
bool readFrameFromSerial() {
  size_t need = NUM_SAMPLES * sizeof(int16_t);
  size_t got = 0;
  unsigned long start = millis();
  uint8_t *p = (uint8_t *)audio;
  while (got < need) {
    if (Serial.available()) {
      got += Serial.readBytes(p + got, need - got);
    }
    if (millis() - start > 10000) return false;   // 10 s timeout
  }
  return true;
}

// ---------------------------- reporting -----------------------------------
void printStatus() {
  Serial.println("STATUS");
  Serial.printf("  D=%d K=%d SR=%d BETA=%.3f L_MIN=%.4f L_MAX=%.4f ALPHA=%.4f KAPPA_REF=%.1f\n",
                D, K, SAMPLE_RATE_HZ, BETA, L_MIN, L_MAX, ALPHA, KAPPA_REF);
  Serial.printf("  updates=%lu lambda=%.5f pi=[%.3f %.3f]\n",
                (unsigned long)model.n_updates, model.lambda, model.pi[0], model.pi[1]);
  for (int k = 0; k < K; k++)
    Serial.printf("  mu[%d]=[%.4f %.4f %.4f %.4f]\n", k,
                  model.mu[k][0], model.mu[k][1], model.mu[k][2], model.mu[k][3]);
  Serial.printf("  sigma=[%.5f %.5f %.5f %.5f]\n",
                model.sigma[0], model.sigma[1], model.sigma[2], model.sigma[3]);
  Serial.printf("  free_heap=%u\n", ESP.getFreeHeap());
}

void printMemory() {
  Serial.println("MEMORY");
  Serial.printf("  audio_buffer_bytes=%u\n", (unsigned)sizeof(audio));
  Serial.printf("  feature_vector_bytes=%u\n", (unsigned)sizeof(feat));
  Serial.printf("  model_bytes=%u\n", (unsigned)sizeof(model));
  Serial.printf("  static_total_bytes=%u\n",
                (unsigned)(sizeof(audio) + sizeof(feat) + sizeof(model)));
  Serial.printf("  free_heap=%u min_free_heap=%u\n", ESP.getFreeHeap(), ESP.getMinFreeHeap());
  Serial.println("  (note: the 1 s audio buffer at 16 kHz is 32,000 bytes; report it as such)");
}

void runFrame(int label, bool do_update, bool from_serial) {
  if (from_serial) {
    if (!readFrameFromSerial()) { Serial.println("ERR frame timeout"); return; }
    t_acq_us = 0;
  } else {
    captureFrame();
  }
  unsigned long t0 = micros();
  extractFeatures(audio, NUM_SAMPLES, feat);
  t_feat_us = micros() - t0;

  float margin;
  t0 = micros();
  int yhat = predict(feat, &margin);
  t_pred_us = micros() - t0;

  t_upd_us = 0;
  if (do_update && (label == 0 || label == 1)) {
    t0 = micros();
    update(feat, label);
    t_upd_us = micros() - t0;
  }
  Serial.printf("PRED %d %.4f %lu %lu %lu %lu\n", yhat, margin,
                t_acq_us, t_feat_us, t_pred_us, t_upd_us);
}

// ---------------------------- main ---------------------------------------
void setup() {
  Serial.begin(921600);
  delay(200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  modelReset();
  Serial.println("READY TinyOL-ARLDA supervised D=4");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;
  char cmd = line.charAt(0);
  int arg = -1;
  if (line.length() >= 3) arg = line.substring(2).toInt();

  switch (cmd) {
    case 'R': modelReset(); Serial.println("OK reset"); break;
    case 'S': printStatus(); break;
    case 'M': printMemory(); break;
    case 'L': runFrame(-1, false, false); break;
    case 'T': runFrame(arg, true,  false); break;
    case 'F': runFrame(arg, true,  true);  break;
    case 'P': runFrame(arg, false, true);  break;
    default:  Serial.println("ERR unknown cmd"); break;
  }
}
