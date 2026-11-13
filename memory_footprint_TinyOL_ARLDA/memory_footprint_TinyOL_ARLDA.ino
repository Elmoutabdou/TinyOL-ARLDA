/*
  memory_footprint_TinyOL-ARLDA.ino
  ----------------------------------
  Mesure l'empreinte mémoire RAM (tas + pile) du modèle TinyOL-ARLDA
  embarqué, pour la réponse au Reviewer 2, commentaire 1 (Discover AI, R2).

  Plateforme : Elecrow Mbits (ESP32-WROVER-B, 8 MB PSRAM)
  Méthode    : delta de heap libre avant/après allocation + initialisation
               du modèle, mesuré avec heap_caps_get_free_size() ; stack
               mesurée via high-water mark FreeRTOS (uxTaskGetStackHighWaterMark).

  MODE D'EMPLOI :
  1) Remplacez le bloc "===== MODELE TinyOL-ARLDA =====" ci-dessous par
     votre propre classe/structure (celle utilisée en production), ou
     incluez votre .h existant à la place de la structure de démonstration.
  2) Flashez sur le Mbits, ouvrez le moniteur série à 115200 bauds.
  3) Notez les 3 valeurs affichées :
       - "Empreinte modele (heap)"   -> à citer dans le papier
       - "Stack high-water mark"      -> marge de sécurité pile
       - "Flash (sketch + libs)"      -> taille programme (pour info)
  4) Répétez la mesure 3-5 fois (reset entre chaque) pour vérifier la
     stabilité de la valeur avant de la reporter dans le manuscrit.
*/

#include <Arduino.h>
#include "esp_heap_caps.h"

// Nombre de features et de classes utilisés par TinyOL-ARLDA
// (rms, zcr, skewness, kurtosis -> 4 features ; normal/anormal -> 2 classes)
#define N_FEATURES 4
#define N_CLASSES  2

// ===================== MODELE TinyOL-ARLDA =====================
// Remplacez ce bloc par votre implémentation réelle (mêmes tailles
// de buffers que le firmware de production) si elle diffère.
struct TinyOL_ARLDA {
  float mean[N_CLASSES][N_FEATURES];       // moyennes par classe
  float cov[N_FEATURES][N_FEATURES];       // covariance partagée (regularisee)
  float lambda;                            // parametre de regularisation adaptatif
  float beta;                              // taux EMA
  float kappa_ref;                         // reference du conditionnement
  uint32_t n_updates;

  void begin() {
    memset(mean, 0, sizeof(mean));
    memset(cov, 0, sizeof(cov));
    // cible : tr(Sigma)/D * I  (cf. correction R2 du shrinkage)
    for (int i = 0; i < N_FEATURES; i++) cov[i][i] = 1.0f;
    lambda    = 0.01f;   // valeur initiale (grille : beta=0.15, lam in [0.001,0.05])
    beta      = 0.15f;
    kappa_ref = 100.0f;
    n_updates = 0;
  }

  // Mise a jour incrementale (EMA + regle sigmoide lambda_t) -- stub
  void update(const float x[N_FEATURES], int label) {
    for (int i = 0; i < N_FEATURES; i++) {
      mean[label][i] += beta * (x[i] - mean[label][i]);
    }
    n_updates++;
  }

  int predict(const float x[N_FEATURES]) {
    // Discriminant LDA simplifie -- stub pour la mesure memoire uniquement
    float best_score = -1e30f;
    int best_class = 0;
    for (int c = 0; c < N_CLASSES; c++) {
      float score = 0.0f;
      for (int i = 0; i < N_FEATURES; i++) {
        float d = x[i] - mean[c][i];
        score -= d * d;
      }
      if (score > best_score) { best_score = score; best_class = c; }
    }
    return best_class;
  }
};
// =================================================================

TinyOL_ARLDA *model = nullptr;

size_t freeHeapBefore = 0;
size_t freeHeapAfterAlloc = 0;

void printMemoryReport() {
  Serial.println(F("\n================ RAPPORT MEMOIRE ================"));

  Serial.printf("Heap libre AVANT init modele   : %10u octets\n", (unsigned)freeHeapBefore);
  Serial.printf("Heap libre APRES init modele   : %10u octets\n", (unsigned)freeHeapAfterAlloc);

  long footprint = (long)freeHeapBefore - (long)freeHeapAfterAlloc;
  Serial.printf("Empreinte modele (heap)        : %10ld octets  (%.2f KB)\n",
                footprint, footprint / 1024.0f);

  Serial.printf("Taille statique sizeof(model)  : %10u octets  (%.2f KB)\n",
                (unsigned)sizeof(TinyOL_ARLDA), sizeof(TinyOL_ARLDA) / 1024.0f);

  UBaseType_t hwm = uxTaskGetStackHighWaterMark(NULL);
  Serial.printf("Stack high-water mark (loop)   : %10u mots libres restants\n", (unsigned)hwm);

  Serial.printf("Flash (sketch)                 : %10u octets  (%.2f KB)\n",
                (unsigned)ESP.getSketchSize(), ESP.getSketchSize() / 1024.0f);
  Serial.printf("Flash libre restante            : %10u octets  (%.2f KB)\n",
                (unsigned)ESP.getFreeSketchSpace(), ESP.getFreeSketchSpace() / 1024.0f);

  Serial.printf("PSRAM libre (si utilisee)      : %10u octets\n",
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  Serial.println(F("===================================================\n"));
  Serial.println(F("-> A reporter dans le manuscrit : la ligne 'Empreinte modele (heap)'."));
  Serial.println(F("   Repetez la mesure 3-5x (reset entre chaque) pour verifier la stabilite."));
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println(F("Mesure de l'empreinte memoire TinyOL-ARLDA -- Elecrow Mbits (ESP32-WROVER-B)"));

  // 1) Heap libre avant toute allocation liee au modele
  freeHeapBefore = heap_caps_get_free_size(MALLOC_CAP_8BIT);

  // 2) Allocation + initialisation du modele (comme au demarrage reel du firmware)
  model = new TinyOL_ARLDA();
  model->begin();

  // 3) Quelques mises a jour + inferences pour s'assurer qu'aucune allocation
  //    dynamique supplementaire n'apparait en cours de fonctionnement
  float dummy[N_FEATURES] = {0.1f, 0.2f, 0.3f, 0.4f};
  for (int i = 0; i < 50; i++) {
    model->update(dummy, i % N_CLASSES);
    model->predict(dummy);
  }

  freeHeapAfterAlloc = heap_caps_get_free_size(MALLOC_CAP_8BIT);

  printMemoryReport();
}

void loop() {
  // Rien a faire : rapport imprime une seule fois dans setup().
  // Decommentez pour re-imprimer periodiquement et verifier la stabilite :
  // delay(5000);
  // printMemoryReport();
}
