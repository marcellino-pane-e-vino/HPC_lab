#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char **argv) {

   if (argc < 3) {
      printf("Errore: devi passare N e NUM_THREADS come argomenti!\n");
      return 1;
  }
  
  // Leggiamo N e il numero di thread da terminale
  int n = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  
  // --- PADDING LOGIC ---
  // Arrotondiamo N al multiplo di 8 successivo. 
  // Garantisce al compilatore vettorizzazione pura, senza remainder loop.
  int n_pad = (n + 7) & ~7; 
  
  // Imponiamo a OpenMP quanti thread usare per questo run
  omp_set_num_threads(num_threads);
  
  /********* IMPROVEMENT *********/
  // 1. Calcoliamo la memoria esatta richiesta dalla matrice PADDATA
  size_t bytes = sizeof(double[n_pad][n_pad]);
  
  // 2. Arrotondiamo la dimensione al multiplo di 64 successivo
  // (Requisito rigido dello standard C per aligned_alloc)
  size_t aligned_bytes = (bytes + 63) & ~63;

  // 3. Allocazione con memoria allineata a 64-byte e puntatori "restrict".
  // NOTA BENE: Ora usiamo n_pad per definire la dimensione contigua della riga!
  double (* restrict a)[n_pad] = aligned_alloc(64, aligned_bytes);
  double (* restrict b)[n_pad] = aligned_alloc(64, aligned_bytes);
  double (* restrict c)[n_pad] = aligned_alloc(64, aligned_bytes);

  if (!a || !b || !c) {
      printf("Errore: Memoria insufficiente per N=%d (Allocati per n_pad=%d)!\n", n, n_pad);
      return 1;
  }

  printf("Inizializzazione parallela con %d thread...\n", num_threads);
  
  #pragma omp parallel for
  for (int i = 0; i < n_pad; i++) {
     for (int j = 0; j < n_pad; j++) {
        // Riempiamo con dati reali la sottomatrice utile [0..n-1].
        // Tutto il "cuscinetto" extra viene inizializzato rigorosamente a 0.0.
        if (i < n && j < n) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
        } else {
            a[i][j] = 0.0;
            b[i][j] = 0.0;
        }
        c[i][j] = 0.0;
     }
  }

  printf("Starting the computation...\n");
  
  double start_time = omp_get_wtime(); 

  // Qui gestiamo i thread sui vari core
  #pragma omp parallel for schedule(dynamic, 4)
  // I cicli ora iterano su n_pad. Il compilatore ringrazia!
  for (int i = 0; i < n_pad; ++i) {
     for (int k = 0; k < n_pad; k++) {
        for (int j = 0; j < n_pad; ++j) {
           c[i][j] += a[i][k] * b[k][j];
        }
     }
  }

  double end_time = omp_get_wtime(); 

  printf("Tempo di esecuzione OpenMP: %f secondi\n", end_time - start_time);

  FILE *f = fopen("mat-res.txt", "w");
  if (!f) {
     perror("fopen");
     return 1;
  }

  // L'utente finale e i test non si accorgeranno del padding.
  // Stampiamo N originale e leggiamo i risultati solo fino al limite reale.
  fprintf(f, "%d\n\n", n);  
  
  int limit = (n < 1000) ? n : 1000;
  for (int row = 0; row < limit; row++) {
     for (int col = 0; col < limit; col++) {
        fprintf(f, "%.0f ", c[row][col]);
     }
     fprintf(f, "\n");
  }

  fclose(f);
  free(a);
  free(b);
  free(c);
  
  return 0;
}