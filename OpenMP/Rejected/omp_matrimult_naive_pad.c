#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char **argv) {

   if (argc < 3) {
      printf("Errore: devi passare N e NUM_THREADS come argomenti!\n");
      return 1;
  }
  
  int n = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  
  // Padding al multiplo di 8 per riempire i registri AVX2 e allineare le cache line
  int n_pad = (n + 7) & ~7; 
  
  omp_set_num_threads(num_threads);
  
  // Memoria totale per la matrice appiattita
  size_t total_elements = (size_t)n_pad * n_pad;
  size_t bytes = total_elements * sizeof(double);
  size_t aligned_bytes = (bytes + 63) & ~63;

  // OPZIONE NUCLEARE 1: Array 1D puri. 
  // Niente più puntatori a matrici dinamiche. È un blocco unico.
  double * restrict a = aligned_alloc(64, aligned_bytes);
  double * restrict b = aligned_alloc(64, aligned_bytes);
  double * restrict c = aligned_alloc(64, aligned_bytes);

  if (!a || !b || !c) {
      printf("Errore di memoria!\n");
      return 1;
  }

  // Inizializzazione (adattata per 1D)
  #pragma omp parallel for
  for (int i = 0; i < n_pad; i++) {
     for (int j = 0; j < n_pad; j++) {
        int idx = i * n_pad + j;
        if (i < n && j < n) {
            a[idx] = 2.0;
            b[idx] = 3.0;
        } else {
            a[idx] = 0.0;
            b[idx] = 0.0;
        }
        c[idx] = 0.0;
     }
  }
  
  double start_time = omp_get_wtime(); 

  // Calcolo con accesso 1D e pragma SIMD forzato
  #pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < n_pad; ++i) {
     for (int k = 0; k < n_pad; k++) {
        
        // OPZIONE NUCLEARE 2: Diciamo al compilatore "Stai zitto e vettorizza, 
        // ti giuro che la memoria è allineata a 64 byte".
        #pragma omp simd aligned(a, b, c : 64)
        for (int j = 0; j < n_pad; ++j) {
           // Accesso manuale riga*colonna
           c[i * n_pad + j] += a[i * n_pad + k] * b[k * n_pad + j];
        }
     }
  }

  double end_time = omp_get_wtime(); 
  printf("Tempo: %f secondi\n", end_time - start_time);

  // Scrittura risultati
  FILE *f = fopen("mat-res.txt", "w");
  if (f) {
      fprintf(f, "%d\n\n", n);  
      int limit = (n < 1000) ? n : 1000;
      for (int row = 0; row < limit; row++) {
         for (int col = 0; col < limit; col++) {
            fprintf(f, "%.0f ", c[row * n_pad + col]);
         }
         fprintf(f, "\n");
      }
      fclose(f);
  }

  free(a);
  free(b);
  free(c);
  
  return 0;
}
