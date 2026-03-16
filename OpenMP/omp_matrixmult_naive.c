#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char **argv) {
  // Controlliamo che Python ci abbia passato sia N che i Thread
  if (argc < 3) {
      printf("Errore: devi passare N e NUM_THREADS come argomenti!\n");
      return 1;
  }
  
  // Leggiamo N e il numero di thread da terminale
  int n = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  
  // Imponiamo a OpenMP quanti thread usare per questo run
  omp_set_num_threads(num_threads);

  int i, j, k;
  
  // Allocazione dinamica
  double (*a)[n] = malloc(sizeof(double[n][n]));
  double (*b)[n] = malloc(sizeof(double[n][n]));
  double (*c)[n] = malloc(sizeof(double[n][n]));

  if (!a || !b || !c) {
      printf("Errore: Memoria insufficiente per N=%d!\n", n);
      return 1;
  }

  printf("Inizializzazione parallela con %d thread...\n", num_threads);
  
  #pragma omp parallel for shared(a, b, c) private(i, j) schedule(static)
  for (i = 0; i < n; i++) {
     for (j = 0; j < n; j++) {
        a[i][j] = 2.0;
        b[i][j] = 3.0;
        c[i][j] = 0.0;
     }
  }

  printf("Starting the computation...\n");
  
  double start_time = omp_get_wtime(); 

  // Qui gestiamo i thread sui vari core
  #pragma omp parallel for shared(a, b, c) private(i, j, k) schedule(static)
  for (i = 0; i < n; ++i) {
     for (k = 0; k < n; k++) {
        
        // Qui forziamo la vettorizzazione SIMD all'interno del singolo core!
        #pragma omp simd
        for (j = 0; j < n; ++j) {
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