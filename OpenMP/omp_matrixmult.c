#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define n 5000

int main(int argc, char **argv) {
  int i, j, k;
  
  double (*a)[n] = malloc(sizeof(double[n][n]));
  double (*b)[n] = malloc(sizeof(double[n][n]));
  double (*c)[n] = malloc(sizeof(double[n][n]));

  printf("Inizializzazione parallela (NUMA First-Touch)...\n");
  
  // Usiamo static per mappare le righe in modo identico al calcolo successivo
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

  // Torniamo a SCHEDULE(STATIC) per azzerare l'overhead del runtime
  #pragma omp parallel for shared(a, b, c) private(i, j, k) schedule(static)
  for (i = 0; i < n; ++i) {
     for (k = 0; k < n; k++) {
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
  
  for (int row = 0; row < 1000; row++) {
     for (int col = 0; col < 1000; col++) {
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