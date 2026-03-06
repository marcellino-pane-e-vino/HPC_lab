#define n 5000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>    // 1. Inclusione obbligatoria

int main(int argc, char **argv) {
  
  // 2. Upgrade ad aligned_alloc e restrict per massimizzare l'efficienza SIMD
  size_t alignment = 64;
  size_t matrix_size = sizeof(double[n][n]);

  double (* a)[n] = aligned_alloc(alignment, matrix_size);
  double (* b)[n] = aligned_alloc(alignment, matrix_size);
  double (* c)[n] = aligned_alloc(alignment, matrix_size);

  // 3. Inizializzazione parallela (First-Touch Policy)
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i=0; i<n; i++) {
     for (int j=0; j<n; j++) {
        a[i][j] = 2.0;
        b[i][j] = 3.0;
        c[i][j] = 0.0;
     }
  }

  printf("Starting the computation...\n");
  
  // 4. Sostituzione di clock() con il timer Real-Time
  double start = omp_get_wtime();

  // computation
  // 5. MOTORE HPC BASE: Parallelizziamo SOLO il ciclo 'i' con schedule dynamic
  #pragma omp parallel for schedule(dynamic)
  for (int i=0; i<n; ++i) {
     for (int k=0; k<n; k++) {
        
        // 6. Forzatura della vettorizzazione sul ciclo più interno
        #pragma omp simd
        for (int j=0; j<n; ++j) {
           c[i][j] += a[i][k]*b[k][j];
        }
     }
  }

  double end = omp_get_wtime();

  double duration = end - start;
  printf("Execution Time: %.4f seconds\n", duration);

  FILE *f = fopen("mat-res.txt", "w");
  if (!f) {
     perror("fopen");
      return 1;
  }

  fprintf(f, "%d\n\n", n);  
  for (int i = 0; i < 1000; i++) {
     for (int j = 0; j < 1000; j++) {
        fprintf(f, "%.0f ", c[i][j]);
     }
     fprintf(f, "\n");
  }

  fclose(f);

  free(a);
  free(b);
  free(c);
  return 0;
}