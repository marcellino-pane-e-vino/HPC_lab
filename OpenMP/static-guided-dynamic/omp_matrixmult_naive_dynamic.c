#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char **argv) {

   if (argc < 3) {
      printf("Error: you must provide N and NUM_THREADS as arguments!\n");
      return 1;
  }
  
  // We read N and the number of threads
  int n = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  
  // We fix the number of threads to be used
  omp_set_num_threads(num_threads);
  
  //double (*a)[n] = malloc(sizeof(double[n][n]));
  //double (*b)[n] = malloc(sizeof(double[n][n]));
  //double (*c)[n] = malloc(sizeof(double[n][n]));


  /********* IMPROVEMENT *********/
   // We get the exact memory needed requested by the matrix
  size_t bytes = sizeof(double[n][n]);
  
  // We round the dimension as multiple of 64: strong requisite of standard C for aligned_alloc
  size_t aligned_bytes = (bytes + 63) & ~63;

  // Aligned memory allocation to 64-byte and "restrict" pointers
  double (* restrict a)[n] = aligned_alloc(64, aligned_bytes);
  double (* restrict b)[n] = aligned_alloc(64, aligned_bytes);
  double (* restrict c)[n] = aligned_alloc(64, aligned_bytes);

  if (!a || !b || !c) {
      printf("Error: insufficient memory for N=%d!\n", n);
      return 1;
  }

  printf("Parallel inizialization with %d threads...\n", num_threads);
  
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
     for (int j = 0; j < n; j++) {
        a[i][j] = 2.0;
        b[i][j] = 3.0;
        c[i][j] = 0.0;
     }
  }

  printf("Starting the computation...\n");
  
  double start_time = omp_get_wtime(); 

  // Here we handle the threads accross the cores
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
     for (int k = 0; k < n; k++) {
        
        // We force SIMD vectorization inside each core
        //#pragma omp simd
        for (int j = 0; j < n; ++j) {
           c[i][j] += a[i][k] * b[k][j];
        }
     }
  }

  double end_time = omp_get_wtime(); 

  printf("OpenMP execution time: %f seconds\n", end_time - start_time);

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
