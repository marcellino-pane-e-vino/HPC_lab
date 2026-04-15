#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Necessary library
#include <omp.h>

int main(int argc, char **argv) {

   if (argc < 3) {
      printf("Error: you must provide n and NUM_THREADS as arguments!\n");
      return 1;
   }
  
   int n = atoi(argv[1]);
   int num_threads = atoi(argv[2]);
  
   // Impose OpenMP how many threads to use for this run
   omp_set_num_threads(num_threads);
  

   // Aligned memory allocation to 64-byte and "restrict" pointers
   size_t alignment = 64;
   size_t matrix_size = sizeof(double[n][n]);

   if (matrix_size % alignment != 0) {
      matrix_size = ((matrix_size / alignment) + 1) * alignment;
   }

   double (* restrict a)[n] = aligned_alloc(alignment, matrix_size);
   double (* restrict b)[n] = aligned_alloc(alignment, matrix_size);
   double (* restrict c)[n] = aligned_alloc(alignment, matrix_size);

   if (!a || !b || !c) {
         printf("Error: insufficient memory for N=%d!\n", n);
         return 1;
   }

   printf("Parallel inizialization with %d threads...\n", num_threads);
   
   // Parallelized initialization
   #pragma omp parallel for
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
         a[i][j] = 2.0;
         b[i][j] = 3.0;
         c[i][j] = 0.0;
      }
   }

   // Timer for the computation: using the specific omp_get_wtime function
   printf("Starting the computation...\n");
   double start_time = omp_get_wtime(); 


   // Handling the matrix multiplication making explicit the policy scheduling 
   #pragma omp parallel for schedule(guided)
   for (int i = 0; i < n; ++i) {
      for (int k = 0; k < n; k++) {
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

   // Memory deallocation
   free(a);
   free(b);
   free(c);
   
   return 0;
}
