#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc,char **argv) {

   // Timer for Total Execution Time
   printf("Starting the computation...\n");
   clock_t start_total = clock();

   // Added for better parameter management
   if (argc < 2) {
      fprintf(stderr, "Error: missing n argument.\n");
      return 1;
   }

   int n = atoi(argv[1]);
   if (n <= 0) {
      fprintf(stderr, "Error: provided an invalid n!.\n");
      return 1;
   }

   int i, j, k;

   // Memory allocation 
   double ( *a )[n] = malloc(sizeof(double[n][n]));
   double ( *b )[n] = malloc(sizeof(double[n][n]));
   double ( *c )[n] = malloc(sizeof(double[n][n]));

   // Initialization
   for (i=0; i<n; i++)
      for (j=0; j<n; j++) {
         a[i][j] = 2.0;
         b[i][j] = 3.0;
         c[i][j] = 0.0;
      }


   // Timer for critical section: hotspot identification
   printf("Starting the critical section...\n");
   clock_t start_critical = clock();

   // Matrix multiplication through i-k-j order.
   for (i=0; i<n; ++i)
      for (k=0; k<n; k++)
         for (j=0; j<n; ++j)
            c[i][j] += a[i][k]*b[k][j];

   clock_t end_critical = clock();
   double duration = (double)(end_critical - start_critical) / CLOCKS_PER_SEC;
   printf("Execution Time for the Critical Section: %.4f seconds\n", duration);

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

   // Memory deallocation
   free(a);
   free(b);
   free(c);
   
   clock_t end_total = clock();
   double total_time = (double)(end_total - start_total) / CLOCKS_PER_SEC;
   printf("Total Execution Time: %.4f seconds\n", total_time);

   return 0;
}
