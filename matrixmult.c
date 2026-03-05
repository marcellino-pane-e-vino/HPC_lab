#define n 5000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <time.h>


int main(int argc,char **argv) {
  int i, j, k;

  double ( *a )[n] = malloc(sizeof(double[n][n]));
  double ( *b )[n] = malloc(sizeof(double[n][n]));
  double ( *c )[n] = malloc(sizeof(double[n][n]));

  // initialization
  for (i=0; i<n; i++)
     for (j=0; j<n; j++) {
        a[i][j] = 2.0;
        b[i][j] = 3.0;
        c[i][j] = 0.0;
     }

   printf("Starting the computation...\n");
  clock_t start = clock();
  // computation
  // note: loop order i-k-j instead of the standard i-j-k --> optimization technique: linear access memory & efficient cache use
  for (i=0; i<n; ++i)
     for (k=0; k<n; k++)
        for (j=0; j<n; ++j)
           c[i][j] += a[i][k]*b[k][j];

   clock_t end = clock();


   double duration = (double)(end - start) / CLOCKS_PER_SEC;
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


// Starting point: sequential complexity is O(n^3)

