// Standard library inclusions for I/O operations, memory allocation, math, and timing
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc,char **argv) {
   // Ensure the user provides the matrix dimension 'n' as a command-line argument
   if (argc < 2) {
        fprintf(stderr, "Error: missing n argument.\n");
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }

    // Convert the argument to an integer and validate it
    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Error: You forgot to provide n!.\n");
        return 1;
    }

  int i, j, k;

  // Dynamically allocate memory for three n x n matrices. 
  // Using pointers to Variable Length Arrays (VLAs) to handle large sizes and prevent stack overflow.
  double ( *a )[n] = malloc(sizeof(double[n][n]));
  double ( *b )[n] = malloc(sizeof(double[n][n]));
  double ( *c )[n] = malloc(sizeof(double[n][n]));

  // initialization
  // Populate matrices: 'a' and 'b' get constant values, 'c' (the result matrix) is zeroed out.
  for (i=0; i<n; i++)
     for (j=0; j<n; j++) {
        a[i][j] = 2.0;
        b[i][j] = 3.0;
        c[i][j] = 0.0;
     }

   printf("Starting the computation...\n");
  clock_t start = clock(); // Start the timer
  
  // computation
  // note: loop order i-k-j instead of the standard i-j-k --> optimization technique: linear access memory & efficient cache use
  // Explanation: C stores matrices in row-major order. The i-k-j order ensures that the innermost loop accesses contiguous memory locations for both 'c' and 'b', drastically reducing CPU cache misses.
  for (i=0; i<n; ++i)
     for (k=0; k<n; k++)
        for (j=0; j<n; ++j)
           c[i][j] += a[i][k]*b[k][j];

   clock_t end = clock(); // Stop the timer


   // Calculate and print the total execution time in seconds
   double duration = (double)(end - start) / CLOCKS_PER_SEC;
   printf("Execution Time: %.4f seconds\n", duration);

  // Open a file to save the results.
  FILE *f = fopen("mat-res.txt", "w");
  if (!f) {
     perror("fopen");
      return 1;
  }

  fprintf(f, "%d\n\n", n);  
  
  // Write a 1000x1000 chunk of the result matrix to the file.
  // Note: This block assumes 'n' is at least 1000; otherwise, it will cause an out-of-bounds access.
  for (int i = 0; i < 1000; i++) {
     for (int j = 0; j < 1000; j++) {
        fprintf(f, "%.0f ", c[i][j]);
     }
     fprintf(f, "\n");
  }

  fclose(f); // Close the file stream

  // Free the dynamically allocated memory to prevent memory leaks
  free(a);
  free(b);
  free(c);
  
  return 0;
}