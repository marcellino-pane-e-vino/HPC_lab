#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// Definition of a Macro to replace the usage of a system function
#define min(a,b) (((a) < (b)) ? (a) : (b))       

int main(int argc,char **argv) {

    // Timer for Total Execution Time
    printf("Starting the computation...\n");
    clock_t start_clock_total = clock();                  
    
    int i, j, k;

    if (argc < 2) {
      fprintf(stderr, "Error: missing n argument.\n");
      return 1;
    }

    // Parameters: matrix size and tile size
    int n = atoi(argv[1]);          
    int B;

    if (n <= 0) {
      fprintf(stderr, "Error: provided an invalid n!.\n");
      return 1;
    }

    // Default choice for B if it is not inserted
    if (argc >= 3) 
        B = atoi(argv[2]);
     else 
        B = 64;


    // Memory alignment and allocation
    // 64 bytes to match the Cache Line size and SIMD vectorization
    size_t alignment = 64;
    size_t matrix_size = sizeof(double[n][n]);

    // Padding: force the matrix_size to be a multiple of the alignment
    if (matrix_size % alignment != 0) {
        matrix_size = ((matrix_size / alignment) + 1) * alignment;
    }
    
    // Memory allocation
    double (* a)[n] = aligned_alloc(alignment, matrix_size);
    double (* b)[n] = aligned_alloc(alignment, matrix_size);
    double (* c)[n] = aligned_alloc(alignment, matrix_size);


    // Initialization
    for (i=0; i<n; i++)
        for (j=0; j<n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    

    // Timer for Critical Section
    printf("Starting the critical section...\n");
    clock_t start_clock = clock();                  

    int ii, kk, jj;
    int i_limit, k_limit, j_limit;
    double temp_a;
    
    // Matrix multiplication through loop tiling technique
    // External loops: Block level, navigation throgh the tiles with a step of B
    for (ii = 0; ii < n; ii += B) {                
        i_limit = min(ii + B, n);
        for (kk = 0; kk < n; kk += B) {         
            k_limit = min(kk + B, n);
            for (jj = 0; jj < n; jj += B) {     
                j_limit = min(jj + B, n);

                // Internal loops: computation within the single Tile 
                for (i = ii; i < i_limit; i++) {            
                    for (k = kk; k < k_limit; k++) {
                        //temp_a = a[i][k];                     // Register Promotion technique
                        for (j = jj; j < j_limit; j++) {    
                            //c[i][j] += temp_a * b[k][j];
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }
    }

    clock_t end_clock = clock();                
    double duration = (double)(end_clock - start_clock) / CLOCKS_PER_SEC;   
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

    clock_t end_clock_total = clock();               
    double total_time = (double)(end_clock_total - start_clock_total) / CLOCKS_PER_SEC;   
    printf("Total Execution Time: %.4f seconds\n", total_time);

    return 0;
}
