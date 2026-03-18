#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define min(a,b) (((a) < (b)) ? (a) : (b))          // Usage of a MACRO rather than a function should be better 
//#define n 10000
//#define B 64 

int main(int argc,char **argv) {

    printf("Starting the computation...\n");
    clock_t start_clock_total = clock();                  // start time
    
    int i, j, k;

    int n = atoi(argv[1]);          //matrix size
    int B ;                         //tile size

    // Default choice for B
    if (argc >= 3) 
        B = atoi(argv[2]);
     else 
        B = 64;

    
    size_t alignment = 64;
    size_t matrix_size = sizeof(double[n][n]);

    if (matrix_size % alignment != 0) {
        matrix_size = ((matrix_size / alignment) + 1) * alignment;
    }
    
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
         
    printf("Starting the critical section...\n");
    clock_t start_clock = clock();                  // start time

    int ii, kk, jj;
    int i_limit, k_limit, j_limit;
    double temp_a;
    
    
    // 1. Spostiamo JJ fuori e KK dentro per isolare l'aggiornamento di C
    for (ii = 0; ii < n; ii += B) {
        i_limit = min(ii + B, n);
        for (jj = 0; jj < n; jj += B) {
            j_limit = min(jj + B, n);
            for (kk = 0; kk < n; kk += B) {
                k_limit = min(kk + B, n);

                // Loop interni: qui avviene il calcolo vero e proprio
                for (i = ii; i < i_limit; i++) {
                    for (j = jj; j < j_limit; j++) {
                        
                        // ACCUMULATORE: Carichiamo C[i][j] in un registro una volta sola
                        double sum = c[i][j]; 
                        
                        for (k = kk; k < k_limit; k++) {
                            // RIUTILIZZO: sum resta nel registro della CPU
                            sum += a[i][k] * b[k][j];
                        }
                        
                        // SCRITTURA: Aggiorniamo la memoria solo alla fine del tile K
                        c[i][j] = sum; 
                    }
                }
            }
        }
    }


    clock_t end_clock = clock();                // end time

    double duration = (double)(end_clock - start_clock) / CLOCKS_PER_SEC;   
    printf("Critical section - Execution Time: %.4f seconds\n", duration);

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

    clock_t end_clock_total = clock();                // end time

    double total_time = (double)(end_clock_total - start_clock_total) / CLOCKS_PER_SEC;   
    printf("Total Execution Time: %.4f seconds\n", total_time);

    return 0;
}