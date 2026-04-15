#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Min macro and size for the tiles
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define BLOCK_SIZE 128

int main(int argc, char **argv) {

    if (argc < 3) {
        printf("Error: you must provide n and NUM_THREADS as arguments!\n");
        return 1;
    }
    
    int n = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));

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

    // Parallelized Inizialization using collapse directive
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }

    // Timer 
    printf("Starting OpenMP computation...\n");
    double start_time = omp_get_wtime();

    // Matrix multiplication adopting loop tiling technique, collapse directive and a specific scheduling policy
    #pragma omp parallel for schedule(dynamic, 4) collapse(2)
    // External Loops
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                int i_end = MIN(ii + BLOCK_SIZE, n);
                int j_end = MIN(jj + BLOCK_SIZE, n);
                int k_end = MIN(kk + BLOCK_SIZE, n);
                // Internal Loops: computation in i-k-j order
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        for (int j = jj; j < j_end; j++) {
                            c[i][j] += a[i][k] * b[k][j];
                        }   
                    }
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    printf("Total Execution Time: %f seconds\n", end_time - start_time);


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
