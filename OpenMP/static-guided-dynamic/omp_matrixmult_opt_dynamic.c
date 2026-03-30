#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define BLOCK_SIZE 64
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Missing arguments\n");
        return 1;
    }
    
    int N = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));

    double * restrict a = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict b = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict c = (double *)aligned_alloc(64, N * N * sizeof(double));
    // Memory allocation for transposed B
    double * restrict b_t = (double *)aligned_alloc(64, N * N * sizeof(double));

    #pragma omp parallel //for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 2.0;
            b[i * N + j] = 3.0;
            c[i * N + j] = 0.0;
        }
    }

    printf("Starting computation with transposition and tiling...\n");
    double start_time = omp_get_wtime();

    // Parallel transposition of B
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b_t[j * N + i] = b[i * N + j];
        }
    }

    // Multiplication with sequential access
    #pragma omp parallel for schedule(static) collapse(2) 
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                int i_end = MIN(ii + BLOCK_SIZE, N);
                int j_end = MIN(jj + BLOCK_SIZE, N);
                int k_end = MIN(kk + BLOCK_SIZE, N);
                
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        
                        // Local accumulator: partial results inside a register
                        double sum = 0.0; 
                        
                        // A and B_T read linearly(k increases)
                        // Simple "scalar product": icx vectorize it perfectly
                        for (int k = kk; k < k_end; k++) {
                            sum += a[i * N + k] * b_t[j * N + k];
                        }
                        
                        c[i * N + j] += sum;
                    }   
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    printf("Total time (transposition included): %f seconds\n", end_time - start_time);

    free(a); free(b); free(c); free(b_t);
    return 0;
}
