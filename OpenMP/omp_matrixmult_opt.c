#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// By using i-k-j order the cache efficiency is high.
// We can experiment also with BLOCK_SIZE 64. 
#define BLOCK_SIZE 128
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Missing arguments\n");
        return 1;
    }
    
    int N = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));

    // Aligned 1D allocation to 64 byte
    double * restrict a = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict b = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict c = (double *)aligned_alloc(64, N * N * sizeof(double));

    // Inizialization with schedule(static).
    // Added default(none) and shared for best practice.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 2.0;
            b[i * N + j] = 3.0;
            c[i * N + j] = 0.0;
        }
    }

    printf("Starting OpenMP computation with tiling (i-k-j order, WITHOUT transposition)...\n");
    double start_time = omp_get_wtime();

    // OUTER PARALLELIZATION:
    // We collapse ii and jj blocks. Each thread handles a umique "quadrant" C[ii..][jj..].
    // The quadrants are separated, hence no race conditions
    #pragma omp parallel for schedule(dynamic, 4) collapse(2)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            
            // The thread enters in his secure block.
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                int i_end = MIN(ii + BLOCK_SIZE, N);
                int j_end = MIN(jj + BLOCK_SIZE, N);
                int k_end = MIN(kk + BLOCK_SIZE, N);
                
                // CORE COMPUTATION: i-k-j order
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        
                        // We extract the fixed element of A. It's inside the CPU register
                        double temp_a = a[i * N + k]; 
                        
                        // Key point: both C and B now are moving wrt j so they go straight on by a horizontal way +1. 
                        // Cache L1 used at 100%: prefetching helps us
                        //#pragma omp simd
                        for (int j = jj; j < j_end; j++) {
                            c[i * N + j] += temp_a * b[k * N + j];
                        }   
                    }
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    printf("Total time (Tiling i-k-j): %f seconds\n", end_time - start_time);

    free(a); free(b); free(c);
    return 0;
}
