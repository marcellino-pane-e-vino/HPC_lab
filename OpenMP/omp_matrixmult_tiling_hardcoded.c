#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Parametri fissi per il profiling
#define N 15000
#define NUM_THREADS 24
#define BLOCK_SIZE 64

int main() {
    // Impostiamo il numero di thread fisso
    omp_set_num_threads(NUM_THREADS);

    // Allocazione allineata a 64 byte per AVX-512 [cite: 421, 215]
    double * restrict a = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict b = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict c = (double *)aligned_alloc(64, N * N * sizeof(double));

    if (!a || !b || !c) {
        printf("Errore memoria!\n");
        return 1;
    }

    // Inizializzazione parallela
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 2.0;
            b[i * N + j] = 3.0;
            c[i * N + j] = 0.0;
        }
    }

    printf("Esecuzione Roofline su N=%d con %d thread...\n", N, NUM_THREADS);
    double start_time = omp_get_wtime();

    // Loop con Tiling e SIMD
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                for (int i = ii; i < (ii + BLOCK_SIZE < N ? ii + BLOCK_SIZE : N); i++) {
                    for (int k = kk; kk + BLOCK_SIZE < N ? k < kk + BLOCK_SIZE : k < N; k++) {
                        
                        double temp = a[i * N + k]; 
                        // Forza la vettorializzazione del loop interno [cite: 1384, 1582]
                        #pragma omp simd
                        for (int j = jj; j < (jj + BLOCK_SIZE < N ? jj + BLOCK_SIZE : N); j++) {
                            c[i * N + j] += temp * b[k * N + j];
                        }
                    }
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    printf("Tempo di esecuzione: %f secondi\n", end_time - start_time);

    free(a); free(b); free(c);
    return 0;
}