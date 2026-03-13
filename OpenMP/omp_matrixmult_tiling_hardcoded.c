#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Definiamo una dimensione per il blocco (Tile Size)
// 64 è spesso un buon compromesso per le cache moderne (64 byte * 8 double = 512 byte)

#define N 5000
#define BLOCK_SIZE 64

int main(int argc, char **argv) {
    omp_set_num_threads(num_threads);

    // Allocazione allineata per favorire le istruzioni SIMD (AVX/AVX-512)
    // Usiamo aligned_alloc (C11) per allineare a 64 byte (dimensione cache line)
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

    printf("Calcolo in corso con Tiling e SIMD...\n");
    double start_time = omp_get_wtime();

    // Core del calcolo: Tiling + OpenMP
   // Correzione: invertiamo jj e kk nei cicli esterni!
#pragma omp parallel for schedule(static) collapse(2) 
for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            
            // Cicli interni: L'ordine I-K-J qui dentro va benissimo
            for (int i = ii; i < (ii + BLOCK_SIZE < N ? ii + BLOCK_SIZE : N); i++) {
                for (int k = kk; k < (kk + BLOCK_SIZE < N ? kk + BLOCK_SIZE : N); k++) {
                    
                    double temp = a[i * N + k]; 
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

    // Pulizia
    free(a); free(b); free(c);
    return 0;
}
