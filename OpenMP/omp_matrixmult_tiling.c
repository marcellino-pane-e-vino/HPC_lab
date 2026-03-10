#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Definiamo una dimensione per il blocco (Tile Size)
// 64 è spesso un buon compromesso per le cache moderne (64 byte * 8 double = 512 byte)
#define BLOCK_SIZE 64

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Errore: devi passare N e NUM_THREADS come argomenti!\n");
        return 1;
    }
    
    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    // Allocazione allineata per favorire le istruzioni SIMD (AVX/AVX-512)
    // Usiamo aligned_alloc (C11) per allineare a 64 byte (dimensione cache line)
    double *a = (double *)aligned_alloc(64, n * n * sizeof(double));
    double *b = (double *)aligned_alloc(64, n * n * sizeof(double));
    double *c = (double *)aligned_alloc(64, n * n * sizeof(double));

    if (!a || !b || !c) {
        printf("Errore memoria!\n");
        return 1;
    }

    // Inizializzazione parallela
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = 2.0;
            b[i * n + j] = 3.0;
            c[i * n + j] = 0.0;
        }
    }

    printf("Calcolo in corso con Tiling e SIMD...\n");
    double start_time = omp_get_wtime();

    // Core del calcolo: Tiling + OpenMP
    // Usiamo 'collapse(2)' per parallelizzare i primi due cicli dei blocchi
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
                
                // Cicli interni sui blocchi
                for (int i = ii; i < (ii + BLOCK_SIZE < n ? ii + BLOCK_SIZE : n); i++) {
                    for (int k = kk; k < (kk + BLOCK_SIZE < n ? kk + BLOCK_SIZE : n); k++) {
                        
                        double temp = a[i * n + k]; // Carichiamo in registro
                        #pragma omp simd
                        for (int j = jj; j < (jj + BLOCK_SIZE < n ? jj + BLOCK_SIZE : n); j++) {
                            c[i * n + j] += temp * b[k * n + j];
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