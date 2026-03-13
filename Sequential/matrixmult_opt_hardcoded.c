#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// Definiamo i parametri fissi come richiesto [cite: 1697, 1698]
#define N 5000
#define B 64

#define min(a,b) (((a) < (b)) ? (a) : (b))

int main() {
    
    int i, j, k;
    int ii, kk, jj;
    int i_limit, k_limit, j_limit;

    // Allineamento a 64 byte per favorire AVX-512 [cite: 421, 452]
    size_t alignment = 64;
    size_t matrix_mem_size = N * N * sizeof(double);
    
    // Arrotondamento della dimensione per aligned_alloc [cite: 450, 453]
    if (matrix_mem_size % alignment != 0) {
        matrix_mem_size = ((matrix_mem_size / alignment) + 1) * alignment;
    }
    
    // Utilizzo di restrict per informare il compilatore che non c'è aliasing 
    double (*restrict a)[N] = aligned_alloc(alignment, matrix_mem_size);
    double (*restrict b)[N] = aligned_alloc(alignment, matrix_mem_size);
    double (*restrict c)[N] = aligned_alloc(alignment, matrix_mem_size);
    
    if (!a || !b || !c) {
        printf("Errore: Memoria insufficiente!\n");
        return 1;
    }

    // Inizializzazione
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }
         
    printf("Starting the computation on N=%d, B=%d...\n", N, B);
    
    // Utilizziamo un timer accurato per il benchmarking [cite: 1629, 1630]
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Struttura a 6 cicli (Tiling) per ottimizzare la cache [cite: 1354, 1367]
    for (ii = 0; ii < N; ii += B) {
        i_limit = min(ii + B, N);
        for (kk = 0; kk < N; kk += B) {
            k_limit = min(kk + B, N);
            for (jj = 0; jj < N; jj += B) {
                j_limit = min(jj + B, N);

                // Cicli interni: lavorano sulla singola Tile
                for (i = ii; i < i_limit; i++) {
                    for (k = kk; k < k_limit; k++) {
                        double temp_a = a[i][k];
                        for (j = jj; j < j_limit; j++) {
                            c[i][j] += temp_a * b[k][j];
                        }
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    
    printf("Execution Time: %.4f seconds\n", duration);
    printf("Performance: %.2f GFLOPS\n", (2.0 * N * N * N) / (duration * 1e9));

    // Scrittura dei risultati (limitata ai primi 1000 per evitare file enormi)
    FILE *f = fopen("mat-res.txt", "w");
    if (f) {
        fprintf(f, "%d\n\n", N);  
        int limit = min(N, 1000);
        for (int r = 0; r < limit; r++) {
            for (int col = 0; col < limit; col++) {
                fprintf(f, "%.0f ", c[r][col]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }

    free(a);
    free(b);
    free(c);
    
    return 0;
}