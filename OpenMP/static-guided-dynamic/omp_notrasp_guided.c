#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define BLOCK_SIZE 32
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Errore argomenti\n");
        return 1;
    }
    
    int N = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));

    // Allocazione a 64 byte. Rimuoviamo b_t.
    double * restrict a = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict b = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict c = (double *)aligned_alloc(64, N * N * sizeof(double));

    // Inizializzazione con schedule(static) per la First-Touch policy
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 2.0;
            b[i * N + j] = 3.0;
            c[i * N + j] = 0.0;
        }
    }

    printf("Inizio calcolo con Tiling (SENZA Trasposizione)...\n");
    double start_time = omp_get_wtime();

    #pragma omp parallel for schedule(guided) collapse(2)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                int i_end = MIN(ii + BLOCK_SIZE, N);
                int j_end = MIN(jj + BLOCK_SIZE, N);
                int k_end = MIN(kk + BLOCK_SIZE, N);
                
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        
                        double sum = 0.0; 
                        
                        // IL PUNTO CHIAVE: Notare l'accesso b[k * N + j]
                        //#pragma omp simd reduction(+:sum)
                        for (int k = kk; k < k_end; k++) {
                            // a[] viene letta linearmente (stride-1)
                            // b[] subisce salti di dimensione N ad ogni incremento di k (stride-N)
                            sum += a[i * N + k] * b[k * N + j];
                        }
                        
                        c[i * N + j] += sum;
                    }   
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    printf("Tempo totale (Senza Trasposizione): %f secondi\n", end_time - start_time);

    free(a); free(b); free(c);
    return 0;
}