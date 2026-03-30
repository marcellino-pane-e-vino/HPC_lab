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
    // Allochiamo spazio per la matrice B trasposta
    double * restrict b_t = (double *)aligned_alloc(64, N * N * sizeof(double));

    #pragma omp parallel //for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 2.0;
            b[i * N + j] = 3.0;
            c[i * N + j] = 0.0;
        }
    }

    printf("Inizio calcolo con Trasposizione e Tiling...\n");
    double start_time = omp_get_wtime();

    // 1. TRASPOSIZIONE PARALLELA DI B
    // Ci mette una frazione di secondo, ma ci salva la vita dopo.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b_t[j * N + i] = b[i * N + j];
        }
    }

    // 2. MOLTIPLICAZIONE CON ACCESSO SEQUENZIALE
    #pragma omp parallel for schedule(static) collapse(2) 
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                int i_end = MIN(ii + BLOCK_SIZE, N);
                int j_end = MIN(jj + BLOCK_SIZE, N);
                int k_end = MIN(kk + BLOCK_SIZE, N);
                
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        
                        // Accumulatore locale: tiene il parziale in un registro velocissimo!
                        double sum = 0.0; 
                        
                        // Ora sia A che B_T vengono lette linearmente (k aumenta)
                        // Questo ciclo è un banale "Prodotto Scalare", icx lo vettorizza alla perfezione
                        //#pragma omp simd reduction(+:sum)
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
    printf("Tempo totale (inclusa trasposizione): %f secondi\n", end_time - start_time);

    free(a); free(b); free(c); free(b_t);
    return 0;
}
