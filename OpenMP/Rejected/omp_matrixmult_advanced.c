#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define BLOCK_SIZE 128
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Errore argomenti\n");
        return 1;
    }
    
    int N = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));

    double * restrict a = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict b = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict c = (double *)aligned_alloc(64, N * N * sizeof(double));
    // Allochiamo spazio per la matrice B trasposta
    double * restrict b_t = (double *)aligned_alloc(64, N * N * sizeof(double));


    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for schedule(dynamic) collapse(2) 
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                int i_end = MIN(ii + BLOCK_SIZE, N);
                int j_end = MIN(jj + BLOCK_SIZE, N);
                int k_end = MIN(kk + BLOCK_SIZE, N);
                
                /*
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
                        */
                // --- INIZIO DEL MICRO-KERNEL SROTOLATO ---
                // Avanziamo a passi di 4 sia sulle righe che sulle colonne
                for (int i = ii; i < i_end; i += 4) {
                    for (int j = jj; j < j_end; j += 4) {
                        
                        // 16 Accumulatori locali nei registri veloci
                        double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                        double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                        double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
                        double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
                        
                        // Ciclo più interno: macina i calcoli a 16 alla volta
                        for (int k = kk; k < k_end; k++) {
                            // Lettura di 4 elementi dalla matrice A
                            double a0 = a[(i + 0) * N + k];
                            double a1 = a[(i + 1) * N + k];
                            double a2 = a[(i + 2) * N + k];
                            double a3 = a[(i + 3) * N + k];
                            
                            // Lettura di 4 elementi dalla matrice B_T
                            double b0 = b_t[(j + 0) * N + k];
                            double b1 = b_t[(j + 1) * N + k];
                            double b2 = b_t[(j + 2) * N + k];
                            double b3 = b_t[(j + 3) * N + k];
                            
                            // 16 operazioni FMA incrociate
                            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
                        }
                        
                        // Scriviamo il risultato in RAM solo alla fine del blocco!
                        c[(i + 0) * N + (j + 0)] += c00; c[(i + 0) * N + (j + 1)] += c01; c[(i + 0) * N + (j + 2)] += c02; c[(i + 0) * N + (j + 3)] += c03;
                        c[(i + 1) * N + (j + 0)] += c10; c[(i + 1) * N + (j + 1)] += c11; c[(i + 1) * N + (j + 2)] += c12; c[(i + 1) * N + (j + 3)] += c13;
                        c[(i + 2) * N + (j + 0)] += c20; c[(i + 2) * N + (j + 1)] += c21; c[(i + 2) * N + (j + 2)] += c22; c[(i + 2) * N + (j + 3)] += c23;
                        c[(i + 3) * N + (j + 0)] += c30; c[(i + 3) * N + (j + 1)] += c31; c[(i + 3) * N + (j + 2)] += c32; c[(i + 3) * N + (j + 3)] += c33;
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
