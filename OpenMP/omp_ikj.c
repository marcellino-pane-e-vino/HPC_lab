#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Con l'ordine i-k-j l'efficienza della cache è altissima (Stride-1).
// Puoi sperimentare tranquillamente anche con BLOCK_SIZE 64 o 128.
#define BLOCK_SIZE 128
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Errore argomenti\n");
        return 1;
    }
    
    int N = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));

    // Allocazione 1D allineata a 64 byte per la massima sicurezza HPC
    double * restrict a = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict b = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict c = (double *)aligned_alloc(64, N * N * sizeof(double));

    // Inizializzazione con schedule(static) per la First-Touch policy.
    // Aggiunto default(none) e shared per best practice.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 2.0;
            b[i * N + j] = 3.0;
            c[i * N + j] = 0.0;
        }
    }

    printf("Inizio calcolo OpenMP con Tiling (Ordine i-k-j, SENZA Trasposizione)...\n");
    double start_time = omp_get_wtime();

    // PARALLELIZZAZIONE ESTERNA:
    // Collassiamo i blocchi ii e jj. Ogni thread riceve un "quadrante" C[ii..][jj..] unico.
    // Essendo i quadranti disgiunti, non c'è nessuna Race Condition!
    #pragma omp parallel for schedule(dynamic, 8) collapse(2)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            
            // Il thread entra nel suo blocco sicuro.
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                int i_end = MIN(ii + BLOCK_SIZE, N);
                int j_end = MIN(jj + BLOCK_SIZE, N);
                int k_end = MIN(kk + BLOCK_SIZE, N);
                
                // CUORE DEL CALCOLO: Ordine i-k-j
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        
                        // Estraiamo l'elemento fisso di A per questo ciclo J.
                        // Sta comodamente in un registro della CPU.
                        double temp_a = a[i * N + k]; 
                        
                        // IL PUNTO CHIAVE: Sia C che B ora si muovono su 'j'.
                        // Entrambi avanzano in orizzontale di +1 (Stride-1).
                        // La Cache L1 viene sfruttata al 100% e il prefetecher vola.
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
    printf("Tempo totale (Tiling i-k-j): %f secondi\n", end_time - start_time);

    free(a); free(b); free(c);
    return 0;
}