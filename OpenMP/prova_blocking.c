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
    
    // NOTA: Per questo Unrolling a fattore 4, assumiamo che N sia un multiplo di 4.
    // In un codice di produzione metteresti un loop "remainder" alla fine.
    int N = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));

    double * restrict a = (double *)aligned_alloc(64, (size_t)N * N * sizeof(double));
    double * restrict b = (double *)aligned_alloc(64, (size_t)N * N * sizeof(double));
    double * restrict c = (double *)aligned_alloc(64, (size_t)N * N * sizeof(double));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 2.0;
            b[i * N + j] = 3.0;
            c[i * N + j] = 0.0;
        }
    }

    printf("Inizio calcolo con Register Blocking (Unroll 4x) e Cache Tiling...\n");
    double start_time = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 4) collapse(2)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                int i_end = MIN(ii + BLOCK_SIZE, N);
                int j_end = MIN(jj + BLOCK_SIZE, N);
                int k_end = MIN(kk + BLOCK_SIZE, N);
                
                // UNROLLING: Avanziamo di 4 in 4 sull'asse 'i'
                for (int i = ii; i < i_end; i += 4) {
                    for (int k = kk; k < k_end; k++) {
                        
                        // 1. REGISTER BLOCKING: Carichiamo 4 valori scalari di A 
                        // in 4 registri fisici della CPU.
                        double a0 = a[(i + 0) * N + k];
                        double a1 = a[(i + 1) * N + k];
                        double a2 = a[(i + 2) * N + k];
                        double a3 = a[(i + 3) * N + k];
                        
                        // Direttiva SIMD per vettorializzare il ciclo più interno
                        #pragma omp simd
                        for (int j = jj; j < j_end; j++) {
                            // 2. DATA REUSE: Leggiamo B una volta sola dalla L1 Cache...
                            double b_val = b[k * N + j];
                            
                            // ...e la spremiamo per fare 4 moltiplicazioni e 4 addizioni!
                            c[(i + 0) * N + j] += a0 * b_val;
                            c[(i + 1) * N + j] += a1 * b_val;
                            c[(i + 2) * N + j] += a2 * b_val;
                            c[(i + 3) * N + j] += a3 * b_val;
                        }   
                    }
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    printf("Tempo totale (Tiling + Register Blocking): %f secondi\n", end_time - start_time);

    free(a); free(b); free(c);
    return 0;
}