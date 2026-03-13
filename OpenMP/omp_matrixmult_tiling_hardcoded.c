#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// --- PARAMETRI MODIFICABILI ---
#define N 5000
#define BLOCK_SIZE 64  // Dimensione della Tile
#define NUM_THREADS 24 // CAMBIA QUI il numero di core da usare 
// ------------------------------

#define min(a,b) (((a)<(b))?(a):(b))

int main(int argc, char **argv) {
    
    // Impostiamo il numero di thread direttamente nel codice come richiesto
    omp_set_num_threads(NUM_THREADS); 

    printf("Inizializzazione matrici N=%d con %d threads...\n", N, NUM_THREADS);

    // Allocazione allineata a 64 byte per favorire AVX2/AVX-512 [cite: 15, 31]
    // Usiamo restrict per dire al compilatore: "tranquillo, non ci sono sovrapposizioni in memoria"
    double * restrict a = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict b = (double *)aligned_alloc(64, N * N * sizeof(double));
    double * restrict c = (double *)aligned_alloc(64, N * N * sizeof(double));

    if (!a || !b || !c) {
        printf("Errore memoria!\n");
        return 1;
    }

    // Inizializzazione parallela: fondamentale per il "First Touch Policy" in Linux
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 2.0;
            b[i * N + j] = 3.0;
            c[i * N + j] = 0.0;
        }
    }

    printf("Calcolo in corso (Tiling + OpenMP + SIMD)...\n");
    double start_time = omp_get_wtime();

    // CORE DEL CALCOLO
    // Parallelizziamo sui blocchi esterni per distribuire il carico di lavoro
    #pragma omp parallel for schedule(static) collapse(2) 
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            // Questo ciclo kk intermedio aiuta a mantenere i dati di B nella cache
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                // Cicli interni: lavorano sulla singola Tile
                // L'ordine I-K-J è il migliore per l'accesso contiguo alla memoria (andiamo a DESTRA!)
                for (int i = ii; i < min(ii + BLOCK_SIZE, N); i++) {
                    for (int k = kk; k < min(kk + BLOCK_SIZE, N); k++) {
                        
                        double temp = a[i * N + k]; 
                        
                        // Suggeriamo al compilatore di vettorializzare il ciclo più interno
                        #pragma omp simd
                        for (int j = jj; j < min(jj + BLOCK_SIZE, N); j++) {
                            c[i * N + j] += temp * b[k * N + j];
                        }
                    }
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    
    double duration = end_time - start_time;
    double gflops = (2.0 * N * N * N) / (duration * 1e9);

    printf("Tempo di esecuzione: %.4f secondi\n", duration);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // Pulizia
    free(a); 
    free(b); 
    free(c);
    
    return 0;
}
