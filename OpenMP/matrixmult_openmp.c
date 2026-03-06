#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

#define min(a,b) (((a) < (b)) ? (a) : (b))

int main(int argc, char **argv) {
    
    int n = atoi(argv[1]);
    int B = (argc >= 3) ? atoi(argv[2]) : 64;

    size_t alignment = 64;
    size_t matrix_size = sizeof(double[n][n]);

    if (matrix_size % alignment != 0) {
        matrix_size = ((matrix_size / alignment) + 1) * alignment;
    }

    // Aggiunta della keyword 'restrict' per aiutare i compilatori con l'Alias Analysis
    double (* a)[n] = aligned_alloc(alignment, matrix_size);
    double (* b)[n] = aligned_alloc(alignment, matrix_size);
    double (* c)[n] = aligned_alloc(alignment, matrix_size);

    // Parallelizziamo anche l'inizializzazione per azzerare i tempi morti
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }
         
    printf("Starting the computation...\n");
    double start_time = omp_get_wtime();                  

    // IL MOTORE HPC DEFINITIVO
    // Collapse(2) crea una griglia perfetta di blocchi C indipendenti.
    // dynamic bilancia il carico sui processori multicore ad alte prestazioni.
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < n; ii += B) {             
        for (int jj = 0; jj < n; jj += B) {     
            
            // Calcoliamo i limiti del singolo blocco C(ii, jj)
            int i_limit = min(ii + B, n);
            int j_limit = min(jj + B, n);

            // Ora scorriamo K tenendo fisso il blocco C in cache L1
            for (int kk = 0; kk < n; kk += B) {         
                int k_limit = min(kk + B, n);

                // Moltiplicazione all'interno del singolo Tile
                for (int i = ii; i < i_limit; i++) {            
                    for (int k = kk; k < k_limit; k++) {        
                        //double temp_a = a[i][k]; 
                        
                        // Forziamo brutalmente la CPU a usare le istruzioni SIMD vettoriali
                        #pragma omp simd
                        for (int j = jj; j < j_limit; j++) {    
                            //c[i][j] += temp_a * b[k][j];
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }
    }

    double end_time = omp_get_wtime();                
    double duration = end_time - start_time;   
    
    printf("Execution Time: %.4f seconds\n", duration);

    FILE *f = fopen("mat-res.txt", "w");
    if (!f) {
        perror("fopen");
        return 1;
    }

    fprintf(f, "%d\n\n", n);  
    // Evitiamo di stampare enormi matrici per N=15000, limitiamo il log a 10x10 per verifica
    int print_limit = min(n, 10);
    for (int i = 0; i < print_limit; i++) {
        for (int j = 0; j < print_limit; j++) {
            fprintf(f, "%.0f ", c[i][j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    free(a);
    free(b);
    free(c);
    
    return 0;
}