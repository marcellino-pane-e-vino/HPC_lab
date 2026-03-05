//#define n 5000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define B 64                                        // Tile (Block) size
#define min(a,b) (((a) < (b)) ? (a) : (b))          // Usage of a MACRO rather than a function should be better 

int main(int argc,char **argv) {
    
    if (argc < 2) {
        fprintf(stderr, "Error: missing n argument.\n");
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Error: You forgot to provide n!.\n");
        return 1;
    }

    int i, j, k;

    double ( *a )[n] = malloc(sizeof(double[n][n]));
    double ( *b )[n] = malloc(sizeof(double[n][n]));
    double ( *c )[n] = malloc(sizeof(double[n][n]));

/*
   double (*a)[n] = aligned_alloc(64, sizeof(double[n][n]));
   double (*b)[n] = aligned_alloc(64, sizeof(double[n][n]));
   double (*c)[n] = aligned_alloc(64, sizeof(double[n][n]));
*/



  // initialization
    for (i=0; i<n; i++)
        for (j=0; j<n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }

    printf("Starting the computation...\n");
    clock_t start_clock = clock();                  // start time

   
   // Loop Tiling ottimizzato
   /*
    for (int ii = 0; ii < n; ii += B) {
        int i_limit = fmin(ii + B, n);
        for (int kk = 0; kk < n; kk += B) {
            int k_limit = fmin(kk + B, n);
            for (int jj = 0; jj < n; jj += B) {
                int j_limit = fmin(jj + B, n);

                // Cicli interni ottimizzati
                for (int i = ii; i < i_limit; i++) {
                    for (int k = kk; k < k_limit; k++) {
                        double temp_a = a[i][k]; 
                        for (int j = jj; j < j_limit; j++) {
                            c[i][j] += temp_a * b[k][j];
                        }
                    }
                }
            }
        }
    }
    */




    // External loops: Block level
    for (int ii = 0; ii < n; ii += B) {             // Along rows of C and A        
        for (int kk = 0; kk < n; kk += B) {         // Along columns of A and rows of B
            for (int jj = 0; jj < n; jj += B) {     // Along columns of C and B
                
                // Internal loops: within the single Tile (Block)
                for (int i = ii; i < fmin(ii + B, n); i++) {                 // Along the rows of the current block
                    for (int k = kk; k < fmin(kk + B, n); k++) {             // Efficiency: along the elements needed for the scalar product but only for the width of the tile B
                        for (int j = jj; j < fmin(jj + B, n); j++) {         // Along the columns of the current block
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
                
            }
        }
    }
   

    // Note:
    // min(...): Se la tua matrice è $100 \times 100$ e il tuo blocco è $32$, 
    // l'ultimo blocco inizierebbe a $96$ e finirebbe a $128$. 
    // Il min serve a fermarsi a $100$ per evitare di leggere memoria fuori dai limiti (Segmentation Fault).


    // Note: for (int k = kk; k < min(kk + B, n); k++)
    // Questo è il segreto dell'efficienza: a[i][k] e b[k][j] rimangono in cache perché stiamo lavorando su un range di k molto piccolo.


    // Note: c[i][j] += a[i][k] * b[k][j];
    // Questa è l'operazione standard. 
    // La differenza è che, grazie ai cicli sopra, quando il programma chiede al processore il valore di b[k][j], 
    // è molto probabile che quel valore sia già nella Cache L1 (vicinissima al processore) 
    // perché è stato usato un istante prima per il calcolo dell'elemento precedente del tile.

    clock_t end_clock = clock();                // end time

    double duration = (double)(end_clock - start_clock) / CLOCKS_PER_SEC;   
    printf("Execution Time: %.4f seconds\n", duration);

    FILE *f = fopen("mat-res.txt", "w");
    if (!f) {
        perror("fopen");
        return 1;
    }

    fprintf(f, "%d\n\n", n);  
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000; j++) {
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


// Starting point: sequential complexity is O(n^3)


// Hint: To perform loop tiling, the loops have to be perfectly nested and a certain type of loops with loop-carried dependencies cannot be tiled.