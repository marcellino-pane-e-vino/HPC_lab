#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Necessary libraries
#include <cblas.h>                  
#include <openblas_config.h>

int main(int argc, char **argv) {

    // Timer for Total Execution Time
    printf("Starting the computation...\n");
    clock_t start_clock_total = clock();   
 
    if (argc < 2) {
        fprintf(stderr, "Error: missing n argument.\n");
        return 1;
    }

    int n = atoi(argv[1]);

    if (n <= 0) {
        fprintf(stderr, "Error: invalid n.\n");
        return 1;
    }

    // Force single-thread execution for fair benchmarking
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("GOTO_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
    openblas_set_num_threads(1);

    // Memory allocation
    double (*a)[n] = malloc(sizeof(double[n][n]));
    double (*b)[n] = malloc(sizeof(double[n][n]));
    double (*c)[n] = malloc(sizeof(double[n][n]));

    if (!a || !b || !c) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    // Initialization
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }


    // Timer for Critical Section
    printf("Starting the critical section...\n");
    clock_t start_clock = clock();  

    // Necessary for our objective: simple matrix multiplication
    double alpha = 1.0;
    double beta = 0.0;

    // OpenBLAS highly-optimized matrix multiplication: C = (alpha*A*B) + (beta*C)
    // - CblasRowMajor: aligns with C-style 2D arrays memory layout
    // - CblasNoTrans: matrices are passed as-is (no hardware transposition)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                alpha,
                &a[0][0], n,
                &b[0][0], n,
                beta,
                &c[0][0], n);

    clock_t end_clock = clock();                
    double duration = (double)(end_clock - start_clock) / CLOCKS_PER_SEC;   
    printf("Execution Time for the Critical Section: %.4f seconds\n", duration);

    FILE *f = fopen("mat-res.txt", "w");
    if (!f) {
        perror("fopen");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    fprintf(f, "%d\n\n", n);
    for (int i = 0; i < n && i < 1000; i++) {
        for (int j = 0; j < n && j < 1000; j++) {
            fprintf(f, "%.0f ", c[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    
    // Memory deallocation
    free(a);
    free(b);
    free(c);

    clock_t end_clock_total = clock();               
    double total_time = (double)(end_clock_total - start_clock_total) / CLOCKS_PER_SEC;   
    printf("Total Execution Time: %.4f seconds\n", total_time);

    return 0;
}
