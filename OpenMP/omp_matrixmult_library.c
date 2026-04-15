#include <stdio.h>
#include <stdlib.h>

// Necessary libraries
#include <mkl.h>    
#include <omp.h>    

int main(int argc, char **argv) {

    if (argc < 2) {
        fprintf(stderr, "Error: missing n argument.\n");
        return 1;
    }

    int n = atoi(argv[1]);

    if (n <= 0) {
        fprintf(stderr, "Error: You forgot to provide n!.\n");
        return 1;
    }

    int i, j;

    // Memory allocation
    double (*a)[n] = malloc(sizeof(double[n][n]));
    double (*b)[n] = malloc(sizeof(double[n][n]));
    double (*c)[n] = malloc(sizeof(double[n][n]));

    if (!a || !b || !c) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return 1;
    }

    // Initialization
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }
    
    // Timer
    double start = omp_get_wtime();
    printf("Starting the computation...\n");

    // Parameters settings - C = alpha * A * B + beta * C
    double alpha = 1.0;
    double beta = 0.0;

    // Matrix multiplication through CBLAS library
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                alpha,
                &a[0][0], n,
                &b[0][0], n,
                beta,
                &c[0][0], n);

    double end = omp_get_wtime();
    double duration = end - start;
    printf("Execution Time: %.4f seconds\n", duration);

    
    FILE *f = fopen("mat-res.txt", "w");
    if (!f) {
        perror("fopen");
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
    
    return 0;
}
