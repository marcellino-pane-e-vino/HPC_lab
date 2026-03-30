#include <stdio.h>
#include <stdlib.h>

// mkl.h replaces cblas.h. It provides the highly optimized, 
// multi-threaded Intel MKL implementation of cblas_dgemm.
#include <mkl.h>    

// omp.h is required for omp_get_wtime(), which measures real "wall-clock" time 
// rather than CPU time (which inflates when using multiple threads).
#include <omp.h>    

int main(int argc, char **argv) {
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

    int i, j;

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

    printf("Starting the computation with MKL Parallel...\n");
    
    // START TIMER: Using OpenMP wall-clock time
    double start = omp_get_wtime();

    // Use BLAS library: cblas_dgemm
    // C = alpha * A * B + beta * C
    double alpha = 1.0;
    double beta = 0.0;

    // Because we compile with -qmkl=parallel, this function will automatically 
    // spin up multiple threads via Intel's OpenMP runtime.
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                alpha,
                &a[0][0], n,
                &b[0][0], n,
                beta,
                &c[0][0], n);

    // END TIMER
    double end = omp_get_wtime();
    double duration = end - start;
    
    printf("Execution Time (Wall-Clock): %.4f seconds\n", duration);

    FILE *f = fopen("mat-res.txt", "w");
    if (!f) {
        perror("fopen");
        // Free memory before exiting on error
        free(a); free(b); free(c);
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

    free(a);
    free(b);
    free(c);
    
    return 0;
}
