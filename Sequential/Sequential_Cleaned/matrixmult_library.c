#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <openblas_config.h>

int main(int argc, char **argv) {
 
    if (argc < 2) {
        fprintf(stderr, "Error: missing n argument.\n");
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
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

    // Allocate memory
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

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }

    printf("Starting the computation...\n");

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    double alpha = 1.0;
    double beta = 0.0;

    // OpenBLAS highly-optimized matrix multiply: C = (alpha*A*B) + (beta*C)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                alpha,
                &a[0][0], n,
                &b[0][0], n,
                beta,
                &c[0][0], n);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double duration =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Execution Time: %.6f seconds\n", duration);

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

    // Cleanup
    fclose(f);
    
    free(a);
    free(b);
    free(c);

    return 0;
}
