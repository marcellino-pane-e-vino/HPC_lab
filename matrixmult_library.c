#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// If your system has cblas, include this
#include <cblas.h>

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

    int i, j;

    double ( *a )[n] = malloc(sizeof(double[n][n]));
    double ( *b )[n] = malloc(sizeof(double[n][n]));
    double ( *c )[n] = malloc(sizeof(double[n][n]));

    // initialization
    for (i=0; i<n; i++)
        for (j=0; j<n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }

    printf("Starting the computation...\n");
    clock_t start = clock();

    // Use BLAS library: cblas_dgemm
    // C = alpha * A * B + beta * C
    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                alpha,
                &a[0][0], n,
                &b[0][0], n,
                beta,
                &c[0][0], n);

    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
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

    free(a);
    free(b);
    free(c);
    return 0;
}