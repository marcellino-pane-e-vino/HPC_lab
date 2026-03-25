#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Include the OpenBLAS headers. BLAS (Basic Linear Algebra Subprograms) 
// is a standard interface for highly optimized linear algebra operations.
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

    /* THREADING CONTROL */
    /* Force sequential execution at runtime too */
    // OpenBLAS normally tries to use all available CPU cores by default. 
    // Since we likely want to compare this single-core performance strictly 
    // against our previous naive/tiled handwritten C codes, we must force it to use exactly 1 thread.
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("GOTO_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
    openblas_set_num_threads(1);

    // Standard dynamic allocation using Variable Length Arrays (VLAs)
    double (*a)[n] = malloc(sizeof(double[n][n]));
    double (*b)[n] = malloc(sizeof(double[n][n]));
    double (*c)[n] = malloc(sizeof(double[n][n]));

    // Safety check: always ensure memory was successfully allocated
    if (!a || !b || !c) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    // Initialization: Fill matrices with standard constants
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }

    printf("Starting the computation...\n");

    // HIGH PRECISION TIMING
    // Using clock_gettime with CLOCK_MONOTONIC instead of the standard clock().
    // Monotonic clocks represent absolute elapsed wall-clock time and are immune to system time changes.
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // BLAS operations usually calculate: C = (alpha * A * B) + (beta * C)
    // By setting alpha=1.0 and beta=0.0, we get the standard matrix multiplication: C = A * B
    double alpha = 1.0;
    double beta = 0.0;

    // THE CORE COMPUTATION: cblas_dgemm
    // dgemm stands for Double-precision General Matrix Multiply.
    // This single call replaces our deeply nested loops and handles all hardware-specific 
    // optimizations (SIMD vectorization, cache blocking, register allocation) automatically.
    cblas_dgemm(CblasRowMajor,   // Specifies that our C-matrices are stored row-by-row in memory
                CblasNoTrans,    // Do not transpose matrix A
                CblasNoTrans,    // Do not transpose matrix B
                n, n, n,         // The dimensions of the matrices (M, N, K)
                alpha,           // Scalar multiplier for A*B
                &a[0][0], n,     // Pointer to matrix A and its leading dimension (stride)
                &b[0][0], n,     // Pointer to matrix B and its leading dimension
                beta,            // Scalar multiplier for C
                &c[0][0], n);    // Pointer to result matrix C and its leading dimension

    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate duration using nanosecond precision converted to seconds
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
    
    // Output loop safely constrained by both matrix size 'n' and a 1000x1000 upper limit
    for (int i = 0; i < n && i < 1000; i++) {
        for (int j = 0; j < n && j < 1000; j++) {
            fprintf(f, "%.0f ", c[i][j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    
    // Clean up
    free(a);
    free(b);
    free(c);
    
    return 0;
}