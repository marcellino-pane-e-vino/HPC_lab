//# I_MPI_FABRICS=shm mpirun -np <NUM_PROCS> ./matrixmult_mpi_naive N

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * Naive MPI matrix multiplication in comparison-first style.
 *
 * Same high-level stages as the optimized version:
 *   1. Setup
 *   2. Memory allocation
 *   3. Initialization
 *   4. Input communication
 *   5. Local computation
 *   6. Output collection
 *
 * Difference from the optimized version:
 * - naive: 1D row distribution + full B broadcast + local full-row GEMM
 * - optimized: 2D block-cyclic distribution + SUMMA panel broadcasts
 */

/* =========================================================
 * 4. INPUT COMMUNICATION
 *   distribute A by rows, broadcast full B
 * ========================================================= */

static void communicate_input_data(
    const double *global_A,
    double *local_A,
    double *B,
    int rows_per_proc,
    int n,
    int rank
) {
    /* Broadcast full matrix B to all processes */
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Scatter contiguous row blocks of A */
    MPI_Scatter((void *)global_A,
                rows_per_proc * n, MPI_DOUBLE,
                local_A,
                rows_per_proc * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

/* =========================================================
 * 5. LOCAL COMPUTATION
 *   each process multiplies its local rows of A by full B
 * ========================================================= */

static void local_multiply_rows(
    const double *local_A,
    const double *B,
    double *local_C,
    int rows_per_proc,
    int n
) {
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < n; j++) {
            local_C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }
}

/* =========================================================
 * 6. OUTPUT COLLECTION
 *   gather row blocks of C back to rank 0
 * ========================================================= */

static void collect_output_data(
    const double *local_C,
    double *global_C,
    int rows_per_proc,
    int n
) {
    MPI_Gather((void *)local_C,
               rows_per_proc * n, MPI_DOUBLE,
               global_C,
               rows_per_proc * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
}

/* =========================================================
 * Main
 * ========================================================= */

int main(int argc, char **argv) {
    int rank, size, n;

    /* 1. SETUP */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = (argc < 2) ? 1000 : atoi(argv[1]);
    if (n <= 0) {
        if (rank == 0) fprintf(stderr, "Error: n must be > 0\n");
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = n / size;

    /*
     * Note:
     * this naive version assumes n is divisible by size.
     * Otherwise the last rows are ignored.
     *
     * A more robust version would use MPI_Scatterv / MPI_Gatherv.
     */

    /* 2. MEMORY ALLOCATION */
    double *local_A = (double *)malloc((size_t)rows_per_proc * n * sizeof(double));
    double *local_C = (double *)malloc((size_t)rows_per_proc * n * sizeof(double));
    double *B       = (double *)malloc((size_t)n * n * sizeof(double));

    double *matrix_A = NULL;
    double *matrix_C = NULL;

    if (!local_A || !local_C || !B) {
        fprintf(stderr, "Rank %d: local allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    /* 3. INITIALIZATION */
    if (rank == 0) {
        matrix_A = (double *)malloc((size_t)n * n * sizeof(double));
        matrix_C = (double *)malloc((size_t)n * n * sizeof(double));

        if (!matrix_A || !matrix_C) {
            fprintf(stderr, "Root: global allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        for (int i = 0; i < n * n; i++) {
            matrix_A[i] = 2.0;
            B[i] = 3.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    /* 4. INPUT COMMUNICATION */
    communicate_input_data(matrix_A, local_A, B, rows_per_proc, n, rank);

    /* 5. LOCAL COMPUTATION
     *
     * Naive version:
     *   - every process receives a contiguous set of rows of A
     *   - every process receives the full matrix B
     *   - local computation is a standard triple loop on those rows
     *
     * Optimized version:
     *   - 2D block-cyclic distribution
     *   - SUMMA panel broadcasts
     *   - local block multiplications
     */
    local_multiply_rows(local_A, B, local_C, rows_per_proc, n);

    /* 6. OUTPUT COLLECTION */
    collect_output_data(local_C, matrix_C, rows_per_proc, n);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("--------------------------------------\n");
        printf("Naive MPI matrix multiplication\n");
        printf("Method: 1D row distribution + full B broadcast\n");
        printf("Initialization: static/naive root-based\n");
        printf("Matrix: %d x %d\n", n, n);
        printf("Processes: %d\n", size);
        printf("Time: %.6f seconds\n", end - start);
        printf("--------------------------------------\n");

        free(matrix_A);
        free(matrix_C);
    }

    free(local_A);
    free(local_C);
    free(B);

    MPI_Finalize();
    return 0;
}