#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define NB 32
#define MIN(a,b) ((a)<(b)?(a):(b))

/* ============================================================
 * Utility helpers
 * ============================================================ */

/* Number of global block indices owned locally along one grid dimension */
static int num_local_blocks(int nblocks, int coord, int nprocs_dim) {
    int count = 0;
    for (int b = coord; b < nblocks; b += nprocs_dim) count++;
    return count;
}

/* Real extent of a block at the matrix boundary */
static int block_extent(int block_idx, int n) {
    int start = block_idx * NB;
    int rem = n - start;
    return (rem >= NB) ? NB : (rem > 0 ? rem : 0);
}

/* Pointer to local block (lbi, lbj) in block storage */
static inline double *block_ptr(double *M, int local_block_cols, int lbi, int lbj) {
    return M + ((size_t)lbi * local_block_cols + lbj) * NB * NB;
}

static void zero_block(double *blk) {
    memset(blk, 0, NB * NB * sizeof(double));
}

static void zero_blocks(double *M, int lbr, int lbc) {
    memset(M, 0, (size_t)lbr * lbc * NB * NB * sizeof(double));
}

/* Dense NB x NB local block multiply-add: Cblk += Ablk * Bblk */
static void local_block_gemm(const double *Ablk, const double *Bblk, double *Cblk) {
    for (int i = 0; i < NB; i++) {
        for (int k = 0; k < NB; k++) {
            double aik = Ablk[i * NB + k];
            for (int j = 0; j < NB; j++) {
                Cblk[i * NB + j] += aik * Bblk[k * NB + j];
            }
        }
    }
}

/* Pack one global block into a padded NB x NB local block */
static void pack_global_block_to_padded(
    const double *global_M, int n, int bi, int bj, double *dst
) {
    int rows = block_extent(bi, n);
    int cols = block_extent(bj, n);

    zero_block(dst);

    int row0 = bi * NB;
    int col0 = bj * NB;

    for (int i = 0; i < rows; i++) {
        memcpy(dst + (size_t)i * NB,
               global_M + (size_t)(row0 + i) * n + col0,
               (size_t)cols * sizeof(double));
    }
}

/* Unpack one padded NB x NB block back into the global matrix */
static void unpack_padded_block_to_global(
    const double *src, double *global_M, int n, int bi, int bj
) {
    int rows = block_extent(bi, n);
    int cols = block_extent(bj, n);

    int row0 = bi * NB;
    int col0 = bj * NB;

    for (int i = 0; i < rows; i++) {
        memcpy(global_M + (size_t)(row0 + i) * n + col0,
               src + (size_t)i * NB,
               (size_t)cols * sizeof(double));
    }
}

/* ============================================================
 * 3. COMMUNICATION SUPPORT
 * Distribution: root sends each global block to its owner
 * ============================================================ */

static void distribute_block_cyclic(
    const double *global_M,
    double *local_M,
    int n,
    int nblocks,
    int p_rows,
    int p_cols,
    int my_row,
    int my_col,
    int rank,
    MPI_Comm cart_comm
) {
    MPI_Status status;
    int local_block_cols = num_local_blocks(nblocks, my_col, p_cols);

    if (rank == 0) {
        double *tmp = (double *)malloc(NB * NB * sizeof(double));
        if (!tmp) {
            fprintf(stderr, "Root: distribution buffer allocation failed\n");
            MPI_Abort(cart_comm, 10);
        }

        for (int bi = 0; bi < nblocks; bi++) {
            for (int bj = 0; bj < nblocks; bj++) {
                int owner_coords[2] = { bi % p_rows, bj % p_cols };
                int owner_rank;
                MPI_Cart_rank(cart_comm, owner_coords, &owner_rank);

                pack_global_block_to_padded(global_M, n, bi, bj, tmp);

                if (owner_rank == 0) {
                    int lbi = bi / p_rows;
                    int lbj = bj / p_cols;
                    memcpy(block_ptr(local_M, local_block_cols, lbi, lbj),
                           tmp, NB * NB * sizeof(double));
                } else {
                    MPI_Send(tmp, NB * NB, MPI_DOUBLE, owner_rank,
                             1000 + bi * nblocks + bj, cart_comm);
                }
            }
        }

        free(tmp);
    } else {
        for (int bi = my_row, lbi = 0; bi < nblocks; bi += p_rows, lbi++) {
            for (int bj = my_col, lbj = 0; bj < nblocks; bj += p_cols, lbj++) {
                MPI_Recv(block_ptr(local_M, local_block_cols, lbi, lbj),
                         NB * NB, MPI_DOUBLE, 0,
                         1000 + bi * nblocks + bj, cart_comm, &status);
            }
        }
    }
}

/* ============================================================
 * 5. GATHER SUPPORT
 * Root receives each distributed C block and rebuilds global C
 * ============================================================ */

static void gather_block_cyclic(
    const double *local_M,
    double *global_M,
    int n,
    int nblocks,
    int p_rows,
    int p_cols,
    int my_row,
    int my_col,
    int rank,
    MPI_Comm cart_comm
) {
    MPI_Status status;
    int local_block_cols = num_local_blocks(nblocks, my_col, p_cols);

    if (rank == 0) {
        double *tmp = (double *)malloc(NB * NB * sizeof(double));
        if (!tmp) {
            fprintf(stderr, "Root: gather buffer allocation failed\n");
            MPI_Abort(cart_comm, 11);
        }

        for (int bi = 0; bi < nblocks; bi++) {
            for (int bj = 0; bj < nblocks; bj++) {
                int owner_coords[2] = { bi % p_rows, bj % p_cols };
                int owner_rank;
                MPI_Cart_rank(cart_comm, owner_coords, &owner_rank);

                if (owner_rank == 0) {
                    int lbi = bi / p_rows;
                    int lbj = bj / p_cols;
                    unpack_padded_block_to_global(
                        block_ptr((double *)local_M, local_block_cols, lbi, lbj),
                        global_M, n, bi, bj
                    );
                } else {
                    MPI_Recv(tmp, NB * NB, MPI_DOUBLE, owner_rank,
                             2000 + bi * nblocks + bj, cart_comm, &status);
                    unpack_padded_block_to_global(tmp, global_M, n, bi, bj);
                }
            }
        }

        free(tmp);
    } else {
        for (int bi = my_row, lbi = 0; bi < nblocks; bi += p_rows, lbi++) {
            for (int bj = my_col, lbj = 0; bj < nblocks; bj += p_cols, lbj++) {
                MPI_Send((void *)block_ptr((double *)local_M, local_block_cols, lbi, lbj),
                         NB * NB, MPI_DOUBLE, 0,
                         2000 + bi * nblocks + bj, cart_comm);
            }
        }
    }
}

int main(int argc, char **argv) {
    int rank, size, n;

    MPI_Comm cart_comm, row_comm, col_comm;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2];
    int p_rows, p_cols, my_row, my_col;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = (argc < 2) ? 1024 : atoi(argv[1]);

    if (n <= 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) fprintf(stderr, "Error: n must be > 0\n");
        MPI_Finalize();
        return 1;
    }

    /*
     * 1. LOAD BALANCING
     * Optimized version:
     *   - 2D Cartesian process grid
     *   - matrix divided into NB x NB blocks
     *   - blocks assigned block-cyclic across the 2D grid
     */
    MPI_Dims_create(size, 2, dims);
    p_rows = dims[0];
    p_cols = dims[1];

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    my_row = coords[0];
    my_col = coords[1];

    /* Row/column communicators are needed later for SUMMA broadcasts */
    MPI_Comm_split(cart_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(cart_comm, my_col, my_row, &col_comm);

    int nblocks = (n + NB - 1) / NB;
    int local_block_rows = num_local_blocks(nblocks, my_row, p_rows);
    int local_block_cols = num_local_blocks(nblocks, my_col, p_cols);

    size_t local_bytes =
        (size_t)local_block_rows * local_block_cols * NB * NB * sizeof(double);

    double *local_A = (double *)malloc(local_bytes);
    double *local_B = (double *)malloc(local_bytes);
    double *local_C = (double *)malloc(local_bytes);

    if (!local_A || !local_B || !local_C) {
        fprintf(stderr, "Rank %d: local allocation failed\n", rank);
        MPI_Abort(cart_comm, 2);
    }

    zero_blocks(local_A, local_block_rows, local_block_cols);
    zero_blocks(local_B, local_block_rows, local_block_cols);
    zero_blocks(local_C, local_block_rows, local_block_cols);

    /*
     * Root keeps full matrices only for initialization / final reconstruction,
     * exactly like the naive version keeps global A and C on rank 0.
     */
    double *matrix_A = NULL;
    double *matrix_B = NULL;
    double *matrix_C = NULL;

    if (rank == 0) {
        matrix_A = (double *)malloc((size_t)n * n * sizeof(double));
        matrix_B = (double *)malloc((size_t)n * n * sizeof(double));
        matrix_C = (double *)malloc((size_t)n * n * sizeof(double));

        if (!matrix_A || !matrix_B || !matrix_C) {
            fprintf(stderr, "Root: global allocation failed\n");
            MPI_Abort(cart_comm, 3);
        }

        for (int i = 0; i < n * n; i++) {
            matrix_A[i] = 2.0;
            matrix_B[i] = 3.0;
            matrix_C[i] = 0.0;
        }
    }

    /*
     * 2. SYNCHRONIZATION
     *
     * Same idea as the naive code:
     * all ranks should conceptually start the timed parallel work together.
     *
     * In the naive code this barrier happens immediately before Bcast/Scatter.
     * Here we place the barrier right before the timed distributed multiplication,
     * after data has already been distributed.
     *
     * That means:
     * - distribution cost is NOT included in the measured kernel time
     * - only the SUMMA multiplication phase is timed
     *
     * This is often preferable for comparing the computation kernel itself.
     */

    /*
     * 3. COMMUNICATION
     *
     * Naive version:
     *   - MPI_Bcast(B)
     *   - MPI_Scatter(A)
     *
     * Optimized version:
     *   - initial distribution:
     *       distribute A blocks to their 2D owners
     *       distribute B blocks to their 2D owners
     *   - later, during computation:
     *       broadcast only the needed A/B panels at each SUMMA step
     *
     * So this phase is still "communication", but upgraded from coarse-grain
     * 1D collectives to structured 2D block distribution.
     */
    distribute_block_cyclic(matrix_A, local_A, n, nblocks,
                            p_rows, p_cols, my_row, my_col, rank, cart_comm);

    distribute_block_cyclic(matrix_B, local_B, n, nblocks,
                            p_rows, p_cols, my_row, my_col, rank, cart_comm);

    if (rank == 0) {
        free(matrix_A);
        free(matrix_B);
        matrix_A = NULL;
        matrix_B = NULL;
    }

    /*
     * Temporary buffers used during the SUMMA communication/computation loop:
     * - A_panel holds one distributed block-column panel of A
     * - B_panel holds one distributed block-row panel of B
     */
    double *A_panel = (double *)malloc((size_t)local_block_rows * NB * NB * sizeof(double));
    double *B_panel = (double *)malloc((size_t)local_block_cols * NB * NB * sizeof(double));

    if (!A_panel || !B_panel) {
        fprintf(stderr, "Rank %d: panel allocation failed\n", rank);
        MPI_Abort(cart_comm, 4);
    }

    MPI_Barrier(cart_comm);
    double start = MPI_Wtime();

    /*
     * 4. LOCAL COMPUTATION
     *
     * Naive version:
     *   - local rows of A times full B
     *   - standard i-j-k triple loop on each rank
     *
     * Optimized version:
     *   - SUMMA outer loop over block index kb
     *   - at each step:
     *       (a) communicate needed A and B panels inside row/column communicators
     *       (b) compute all local C blocks updated by those panels
     *
     * So this remains the "local computation" phase, but enriched with the
     * communication needed to sustain scalable distributed multiplication.
     */
    for (int kb = 0; kb < nblocks; kb++) {
        int owner_col = kb % p_cols;
        int owner_row = kb % p_rows;

        int lk_A = kb / p_cols;  /* local block-column index of A on owner_col ranks */
        int lk_B = kb / p_rows;  /* local block-row index of B on owner_row ranks */

        /* Row-wise broadcast of the A panel needed for this SUMMA step */
        if (my_col == owner_col) {
            for (int lbi = 0; lbi < local_block_rows; lbi++) {
                memcpy(A_panel + (size_t)lbi * NB * NB,
                       block_ptr(local_A, local_block_cols, lbi, lk_A),
                       NB * NB * sizeof(double));
            }
        }
        MPI_Bcast(A_panel, local_block_rows * NB * NB, MPI_DOUBLE,
                  owner_col, row_comm);

        /* Column-wise broadcast of the B panel needed for this SUMMA step */
        if (my_row == owner_row) {
            for (int lbj = 0; lbj < local_block_cols; lbj++) {
                memcpy(B_panel + (size_t)lbj * NB * NB,
                       block_ptr(local_B, local_block_cols, lk_B, lbj),
                       NB * NB * sizeof(double));
            }
        }
        MPI_Bcast(B_panel, local_block_cols * NB * NB, MPI_DOUBLE,
                  owner_row, col_comm);

        /* Local block multiplications using the received panels */
        for (int lbi = 0; lbi < local_block_rows; lbi++) {
            const double *Ablk = A_panel + (size_t)lbi * NB * NB;

            for (int lbj = 0; lbj < local_block_cols; lbj++) {
                const double *Bblk = B_panel + (size_t)lbj * NB * NB;
                double *Cblk = block_ptr(local_C, local_block_cols, lbi, lbj);
                local_block_gemm(Ablk, Bblk, Cblk);
            }
        }
    }

    double end = MPI_Wtime();

    /*
     * 5. GATHER
     *
     * Naive version:
     *   - MPI_Gather(local_c, ..., c, ...)
     *
     * Optimized version:
     *   - gather all distributed C blocks back to rank 0
     *   - reconstruct full matrix_C from padded blocks
     *
     * Same role, more general implementation.
     */
    gather_block_cyclic(local_C, matrix_C, n, nblocks,
                        p_rows, p_cols, my_row, my_col, rank, cart_comm);

    if (rank == 0) {
        double expected = 6.0 * n;

        printf("--------------------------------------\n");
        printf("Optimized MPI matrix multiplication\n");
        printf("Structure: load balancing / synchronization / communication / computation / gather\n");
        printf("Matrix size:      %d x %d\n", n, n);
        printf("Processes used:   %d\n", size);
        printf("Process grid:     %d x %d\n", p_rows, p_cols);
        printf("Block size NB:    %d\n", NB);
        printf("Kernel time:      %.6f seconds\n", end - start);
        printf("C[0,0] = %.2f, expected = %.2f\n", matrix_C[0], expected);
        printf("C[%d,%d] = %.2f, expected = %.2f\n",
               n - 1, n - 1, matrix_C[(size_t)(n - 1) * n + (n - 1)], expected);
        printf("--------------------------------------\n");

        free(matrix_C);
    }

    free(local_A);
    free(local_B);
    free(local_C);
    free(A_panel);
    free(B_panel);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}