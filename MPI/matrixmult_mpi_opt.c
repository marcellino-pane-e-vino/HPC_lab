#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define NB 32
#define MIN(a,b) ((a)<(b)?(a):(b))

/*
 * More general ScaLAPACK-like distributed matrix multiplication using:
 * - 2D Cartesian process grid
 * - row/column communicators
 * - 2D block-cyclic distribution
 * - SUMMA-style panel broadcasts
 * - support for arbitrary n via padded edge blocks
 *
 * Notes:
 * - square matrices only
 * - no external libraries
 * - local storage keeps each block as NB x NB, even edge blocks
 *   (invalid entries are padded with zeros)
 */

static int num_local_blocks(int nblocks, int coord, int nprocs_dim) {
    int count = 0;
    for (int b = coord; b < nblocks; b += nprocs_dim) count++;
    return count;
}

static int block_extent(int block_idx, int n) {
    int start = block_idx * NB;
    int rem = n - start;
    return (rem >= NB) ? NB : (rem > 0 ? rem : 0);
}

static inline double *block_ptr(double *M, int local_block_cols, int lbi, int lbj) {
    return M + ((size_t)lbi * local_block_cols + lbj) * NB * NB;
}

static void zero_blocks(double *M, int lbr, int lbc) {
    memset(M, 0, (size_t)lbr * lbc * NB * NB * sizeof(double));
}

static void zero_block(double *blk) {
    memset(blk, 0, NB * NB * sizeof(double));
}

static void local_block_gemm(const double *Ablk, const double *Bblk, double *Cblk) {
    // Fixed NB x NB multiply; edge blocks are zero-padded.
    for (int i = 0; i < NB; i++) {
        for (int k = 0; k < NB; k++) {
            double aik = Ablk[i * NB + k];
            for (int j = 0; j < NB; j++) {
                Cblk[i * NB + j] += aik * Bblk[k * NB + j];
            }
        }
    }
}

static void pack_global_block_to_padded(
    const double *global_M,
    int n,
    int bi,
    int bj,
    double *dst_padded
) {
    int rows = block_extent(bi, n);
    int cols = block_extent(bj, n);

    zero_block(dst_padded);

    int global_row0 = bi * NB;
    int global_col0 = bj * NB;

    for (int i = 0; i < rows; i++) {
        memcpy(dst_padded + i * NB,
               global_M + (size_t)(global_row0 + i) * n + global_col0,
               cols * sizeof(double));
    }
}

static void unpack_padded_block_to_global(
    const double *src_padded,
    double *global_M,
    int n,
    int bi,
    int bj
) {
    int rows = block_extent(bi, n);
    int cols = block_extent(bj, n);

    int global_row0 = bi * NB;
    int global_col0 = bj * NB;

    for (int i = 0; i < rows; i++) {
        memcpy(global_M + (size_t)(global_row0 + i) * n + global_col0,
               src_padded + i * NB,
               cols * sizeof(double));
    }
}

static void distribute_block_cyclic(
    const double *global_M,   // only valid on rank 0
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

    int local_block_rows = num_local_blocks(nblocks, my_row, p_rows);
    int local_block_cols = num_local_blocks(nblocks, my_col, p_cols);

    if (rank == 0) {
        double *tmp = (double *)malloc(NB * NB * sizeof(double));
        if (!tmp) {
            fprintf(stderr, "Root: temporary pack buffer allocation failed\n");
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
                    double *dst = block_ptr(local_M, local_block_cols, lbi, lbj);
                    memcpy(dst, tmp, NB * NB * sizeof(double));
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
                double *dst = block_ptr(local_M, local_block_cols, lbi, lbj);
                MPI_Recv(dst, NB * NB, MPI_DOUBLE, 0,
                         1000 + bi * nblocks + bj, cart_comm, &status);
            }
        }
    }
}

static void gather_block_cyclic(
    const double *local_M,
    double *global_M,      // only valid on rank 0
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

    int local_block_rows = num_local_blocks(nblocks, my_row, p_rows);
    int local_block_cols = num_local_blocks(nblocks, my_col, p_cols);

    if (rank == 0) {
        double *tmp = (double *)malloc(NB * NB * sizeof(double));
        if (!tmp) {
            fprintf(stderr, "Root: temporary unpack buffer allocation failed\n");
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
                    const double *src = block_ptr((double *)local_M, local_block_cols, lbi, lbj);
                    unpack_padded_block_to_global(src, global_M, n, bi, bj);
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
                const double *src = block_ptr((double *)local_M, local_block_cols, lbi, lbj);
                MPI_Send((void *)src, NB * NB, MPI_DOUBLE, 0,
                         2000 + bi * nblocks + bj, cart_comm);
            }
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    int n;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = (argc < 2) ? 1024 : atoi(argv[1]);

    if (n <= 0) {
        int wrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
        if (wrank == 0) {
            fprintf(stderr, "Error: n must be > 0\n");
        }
        MPI_Finalize();
        return 1;
    }

    // General version: ceil(n / NB)
    int nblocks = (n + NB - 1) / NB;

    // 1) 2D process grid
    MPI_Dims_create(size, 2, dims);
    int p_rows = dims[0];
    int p_cols = dims[1];

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int my_row = coords[0];
    int my_col = coords[1];

    // 2) Row/column communicators
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(cart_comm, my_col, my_row, &col_comm);

    int local_block_rows = num_local_blocks(nblocks, my_row, p_rows);
    int local_block_cols = num_local_blocks(nblocks, my_col, p_cols);

    double *local_A = (double *)malloc((size_t)local_block_rows * local_block_cols * NB * NB * sizeof(double));
    double *local_B = (double *)malloc((size_t)local_block_rows * local_block_cols * NB * NB * sizeof(double));
    double *local_C = (double *)malloc((size_t)local_block_rows * local_block_cols * NB * NB * sizeof(double));

    if (!local_A || !local_B || !local_C) {
        fprintf(stderr, "Rank %d: allocation failed\n", rank);
        MPI_Abort(cart_comm, 2);
    }

    zero_blocks(local_A, local_block_rows, local_block_cols);
    zero_blocks(local_B, local_block_rows, local_block_cols);
    zero_blocks(local_C, local_block_rows, local_block_cols);

    double *matrix_A = NULL, *matrix_B = NULL, *matrix_C = NULL;
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
        }
    }

    // 3) Distribute with 2D block-cyclic ownership
    distribute_block_cyclic(matrix_A, local_A, n, nblocks, p_rows, p_cols,
                            my_row, my_col, rank, cart_comm);
    distribute_block_cyclic(matrix_B, local_B, n, nblocks, p_rows, p_cols,
                            my_row, my_col, rank, cart_comm);

    // SUMMA panel buffers
    double *A_panel = (double *)malloc((size_t)local_block_rows * NB * NB * sizeof(double));
    double *B_panel = (double *)malloc((size_t)local_block_cols * NB * NB * sizeof(double));

    if (!A_panel || !B_panel) {
        fprintf(stderr, "Rank %d: panel allocation failed\n", rank);
        MPI_Abort(cart_comm, 4);
    }

    MPI_Barrier(cart_comm);
    double start_time = MPI_Wtime();

    // 4) SUMMA over block index k
    for (int kb = 0; kb < nblocks; kb++) {
        int owner_col = kb % p_cols;  // owner process-column for A(:,kb)
        int owner_row = kb % p_rows;  // owner process-row for B(kb,:)

        int lk_A = kb / p_cols;       // local block-column index on owner_col
        int lk_B = kb / p_rows;       // local block-row index on owner_row

        // Pack A-panel on the row root, then broadcast across row_comm
        if (my_col == owner_col) {
            for (int lbi = 0; lbi < local_block_rows; lbi++) {
                double *src = block_ptr(local_A, local_block_cols, lbi, lk_A);
                memcpy(A_panel + (size_t)lbi * NB * NB, src, NB * NB * sizeof(double));
            }
        }
        MPI_Bcast(A_panel, local_block_rows * NB * NB, MPI_DOUBLE, owner_col, row_comm);

        // Pack B-panel on the column root, then broadcast across col_comm
        if (my_row == owner_row) {
            for (int lbj = 0; lbj < local_block_cols; lbj++) {
                double *src = block_ptr(local_B, local_block_cols, lk_B, lbj);
                memcpy(B_panel + (size_t)lbj * NB * NB, src, NB * NB * sizeof(double));
            }
        }
        MPI_Bcast(B_panel, local_block_cols * NB * NB, MPI_DOUBLE, owner_row, col_comm);

        // Local update of all owned C blocks
        for (int lbi = 0; lbi < local_block_rows; lbi++) {
            const double *Ablk = A_panel + (size_t)lbi * NB * NB;
            for (int lbj = 0; lbj < local_block_cols; lbj++) {
                const double *Bblk = B_panel + (size_t)lbj * NB * NB;
                double *Cblk = block_ptr(local_C, local_block_cols, lbi, lbj);
                local_block_gemm(Ablk, Bblk, Cblk);
            }
        }
    }

    double end_time = MPI_Wtime();

    // 5) Gather result back
    gather_block_cyclic(local_C, matrix_C, n, nblocks, p_rows, p_cols,
                        my_row, my_col, rank, cart_comm);

    if (rank == 0) {
        printf("ScaLAPACK-like SUMMA (general n, padded edge blocks) - Matrix: %d, Procs: %d (%dx%d), NB: %d, Time: %f s\n",
               n, size, p_rows, p_cols, NB, end_time - start_time);

        // For the current initialization, each valid entry should be 6*n
        double expected = 6.0 * n;
        printf("C[0,0] = %.2f, expected = %.2f\n", matrix_C[0], expected);
        printf("C[%d,%d] = %.2f, expected = %.2f\n", n-1, n-1, matrix_C[(size_t)(n-1)*n + (n-1)], expected);

        free(matrix_A);
        free(matrix_B);
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