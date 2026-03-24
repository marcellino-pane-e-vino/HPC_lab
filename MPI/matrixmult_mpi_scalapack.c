#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define TILE_SIZE 32

// Helper function to find the best dimensions for the process grid
void find_best_grid(int size, int *dims) {
    dims[0] = (int)sqrt(size);
    while (size % dims[0] != 0) dims[0]--;
    dims[1] = size / dims[0];
}

int main(int argc, char **argv) {
    int rank, size, n;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0}; // No wrap-around (non-periodic)
    int coords[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) n = 1024; else n = atoi(argv[1]);

    // 1. DYNAMIC RECTANGULAR GRID SETUP
    find_best_grid(size, dims);
    int p_rows = dims[0];
    int p_cols = dims[1];

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int my_row = coords[0];
    int my_col = coords[1];

    int local_n_rows = n / p_rows;
    int local_n_cols = n / p_cols;

    // 2. MEMORY ALLOCATION
    double *local_A = malloc(local_n_rows * local_n_cols * sizeof(double));
    double *local_B = malloc(local_n_rows * local_n_cols * sizeof(double));
    double *local_C = calloc(local_n_rows * local_n_cols, sizeof(double));

    double *matrix_A = NULL, *matrix_B = NULL, *matrix_C = NULL;

    if (rank == 0) {
        matrix_A = malloc(n * n * sizeof(double));
        matrix_B = malloc(n * n * sizeof(double));
        matrix_C = malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) { matrix_A[i] = 2.0; matrix_B[i] = 3.0; }
    }

    // 3. SCALAPACK-STYLE 2D DISTRIBUTION
    // Using Subarray types to handle any rectangular grid
    MPI_Datatype block_type, resized_block_type;
    int gsize[2] = {n, n};
    int lsize[2] = {local_n_rows, local_n_cols};
    int start[2] = {0, 0};

    MPI_Type_create_subarray(2, gsize, lsize, start, MPI_ORDER_C, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double), &resized_block_type);
    MPI_Type_commit(&resized_block_type);

    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int i = 0; i < p_rows; i++) {
            for (int j = 0; j < p_cols; j++) {
                int r = i * p_cols + j;
                counts[r] = 1;
                displs[r] = i * n * local_n_rows + j * local_n_cols;
            }
        }
    }

    MPI_Scatterv(matrix_A, counts, displs, resized_block_type, local_A, local_n_rows * local_n_cols, MPI_DOUBLE, 0, cart_comm);
    MPI_Scatterv(matrix_B, counts, displs, resized_block_type, local_B, local_n_rows * local_n_cols, MPI_DOUBLE, 0, cart_comm);

    // 4. ROW & COLUMN COMMUNICATORS (for Bcast-based multiplication)
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(cart_comm, my_col, my_row, &col_comm);

    MPI_Barrier(cart_comm);
    double start_time = MPI_Wtime();

    // 5. BLOCK MULTIPLICATION (General SUMMA Logic)
    double *temp_A = malloc(local_n_rows * local_n_cols * sizeof(double));
    double *temp_B = malloc(local_n_rows * local_n_cols * sizeof(double));

    for (int k = 0; k < p_cols; k++) {
        // Broadcast along rows and columns
        if (my_col == k % p_cols) for(int i=0; i<local_n_rows*local_n_cols; i++) temp_A[i] = local_A[i];
        MPI_Bcast(temp_A, local_n_rows * local_n_cols, MPI_DOUBLE, k % p_cols, row_comm);

        if (my_row == k % p_rows) for(int i=0; i<local_n_rows*local_n_cols; i++) temp_B[i] = local_B[i];
        MPI_Bcast(temp_B, local_n_rows * local_n_cols, MPI_DOUBLE, k % p_rows, col_comm);

        // Tiled local compute
        for (int i = 0; i < local_n_rows; i++)
            for (int kk = 0; kk < local_n_cols; kk++)
                for (int j = 0; j < local_n_cols; j++)
                    local_C[i * local_n_cols + j] += temp_A[i * local_n_cols + kk] * temp_B[kk * local_n_cols + j];
    }

    double end_time = MPI_Wtime();

    // 6. GATHER
    MPI_Gatherv(local_C, local_n_rows * local_n_cols, MPI_DOUBLE, matrix_C, counts, displs, resized_block_type, 0, cart_comm);

    if (rank == 0) {
        printf("ScaLAPACK-Style - Matrix: %d, Procs: %d (%dx%d), Time: %f s\n", n, size, p_rows, p_cols, end_time - start_time);
        free(matrix_A); free(matrix_B); free(matrix_C); free(counts); free(displs);
    }

    free(local_A); free(local_B); free(local_C); free(temp_A); free(temp_B);
    MPI_Type_free(&resized_block_type);
    MPI_Finalize();
    return 0;
}