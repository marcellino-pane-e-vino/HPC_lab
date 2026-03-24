#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define TILE_SIZE 32  // cache tile size

int main(int argc, char **argv) {
    int world_rank, world_size, matrix_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) matrix_size = 1024;
    else matrix_size = atoi(argv[1]);

    // --- 1. CREATE 2D PROCESS GRID ---
    int process_grid_dim = (int)sqrt(world_size);
    if (process_grid_dim * process_grid_dim != world_size) {
        if (world_rank == 0)
            printf("Number of processes must be a perfect square\n");
        MPI_Finalize();
        return 0;
    }

    int process_row = world_rank / process_grid_dim;
    int process_col = world_rank % process_grid_dim;

    int local_block_size = matrix_size / process_grid_dim;

    // --- 2. MEMORY ALLOCATION ---
    double *matrix_A = NULL, *matrix_B = NULL, *matrix_C = NULL;

    double *local_A = malloc(local_block_size * local_block_size * sizeof(double));
    double *local_B = malloc(local_block_size * local_block_size * sizeof(double));
    double *local_C = calloc(local_block_size * local_block_size, sizeof(double));

    if (world_rank == 0) {
        matrix_A = malloc(matrix_size * matrix_size * sizeof(double));
        matrix_B = malloc(matrix_size * matrix_size * sizeof(double));
        matrix_C = malloc(matrix_size * matrix_size * sizeof(double));

        for (int idx = 0; idx < matrix_size * matrix_size; idx++) {
            matrix_A[idx] = 2.0;
            matrix_B[idx] = 3.0;
        }
    }

    // --- 3. TRUE 2D SCATTER ---
    MPI_Datatype block_type, resized_block_type;

    int global_sizes[2] = {matrix_size, matrix_size};
    int local_sizes[2] = {local_block_size, local_block_size};
    int start_indices[2] = {0, 0};

    MPI_Type_create_subarray(2, global_sizes, local_sizes, start_indices,
                             MPI_ORDER_C, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double), &resized_block_type);
    MPI_Type_commit(&resized_block_type);

    int *send_counts = NULL;
    int *displacements = NULL;

    if (world_rank == 0) {
        send_counts = malloc(world_size * sizeof(int));
        displacements = malloc(world_size * sizeof(int));

        for (int grid_row = 0; grid_row < process_grid_dim; grid_row++) {
            for (int grid_col = 0; grid_col < process_grid_dim; grid_col++) {
                int proc_id = grid_row * process_grid_dim + grid_col;
                send_counts[proc_id] = 1;
                displacements[proc_id] =
                    grid_row * matrix_size * local_block_size +
                    grid_col * local_block_size;
            }
        }
    }

    MPI_Scatterv(matrix_A, send_counts, displacements, resized_block_type,
                 local_A, local_block_size * local_block_size, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(matrix_B, send_counts, displacements, resized_block_type,
                 local_B, local_block_size * local_block_size, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // --- 4. ROW & COLUMN COMMUNICATORS ---
    MPI_Comm row_communicator, col_communicator;

    MPI_Comm_split(MPI_COMM_WORLD, process_row, process_col, &row_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, process_col, process_row, &col_communicator);

    double *A_broadcast_block = malloc(local_block_size * local_block_size * sizeof(double));
    double *B_broadcast_block = malloc(local_block_size * local_block_size * sizeof(double));

    // --- 5. SUMMA ALGORITHM ---
    for (int step_k = 0; step_k < process_grid_dim; step_k++) {

        // Broadcast A block along row
        if (process_col == step_k) {
            for (int i = 0; i < local_block_size * local_block_size; i++)
                A_broadcast_block[i] = local_A[i];
        }

        MPI_Bcast(A_broadcast_block,
                  local_block_size * local_block_size,
                  MPI_DOUBLE,
                  step_k,
                  row_communicator);

        // Broadcast B block along column
        if (process_row == step_k) {
            for (int i = 0; i < local_block_size * local_block_size; i++)
                B_broadcast_block[i] = local_B[i];
        }

        MPI_Bcast(B_broadcast_block,
                  local_block_size * local_block_size,
                  MPI_DOUBLE,
                  step_k,
                  col_communicator);

        // ============================================================
        // OPTIMIZED VERSION: TILED LOCAL MULTIPLICATION
        // ============================================================
        for (int tile_i = 0; tile_i < local_block_size; tile_i += TILE_SIZE) {
            for (int tile_k = 0; tile_k < local_block_size; tile_k += TILE_SIZE) {
                for (int tile_j = 0; tile_j < local_block_size; tile_j += TILE_SIZE) {

                    int i_end = (tile_i + TILE_SIZE < local_block_size) ? tile_i + TILE_SIZE : local_block_size;
                    int k_end = (tile_k + TILE_SIZE < local_block_size) ? tile_k + TILE_SIZE : local_block_size;
                    int j_end = (tile_j + TILE_SIZE < local_block_size) ? tile_j + TILE_SIZE : local_block_size;

                    for (int i = tile_i; i < i_end; i++) {
                        for (int k_inner = tile_k; k_inner < k_end; k_inner++) {
                            double a_val = A_broadcast_block[i * local_block_size + k_inner];
                            for (int j = tile_j; j < j_end; j++) {
                                local_C[i * local_block_size + j] +=
                                    a_val * B_broadcast_block[k_inner * local_block_size + j];
                            }
                        }
                    }
                }
            }
        }

        // ============================================================
        // BASE VERSION (NO TILING) — COMMENTED
        // ============================================================
        /*
        for (int i = 0; i < local_block_size; i++) {
            for (int j = 0; j < local_block_size; j++) {
                for (int k_inner = 0; k_inner < local_block_size; k_inner++) {
                    local_C[i * local_block_size + j] +=
                        A_broadcast_block[i * local_block_size + k_inner] *
                        B_broadcast_block[k_inner * local_block_size + j];
                }
            }
        }
        */
    }

    double end_time = MPI_Wtime();

    // --- 6. GATHER RESULT ---
    MPI_Gatherv(local_C, local_block_size * local_block_size, MPI_DOUBLE,
                matrix_C, send_counts, displacements, resized_block_type,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("SUMMA + Tiling - n=%d, p=%d, tile=%d, time=%f s\n",
               matrix_size, world_size, TILE_SIZE, end_time - start_time);
    }

    // --- CLEANUP ---
    free(local_A); free(local_B); free(local_C);
    free(A_broadcast_block); free(B_broadcast_block);

    MPI_Type_free(&resized_block_type);

    if (world_rank == 0) {
        free(matrix_A); free(matrix_B); free(matrix_C);
        free(send_counts); free(displacements);
    }

    MPI_Finalize();
    return 0;
}