#define NB 32
#define MIN(a,b) ((a)<(b)?(a):(b))

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
void check_result(double *c_global, int n, double val_a, double val_b);
int main(int argc, char **argv) {
    // 1. INITIALIZATION + LOAD BALANCING
    int rank, size, n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 2) n = 1000; else n = atoi(argv[1]);
    // creating 2d grid
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
    MPI_Dims_create(size, 2, dims);
    int p_rows = dims[0];
    int p_cols = dims[1];
    // creating communicators
    MPI_Comm cart_comm, row_comm, col_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];
    MPI_Comm_split(cart_comm, my_row, my_col, &row_comm); // Creating custom communicator for column brodcast
    MPI_Comm_split(cart_comm, my_col, my_row, &col_comm); // Creating custom communicator for row brodcast
    // Simplified load balancing: exact 2D blocks
    int local_rows = n / p_rows;
    int local_cols = n / p_cols;
    int block_size = local_rows * local_cols;

    // 2. MEMORY ALLOCATION
    // local matrices blocks
    double *a_local = malloc(block_size * sizeof(double));
    double *b_local = malloc(block_size * sizeof(double));
    double *c_local = calloc(block_size, sizeof(double)); // every process is responsible of computing this
    // blocks to be brodcasted to/from
    double *a_buffer = malloc(block_size * sizeof(double));
    double *b_buffer = malloc(block_size * sizeof(double));
    // rank 0 initialises the whole global matrices
    double *a_global = NULL, *b_global = NULL, *c_global = NULL;
    if (rank == 0) {
        a_global = malloc(n * n * sizeof(double));
        b_global = malloc(n * n * sizeof(double));
        c_global = calloc(n * n, sizeof(double));
        for (int i = 0; i < n * n; i++) { a_global[i] = 2.0; b_global[i] = 3.0; }
    }

    // 3. SYNCHRONIZATION AND DISTRIBUTION
    MPI_Barrier(cart_comm); 
    double start = MPI_Wtime();
    // Manual 2D distribution from rank 0 from global to local matrices
    if (rank == 0) {
        double *tmp_buf = malloc(block_size * sizeof(double));
        for (int pr = 0; pr < p_rows; pr++) {
            for (int pc = 0; pc < p_cols; pc++) {
                int dest_coords[2] = {pr, pc};
                int dest_rank;
                MPI_Cart_rank(cart_comm, dest_coords, &dest_rank);
                // Extract local blocks from a_global and b_global
                for(int m = 0; m < 2; m++) {
                    double *source_matrix = (m == 0) ? a_global : b_global;
                    for (int i = 0; i < local_rows; i++) {
                        memcpy(tmp_buf + i * local_cols, 
                               source_matrix + ((pr * local_rows + i) * n) + (pc * local_cols), 
                               local_cols * sizeof(double));
                    }
                    if (dest_rank == 0) {
                        memcpy((m == 0) ? a_local : b_local, tmp_buf, block_size * sizeof(double));
                    } else {
                        MPI_Send(tmp_buf, block_size, MPI_DOUBLE, dest_rank, m, cart_comm);
                    }
                }
            }
        }
        free(tmp_buf);
    // ranks different from 0 recieve their respective local blocs
    } else {
        MPI_Recv(a_local, block_size, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
        MPI_Recv(b_local, block_size, MPI_DOUBLE, 0, 1, cart_comm, MPI_STATUS_IGNORE);
    }
    // 4. LOCAL COMPUTATION (SUMMA)
    for (int k = 0; k < p_cols; k++) {
        // 4.1 Row broadcast to get block for A
        if (my_col == k) memcpy(a_buffer, a_local, block_size * sizeof(double)); // if this process owns the block correspondint to K i broadcast it to the others 
        MPI_Bcast(a_buffer, block_size, MPI_DOUBLE, k, row_comm); // depending on K value i "give" or "take" the block needed for this iteration
        //4.1 Column broadcast to get block for B
        if (my_row == k) memcpy(b_buffer, b_local, block_size * sizeof(double)); // if this process owns the block correspondint to K i broadcast it to the others 
        MPI_Bcast(b_buffer, block_size, MPI_DOUBLE, k, col_comm); // depending on K value i "give" or "take" the block needed for this iteration

        // 4.3 Tiled local multiplication
        for (int ii = 0; ii < local_rows; ii += NB) {
            for (int kk = 0; kk < local_cols; kk += NB) {
                for (int jj = 0; jj < local_cols; jj += NB) {
                    // Clamp tile boundaries to prevent out-of-bounds access for non-multiples of NB.
                    int i_max = MIN(ii + NB, local_rows);
                    int k_max = MIN(kk + NB, local_cols);
                    int j_max = MIN(jj + NB, local_cols);
                    // internal loop (as seen in sequential optimization)
                    for (int i = ii; i < i_max; i++) {
                        for (int k_idx = kk; k_idx < k_max; k_idx++) {
                            double a_ik = a_buffer[i * local_cols + k_idx];
                            for (int j = jj; j < j_max; j++) {
                                c_local[i * local_cols + j] += a_ik * b_buffer[k_idx * local_cols + j];
                            }
                        }
                    }
                }
            }
        }
    }

    // 5. GATHER + OUTPUT
    if (rank == 0) {
        double *tmp_buf = malloc(block_size * sizeof(double));
        for (int pr = 0; pr < p_rows; pr++) {
            for (int pc = 0; pc < p_cols; pc++) {
                int src_coords[2] = {pr, pc};
                int src_rank;
                MPI_Cart_rank(cart_comm, src_coords, &src_rank);

                if (src_rank == 0) {
                    memcpy(tmp_buf, c_local, block_size * sizeof(double));
                } else {
                    MPI_Recv(tmp_buf, block_size, MPI_DOUBLE, src_rank, 2, cart_comm, MPI_STATUS_IGNORE);
                }

                // Paste c blocks into global output
                for (int i = 0; i < local_rows; i++) {
                    memcpy(c_global + ((pr * local_rows + i) * n) + (pc * local_cols), 
                           tmp_buf + i * local_cols, 
                           local_cols * sizeof(double));
                }
            }
        }
        free(tmp_buf);
        
        double end = MPI_Wtime();
        printf("--------------------------------------\n");
        printf("Matrix Size: %d x %d (SUMMA Template)\n", n, n);
        printf("Processes used: %d (Grid %dx%d)\n", size, p_rows, p_cols);
        printf("Total time:   %.6f sec\n", end - start);
        printf("C[0,0] = %.2f\n", c_global[0]);
        printf("--------------------------------------\n");
        
        check_result(c_global, n, 2.0, 3.0);

        free(a_global); free(b_global); free(c_global);
    } else {
        MPI_Send(c_local, block_size, MPI_DOUBLE, 0, 2, cart_comm);
    }

    // 6. CLEANUP
    free(a_local); free(b_local); free(c_local);
    free(a_buffer); free(b_buffer);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    
    MPI_Finalize();
    return 0;
}

void check_result(double *c_global, int n, double val_a, double val_b) {
    if (c_global == NULL) return;

    double expected = (double)n * val_a * val_b;
    int errors = 0;

    // Define strategic indices to sample
    size_t indices_to_check[5];
    const char *labels[5] = {
        "Top-Left (0,0)", 
        "Top-Right (0, n-1)", 
        "Center (n/2, n/2)", 
        "Bottom-Left (n-1, 0)", 
        "Bottom-Right (n-1, n-1)"
    };

    indices_to_check[0] = 0;                                 // (0,0)
    indices_to_check[1] = (size_t)n - 1;                     // (0, n-1)
    indices_to_check[2] = (size_t)(n / 2) * n + (n / 2);     // Middle
    indices_to_check[3] = (size_t)(n - 1) * n;               // (n-1, 0)
    indices_to_check[4] = (size_t)n * n - 1;                 // (n-1, n-1)

    printf(">>> STARTING SAMPLED VERIFICATION (Checking 5 points)...\n");

    for (int i = 0; i < 5; i++) {
        size_t idx = indices_to_check[i];
        if (c_global[idx] != expected) {
            printf("Verification FAILED at %s (index %zu): expected %.2f, found %.2f\n", 
                   labels[i], idx, expected, c_global[idx]);
            errors++;
        } else {
            printf("Verification PASSED at %s: %.2f\n", labels[i], c_global[idx]);
        }
    }

    if (errors == 0) {
        printf(">>> SAMPLED VERIFICATION SUCCESSFUL.\n");
    } else {
        printf(">>> SAMPLED VERIFICATION FAILED: %d errors found.\n", errors);
    }
}