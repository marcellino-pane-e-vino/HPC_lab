#define NB 32
#define MIN(a,b) ((a)<(b)?(a):(b))

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char **argv) {
    // ==========================================
    // 1. Initialization + Load balancing
    // ==========================================
    int rank, size, n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 2) n = 1000; else n = atoi(argv[1]);

    // Creazione della griglia cartesiana 2D
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
    MPI_Dims_create(size, 2, dims); // Crea automaticamente una griglia (es. 2x2 se size=4)
    int p_rows = dims[0];
    int p_cols = dims[1];

    MPI_Comm cart_comm, row_comm, col_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    // Sdoppiamento comunicatori per i broadcast di riga e colonna di SUMMA
    MPI_Comm_split(cart_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(cart_comm, my_col, my_row, &col_comm);

    // Load Balancing semplificato per il template: blocchi 2D esatti
    int local_rows = n / p_rows;
    int local_cols = n / p_cols;
    int block_size = local_rows * local_cols;

    // ==========================================
    // 2. Memory allocation
    // ==========================================
    double *local_a = malloc(block_size * sizeof(double));
    double *local_b = malloc(block_size * sizeof(double));
    double *local_c = calloc(block_size, sizeof(double)); // Inizializzato a 0
    
    // Buffer per ricevere i "pannelli" durante SUMMA
    double *panel_a = malloc(block_size * sizeof(double));
    double *panel_b = malloc(block_size * sizeof(double));

    double *a = NULL, *b_global = NULL, *c = NULL;
    if (rank == 0) {
        a = malloc(n * n * sizeof(double));
        b_global = malloc(n * n * sizeof(double));
        c = calloc(n * n, sizeof(double));
        for (int i = 0; i < n * n; i++) { a[i] = 2.0; b_global[i] = 3.0; }
    }

    // ==========================================
    // 3. Synchronization and distribution
    // ==========================================
    MPI_Barrier(cart_comm); 
    double start = MPI_Wtime();

    // Distribuzione manuale dei blocchi 2D da rank 0 a tutti gli altri
    if (rank == 0) {
        double *tmp_buf = malloc(block_size * sizeof(double));
        for (int pr = 0; pr < p_rows; pr++) {
            for (int pc = 0; pc < p_cols; pc++) {
                int dest_coords[2] = {pr, pc};
                int dest_rank;
                MPI_Cart_rank(cart_comm, dest_coords, &dest_rank);

                // Estrai il blocco per A e per B
                for(int m = 0; m < 2; m++) {
                    double *source_matrix = (m == 0) ? a : b_global;
                    for (int i = 0; i < local_rows; i++) {
                        memcpy(tmp_buf + i * local_cols, 
                               source_matrix + ((pr * local_rows + i) * n) + (pc * local_cols), 
                               local_cols * sizeof(double));
                    }
                    if (dest_rank == 0) {
                        memcpy((m == 0) ? local_a : local_b, tmp_buf, block_size * sizeof(double));
                    } else {
                        MPI_Send(tmp_buf, block_size, MPI_DOUBLE, dest_rank, m, cart_comm);
                    }
                }
            }
        }
        free(tmp_buf);
    } else {
        MPI_Recv(local_a, block_size, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
        MPI_Recv(local_b, block_size, MPI_DOUBLE, 0, 1, cart_comm, MPI_STATUS_IGNORE);
    }

    // ==========================================
    // 4. Local computation (SUMMA Core con Tassellazione)
    // ==========================================
    // Assumiamo griglia quadrata p_rows == p_cols per la moltiplicazione standard
    for (int k = 0; k < p_cols; k++) {
        // 4.1 Broadcast riga del blocco di A 
        if (my_col == k) memcpy(panel_a, local_a, block_size * sizeof(double)); //se sono il proprietario del blocco corrispondente a K lo metto nella variabile da broadcastare
        MPI_Bcast(panel_a, block_size, MPI_DOUBLE, k, row_comm); // in base al valore di K ""prende dalla piazza" o "mette "in piazza" il blocco di A che serve per questa iterazione

        // 4.2 Broadcast colonna del blocco di B
        if (my_row == k) memcpy(panel_b, local_b, block_size * sizeof(double)); //se sono il proprietario del blocco corrispondente a K lo metto nella variabile da broadcastare
        MPI_Bcast(panel_b, block_size, MPI_DOUBLE, k, col_comm); //in base al valore di K "prende dalla piazza" o "mette in piazza" il blocco di B che serve per questa iterazione

        // 4.3 Moltiplicazione locale Tassellata (Cache Blocking)
        for (int ii = 0; ii < local_rows; ii += NB) {
            for (int kk = 0; kk < local_cols; kk += NB) {
                for (int jj = 0; jj < local_cols; jj += NB) {
                    
                    // Calcoliamo i limiti della finestrella per non uscire fuori dai bordi (Utile se la sottomatrice non è un multiplo esatto di NB)
                    int i_max = MIN(ii + NB, local_rows);
                    int k_max = MIN(kk + NB, local_cols);
                    int j_max = MIN(jj + NB, local_cols);

                    // I 3 cicli INTERNI lavorano solo nel micro-blocco che sta in Cache L1
                    for (int i = ii; i < i_max; i++) {
                        for (int k_idx = kk; k_idx < k_max; k_idx++) {
                            double a_ik = panel_a[i * local_cols + k_idx];
                            for (int j = jj; j < j_max; j++) {
                                local_c[i * local_cols + j] += a_ik * panel_b[k_idx * local_cols + j];
                            }
                        }
                    }
                }
            }
        }
    }

    // ==========================================
    // 5. Gather + output
    // ==========================================
    if (rank == 0) {
        double *tmp_buf = malloc(block_size * sizeof(double));
        for (int pr = 0; pr < p_rows; pr++) {
            for (int pc = 0; pc < p_cols; pc++) {
                int src_coords[2] = {pr, pc};
                int src_rank;
                MPI_Cart_rank(cart_comm, src_coords, &src_rank);

                if (src_rank == 0) {
                    memcpy(tmp_buf, local_c, block_size * sizeof(double));
                } else {
                    MPI_Recv(tmp_buf, block_size, MPI_DOUBLE, src_rank, 2, cart_comm, MPI_STATUS_IGNORE);
                }

                // Incolla il blocco nella matrice globale C
                for (int i = 0; i < local_rows; i++) {
                    memcpy(c + ((pr * local_rows + i) * n) + (pc * local_cols), 
                           tmp_buf + i * local_cols, 
                           local_cols * sizeof(double));
                }
            }
        }
        free(tmp_buf);
        
        double end = MPI_Wtime();
        printf("--------------------------------------\n");
        printf("Taglia Matrice: %d x %d (SUMMA Template)\n", n, n);
        printf("Processi usati: %d (Griglia %dx%d)\n", size, p_rows, p_cols);
        printf("Tempo Totale:   %.6f secondi\n", end - start);
        printf("C[0,0] = %.2f\n", c[0]);
        printf("--------------------------------------\n");
        
        free(a); free(b_global); free(c);
    } else {
        MPI_Send(local_c, block_size, MPI_DOUBLE, 0, 2, cart_comm);
    }

    // ==========================================
    // 6. Cleanup
    // ==========================================
    free(local_a); free(local_b); free(local_c);
    free(panel_a); free(panel_b);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    
    MPI_Finalize();
    return 0;
}