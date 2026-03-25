#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define NB 32

int main(int argc, char **argv) {
    // ==========================================
    // 1. Initialization + Load balancing (con Padding)
    // ==========================================
    int rank, size, n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 2) n = 1000; else n = atoi(argv[1]);

    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
    MPI_Dims_create(size, 2, dims);
    int p_rows = dims[0];
    int p_cols = dims[1];

    MPI_Comm cart_comm, row_comm, col_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    MPI_Comm_split(cart_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(cart_comm, my_col, my_row, &col_comm);

    // LOGICA DI PADDING: n_pad deve essere un multiplo perfetto di (NB * griglia)
    // Questo garantisce che ogni processo riceva ESATTAMENTE lo stesso numero di blocchi.
    int pad_factor = NB * p_rows * p_cols; 
    int n_pad = ((n + pad_factor - 1) / pad_factor) * pad_factor;

    int local_rows = n_pad / p_rows;
    int local_cols = n_pad / p_cols;
    int local_size = local_rows * local_cols;

    // ==========================================
    // 2. Memory allocation
    // ==========================================
    // Usiamo calloc per garantire che il "padding" vuoto sia inizializzato a 0.0
    double *local_a = calloc(local_size, sizeof(double));
    double *local_b = calloc(local_size, sizeof(double));
    double *local_c = calloc(local_size, sizeof(double));
    
    // Pannelli per il broadcast durante SUMMA
    double *panel_a = calloc(local_rows * NB, sizeof(double));
    double *panel_b = calloc(NB * local_cols, sizeof(double));

    double *a_global = NULL, *b_global = NULL, *c_global = NULL;
    if (rank == 0) {
        a_global = calloc(n_pad * n_pad, sizeof(double));
        b_global = calloc(n_pad * n_pad, sizeof(double));
        c_global = calloc(n_pad * n_pad, sizeof(double));
        
        // Inizializziamo SOLO la matrice reale "n x n". 
        // Il resto della memoria (il padding) rimane a 0.0 e non intaccherà i calcoli.
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a_global[i * n_pad + j] = 2.0;
                b_global[i * n_pad + j] = 3.0;
            }
        }
    }

    // ==========================================
    // 3. Synchronization and distribution (MPI NATIVO)
    // ==========================================
    MPI_Barrier(cart_comm); 
    double start = MPI_Wtime();

    // Definiamo i parametri per la magia del tipo DARRAY (Distributed Array) di MPI
    int gsize[2] = {n_pad, n_pad};
    int distrib[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC};
    int dargs[2] = {NB, NB};
    int psize[2] = {p_rows, p_cols};

    if (rank == 0) {
        for (int p = 0; p < size; p++) {
            MPI_Datatype darray;
            // MPI crea la mappa Block-Cyclic perfetta per il rank 'p'
            MPI_Type_create_darray(size, p, 2, gsize, distrib, dargs, psize, MPI_ORDER_C, MPI_DOUBLE, &darray);
            MPI_Type_commit(&darray);

            if (p == 0) {
                // Il rank 0 scambia i dati con se stesso usando Sendrecv per evitare deadlock
                MPI_Sendrecv(a_global, 1, darray, 0, 0, local_a, local_size, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
                MPI_Sendrecv(b_global, 1, darray, 0, 1, local_b, local_size, MPI_DOUBLE, 0, 1, cart_comm, MPI_STATUS_IGNORE);
            } else {
                MPI_Send(a_global, 1, darray, p, 0, cart_comm);
                MPI_Send(b_global, 1, darray, p, 1, cart_comm);
            }
            MPI_Type_free(&darray);
        }
    } else {
        MPI_Recv(local_a, local_size, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
        MPI_Recv(local_b, local_size, MPI_DOUBLE, 0, 1, cart_comm, MPI_STATUS_IGNORE);
    }

    // ==========================================
    // 4. Local computation (SUMMA Block-Cyclic)
    // ==========================================
    int num_blocks = n_pad / NB; // Passi totali dell'algoritmo SUMMA
    
    for (int k = 0; k < num_blocks; k++) {
        int owner_col = k % p_cols;
        int owner_row = k % p_rows;
        int lk_col = k / p_cols; // Indice locale della colonna per chi la possiede
        int lk_row = k / p_rows; // Indice locale della riga per chi la possiede

        // 4.1 Broadcast del Pannello A (Striscia Verticale)
        if (my_col == owner_col) {
            // Estraiamo la colonna di blocchi saltando tra le righe locali
            for (int i = 0; i < local_rows; i++) {
                memcpy(&panel_a[i * NB], &local_a[i * local_cols + (lk_col * NB)], NB * sizeof(double));
            }
        }
        MPI_Bcast(panel_a, local_rows * NB, MPI_DOUBLE, owner_col, row_comm);

        // 4.2 Broadcast del Pannello B (Striscia Orizzontale)
        if (my_row == owner_row) {
            // La riga di blocchi è fortunatamente contigua in memoria (Row-Major)
            memcpy(panel_b, &local_b[(lk_row * NB) * local_cols], NB * local_cols * sizeof(double));
        }
        MPI_Bcast(panel_b, NB * local_cols, MPI_DOUBLE, owner_row, col_comm);

        // 4.3 GEMM Tassellata (Il Padding ci salva dai controlli sui bordi!)
        for (int ii = 0; ii < local_rows; ii += NB) {
            for (int jj = 0; jj < local_cols; jj += NB) {
                for (int i = ii; i < ii + NB; i++) {
                    for (int k_idx = 0; k_idx < NB; k_idx++) {
                        double a_ik = panel_a[i * NB + k_idx];
                        for (int j = jj; j < jj + NB; j++) {
                            local_c[i * local_cols + j] += a_ik * panel_b[k_idx * local_cols + j];
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
        for (int p = 0; p < size; p++) {
            MPI_Datatype darray;
            MPI_Type_create_darray(size, p, 2, gsize, distrib, dargs, psize, MPI_ORDER_C, MPI_DOUBLE, &darray);
            MPI_Type_commit(&darray);

            if (p == 0) {
                MPI_Sendrecv(local_c, local_size, MPI_DOUBLE, 0, 2, c_global, 1, darray, 0, 2, cart_comm, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(c_global, 1, darray, p, 2, cart_comm, MPI_STATUS_IGNORE);
            }
            MPI_Type_free(&darray);
        }
        
        double end = MPI_Wtime();
        printf("--------------------------------------\n");
        printf("Taglia Originale:   %d x %d\n", n, n);
        printf("Taglia con Padding: %d x %d (Block-Cyclic NATIVO)\n", n_pad, n_pad);
        printf("Processi usati:     %d (Griglia %dx%d)\n", size, p_rows, p_cols);
        printf("Tempo Totale:       %.6f secondi\n", end - start);
        // Indice 0,0 è garantito essere intatto dal padding
        printf("Controllo C[0,0] =  %.2f\n", c_global[0]); 
        printf("--------------------------------------\n");
        
        free(a_global); free(b_global); free(c_global);
    } else {
        MPI_Send(local_c, local_size, MPI_DOUBLE, 0, 2, cart_comm);
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