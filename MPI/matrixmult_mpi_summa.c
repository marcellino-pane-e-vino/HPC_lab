#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size, n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) n = 1024;
    else n = atoi(argv[1]);

    // --- 1. CREA GRIGLIA 2D ---
    int q = (int)sqrt(size);
    if (q * q != size) {
        if (rank == 0) printf("Numero di processi deve essere quadrato perfetto\n");
        MPI_Finalize();
        return 0;
    }

    int row = rank / q;
    int col = rank % q;

    int block_size = n / q;

    // --- 2. ALLOCAZIONE ---
    double *A = NULL, *B = NULL, *C = NULL;
    double *A_local = malloc(block_size * block_size * sizeof(double));
    double *B_local = malloc(block_size * block_size * sizeof(double));
    double *C_local = calloc(block_size * block_size, sizeof(double));

    if (rank == 0) {
        A = malloc(n * n * sizeof(double));
        B = malloc(n * n * sizeof(double));
        C = malloc(n * n * sizeof(double));

        for (int i = 0; i < n*n; i++) {
            A[i] = 2.0;
            B[i] = 3.0;
        }
    }

    // --- 3. DISTRIBUZIONE A BLOCCHI ---
    // (semplificata: scatter manuale)
    for (int i = 0; i < block_size; i++) {
        MPI_Scatter(
            rank == 0 ? A + (i + row*block_size)*n : NULL,
            block_size, MPI_DOUBLE,
            A_local + i*block_size,
            block_size, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        MPI_Scatter(
            rank == 0 ? B + (i + row*block_size)*n : NULL,
            block_size, MPI_DOUBLE,
            B_local + i*block_size,
            block_size, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // --- 4. CREA COMMUNICATORI ---
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    double *A_panel = malloc(block_size * block_size * sizeof(double));
    double *B_panel = malloc(block_size * block_size * sizeof(double));

    // --- 5. SUMMA ---
    for (int k = 0; k < q; k++) {

        // broadcast A sulla riga
        if (col == k)
            for (int i = 0; i < block_size * block_size; i++)
                A_panel[i] = A_local[i];

        MPI_Bcast(A_panel, block_size*block_size, MPI_DOUBLE, k, row_comm);

        // broadcast B sulla colonna
        if (row == k)
            for (int i = 0; i < block_size * block_size; i++)
                B_panel[i] = B_local[i];

        MPI_Bcast(B_panel, block_size*block_size, MPI_DOUBLE, k, col_comm);

        // moltiplicazione locale
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int kk = 0; kk < block_size; kk++) {
                    C_local[i*block_size + j] +=
                        A_panel[i*block_size + kk] *
                        B_panel[kk*block_size + j];
                }
            }
        }
    }

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("SUMMA - n=%d, p=%d, tempo=%f\n", n, size, end - start);
    }

    free(A_local); free(B_local); free(C_local);
    free(A_panel); free(B_panel);

    if (rank == 0) {
        free(A); free(B); free(C);
    }

    MPI_Finalize();
    return 0;
}