#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define BS 32  // Dimensione del blocco (Tile)

int main(int argc, char **argv) {
    int rank, size, n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) n = 1024; else n = atoi(argv[1]);

    int rows_per_proc = n / size;
    double *local_a = malloc(rows_per_proc * n * sizeof(double));
    double *local_c = malloc(rows_per_proc * n * sizeof(double));
    double *b = malloc(n * n * sizeof(double));
    double *a = NULL, *c = NULL;

    if (rank == 0) {
        a = malloc(n * n * sizeof(double));
        c = malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) { a[i] = 1.0; b[i] = 2.0; }
    }

    MPI_Bcast(b, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(a, rows_per_proc * n, MPI_DOUBLE, local_a, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Inizializza C locale
    for(int i=0; i < rows_per_proc * n; i++) local_c[i] = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // --- TILED ---
    // Dividiamo i loop i, j, k in blocchi di dimensione BS
    for (int kk = 0; kk < n; kk += BS) {
        for (int jj = 0; jj < n; jj += BS) {
            for (int i = 0; i < rows_per_proc; i++) {
                for (int k = kk; k < (kk + BS < n ? kk + BS : n); k++) {
                    double r = local_a[i * n + k];
                    for (int j = jj; j < (jj + BS < n ? jj + BS : n); j++) {
                        local_c[i * n + j] += r * b[k * n + j];
                    }
                }
            }
        }
    }

    MPI_Gather(local_c, rows_per_proc * n, MPI_DOUBLE, c, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Tiled MPI - n=%d, np=%d, BS=%d, Tempo: %f s\n", n, size, BS, end - start);
        free(a); free(c);
    }

    free(local_a); free(local_c); free(b);
    MPI_Finalize();
    return 0;
}