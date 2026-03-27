#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    // 1. Initialization
    int rank, size, n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 2) n = 1000; else n = atoi(argv[1]);

    // 1.5 Load balancing
    int remainder = n % size; 
    int rows_per_proc = (n / size) + (rank < remainder ? 1 : 0); 
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int current_displ = 0;
    for (int i = 0; i < size; i++) {
        int r = (n / size) + (i < remainder ? 1 : 0); // Rows assigned to rank 'i'
        sendcounts[i] = r * n; // Total elements (rows * columns)
        displs[i] = current_displ;
        current_displ += sendcounts[i]; // Update offset for the next rank
    }

    // 2. Memory allocation
    double *local_a = malloc(rows_per_proc * n * sizeof(double));
    double *local_c = malloc(rows_per_proc * n * sizeof(double));
    double *b = malloc(n * n * sizeof(double));
    double *a = NULL, *c = NULL;

    if (rank == 0) {
        a = malloc(n * n * sizeof(double));
        c = malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) { a[i] = 2.0; b[i] = 3.0; }
    }

    // 3. Synchronization and distribution
    MPI_Barrier(MPI_COMM_WORLD); 
    double start = MPI_Wtime(); 
    MPI_Bcast(b, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE, 
                 local_a, rows_per_proc * n, MPI_DOUBLE, 
                 0, MPI_COMM_WORLD);

    // 4. Local computation
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < n; j++) {
            local_c[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                local_c[i * n + j] += local_a[i * n + k] * b[k * n + j];
            }
        }
    }

    // 5. Gather + output
    MPI_Gatherv(local_c, rows_per_proc * n, MPI_DOUBLE, 
                c, sendcounts, displs, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double duration = end - start;

    if (rank == 0) {
        printf("--------------------------------------\n");
        printf("Taglia Matrice: %d x %d\n", n, n);
        printf("Processi usati: %d\n", size);
        printf("Tempo Totale:   %.6f secondi\n", duration);
        printf("--------------------------------------\n");
        free(a); free(c);
    }

    // 6. Cleanup
    free(sendcounts);
    free(displs); 
    free(local_a); 
    free(local_c); 
    free(b); 
    MPI_Finalize();
    return 0;
}