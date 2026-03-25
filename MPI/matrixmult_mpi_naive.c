//# I_MPI_FABRICS=shm mpirun -np (NUMERO DI PROCESSI) ./matrixmult_mpi N

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size, n;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) n = 1000; else n = atoi(argv[1]);

    // Distribuzione del carico (Load Balancing)
    int rows_per_proc = n / size; 
    
    double *local_a = malloc(rows_per_proc * n * sizeof(double));
    double *local_c = malloc(rows_per_proc * n * sizeof(double));
    double *b = malloc(n * n * sizeof(double));
    double *a = NULL, *c = NULL;

    if (rank == 0) {
        a = malloc(n * n * sizeof(double));
        c = malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) { a[i] = 2.0; b[i] = 3.0; }
    }

    // 1. SINCRONIZZAZIONE (MPI_Barrier)
    // Forza tutti i processi ad aspettare qui. 
    // Serve per far partire il cronometro nello stesso istante per tutti.
    MPI_Barrier(MPI_COMM_WORLD); 
    double start = MPI_Wtime(); // 2. MPI_Wtime restituisce il tempo "reale"

    // 3. COMUNICAZIONE COLLETTIVA
    // Bcast: invia la matrice B intera a tutti i processi
    MPI_Bcast(b, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Scatter: divide la matrice A e ne invia un pezzo a ciascuno
    MPI_Scatter(a, rows_per_proc * n, MPI_DOUBLE, local_a, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 4. CALCOLO LOCALE
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < n; j++) {
            local_c[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                local_c[i * n + j] += local_a[i * n + k] * b[k * n + j];
            }
        }
    }

    // 5. RACCOLTA DATI (Gather)
    MPI_Gather(local_c, rows_per_proc * n, MPI_DOUBLE, c, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

    free(local_a); free(local_c); free(b);
    MPI_Finalize();
    return 0;
}