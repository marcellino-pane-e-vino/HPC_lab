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

    int remainder = n % size; 
    int rows_per_proc = (n / size) + (rank < remainder ? 1 : 0); 
    
    // MODIFICA 2: Array per le mappe di Scatterv e Gatherv
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    
    int current_displ = 0;
    for (int i = 0; i < size; i++) {
        // Calcolo quante righe spettano al processo 'i'
        int r = (n / size) + (i < remainder ? 1 : 0);
        // Moltiplico per 'n' perché MPI conta i singoli double, non le righe!
        sendcounts[i] = r * n; 
        displs[i] = current_displ;
        current_displ += sendcounts[i]; // Aggiorno l'offset per il prossimo
    }

    // Le allocazioni usano il nuovo rows_per_proc calcolato su misura
    double *local_a = malloc(rows_per_proc * n * sizeof(double));
    double *local_c = malloc(rows_per_proc * n * sizeof(double));
    double *b = malloc(n * n * sizeof(double));
    double *a = NULL, *c = NULL;

    if (rank == 0) {
        a = malloc(n * n * sizeof(double));
        c = malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) { a[i] = 2.0; b[i] = 3.0; }
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    double start = MPI_Wtime(); 

    MPI_Bcast(b, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // MODIFICA 3: Sostituzione con MPI_Scatterv
    MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE, 
                 local_a, rows_per_proc * n, MPI_DOUBLE, 
                 0, MPI_COMM_WORLD);

    // CALCOLO LOCALE (Invariato, funziona da solo grazie al nuovo rows_per_proc)
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < n; j++) {
            local_c[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                local_c[i * n + j] += local_a[i * n + k] * b[k * n + j];
            }
        }
    }

    // MODIFICA 4: Sostituzione con MPI_Gatherv
    // Usiamo comodamente gli stessi sendcounts e displs perché 
    // la matrice C ha la stessa forma della A!
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

    // MODIFICA 5: Pulizia della memoria extra
    free(sendcounts); 
    free(displs);
    
    free(local_a); free(local_c); free(b);
    MPI_Finalize();
    return 0;
}