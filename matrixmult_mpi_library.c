#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Prototipi delle funzioni ScaLAPACK (interfaccia Fortran)
extern void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
extern void pdgemm_(char*, char*, int*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
extern void Cblacs_pinfo(int*, int*);
extern void Cblacs_get(int, int, int*);
extern void Cblacs_gridinit(int*, char*, int, int);
extern void Cblacs_gridinfo(int, int*, int*, int*, int*);

int main(int argc, char **argv) {
    int rank, size, n;
    int i_zero = 0, i_one = 1;
    double d_one = 1.0, d_zero = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) n = 1024; else n = atoi(argv[1]);

    // 1. SETUP BLACS GRID (La scacchiera dei processi)
    int icontext, myrow, mycol, nprow = 2, npcol = size/2; 
    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &icontext);
    Cblacs_gridinit(&icontext, "Row-major", nprow, npcol);
    Cblacs_gridinfo(icontext, &nprow, &npcol, &myrow, &mycol);

    // 2. DESCRIPTOR SETUP (La "carta d'identità" delle matrici)
    int descA[9], descB[9], descC[9], info;
    int nb = 64; // Taglia del blocco (fondamentale per ScaLAPACK)
    
    // Inizializziamo i descrittori per A, B e C
    descinit_(descA, &n, &n, &nb, &nb, &i_zero, &i_zero, &icontext, &n, &info);
    descinit_(descB, &n, &n, &nb, &nb, &i_zero, &i_zero, &icontext, &n, &info);
    descinit_(descC, &n, &n, &nb, &nb, &i_zero, &i_zero, &icontext, &n, &info);

    // 3. MEMORY ALLOCATION (Solo la parte locale!)
    int local_rows = n / nprow;
    int local_cols = n / npcol;
    double *local_a = malloc(local_rows * local_cols * sizeof(double));
    double *local_b = malloc(local_rows * local_cols * sizeof(double));
    double *local_c = malloc(local_rows * local_cols * sizeof(double));

    // Riempimento locale (per semplicità qui lo facciamo diretto)
    for(int i=0; i < local_rows * local_cols; i++) {
        local_a[i] = 2.0;
        local_b[i] = 3.0;
        local_c[i] = 0.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // 4. THE CROWN JEWEL: pdgemm
    // Questa singola chiamata fa tutto: comunicazione, calcolo e ottimizzazione.
    pdgemm_("N", "N", &n, &n, &n, &d_one, 
            local_a, &i_one, &i_one, descA, 
            local_b, &i_one, &i_one, descB, 
            &d_zero, 
            local_c, &i_one, &i_one, descC);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("--------------------------------------\n");
        printf("Libreria: ScaLAPACK (pdgemm)\n");
        printf("Taglia Matrice: %d x %d\n", n, n);
        printf("Tempo Totale: %.6f secondi\n", end - start);
        printf("--------------------------------------\n");
    }

    free(local_a); free(local_b); free(local_c);
    MPI_Finalize();
    return 0;
}