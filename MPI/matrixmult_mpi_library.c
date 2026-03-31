#define NB 64

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// 0. ScaLAPACK / BLACS / tools
extern void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
extern void pdgemm_(char*, char*, int*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
extern void Cblacs_pinfo(int*, int*);
extern void Cblacs_get(int, int, int*);
extern void Cblacs_gridinit(int*, char*, int, int);
extern void Cblacs_gridinfo(int, int*, int*, int*, int*);
extern void Cblacs_gridexit(int);
extern int numroc_(int*, int*, int*, int*, int*);

int main(int argc, char **argv) {
    // 1. INITIALIZATION + LOAD BALANCING
    int rank, size, n;
    int i_zero = 0, i_one = 1;
    double d_one = 1.0, d_zero = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#define NB 64
    n = (argc < 2) ? 1024 : atoi(argv[1]);

    if (n <= 0) {
        if (rank == 0) fprintf(stderr, "Error: n must be > 0\n");
        MPI_Finalize();
        return 1;
    }

    // building 2d grid (BLACS)
    int icontext, myrow, mycol;
    int nprow = (int)sqrt((double)size);
    while (size % nprow != 0) nprow--;
    int npcol = size / nprow;
    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &icontext);
    Cblacs_gridinit(&icontext, "Row-major", nprow, npcol);
    Cblacs_gridinfo(icontext, &nprow, &npcol, &myrow, &mycol);

    // building descriptors who encode matrices as block-cyclic
    int descA[9], descB[9], descC[9], info;
    int nb = NB;  // local alias needed to pass NB by pointer to Fortran routines //cannot use a constant because of how Fortan interops with C

    int local_rows = numroc_(&n, &nb, &myrow, &i_zero, &nprow);
    int local_cols = numroc_(&n, &nb, &mycol, &i_zero, &npcol);
    int lld = (local_rows > 1) ? local_rows : 1;
    
    // filling descriptors
    descinit_(descA, &n, &n, &nb, &nb, &i_zero, &i_zero, &icontext, &lld, &info);
    descinit_(descB, &n, &n, &nb, &nb, &i_zero, &i_zero, &icontext, &lld, &info);
    descinit_(descC, &n, &n, &nb, &nb, &i_zero, &i_zero, &icontext, &lld, &info);

    if (info != 0) {
        if (rank == 0) printf("Error in descinit: %d\n", info);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. MEMORY ALLOCATION
    size_t local_elems = (size_t)local_rows * local_cols;

    double *local_a = (double *)malloc(local_elems * sizeof(double));
    double *local_b = (double *)malloc(local_elems * sizeof(double));
    double *local_c = (double *)malloc(local_elems * sizeof(double));

    if (!local_a || !local_b || !local_c) {
        printf("Memory allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // parallel initialization (could only be performed for constant matrices)
    for (size_t i = 0; i < local_elems; i++) {
        local_a[i] = 2.0;
        local_b[i] = 3.0;
        local_c[i] = 0.0;
    }

    // 4. COMPUTATION
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    pdgemm_("N", "N", &n, &n, &n, &d_one,
            local_a, &i_one, &i_one, descA,
            local_b, &i_one, &i_one, descB,
            &d_zero,
            local_c, &i_one, &i_one, descC);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    // 5. OUTPUT
    if (rank == 0) {
        printf("--------------------------------------\n");
        printf("Libreria: ScaLAPACK (pdgemm)\n");
        printf("Initialization: static/naive uniform matrix\n");
        printf("Process grid: %d x %d\n", nprow, npcol);
        printf("Taglia Matrice: %d x %d\n", n, n);
        printf("Tempo Totale: %.6f secondi\n", end - start);
        printf("--------------------------------------\n");
    }

    // 6. CLEANUP
    free(local_a);
    free(local_b);
    free(local_c);

    Cblacs_gridexit(icontext);
    MPI_Finalize();
    return 0;
}