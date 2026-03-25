#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// ScaLAPACK / BLACS / tools
extern void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
extern void pdgemm_(char*, char*, int*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
extern void Cblacs_pinfo(int*, int*);
extern void Cblacs_get(int, int, int*);
extern void Cblacs_gridinit(int*, char*, int, int);
extern void Cblacs_gridinfo(int, int*, int*, int*, int*);
extern void Cblacs_gridexit(int);
extern int numroc_(int*, int*, int*, int*, int*);

int main(int argc, char **argv) {
    int rank, size, n;
    int i_zero = 0, i_one = 1;
    double d_one = 1.0, d_zero = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = (argc < 2) ? 1024 : atoi(argv[1]);

    if (n <= 0) {
        if (rank == 0) fprintf(stderr, "Error: n must be > 0\n");
        MPI_Finalize();
        return 1;
    }

    // ==========================================
    // 1. BLACS GRID (robust: near-square)
    // ==========================================
    int icontext, myrow, mycol;
    int nprow = (int)sqrt((double)size);
    while (size % nprow != 0) nprow--;
    int npcol = size / nprow;

    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &icontext);
    Cblacs_gridinit(&icontext, "Row-major", nprow, npcol);
    Cblacs_gridinfo(icontext, &nprow, &npcol, &myrow, &mycol);

    // ==========================================
    // 2. DESCRIPTORS
    // ==========================================
    int descA[9], descB[9], descC[9], info;
    int nb = 64;

    int local_rows = numroc_(&n, &nb, &myrow, &i_zero, &nprow);
    int local_cols = numroc_(&n, &nb, &mycol, &i_zero, &npcol);

    int lld = (local_rows > 1) ? local_rows : 1;

    descinit_(descA, &n, &n, &nb, &nb, &i_zero, &i_zero, &icontext, &lld, &info);
    descinit_(descB, &n, &n, &nb, &nb, &i_zero, &i_zero, &icontext, &lld, &info);
    descinit_(descC, &n, &n, &nb, &nb, &i_zero, &i_zero, &icontext, &lld, &info);

    if (info != 0) {
        if (rank == 0) printf("Error in descinit: %d\n", info);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ==========================================
    // 3. MEMORY ALLOCATION
    // ==========================================
    size_t local_elems = (size_t)local_rows * local_cols;

    double *local_a = (double *)malloc(local_elems * sizeof(double));
    double *local_b = (double *)malloc(local_elems * sizeof(double));
    double *local_c = (double *)malloc(local_elems * sizeof(double));

    if (!local_a || !local_b || !local_c) {
        printf("Memory allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ==========================================
    // 4. STATIC / NAIVE INITIALIZATION (DEFAULT)
    // ==========================================
    // In practice, with ScaLAPACK storage, each process still initializes only
    // its own local buffer. But conceptually this is the "static matrix" case:
    // every entry of the global matrix A is 2.0, every entry of B is 3.0,
    // and C starts from 0.0.
    for (size_t i = 0; i < local_elems; i++) {
        local_a[i] = 2.0;
        local_b[i] = 3.0;
        local_c[i] = 0.0;
    }

    /*
    // ==========================================
    // 4.b LOCAL / DISTRIBUTED INITIALIZATION IDEA (COMMENTED OUT)
    // ==========================================
    // This is where you would put a true local/distributed initialization based
    // on the global coordinates owned by each rank.
    //
    // Example sketch:
    //
    // for each local entry (li, lj):
    //     compute corresponding global indices (gi, gj)
    //     local_a[local_index] = f(gi, gj);
    //     local_b[local_index] = g(gi, gj);
    //     local_c[local_index] = 0.0;
    //
    // That path is more general, but for this benchmark we keep the default
    // initialization explicitly static and uniform.
    */

    // ==========================================
    // 5. COMPUTATION
    // ==========================================
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    pdgemm_("N", "N", &n, &n, &n, &d_one,
            local_a, &i_one, &i_one, descA,
            local_b, &i_one, &i_one, descB,
            &d_zero,
            local_c, &i_one, &i_one, descC);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    // ==========================================
    // 6. OUTPUT
    // ==========================================
    if (rank == 0) {
        printf("--------------------------------------\n");
        printf("Libreria: ScaLAPACK (pdgemm)\n");
        printf("Initialization: static/naive uniform matrix\n");
        printf("Process grid: %d x %d\n", nprow, npcol);
        printf("Taglia Matrice: %d x %d\n", n, n);
        printf("Tempo Totale: %.6f secondi\n", end - start);
        printf("--------------------------------------\n");
    }

    // ==========================================
    // CLEANUP
    // ==========================================
    free(local_a);
    free(local_b);
    free(local_c);

    Cblacs_gridexit(icontext);
    MPI_Finalize();
    return 0;
}