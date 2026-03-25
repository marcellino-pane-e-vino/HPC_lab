#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// Macro used for inline evaluation. 
// Using a macro avoids the overhead of a function call during the intense inner loops.
#define min(a,b) (((a) < (b)) ? (a) : (b))          

int main(int argc,char **argv) {

    printf("Starting the computation...\n");
    clock_t start_clock_total = clock(); // Start tracking total program execution time
    
    int i, j, k;

    // Read matrix size 'n' from command line
    int n = atoi(argv[1]);          
    int B; // Tile (or block) size for cache blocking

    // Default choice for block/tile size 'B'. 
    // 64 is a common choice because it often fits perfectly within the CPU L1 cache limits.
    if (argc >= 3) 
        B = atoi(argv[2]);
     else 
        B = 64;

    // MEMORY ALIGNMENT OPTIMIZATION
    // 64 bytes is the typical size of a CPU cache line. 
    // Aligning data prevents a single data element from spanning two separate cache lines, speeding up memory fetches.
    size_t alignment = 64;
    size_t matrix_size = sizeof(double[n][n]);

    // aligned_alloc requires the total allocated size to be a multiple of the alignment value.
    // This padding calculation ensures the allocation won't fail due to size mismatch.
    if (matrix_size % alignment != 0) {
        matrix_size = ((matrix_size / alignment) + 1) * alignment;
    }
    
    // Using aligned_alloc instead of standard malloc to guarantee the 64-byte memory alignment.
    // This is a crucial step if vectorized SIMD instructions (like AVX) are used by the compiler.
    double (* a)[n] = aligned_alloc(alignment, matrix_size);
    double (* b)[n] = aligned_alloc(alignment, matrix_size);
    double (* c)[n] = aligned_alloc(alignment, matrix_size);

    // Initialization: Fill matrices A and B with constants, and zero out the result matrix C
    for (i=0; i<n; i++)
        for (j=0; j<n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
         
    printf("Starting the critical section...\n");
    clock_t start_clock = clock(); // Start tracking only the matrix multiplication time

    int ii, kk, jj;
    int i_limit, k_limit, j_limit;
    
    // Note: 'temp_a' is declared but currently unused in the inner loop. 
    // It could be used to hold a[i][k] in a register to further reduce memory reads.
    double temp_a; 
    
    // CACHE BLOCKING / TILING IMPLEMENTATION
    // Instead of processing the whole matrix row by row (which causes massive cache misses on large matrices),
    // we break the matrices into smaller BxB chunks (tiles). These chunks fit entirely into the CPU's fast L1/L2 cache.

    // EXTERNAL LOOPS: Block-level traversal. 
    // These slide the BxB "window" across the matrices A, B, and C.
    for (ii = 0; ii < n; ii += B) {             // Traverse blocks along rows of C and A    
        i_limit = min(ii + B, n);               // min() handles edge cases where matrix size 'n' is not perfectly divisible by block size 'B'
        
        for (kk = 0; kk < n; kk += B) {         // Traverse blocks along columns of A and rows of B
            k_limit = min(kk + B, n);
            
            for (jj = 0; jj < n; jj += B) {     // Traverse blocks along columns of C and B
                j_limit = min(jj + B, n);

                // INTERNAL LOOPS: Element-level computation.
                // This strictly computes the dot product for the elements within the current BxB tile.
                // We maintain the optimized i-k-j loop order inside the tile to ensure linear memory access patterns.
                for (i = ii; i < i_limit; i++) {            
                    for (k = kk; k < k_limit; k++) {        
                        for (j = jj; j < j_limit; j++) {    
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }
    }

    clock_t end_clock = clock(); // Stop tracking matrix multiplication time

    double duration = (double)(end_clock - start_clock) / CLOCKS_PER_SEC;   
    printf("Critical section - Execution Time: %.4f seconds\n", duration);

    // I/O Operations: Save a chunk of the result to a file
    FILE *f = fopen("mat-res.txt", "w");
    if (!f) {
        perror("fopen");
        return 1;
    }

    fprintf(f, "%d\n\n", n);  
    
    // Note: This hardcodes a 1000x1000 output chunk. 
    // It assumes the user provided an 'n' of at least 1000.
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000; j++) {
            fprintf(f, "%.0f ", c[i][j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);

    // Free the dynamically allocated and aligned memory
    free(a);
    free(b);
    free(c);

    clock_t end_clock_total = clock(); // Stop tracking total program time

    // Calculate total program time, including memory allocation, initialization, and file I/O overhead
    double total_time = (double)(end_clock_total - start_clock_total) / CLOCKS_PER_SEC;   
    printf("Total Execution Time: %.4f seconds\n", total_time);

    return 0;
}