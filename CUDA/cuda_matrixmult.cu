#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Kernel CUDA super-ottimizzato: Tiling in Shared Memory + Singola Precisione (FP32)
__global__ void matMulTiledFloat(const float *A, const float *B, float *C, int n) {
    // 1. Identifichiamo dove siamo nella matrice globale
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Alloco la Shared Memory per il blocco (ora usiamo i float!)
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // 3. Facciamo scorrere la piastrella lungo la direzione da calcolare
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        // I thread caricano i dati dalla memoria Globale a quella Shared
        if (row < n && (t * TILE_SIZE + threadIdx.x) < n)
            tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && (t * TILE_SIZE + threadIdx.y) < n)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Barriera di sincronizzazione: aspettiamo il caricamento completo
        __syncthreads();

        // 4. Moltiplicazione a manetta sui dati in cache
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Seconda barriera: aspettiamo che tutti abbiano finito prima del prossimo giro
        __syncthreads();
    }

    // 5. Scrittura del risultato finale nella memoria globale
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <n>\\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    // Calcoliamo i byte basandoci su float (4 byte) anziché double (8 byte)
    size_t bytes = n * n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Inizializzazione con letterali float (il suffisso 'f' è importante!)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_a[i * n + j] = 2.0f; 
            h_b[i * n + j] = 3.0f; 
            h_c[i * n + j] = 0.0f;
        }
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE); 
    dim3 numBlocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    clock_t start = clock();
    
    // Lanciamo il kernel in singola precisione!
    matMulTiledFloat<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    cudaDeviceSynchronize();
    clock_t end = clock();

    printf("Execution Time (Tiled FP32): %.4f seconds\\n", (double)(end - start) / CLOCKS_PER_SEC);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Salviamo un piccolo sample per verificare che i conti tornino
    FILE *f = fopen("mat-res.txt", "w");
    if (f) {
        fprintf(f, "%d\\n\\n", n);  
        int sample = (n < 100) ? n : 10;
        for (int i = 0; i < sample; i++) {
            for (int j = 0; j < sample; j++) {
                fprintf(f, "%.0f ", h_c[i * n + j]);
            }
            fprintf(f, "\\n");
        }
        fclose(f);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}
