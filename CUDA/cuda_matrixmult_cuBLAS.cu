#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h> // Ecco la libreria magica!

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <n>\\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    size_t bytes = n * n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

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

    // --- INIZIO MAGIA CUBLAS ---
    
    // 1. Creiamo un "handle", ovvero il contesto per la libreria
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS calcola un'operazione generica: C = (alpha * A * B) + (beta * C)
    // Noi vogliamo solo C = A * B, quindi impostiamo alpha a 1 e beta a 0
    float alpha = 1.0f;
    float beta = 0.0f;

    clock_t start = clock();
    
    // 2. Lanciamo la funzione Sgemm (Single precision GEneral Matrix Multiply)
    // TRUCCHETTO: cuBLAS usa il formato "Column-Major" (colonne in memoria, stile Fortran).
    // Noi in C usiamo "Row-Major" (righe in memoria). Per fare in modo che la matematica 
    // torni perfetta senza dover riordinare a mano le matrici, basta passare a cuBLAS 
    // prima la matrice B e poi la matrice A!
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, n, n, 
                &alpha, 
                d_b, n,  // <--- Nota: passiamo d_b per primo!
                d_a, n, 
                &beta, 
                d_c, n);
    
    cudaDeviceSynchronize();
    clock_t end = clock();
    
    // 3. Chiudiamo l'handle
    cublasDestroy(handle);
    
    // --- FINE MAGIA CUBLAS ---

    printf("Execution Time (cuBLAS FP32): %.4f seconds\\n", (double)(end - start) / CLOCKS_PER_SEC);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Stampa di controllo
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
