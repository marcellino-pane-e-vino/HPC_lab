# Performance Optimization and Scaling of Large-Scale Matrix Multiplication

[cite_start]This project analyzes the optimization and parallelization of square matrix multiplication ($n \times n$), addressing the **Memory Wall** challenge across different architectures and programming paradigms[cite: 13, 16, 20].

## Objectives
[cite_start]The primary goal is the transformation of a baseline sequential algorithm into a suite of high-performance implementations[cite: 18]. [cite_start]Results are validated through comparison with industry-standard libraries (CBLAS, ScaLAPACK, cuBLAS) and analyzed using the **Roofline Model**[cite: 63, 1107].

## Paradigms and Implemented Techniques

### 1. Sequential Optimization (CPU)
* [cite_start]**Loop Tiling**: Partitioning matrices into $B \times B$ tiles to maximize temporal cache locality[cite: 162, 163].
* [cite_start]**Register Promotion**: Using scalar variables to force the use of CPU registers and reduce redundant cache reads[cite: 296, 297].
* [cite_start]**Memory Alignment**: Allocation via `aligned_alloc` (64-byte) to ensure optimal data loading for AVX2 SIMD instructions[cite: 197, 208, 209].

### 2. Shared Memory (OpenMP)
* [cite_start]**Parallelization**: Workload distribution across 24 logical threads using the fork-join model[cite: 30, 392].
* [cite_start]**Scheduling Analysis**: Comparison of *Static*, *Dynamic*, and *Guided* policies to optimize load balancing[cite: 437, 438].
* [cite_start]**Loop Collapse**: Fusing iteration spaces to increase available parallelism and ensure thread saturation[cite: 418, 419, 421].

### 3. Distributed Memory (MPI)
* [cite_start]**SUMMA Algorithm**: Implementation of the *Scalable Universal Matrix Multiplication Algorithm* using a 2D process grid[cite: 593, 594].
* [cite_start]**Communication Optimization**: Utilization of Cartesian row-exclusive and column-exclusive communicators for targeted panel broadcasts[cite: 598, 600].

### 4. Heterogeneous Computing (CUDA)
* [cite_start]**2D Register Tiling**: A multi-level memory hierarchy strategy (Shared Memory and Registers) to overcome the global memory bottleneck[cite: 920, 921].
* [cite_start]**Memory Vectorization**: Using the `float4` data type to enable 128-bit memory transactions[cite: 943, 945].
* [cite_start]**Precision Shift**: Transitioning to single-precision (FP32) to fully exploit the hardware resources of the Turing microarchitecture[cite: 816, 819].

## Performance Analysis (N = 10000)
[cite_start]Benchmarks were conducted on an **Intel i9-12900K** (24 threads) and an **NVIDIA Tesla T4**[cite: 25, 808].

| Paradigm | Implementation | Execution Time (s) | GFLOPS | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential** | Naive Baseline | 76.98 | 25.98 | 1.00x |
| **Sequential** | Optimized (Tiling) | 44.37 | 45.07 | 1.73x |
| **OpenMP** | Optimized (24 Thr) | 6.63 | 301.66 | 11.61x |
| **MPI** | SUMMA (25 Proc) | 7.71 | 259.40 | 9.98x |
| **CUDA** | **Custom float4** | **0.45** | **4444.44** | **171.07x** |
| **CUDA** | **cuBLAS Library** | **0.44** | **4545.45** | **174.95x** |

[cite_start]*[Data source: Table 26 and Section 6.5 of the project report]* [cite: 1107]

## Technical Setup
* [cite_start]**Compilers**: `icx` (Intel OneAPI), `nvcc` (NVIDIA CUDA Compiler), `mpiicc`[cite: 61, 62].
* [cite_start]**Libraries**: OpenBLAS (CBLAS), ScaLAPACK, cuBLAS[cite: 216, 678, 932].
* [cite_start]**Hardware**: HP Z2 Tower G9 (CPU) and NVIDIA Tesla T4 (GPU)[cite: 25, 808].

---
[cite_start]**Authors**: Luca Barbati, Wassim Fatnassi, Roberto Lazzarini[cite: 4, 5].


