# Performance Optimization and Scaling of Large-Scale Matrix Multiplication

This project analyzes the optimization and parallelization of square matrix multiplication ($n \times n$), addressing the **Memory Wall** challenge across different architectures and programming paradigms.

## Objectives
The primary goal is the transformation of a baseline sequential algorithm into a suite of high-performance implementations. Results are validated through comparison with industry-standard libraries (CBLAS, ScaLAPACK, cuBLAS) and analyzed using the **Roofline Model**.

## Paradigms and Implemented Techniques

### 1. Sequential Optimization (CPU)
* **Loop Tiling**: Partitioning matrices into $B \times B$ tiles to maximize temporal cache locality.
* **Register Promotion**: Using scalar variables to force the use of CPU registers and reduce redundant cache reads.
* **Memory Alignment**: Allocation via `aligned_alloc` (64-byte) to ensure optimal data loading for AVX2 SIMD instructions.

### 2. Shared Memory (OpenMP)
* **Parallelization**: Workload distribution across 24 logical threads using the fork-join model.
* **Scheduling Analysis**: Comparison of *Static*, *Dynamic*, and *Guided* policies to optimize load balancing.
* **Loop Collapse**: Fusing iteration spaces to increase available parallelism and ensure thread saturation.

### 3. Distributed Memory (MPI)
* **SUMMA Algorithm**: Implementation of the *Scalable Universal Matrix Multiplication Algorithm* using a 2D process grid.
* **Communication Optimization**: Utilization of Cartesian row-exclusive and column-exclusive communicators for targeted panel broadcasts.

### 4. Heterogeneous Computing (CUDA)
* **2D Register Tiling**: A multi-level memory hierarchy strategy (Shared Memory and Registers) to overcome the global memory bottleneck.
* **Memory Vectorization**: Using the `float4` data type to enable 128-bit memory transactions.
* **Precision Shift**: Transitioning to single-precision (FP32) to fully exploit the hardware resources of the Turing microarchitecture.

## Performance Analysis (N = 10000)
Benchmarks were conducted on an **Intel i9-12900K** (24 threads) and an **NVIDIA Tesla T4**.

| Paradigm | Implementation | Execution Time (s) | GFLOPS | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential** | Naive Baseline | 76.98 | 25.98 | 1.00x |
| **Sequential** | Optimized (Tiling) | 44.37 | 45.07 | 1.73x |
| **OpenMP** | Optimized (24 Thr) | 6.63 | 301.66 | 11.61x |
| **MPI** | SUMMA (25 Proc) | 7.71 | 259.40 | 9.98x |
| **CUDA** | **Custom float4** | **0.45** | **4444.44** | **171.07x** |
| **CUDA** | **cuBLAS Library** | **0.44** | **4545.45** | **174.95x** |

## Technical Setup
* **Compilers**: `icx` (Intel OneAPI), `nvcc` (NVIDIA CUDA Compiler), `mpiicc`.
* **Libraries**: OpenBLAS (CBLAS), ScaLAPACK, cuBLAS.
* **Hardware**: HP Z2 Tower G9 (CPU) and NVIDIA Tesla T4 (GPU).

---
**Authors**: Luca Barbati, Wassim Fatnassi, Roberto Lazzarini.
