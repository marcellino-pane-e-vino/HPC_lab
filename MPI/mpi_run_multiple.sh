#!/bin/bash
source /opt/intel/oneapi/setvars.sh

# Enforce strict error handling
set -euo pipefail

# ==========================================
# CONFIGURATION
# ==========================================
# Sources to benchmark
SOURCES=("matrixmult_mpi_naive.c" "matrixmult_mpi_summa.c")

# Compilers and Flags to test (you can add more to compare)
# Comparison example: """COMPILER_FLAGS=("-O2 -lm" "-O3 -lm")"""
COMPILER_FLAGS=("-O3 -lm")

# Execution settings
NUM_PROCS=4         # Must be a perfect square for SUMMA
MATRIX_SIZE=1024
CSV_OUTPUT="benchmark_results.csv"

# Local MPICH paths
MPICC_BIN="${HOME}/.local/mpich-4.3.0/bin/mpicc"
MPIRUN_BIN="${HOME}/.local/mpich-4.3.0/bin/mpirun"

# ==========================================
# INITIALIZATION
# ==========================================

# Create CSV header if it doesn't exist
if [ ! -f "$CSV_OUTPUT" ]; then
    echo "Source,Flags,MatrixSize,Procs,Time_Seconds" > "$CSV_OUTPUT"
fi

echo "--------------------------------------------------"
echo " Starting Multi-Source HPC Benchmark"
echo "Config: $NUM_PROCS processes, $MATRIX_SIZE matrix size"
echo "--------------------------------------------------"

for SRC in "${SOURCES[@]}"; do
    for FLAGS in "${COMPILER_FLAGS[@]}"; do
        OUT_BIN="${SRC%.c}.out"
        
        echo ">>> CURRENT TASK: $SRC with flags [$FLAGS]"
        
        # 1. Compilation
        echo "    [STEP 1/2] Compiling..."
        if ! $MPICC_BIN "$SRC" -o "$OUT_BIN" $FLAGS; then
            echo "    ERROR: Compilation failed for $SRC"
            continue
        fi

        # 2. Execution
        echo "    [STEP 2/2] Executing benchmark..."
        
        # Execute and capture only the output line containing timing info
        set +e
        EXEC_OUT=$($MPIRUN_BIN -np "$NUM_PROCS" "./$OUT_BIN" "$MATRIX_SIZE" 2>&1)
        RUN_STATUS=$?
        set -e

        if [ $RUN_STATUS -eq 0 ]; then
            # Extract time using regex (works for both "Tempo Totale" and "time=...")
            TIME_VAL=$(echo "$EXEC_OUT" | grep -Ei "time|tempo" | grep -Eo '[0-9]+\.[0-9]+' | tail -1)
            
            echo "    SUCCESS: Execution time: ${TIME_VAL}s"
            # Write to CSV
            echo "$SRC,\"$FLAGS\",$MATRIX_SIZE,$NUM_PROCS,$TIME_VAL" >> "$CSV_OUTPUT"
        else
            echo "    ERROR: Runtime crash (Exit Code $RUN_STATUS)"
            echo "    Check if NUM_PROCS is a perfect square for SUMMA."
        fi
        
        echo "--------------------------------------------------"
    done
done

echo " All tasks complete. Results saved in: $CSV_OUTPUT"
echo "You can view them with: column -s, -t $CSV_OUTPUT"