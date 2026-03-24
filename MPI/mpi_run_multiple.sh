#!/bin/bash

source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

set -euo pipefail

# ==========================================
# CONFIGURATION
# ==========================================
#SOURCES=("matrixmult_mpi_naive.c" "matrixmult_mpi_scalapack.c" "matrixmult_mpi_library.c")
SOURCES=("matrixmult_mpi_scalapack.c" "matrixmult_mpi_library.c")

COMPILER_FLAGS=("-O3")

NUM_PROCS=4
MATRIX_SIZE=10000
CSV_OUTPUT="benchmark_results.csv"

MPICC_BIN="mpicc"
MPIRUN_BIN="mpirun"

# Track failures
ANY_FAILURE=0

# ==========================================
# CHECKS
# ==========================================

if ! command -v $MPICC_BIN &> /dev/null; then
    echo "ERROR: mpicc not found."
    exit 1
fi

if [ -z "${MKLROOT:-}" ]; then
    echo "ERROR: MKL not available."
    exit 1
fi

# ==========================================
# INIT
# ==========================================

if [ ! -f "$CSV_OUTPUT" ]; then
    echo "Source,Flags,MatrixSize,Procs,Time_Seconds" > "$CSV_OUTPUT"
fi

echo "--------------------------------------------------"
echo " Starting Multi-Source HPC Benchmark"
echo "--------------------------------------------------"

# ==========================================
# MAIN LOOP
# ==========================================

for SRC in "${SOURCES[@]}"; do
    for FLAGS in "${COMPILER_FLAGS[@]}"; do

        OUT_BIN="${SRC%.c}.out"

        echo ">>> CURRENT TASK: $SRC"
        echo "    [STEP 1/2] Compiling..."

        if [[ "$SRC" == "matrixmult_mpi_library.c" ]]; then
            echo "    Using ScaLAPACK (MKL manual linking)"

            if ! $MPICC_BIN "$SRC" -o "$OUT_BIN" $FLAGS \
                -I${MKLROOT}/include \
                -L${MKLROOT}/lib/intel64 \
                -lmkl_scalapack_lp64 \
                -lmkl_blacs_intelmpi_lp64 \
                -lmkl_intel_lp64 \
                -lmkl_core \
                -lmkl_sequential \
                -lpthread -lm -ldl; then

                echo "    ERROR: Compilation failed (ScaLAPACK)"
                ANY_FAILURE=1
                continue
            fi

        else
            if ! $MPICC_BIN "$SRC" -o "$OUT_BIN" $FLAGS -lm; then
                echo "    ERROR: Compilation failed"
                ANY_FAILURE=1
                continue
            fi
        fi

        echo "    [STEP 2/2] Executing..."

        set +e
        EXEC_OUT=$($MPIRUN_BIN -np "$NUM_PROCS" "./$OUT_BIN" "$MATRIX_SIZE" 2>&1)
        RUN_STATUS=$?
        set -e

        if [ $RUN_STATUS -eq 0 ]; then
            TIME_VAL=$(echo "$EXEC_OUT" | grep -Ei "time|tempo" | grep -Eo '[0-9]+\.[0-9]+' | tail -1)

            if [ -z "$TIME_VAL" ]; then
                TIME_VAL="NaN"
            fi

            echo "    SUCCESS: ${TIME_VAL}s"
            echo "$SRC,\"$FLAGS\",$MATRIX_SIZE,$NUM_PROCS,$TIME_VAL" >> "$CSV_OUTPUT"

        else
            echo "    ERROR: Runtime crash (Exit Code $RUN_STATUS)"
            echo "$EXEC_OUT"
            ANY_FAILURE=1
        fi

        echo "--------------------------------------------------"
    done
done

# ==========================================
# FINAL STATUS
# ==========================================

if [ "$ANY_FAILURE" -eq 0 ]; then
    echo " All tasks completed successfully."
else
    echo " Completed with ERRORS. Check logs above."
fi

echo " Results: $CSV_OUTPUT"