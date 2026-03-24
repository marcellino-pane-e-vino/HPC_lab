#!/bin/bash

# Enforce strict error handling
# -e: Exit immediately if a command exits with a non-zero status.
# -u: Treat unset variables as an error when substituting.
# -o pipefail: The return value of a pipeline is the status of the last command to exit with a non-zero status.
set -euo pipefail

# ==========================================
# CONFIGURATION
# Modify these variables as needed
# ==========================================
COMPILER="mpicc"
SRC_FILE="matrixmult_mpi_naive.c"
OUTPUT_BIN="test"
EXTRA_FLAGS="-lucp -lucs -lhwloc"

# ==========================================
# EXECUTION
# ==========================================

echo "--------------------------------------------------"
echo "Starting compilation process..."
echo "Compiler    : ${COMPILER}"
echo "Source file : ${SRC_FILE}"
echo "Output file : ${OUTPUT_BIN}"
echo "Extra flags : ${EXTRA_FLAGS}"
echo "--------------------------------------------------"

# Construct the command for logging
COMMAND="${COMPILER} ${SRC_FILE} -o ${OUTPUT_BIN} ${EXTRA_FLAGS}"
echo "Executing: ${COMMAND}"
echo ""

# Execute the compilation and handle the exit status gracefully
if ${COMPILER} ${SRC_FILE} -o "${OUTPUT_BIN}" ${EXTRA_FLAGS}; then
    echo ""
    echo "--------------------------------------------------"
    echo "SUCCESS: Compilation completed successfully."
    echo "Usage example: mpirun -np 4 ./${OUTPUT_BIN}"
    echo "--------------------------------------------------"
else
    echo ""
    echo "--------------------------------------------------"
    echo "ERROR: Compilation failed."
    echo "Please review the compiler/linker output above."
    echo "--------------------------------------------------"
    exit 1
fi