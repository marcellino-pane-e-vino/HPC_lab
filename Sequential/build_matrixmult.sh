#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SRC_FILE="${SRC_FILE:-matrixmult_library.c}"
COMPILER="${COMPILER:-icx}"
EXECUTABLE="${EXECUTABLE:-$SCRIPT_DIR/matrixmult}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/local/OpenBLAS_seq}"
OPENBLAS_VERSION="${OPENBLAS_VERSION:-0.3.23}"
OPENBLAS_TAR="v${OPENBLAS_VERSION}.tar.gz"
OPENBLAS_SRC_DIR="OpenBLAS-${OPENBLAS_VERSION}"
EXTRA_FLAGS="${EXTRA_FLAGS:--O3 -xHost}"

if [ ! -f "$SRC_FILE" ]; then
    echo "Error: $SRC_FILE not found in $SCRIPT_DIR"
    exit 1
fi

if ! command -v "$COMPILER" >/dev/null 2>&1; then
    echo "Error: compiler '$COMPILER' not found in PATH"
    exit 1
fi

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ ! -f "$OPENBLAS_TAR" ]; then
    echo "Downloading OpenBLAS $OPENBLAS_VERSION..."
    wget "https://github.com/xianyi/OpenBLAS/archive/refs/tags/$OPENBLAS_TAR"
fi

if [ ! -d "$OPENBLAS_SRC_DIR" ]; then
    tar xvf "$OPENBLAS_TAR"
fi

cd "$OPENBLAS_SRC_DIR"

echo "Building SINGLE-THREADED OpenBLAS locally..."
make clean || true
make -j4 USE_THREAD=0 USE_OPENMP=0
make PREFIX="$INSTALL_DIR" USE_THREAD=0 USE_OPENMP=0 install

cd "$SCRIPT_DIR"

echo "Compiling $SRC_FILE with $COMPILER..."

COMPILE_CMD=( "$COMPILER" )

if [ -n "$EXTRA_FLAGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_FLAGS_ARRAY=( $EXTRA_FLAGS )
    COMPILE_CMD+=( "${EXTRA_FLAGS_ARRAY[@]}" )
fi

COMPILE_CMD+=(
    "$SRC_FILE"
    "-I$INSTALL_DIR/include"
    "-L$INSTALL_DIR/lib"
    "-Wl,-rpath,$INSTALL_DIR/lib"
    "-lopenblas"
    "-o" "$EXECUTABLE"
)

printf 'Actual compile command:'
printf ' %q' "${COMPILE_CMD[@]}"
printf '\n'

BUILD_LOG="$(mktemp)"

if "${COMPILE_CMD[@]}" >"$BUILD_LOG" 2>&1; then
    echo "Compilation successful!"
    if [ -s "$BUILD_LOG" ]; then
        echo "Compiler messages/warnings:"
        cat "$BUILD_LOG"
    else
        echo "No compiler warnings."
    fi
else
    echo "Compilation failed."
    echo "Compiler output:"
    cat "$BUILD_LOG"
    rm -f "$BUILD_LOG"
    exit 1
fi

rm -f "$BUILD_LOG"

echo "Executable: $EXECUTABLE"
echo "Run example: OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 $EXECUTABLE 5000"