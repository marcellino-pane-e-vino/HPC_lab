#!/bin/bash
set -euo pipefail

# -----------------------------
# Local OpenBLAS + OpenMP Compile Script
# -----------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SRC_FILE="${SRC_FILE:-omp_matrixmult_library.c}"
COMPILER="${COMPILER:-icx}"
EXECUTABLE="${EXECUTABLE:-$SCRIPT_DIR/matrixmult}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/local/OpenBLAS}"
OPENBLAS_VERSION="${OPENBLAS_VERSION:-0.3.23}"
OPENBLAS_TAR="v${OPENBLAS_VERSION}.tar.gz"
OPENBLAS_SRC_DIR="OpenBLAS-${OPENBLAS_VERSION}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

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
echo "Compiling OpenBLAS locally..."
make -j4
make PREFIX="$INSTALL_DIR" install

cd "$SCRIPT_DIR"

echo "Compiling $SRC_FILE with $COMPILER and OpenBLAS..."
# shellcheck disable=SC2086
"$COMPILER" $EXTRA_FLAGS "$SRC_FILE" \
    -I"$INSTALL_DIR/include" \
    -L"$INSTALL_DIR/lib" \
    -Wl,-rpath,"$INSTALL_DIR/lib" \
    -lopenblas \
    -o "$EXECUTABLE"

echo "Compilation successful!"
echo "Executable: $EXECUTABLE"
echo "Run example: $EXECUTABLE 5000 24"