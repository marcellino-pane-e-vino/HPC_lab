#!/bin/bash

# -----------------------------
# Local OpenBLAS + Compile Script
# -----------------------------

# Detect script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

SRC_FILE="matrixmult_library.c"
EXECUTABLE="matrixmult"
INSTALL_DIR="$HOME/local/OpenBLAS"

# Check source file exists
if [ ! -f "$SRC_FILE" ]; then
    echo "Error: $SRC_FILE not found in $SCRIPT_DIR"
    exit 1
fi

# Create local OpenBLAS folder
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR" || exit 1

# Download OpenBLAS if not already present
OPENBLAS_VERSION="0.3.23"
OPENBLAS_TAR="v$OPENBLAS_VERSION.tar.gz"
if [ ! -f "$OPENBLAS_TAR" ]; then
    echo "Downloading OpenBLAS $OPENBLAS_VERSION..."
    wget "https://github.com/xianyi/OpenBLAS/archive/refs/tags/$OPENBLAS_TAR"
fi

# Extract OpenBLAS
if [ ! -d "OpenBLAS-$OPENBLAS_VERSION" ]; then
    tar xvf "$OPENBLAS_TAR"
fi

# Compile and install OpenBLAS locally
cd "OpenBLAS-$OPENBLAS_VERSION" || exit 1
echo "Compiling OpenBLAS locally..."
make -j4
make PREFIX="$INSTALL_DIR" install

# Return to script folder
cd "$SCRIPT_DIR" || exit 1

# Compile the C program
echo "Compiling $SRC_FILE with OpenBLAS..."
gcc -O2 "$SRC_FILE" -I"$INSTALL_DIR/include" -L"$INSTALL_DIR/lib" -Wl,-rpath,"$INSTALL_DIR/lib" -lopenblas -o "$EXECUTABLE"

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "You can now run your program directly:"
    echo "./$EXECUTABLE 5000"
else
    echo "Compilation failed."
    exit 1
fi
