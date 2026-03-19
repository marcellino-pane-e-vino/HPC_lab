#!/bin/bash

# ============================
# CONFIGURATION
# ============================

CC="icx"
CFLAGS="-O3 -xHost -qopenmp -g"

ARG1=15000
ARG2=24

SPECIAL_SRC="omp_matrixmult_library.c"
SPECIAL_BUILD_SCRIPT="./omp_build_matrixmult.sh"

SOURCES=(
    "omp_matrixmult_naive.c"
    "omp_matrixmult_opt.c"
    "omp_matrixmult_library.c"
)

# ============================
# DIRECTORIES
# ============================

ROOT_DIR="$(pwd)"
PROJECTS_DIR="$ROOT_DIR/roofline_projects"   # Folder for Advisor project folders
BIN_DIR="$PROJECTS_DIR/bin"                   # Folder for binaries

mkdir -p "$PROJECTS_DIR"
mkdir -p "$BIN_DIR"

# ============================
# LOAD INTEL ENVIRONMENT
# ============================

if ! command -v advixe-cl >/dev/null 2>&1; then
    if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
        echo "[*] Loading Intel oneAPI environment..."
        source /opt/intel/oneapi/setvars.sh
    else
        echo "Error: Intel oneAPI environment not found."
        exit 1
    fi
fi

# ============================
# BUILD BINARIES
# ============================

for SRC in "${SOURCES[@]}"; do
    NAME=$(basename "$SRC" .c)
    EXEC_PATH="$BIN_DIR/$NAME"

    echo "============================"
    echo "[BUILD] $SRC -> $EXEC_PATH"
    echo "============================"

    if [ "$SRC" == "$SPECIAL_SRC" ]; then
        COMPILER="$CC" \
        EXTRA_FLAGS="$CFLAGS" \
        EXECUTABLE="$EXEC_PATH" \
        SRC_FILE="$SRC" \
        bash "$SPECIAL_BUILD_SCRIPT" || {
            echo "Build failed for $SRC"
            exit 1
        }
    else
        $CC $CFLAGS "$SRC" -o "$EXEC_PATH" || {
            echo "Build failed for $SRC"
            exit 1
        }
    fi
done

# ============================
# RUN ADVISOR PER BINARY (ONE PROJECT PER FILE)
# ============================

for SRC in "${SOURCES[@]}"; do
    NAME=$(basename "$SRC" .c)
    EXEC_PATH="$BIN_DIR/$NAME"
    PROJECT_DIR="$PROJECTS_DIR/$NAME"   # Each project gets its own folder

    mkdir -p "$PROJECT_DIR"

    echo "============================"
    echo "[ADVISING] $NAME -> $PROJECT_DIR"
    echo "============================"

    # Survey
    advixe-cl --collect=survey \
        --project-dir "$PROJECT_DIR" \
        --search-dir all:r="$ROOT_DIR" \
        -- "$EXEC_PATH" $ARG1 $ARG2

    # Tripcounts
    advixe-cl --collect=tripcounts \
        --project-dir "$PROJECT_DIR" \
        --search-dir all:r="$ROOT_DIR" \
        -- "$EXEC_PATH" $ARG1 $ARG2

    echo "[✓] Advisor project ready: $PROJECT_DIR"
done

echo "============================"
echo "[ALL DONE]"
echo "All binaries are in: $BIN_DIR"
echo "All Advisor projects are in: $PROJECTS_DIR"
echo "Each project folder contains its .advixeproj file."
echo "============================"