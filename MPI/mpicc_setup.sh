#!/bin/bash

# Enforce strict error handling
set -euo pipefail

# 0. Save current working directory to return here after shell refresh
CURRENT_DIR="$PWD"

# ==========================================
# CONFIGURATION
# ==========================================
MPICH_VERSION="4.3.0"
INSTALL_DIR="${HOME}/.local/mpich-${MPICH_VERSION}"
SRC_DIR="/tmp/mpich-build-${USER}"
TAR_FILE="mpich-${MPICH_VERSION}.tar.gz"
DOWNLOAD_URL="https://www.mpich.org/static/downloads/${MPICH_VERSION}/${TAR_FILE}"
BASHRC_FILE="${HOME}/.bashrc"

# ==========================================
# EXECUTION
# ==========================================

echo "--------------------------------------------------"
echo "Starting MPICH Local Installation Script"
echo "Version: ${MPICH_VERSION}"
echo "Target : ${INSTALL_DIR}"
echo "--------------------------------------------------"

# 1. Check if it's already installed
if [ -x "${INSTALL_DIR}/bin/mpicc" ]; then
    echo "STEP 1: MPICH is already installed. Skipping build."
else
    echo "STEP 1: Installing MPICH (this happens only once)..."
    mkdir -p "${SRC_DIR}"
    cd "${SRC_DIR}"

    echo "  -> Downloading source..."
    wget -q --show-progress "${DOWNLOAD_URL}"

    echo "  -> Extracting..."
    tar -xzf "${TAR_FILE}"
    cd "mpich-${MPICH_VERSION}"

    echo "  -> Configuring (CH3 device for compatibility)..."
    ./configure --prefix="${INSTALL_DIR}" --disable-fortran --with-device=ch3 > configure.log 2>&1

    echo "  -> Compiling (using $(nproc) threads)..."
    make -j"$(nproc)" > make.log 2>&1

    echo "  -> Installing..."
    make install > install.log 2>&1
    
    cd /tmp
    rm -rf "${SRC_DIR}"
fi

# 2. Update .bashrc for FUTURE sessions (making it permanent)
echo "STEP 2: Hard-coding configuration into ${BASHRC_FILE}..."

# Remove any old auto-installer lines to avoid clutter
sed -i '/# Added by MPICH auto-installer/d' "$BASHRC_FILE" || true
sed -i "/mpich-${MPICH_VERSION}/d" "$BASHRC_FILE" || true

# Append fresh configuration to the end of .bashrc
{
    echo ""
    echo "# Added by MPICH auto-installer"
    echo "export PATH=\"${INSTALL_DIR}/bin:\$PATH\""
    echo "alias mpicc='${INSTALL_DIR}/bin/mpicc'"
    echo "alias mpirun='${INSTALL_DIR}/bin/mpirun'"
} >> "$BASHRC_FILE"

# 3. Create a temporary "Force-Environment" file for the IMMEDIATE shell
echo "STEP 3: Preparing the immediate environment shell..."
TEMP_RC=$(mktemp)
{
    echo "source ~/.bashrc"
    echo "export PATH=\"${INSTALL_DIR}/bin:\$PATH\""
    echo "alias mpicc='${INSTALL_DIR}/bin/mpicc'"
    echo "alias mpirun='${INSTALL_DIR}/bin/mpirun'"
    echo "cd \"$CURRENT_DIR\""
    echo "echo '--------------------------------------------------'"
    echo "echo 'SUCCESS: New environment loaded!'"
    echo "echo \"Current mpicc: \$(which mpicc)\""
    echo "echo 'You can now run: mpicc matrixmult_mpi_naive.c -o test'"
    echo "echo '--------------------------------------------------'"
    echo "rm \"$TEMP_RC\"" # The shell file deletes itself after loading
} > "$TEMP_RC"

echo "--------------------------------------------------"
echo "COMPLETED: Suicide protocol initiated."
echo "Killing old shell and dropping you into the functional one..."
echo "--------------------------------------------------"

# 4. The "Suicide" and Refresh
# Launch bash using our bulletproof temporary RC file
exec bash --rcfile "$TEMP_RC"