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
    echo "INFO: MPICH is already installed at ${INSTALL_DIR}."
else
    # 2. Setup build environment
    echo "Creating build environment in ${SRC_DIR}..."
    mkdir -p "${SRC_DIR}"
    cd "${SRC_DIR}"

    # 3. Download the source code
    if [ ! -f "${TAR_FILE}" ]; then
        echo "Downloading MPICH source from ${DOWNLOAD_URL}..."
        wget -q --show-progress "${DOWNLOAD_URL}"
    fi

    # 4. Extract
    echo "Extracting archive..."
    tar -xzf "${TAR_FILE}"
    cd "mpich-${MPICH_VERSION}"

    # 5. Configure the build (Device ch3 for maximum compatibility)
    echo "Configuring build (this may take a minute)..."
    ./configure \
        --prefix="${INSTALL_DIR}" \
        --disable-fortran \
        --with-device=ch3 \
        > configure.log 2>&1 || { echo "ERROR: Configuration failed."; exit 1; }

    # 6. Compile
    echo "Compiling (using $(nproc) parallel threads)..."
    make -j"$(nproc)" > make.log 2>&1 || { echo "ERROR: Compilation failed."; exit 1; }

    # 7. Install to local directory
    echo "Installing to ${INSTALL_DIR}..."
    make install > install.log 2>&1 || { echo "ERROR: Installation failed."; exit 1; }

    # 8. Cleanup
    echo "Cleaning up temporary files..."
    cd /tmp
    rm -rf "${SRC_DIR}"
fi

# 9. Automate Alias and PATH configuration in .bashrc
# We use aliases to explicitly bypass the broken system wrappers in /usr/bin/
MPICC_ALIAS="alias mpicc='${INSTALL_DIR}/bin/mpicc'"
MPIRUN_ALIAS="alias mpirun='${INSTALL_DIR}/bin/mpirun'"
PATH_EXPORT="export PATH=\"${INSTALL_DIR}/bin:\$PATH\""

echo "Updating environment configuration..."

# Function to add line if not present
add_to_bashrc() {
    local line="$1"
    if ! grep -qF "$line" "$BASHRC_FILE"; then
        echo "$line" >> "$BASHRC_FILE"
    fi
}

echo "" >> "$BASHRC_FILE"
echo "# Added by MPICH auto-installer" >> "$BASHRC_FILE"
add_to_bashrc "$PATH_EXPORT"
add_to_bashrc "$MPICC_ALIAS"
add_to_bashrc "$MPIRUN_ALIAS"

echo "--------------------------------------------------"
echo "SUCCESS: MPICH is ready."
echo "ACTION: Killing current shell and launching a new one..."
echo "ACTION: Navigating back to: $CURRENT_DIR"
echo "ACTION: Command history for this session will be reset."
echo "--------------------------------------------------"

# 10. The "Suicide" and Refresh
# We use 'exec' to replace the current process. 
# We call 'bash' with a command to first cd back, then remain interactive.
exec bash --init-file <(echo "source $BASHRC_FILE; cd '$CURRENT_DIR'")