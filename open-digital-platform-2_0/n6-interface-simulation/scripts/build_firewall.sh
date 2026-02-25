#!/bin/bash

#
# Build script for DOCA Simple Firewall Application
# Compiles the C code with proper DOCA SDK linking
#

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE_FILE="$PROJECT_ROOT/doca-app/simple_firewall.c"
OUTPUT_FILE="$PROJECT_ROOT/doca-app/simple_firewall"

# DOCA SDK paths - adjust these based on your installation
DOCA_INSTALL_DIR="${DOCA_INSTALL_DIR:-/opt/mellanox/doca}"
PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${DOCA_INSTALL_DIR}/lib/x86_64-linux-gnu/pkgconfig"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo ""
echo "=============================================="
echo "    DOCA Simple Firewall - Build Script      "
echo "=============================================="
echo ""

# Check if source file exists
if [[ ! -f "$SOURCE_FILE" ]]; then
    echo_error "Source file not found: $SOURCE_FILE"
    exit 1
fi

echo_info "Source file: $SOURCE_FILE"
echo_info "Output file: $OUTPUT_FILE"
echo_info "DOCA SDK directory: $DOCA_INSTALL_DIR"
echo ""

# Check if DOCA SDK is installed
if [[ ! -d "$DOCA_INSTALL_DIR" ]]; then
    echo_warn "DOCA SDK not found at $DOCA_INSTALL_DIR"
    echo_warn "You may need to:"
    echo_warn "  1. Install the DOCA SDK"
    echo_warn "  2. Set DOCA_INSTALL_DIR environment variable"
    echo_warn ""
    echo_warn "Example installation:"
    echo_warn "  wget https://developer.download.nvidia.com/networking/doca/doca_sdk/2.6.0/doca-sdk-2.6.0-x86_64-ubuntu2204.deb"
    echo_warn "  sudo apt install ./doca-sdk-2.6.0-x86_64-ubuntu2204.deb"
    echo ""
fi

# Check if pkg-config can find DOCA libraries
export PKG_CONFIG_PATH
echo_info "Checking for DOCA libraries..."

if ! pkg-config --exists libdoca 2>/dev/null; then
    echo_warn "pkg-config cannot find DOCA libraries"
    echo_warn "Falling back to manual linking..."
    
    # Manual compilation with explicit paths
    echo_info "Compiling with manual linking..."
    
    gcc -std=c11 -Wall -Wextra -O2 \
        -I"${DOCA_INSTALL_DIR}/include" \
        -I"${DOCA_INSTALL_DIR}/include/dpdk" \
        "$SOURCE_FILE" \
        -o "$OUTPUT_FILE" \
        -L"${DOCA_INSTALL_DIR}/lib/x86_64-linux-gnu" \
        -ldoca_flow \
        -ldoca_common \
        -ldoca_utils \
        -ldpdk \
        -lrte_eal \
        -lrte_ethdev \
        -lrte_mbuf \
        -lrte_mempool \
        -lrte_ring \
        -lrte_flow_classify \
        -lpthread \
        -ldl \
        -lnuma \
        -lm

else
    echo_info "Using pkg-config for DOCA libraries..."
    
    # Compilation with pkg-config
    CFLAGS=$(pkg-config --cflags libdoca)
    LIBS=$(pkg-config --libs libdoca)
    
    echo_info "CFLAGS: $CFLAGS"
    echo_info "LIBS: $LIBS"
    
    gcc -std=c11 -Wall -Wextra -O2 \
        $CFLAGS \
        "$SOURCE_FILE" \
        -o "$OUTPUT_FILE" \
        $LIBS
fi

# Check if compilation was successful
if [[ $? -eq 0 && -f "$OUTPUT_FILE" ]]; then
    echo ""
    echo_success "Compilation completed successfully!"
    echo_info "Executable created: $OUTPUT_FILE"
    
    # Make executable
    chmod +x "$OUTPUT_FILE"
    
    # Show file info
    echo ""
    echo_info "File information:"
    ls -lh "$OUTPUT_FILE"
    file "$OUTPUT_FILE"
    
    echo ""
    echo_success "Build complete! You can now run:"
    echo_success "  $OUTPUT_FILE"
    
else
    echo ""
    echo_error "Compilation failed!"
    echo_error "Please check the error messages above."
    echo ""
    echo_info "Common issues:"
    echo_info "  1. DOCA SDK not installed or wrong path"
    echo_info "  2. Missing development packages (build-essential, pkg-config)"
    echo_info "  3. Incompatible DOCA SDK version"
    exit 1
fi

echo ""
echo "=============================================="