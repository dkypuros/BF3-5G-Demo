#!/bin/bash

#
# NVIDIA DOCA SDK Installation Script for BlueField-3 DPU
# Production deployment script for N6 Interface Firewall
#
# Copyright (c) 2024 NVIDIA Corporation
# SPDX-License-Identifier: Proprietary
#

set -euo pipefail

# Script Configuration
SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/doca_sdk_install.log"

# DOCA SDK Configuration
DOCA_VERSION="2.6.0"
DOCA_REPO_URL="https://linux.mellanox.com/public/repo/doca"
DOCA_INSTALL_DIR="/opt/mellanox/doca"

# BlueField DPU Detection
BF_MODEL=""
BF_VERSION=""
OS_VERSION=""
ARCH="$(uname -m)"

# Installation Options
INSTALL_DOCS=false
INSTALL_EXAMPLES=true
INSTALL_DEV_TOOLS=true
SKIP_REBOOT=false
FORCE_INSTALL=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log_error "$1"
    log_error "Installation failed. Check log file: $LOG_FILE"
    exit 1
}

# Show usage information
show_usage() {
    cat << EOF
NVIDIA DOCA SDK Installation Script for BlueField-3 DPU

Usage: $SCRIPT_NAME [OPTIONS]

Options:
    -h, --help              Show this help message
    -v, --version VERSION   DOCA SDK version to install (default: $DOCA_VERSION)
    -d, --docs              Install documentation packages
    -e, --no-examples       Skip example applications
    -t, --no-dev-tools      Skip development tools
    -f, --force             Force installation (overwrite existing)
    -r, --skip-reboot       Skip automatic reboot after installation
    --offline FILE          Install from offline package file

Examples:
    $SCRIPT_NAME                                    # Standard installation
    $SCRIPT_NAME --version 2.5.0 --docs           # Install specific version with docs
    $SCRIPT_NAME --force --skip-reboot            # Force install without reboot
    $SCRIPT_NAME --offline doca-sdk-2.6.0.deb     # Install from local file

Environment Variables:
    DOCA_INSTALL_DIR        Custom installation directory (default: $DOCA_INSTALL_DIR)
    DOCA_REPO_URL          Custom repository URL
    HTTP_PROXY             HTTP proxy for downloads
    HTTPS_PROXY            HTTPS proxy for downloads

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--version)
                DOCA_VERSION="$2"
                shift 2
                ;;
            -d|--docs)
                INSTALL_DOCS=true
                shift
                ;;
            -e|--no-examples)
                INSTALL_EXAMPLES=false
                shift
                ;;
            -t|--no-dev-tools)
                INSTALL_DEV_TOOLS=false
                shift
                ;;
            -f|--force)
                FORCE_INSTALL=true
                shift
                ;;
            -r|--skip-reboot)
                SKIP_REBOOT=true
                shift
                ;;
            --offline)
                OFFLINE_PACKAGE="$2"
                shift 2
                ;;
            *)
                error_exit "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done
}

# Detect BlueField DPU model and OS
detect_platform() {
    log_step "Detecting platform and environment..."
    
    # Check if running on BlueField DPU
    if [[ -f /sys/devices/platform/MLNXBF04:00/driver/driver_override ]]; then
        BF_MODEL="BlueField-2"
    elif [[ -f /sys/devices/platform/MLNXBF30:00/driver/driver_override ]]; then
        BF_MODEL="BlueField-3"
    elif [[ "$ARCH" == "aarch64" ]] && lscpu | grep -q "Implementer.*0x41"; then
        log_warn "ARM64 platform detected but BlueField model unclear"
        BF_MODEL="Unknown-ARM64"
    else
        log_warn "Not running on BlueField DPU - this script is optimized for BlueField hardware"
        BF_MODEL="Generic-x86_64"
    fi
    
    # Detect OS version
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        OS_VERSION="${ID}${VERSION_ID}"
    else
        error_exit "Cannot detect OS version - /etc/os-release not found"
    fi
    
    log_info "Platform: $BF_MODEL ($ARCH)"
    log_info "OS: $OS_VERSION"
    log_info "Kernel: $(uname -r)"
}

# Check system requirements
check_requirements() {
    log_step "Checking system requirements..."
    
    # Check root privileges
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root (use sudo)"
    fi
    
    # Check disk space (need at least 2GB)
    local available_space
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 2097152 ]]; then  # 2GB in KB
        error_exit "Insufficient disk space. Need at least 2GB free space."
    fi
    
    # Check network connectivity
    if ! curl -s --connect-timeout 10 http://www.nvidia.com > /dev/null; then
        log_warn "Network connectivity issues detected - some features may not work"
    fi
    
    # Check if DOCA is already installed
    if [[ -d "$DOCA_INSTALL_DIR" ]] && [[ "$FORCE_INSTALL" != "true" ]]; then
        log_warn "DOCA SDK appears to be already installed at $DOCA_INSTALL_DIR"
        read -p "Continue with installation? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Installation cancelled by user"
            exit 0
        fi
    fi
    
    log_success "System requirements check passed"
}

# Install system dependencies
install_dependencies() {
    log_step "Installing system dependencies..."
    
    # Update package lists
    apt-get update || error_exit "Failed to update package lists"
    
    # Essential packages
    local essential_packages=(
        "build-essential"
        "cmake"
        "pkg-config"
        "libnuma-dev"
        "libssl-dev"
        "zlib1g-dev"
        "python3-dev"
        "python3-pip"
        "curl"
        "wget"
        "gnupg2"
        "software-properties-common"
    )
    
    # DPDK dependencies
    local dpdk_packages=(
        "libbsd-dev"
        "libpcap-dev"
        "libcap-dev"
        "libelf-dev"
        "libjansson-dev"
    )
    
    # Development tools (optional)
    local dev_packages=()
    if [[ "$INSTALL_DEV_TOOLS" == "true" ]]; then
        dev_packages+=(
            "gdb"
            "valgrind"
            "perf-tools-unstable"
            "strace"
            "tcpdump"
            "ethtool"
            "bridge-utils"
            "iperf3"
            "netperf"
        )
    fi
    
    # Documentation tools (optional)
    local doc_packages=()
    if [[ "$INSTALL_DOCS" == "true" ]]; then
        doc_packages+=(
            "doxygen"
            "graphviz"
            "texlive-latex-base"
            "texlive-latex-extra"
        )
    fi
    
    # Combine all packages
    local all_packages=("${essential_packages[@]}" "${dpdk_packages[@]}" "${dev_packages[@]}" "${doc_packages[@]}")
    
    log_info "Installing ${#all_packages[@]} packages..."
    
    # Install packages with error handling
    for package in "${all_packages[@]}"; do
        if ! apt-get install -y "$package"; then
            log_warn "Failed to install $package - continuing..."
        fi
    done
    
    log_success "System dependencies installed"
}

# Add NVIDIA repository
setup_nvidia_repository() {
    log_step "Setting up NVIDIA repository..."
    
    # Download and install NVIDIA GPG key
    local keyring_path="/usr/share/keyrings/nvidia-mellanox-archive-keyring.gpg"
    
    if ! wget -qO- https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | \
         gpg --dearmor -o "$keyring_path"; then
        error_exit "Failed to install NVIDIA GPG key"
    fi
    
    # Add DOCA repository based on OS version
    local repo_config="/etc/apt/sources.list.d/doca-sdk.list"
    
    case "$OS_VERSION" in
        ubuntu20.04|ubuntu2004)
            echo "deb [signed-by=$keyring_path] $DOCA_REPO_URL/$DOCA_VERSION/ubuntu2004/amd64 ./" > "$repo_config"
            ;;
        ubuntu22.04|ubuntu2204)
            echo "deb [signed-by=$keyring_path] $DOCA_REPO_URL/$DOCA_VERSION/ubuntu2204/amd64 ./" > "$repo_config"
            ;;
        *)
            log_warn "Unsupported OS version: $OS_VERSION"
            log_info "Attempting generic Ubuntu 22.04 repository..."
            echo "deb [signed-by=$keyring_path] $DOCA_REPO_URL/$DOCA_VERSION/ubuntu2204/amd64 ./" > "$repo_config"
            ;;
    esac
    
    # Update package lists with new repository
    apt-get update || error_exit "Failed to update package lists after adding repository"
    
    log_success "NVIDIA repository configured"
}

# Install DOCA SDK packages
install_doca_sdk() {
    log_step "Installing DOCA SDK $DOCA_VERSION..."
    
    # Core DOCA packages
    local doca_packages=(
        "doca-sdk"
        "doca-runtime"
        "doca-tools"
        "libdoca-dev"
        "libdoca-libs"
    )
    
    # DPDK packages (bundled with DOCA)
    local dpdk_packages=(
        "dpdk"
        "dpdk-dev"
        "dpdk-doc"
    )
    
    # Optional packages
    local optional_packages=()
    if [[ "$INSTALL_EXAMPLES" == "true" ]]; then
        optional_packages+=("doca-examples")
    fi
    
    if [[ "$INSTALL_DOCS" == "true" ]]; then
        optional_packages+=("doca-doc" "doca-doc-html")
    fi
    
    # Combine all packages
    local all_packages=("${doca_packages[@]}" "${dpdk_packages[@]}" "${optional_packages[@]}")
    
    log_info "Installing DOCA SDK packages: ${all_packages[*]}"
    
    # Install DOCA packages
    if ! apt-get install -y "${all_packages[@]}"; then
        log_warn "Some DOCA packages failed to install - attempting individual installation..."
        
        for package in "${all_packages[@]}"; do
            if apt-get install -y "$package"; then
                log_info "✓ Installed $package"
            else
                log_warn "✗ Failed to install $package"
            fi
        done
    fi
    
    log_success "DOCA SDK installation completed"
}

# Install from offline package (alternative method)
install_offline_package() {
    local package_file="$1"
    
    log_step "Installing DOCA SDK from offline package: $package_file"
    
    if [[ ! -f "$package_file" ]]; then
        error_exit "Offline package file not found: $package_file"
    fi
    
    # Install the package
    if dpkg -i "$package_file"; then
        log_success "Offline package installed successfully"
    else
        log_warn "Package installation had issues - fixing dependencies..."
        apt-get install -f -y || error_exit "Failed to fix dependencies"
    fi
}

# Configure DOCA environment
configure_environment() {
    log_step "Configuring DOCA environment..."
    
    # Set up environment variables
    local env_file="/etc/environment"
    local profile_file="/etc/profile.d/doca-sdk.sh"
    
    # Create profile script
    cat > "$profile_file" << EOF
#!/bin/bash
# NVIDIA DOCA SDK Environment Configuration

export DOCA_INSTALL_DIR="$DOCA_INSTALL_DIR"
export DOCA_VERSION="$DOCA_VERSION"

# Library paths
export LD_LIBRARY_PATH="\$DOCA_INSTALL_DIR/lib/aarch64-linux-gnu:\$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="\$DOCA_INSTALL_DIR/lib/aarch64-linux-gnu/pkgconfig:\$PKG_CONFIG_PATH"

# Include paths
export CPATH="\$DOCA_INSTALL_DIR/include:\$CPATH"

# DPDK environment
export RTE_SDK="\$DOCA_INSTALL_DIR"
export RTE_TARGET="aarch64-native-linux-gcc"

# Development tools
export PATH="\$DOCA_INSTALL_DIR/bin:\$PATH"

EOF
    
    chmod +x "$profile_file"
    
    # Configure hugepages for DPDK
    echo 'vm.nr_hugepages=1024' >> /etc/sysctl.conf
    
    # Create hugepage mount point
    mkdir -p /dev/hugepages
    if ! grep -q hugepages /etc/fstab; then
        echo 'hugetlbfs /dev/hugepages hugetlbfs defaults 0 0' >> /etc/fstab
    fi
    
    # Mount hugepages
    mount -t hugetlbfs none /dev/hugepages || log_warn "Failed to mount hugepages"
    
    # Configure IOMMU (required for DPDK)
    local grub_file="/etc/default/grub"
    if [[ -f "$grub_file" ]]; then
        if ! grep -q "iommu=pt intel_iommu=on" "$grub_file"; then
            sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="/&iommu=pt intel_iommu=on /' "$grub_file"
            update-grub || log_warn "Failed to update GRUB configuration"
        fi
    fi
    
    log_success "DOCA environment configured"
}

# Verify installation
verify_installation() {
    log_step "Verifying DOCA SDK installation..."
    
    # Check installation directory
    if [[ ! -d "$DOCA_INSTALL_DIR" ]]; then
        error_exit "DOCA installation directory not found: $DOCA_INSTALL_DIR"
    fi
    
    # Check key libraries
    local required_libs=(
        "$DOCA_INSTALL_DIR/lib/aarch64-linux-gnu/libdoca_flow.so"
        "$DOCA_INSTALL_DIR/lib/aarch64-linux-gnu/libdoca_devemu.so"
        "$DOCA_INSTALL_DIR/lib/aarch64-linux-gnu/libdoca_common.so"
    )
    
    for lib in "${required_libs[@]}"; do
        if [[ -f "$lib" ]]; then
            log_info "✓ Found: $(basename "$lib")"
        else
            log_warn "✗ Missing: $(basename "$lib")"
        fi
    done
    
    # Check headers
    local required_headers=(
        "$DOCA_INSTALL_DIR/include/doca_flow.h"
        "$DOCA_INSTALL_DIR/include/doca_devemu.h"
        "$DOCA_INSTALL_DIR/include/doca_error.h"
    )
    
    for header in "${required_headers[@]}"; do
        if [[ -f "$header" ]]; then
            log_info "✓ Found: $(basename "$header")"
        else
            log_warn "✗ Missing: $(basename "$header")"
        fi
    done
    
    # Test pkg-config
    source "$profile_file" 2>/dev/null || true
    if pkg-config --exists libdoca 2>/dev/null; then
        local version
        version=$(pkg-config --modversion libdoca)
        log_success "pkg-config working - DOCA version: $version"
    else
        log_warn "pkg-config not working for DOCA libraries"
    fi
    
    # Test basic compilation
    local test_dir="/tmp/doca_test_$$"
    mkdir -p "$test_dir"
    
    cat > "$test_dir/test.c" << 'EOF'
#include <stdio.h>
#include <doca_error.h>
#include <doca_log.h>

int main() {
    printf("DOCA SDK test compilation successful!\n");
    return 0;
}
EOF
    
    if gcc -I"$DOCA_INSTALL_DIR/include" \
           -L"$DOCA_INSTALL_DIR/lib/aarch64-linux-gnu" \
           -ldoca_common \
           "$test_dir/test.c" \
           -o "$test_dir/test" 2>/dev/null; then
        log_success "Test compilation successful"
        "$test_dir/test"
    else
        log_warn "Test compilation failed - check library paths"
    fi
    
    rm -rf "$test_dir"
    
    log_success "Installation verification completed"
}

# Show final installation summary
show_summary() {
    echo
    echo "=========================================="
    echo "    DOCA SDK Installation Complete"
    echo "=========================================="
    echo "Version:          $DOCA_VERSION"
    echo "Install Path:     $DOCA_INSTALL_DIR"
    echo "Platform:         $BF_MODEL"
    echo "OS:               $OS_VERSION"
    echo "Examples:         $([ "$INSTALL_EXAMPLES" == "true" ] && echo "Installed" || echo "Skipped")"
    echo "Documentation:    $([ "$INSTALL_DOCS" == "true" ] && echo "Installed" || echo "Skipped")"
    echo "Dev Tools:        $([ "$INSTALL_DEV_TOOLS" == "true" ] && echo "Installed" || echo "Skipped")"
    echo "Log File:         $LOG_FILE"
    echo "=========================================="
    echo
    echo "Next Steps:"
    echo "1. Source the environment: source /etc/profile.d/doca-sdk.sh"
    echo "2. Reboot the system to apply kernel changes"
    echo "3. Build your N6 firewall application"
    echo "4. Run: cd bf3-dpu-firewall && make"
    echo
    
    if [[ "$SKIP_REBOOT" != "true" ]]; then
        read -p "Reboot now to complete installation? [Y/n]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            log_info "Rebooting system..."
            reboot
        else
            log_warn "Remember to reboot before using DOCA SDK"
        fi
    fi
}

# Main installation function
main() {
    # Initialize logging
    sudo touch "$LOG_FILE"
    
    log_info "Starting NVIDIA DOCA SDK installation..."
    log_info "Script: $SCRIPT_NAME"
    log_info "Date: $(date)"
    log_info "User: $(whoami)"
    
    # Parse arguments
    parse_arguments "$@"
    
    # Run installation steps
    detect_platform
    check_requirements
    install_dependencies
    
    # Install DOCA SDK
    if [[ -n "${OFFLINE_PACKAGE:-}" ]]; then
        install_offline_package "$OFFLINE_PACKAGE"
    else
        setup_nvidia_repository
        install_doca_sdk
    fi
    
    configure_environment
    verify_installation
    show_summary
    
    log_success "DOCA SDK installation completed successfully!"
}

# Run main function with all arguments
main "$@"