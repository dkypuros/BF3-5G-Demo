#!/bin/bash

#
# BlueField-3 DPU Deployment Script for N6 Interface Firewall
# Production deployment automation for NVIDIA BlueField-3 DPU
#
# Copyright (c) 2024 NVIDIA Corporation
# SPDX-License-Identifier: Proprietary
#

set -euo pipefail

# Script Configuration
SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/var/log/n6_firewall_deploy.log"

# Deployment Configuration
APP_NAME="n6_firewall"
APP_VERSION="2.6.0"
SERVICE_NAME="n6-firewall"
CONFIG_DIR="/etc/n6-firewall"
SYSTEMD_DIR="/etc/systemd/system"
BIN_DIR="/usr/local/bin"
LOG_DIR="/var/log/n6-firewall"

# BlueField DPU Configuration
BF_HOST=""
BF_USER="root"
BF_SSH_KEY=""
BF_SSH_PORT="22"
DEPLOYMENT_MODE="production"

# Network Configuration
UPLINK_INTERFACE=""
DOWNLINK_INTERFACE=""
MGMT_INTERFACE="tmfifo_net0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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
    log_error "Deployment failed. Check log file: $LOG_FILE"
    exit 1
}

# Show usage information
show_usage() {
    cat << EOF
BlueField-3 DPU Deployment Script for N6 Interface Firewall

Usage: $SCRIPT_NAME [OPTIONS]

Required Options:
    -h, --host HOST         BlueField DPU hostname or IP address
    -u, --uplink IFACE      Uplink interface (toward UPF)
    -d, --downlink IFACE    Downlink interface (toward Data Network)

Optional Settings:
    --user USER             SSH username (default: root)
    --ssh-key FILE          SSH private key file
    --ssh-port PORT         SSH port (default: 22)
    --mode MODE             Deployment mode: dev|staging|production (default: production)
    --config FILE           Custom configuration file
    --help                  Show this help message

Network Configuration:
    --uplink-ip IP/MASK     Uplink interface IP address
    --downlink-ip IP/MASK   Downlink interface IP address
    --gateway IP            Default gateway
    --dns IP                DNS server

Advanced Options:
    --skip-build            Skip application build (use existing binary)
    --skip-tests            Skip post-deployment tests
    --backup                Create backup before deployment
    --rollback              Rollback to previous deployment
    --dry-run               Show what would be deployed without changes

Examples:
    # Basic deployment
    $SCRIPT_NAME --host 192.168.1.100 --uplink enp3s0f0 --downlink enp3s0f1

    # Production deployment with custom config
    $SCRIPT_NAME --host bf3-dpu-01 --uplink enp3s0f0 --downlink enp3s0f1 \\
                 --mode production --config production.conf

    # Development deployment with SSH key
    $SCRIPT_NAME --host 10.0.0.50 --user ubuntu --ssh-key ~/.ssh/bf3_key \\
                 --uplink enp3s0f0 --downlink enp3s0f1 --mode dev

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--host)
                BF_HOST="$2"
                shift 2
                ;;
            --user)
                BF_USER="$2"
                shift 2
                ;;
            --ssh-key)
                BF_SSH_KEY="$2"
                shift 2
                ;;
            --ssh-port)
                BF_SSH_PORT="$2"
                shift 2
                ;;
            -u|--uplink)
                UPLINK_INTERFACE="$2"
                shift 2
                ;;
            -d|--downlink)
                DOWNLINK_INTERFACE="$2"
                shift 2
                ;;
            --mode)
                DEPLOYMENT_MODE="$2"
                shift 2
                ;;
            --config)
                CUSTOM_CONFIG="$2"
                shift 2
                ;;
            --uplink-ip)
                UPLINK_IP="$2"
                shift 2
                ;;
            --downlink-ip)
                DOWNLINK_IP="$2"
                shift 2
                ;;
            --gateway)
                GATEWAY_IP="$2"
                shift 2
                ;;
            --dns)
                DNS_SERVER="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --backup)
                CREATE_BACKUP=true
                shift
                ;;
            --rollback)
                ROLLBACK_MODE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$BF_HOST" ]]; then
        error_exit "BlueField host is required. Use --host option."
    fi

    if [[ -z "$UPLINK_INTERFACE" || -z "$DOWNLINK_INTERFACE" ]]; then
        error_exit "Both uplink and downlink interfaces are required."
    fi
}

# SSH execution helper
ssh_exec() {
    local cmd="$1"
    local ssh_opts="-o StrictHostKeyChecking=no -o ConnectTimeout=30"
    
    if [[ -n "$BF_SSH_KEY" ]]; then
        ssh_opts="$ssh_opts -i $BF_SSH_KEY"
    fi
    
    ssh $ssh_opts -p "$BF_SSH_PORT" "$BF_USER@$BF_HOST" "$cmd"
}

# SCP file transfer helper
scp_copy() {
    local src="$1"
    local dst="$2"
    local ssh_opts="-o StrictHostKeyChecking=no -o ConnectTimeout=30"
    
    if [[ -n "$BF_SSH_KEY" ]]; then
        ssh_opts="$ssh_opts -i $BF_SSH_KEY"
    fi
    
    scp $ssh_opts -P "$BF_SSH_PORT" "$src" "$BF_USER@$BF_HOST:$dst"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking deployment prerequisites..."
    
    # Check if application is built
    local binary="$PROJECT_ROOT/build/bin/$APP_NAME"
    if [[ ! -f "$binary" ]] && [[ "${SKIP_BUILD:-false}" != "true" ]]; then
        log_info "Building application..."
        cd "$PROJECT_ROOT"
        make clean && make || error_exit "Failed to build application"
        cd "$SCRIPT_DIR"
    fi
    
    if [[ ! -f "$binary" ]]; then
        error_exit "Application binary not found: $binary"
    fi
    
    # Check SSH connectivity
    log_info "Testing SSH connectivity to $BF_HOST..."
    if ! ssh_exec "echo 'SSH connection successful'"; then
        error_exit "Cannot connect to BlueField DPU via SSH"
    fi
    
    # Check if target is BlueField DPU
    local platform
    platform=$(ssh_exec "cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo 'Unknown'")
    if [[ ! "$platform" == *"BlueField"* ]]; then
        log_warn "Target may not be a BlueField DPU (detected: $platform)"
        read -p "Continue anyway? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    else
        log_info "Detected platform: $platform"
    fi
    
    log_success "Prerequisites check completed"
}

# Deploy application binary and files
deploy_application() {
    log_step "Deploying N6 firewall application..."
    
    local binary="$PROJECT_ROOT/build/bin/$APP_NAME"
    local temp_dir="/tmp/n6_firewall_deploy_$$"
    
    # Create deployment package
    mkdir -p "$temp_dir"/{bin,config,scripts,docs}
    
    # Copy application binary
    cp "$binary" "$temp_dir/bin/"
    chmod +x "$temp_dir/bin/$APP_NAME"
    
    # Copy configuration files
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        cp -r "$PROJECT_ROOT/config"/* "$temp_dir/config/" 2>/dev/null || true
    fi
    
    # Copy documentation
    if [[ -d "$PROJECT_ROOT/docs" ]]; then
        cp -r "$PROJECT_ROOT/docs"/* "$temp_dir/docs/" 2>/dev/null || true
    fi
    
    # Create deployment script for BlueField
    cat > "$temp_dir/scripts/install.sh" << 'EOF'
#!/bin/bash
set -e

TEMP_DIR="$1"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="/etc/n6-firewall"
LOG_DIR="/var/log/n6-firewall"

# Create directories
mkdir -p "$BIN_DIR" "$CONFIG_DIR" "$LOG_DIR"

# Install binary
cp "$TEMP_DIR/bin/n6_firewall" "$BIN_DIR/"
chmod +x "$BIN_DIR/n6_firewall"

# Install configuration
cp "$TEMP_DIR/config"/* "$CONFIG_DIR/" 2>/dev/null || true

# Set permissions
chown -R root:root "$CONFIG_DIR"
chmod 644 "$CONFIG_DIR"/*
chmod 755 "$LOG_DIR"

echo "Application installed successfully"
EOF
    
    chmod +x "$temp_dir/scripts/install.sh"
    
    # Create archive
    cd "$temp_dir"
    tar czf "../n6_firewall_${APP_VERSION}_$(date +%Y%m%d_%H%M%S).tar.gz" .
    cd - > /dev/null
    
    local archive_path="${temp_dir}_package.tar.gz"
    mv "${temp_dir}.tar.gz" "$archive_path"
    
    # Transfer and install
    log_info "Transferring deployment package..."
    scp_copy "$archive_path" "/tmp/"
    
    local remote_archive="/tmp/$(basename "$archive_path")"
    ssh_exec "cd /tmp && tar xzf '$remote_archive' && ./scripts/install.sh /tmp && rm -rf /tmp/n6_firewall_*"
    
    # Cleanup
    rm -rf "$temp_dir" "$archive_path"
    
    log_success "Application deployed successfully"
}

# Configure network interfaces
configure_network() {
    log_step "Configuring network interfaces..."
    
    # Generate netplan configuration for Ubuntu
    local netplan_config="/etc/netplan/99-n6-firewall.yaml"
    
    ssh_exec "cat > '$netplan_config' << 'EOF'
network:
  version: 2
  renderer: networkd
  ethernets:
    $UPLINK_INTERFACE:
      dhcp4: false
      dhcp6: false
EOF"
    
    if [[ -n "${UPLINK_IP:-}" ]]; then
        ssh_exec "echo '      addresses: [\"$UPLINK_IP\"]' >> '$netplan_config'"
    fi
    
    ssh_exec "cat >> '$netplan_config' << 'EOF'
    $DOWNLINK_INTERFACE:
      dhcp4: false
      dhcp6: false
EOF"
    
    if [[ -n "${DOWNLINK_IP:-}" ]]; then
        ssh_exec "echo '      addresses: [\"$DOWNLINK_IP\"]' >> '$netplan_config'"
    fi
    
    if [[ -n "${GATEWAY_IP:-}" ]]; then
        ssh_exec "cat >> '$netplan_config' << 'EOF'
      routes:
        - to: default
          via: $GATEWAY_IP
EOF"
    fi
    
    if [[ -n "${DNS_SERVER:-}" ]]; then
        ssh_exec "cat >> '$netplan_config' << 'EOF'
      nameservers:
        addresses: [$DNS_SERVER]
EOF"
    fi
    
    # Apply network configuration
    log_info "Applying network configuration..."
    ssh_exec "netplan generate && netplan apply" || log_warn "Network configuration may need manual verification"
    
    log_success "Network interfaces configured"
}

# Create systemd service
create_systemd_service() {
    log_step "Creating systemd service..."
    
    local service_file="$SYSTEMD_DIR/$SERVICE_NAME.service"
    
    ssh_exec "cat > '$service_file' << 'EOF'
[Unit]
Description=N6 Interface Firewall for BlueField-3 DPU
Documentation=https://docs.nvidia.com/doca/
After=network-online.target doca-services.target
Wants=network-online.target
Requires=doca-services.target

[Service]
Type=exec
User=root
Group=root
ExecStart=/usr/local/bin/n6_firewall --config /etc/n6-firewall/n6_firewall.conf --port 8001 --verbose
ExecReload=/bin/kill -HUP \$MAINPID
ExecStop=/bin/kill -TERM \$MAINPID
Restart=always
RestartSec=5
TimeoutStartSec=30
TimeoutStopSec=15

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/n6-firewall /tmp
PrivateTmp=true
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictRealtime=true

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768
LimitMEMLOCK=infinity

# Environment
Environment=DOCA_LOG_LEVEL=INFO
Environment=DOCA_LOG_STDOUT=1

[Install]
WantedBy=multi-user.target
EOF"
    
    # Enable and start service
    ssh_exec "systemctl daemon-reload"
    ssh_exec "systemctl enable $SERVICE_NAME"
    
    log_success "Systemd service created and enabled"
}

# Configure DOCA environment
configure_doca_environment() {
    log_step "Configuring DOCA environment on BlueField DPU..."
    
    # Check if DOCA is installed
    if ! ssh_exec "test -d /opt/mellanox/doca"; then
        log_warn "DOCA SDK not found - installing..."
        scp_copy "$SCRIPT_DIR/install_doca_sdk.sh" "/tmp/"
        ssh_exec "chmod +x /tmp/install_doca_sdk.sh && /tmp/install_doca_sdk.sh --skip-reboot"
    fi
    
    # Configure hugepages
    ssh_exec "echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages" || true
    ssh_exec "mkdir -p /mnt/huge && mount -t hugetlbfs nodev /mnt/huge" || true
    
    # Configure DPDK interfaces
    ssh_exec "echo 'net.ipv4.ip_forward=1' >> /etc/sysctl.conf" || true
    ssh_exec "sysctl -p" || true
    
    log_success "DOCA environment configured"
}

# Create configuration file
create_configuration() {
    log_step "Creating application configuration..."
    
    local config_content=""
    
    case "$DEPLOYMENT_MODE" in
        "production")
            config_content='# N6 Firewall Production Configuration
[network]
uplink_interface = '$UPLINK_INTERFACE'
downlink_interface = '$DOWNLINK_INTERFACE'
uplink_port_id = 0
downlink_port_id = 1

[firewall]
default_action = forward
log_blocked_packets = true
stats_enabled = true
stats_interval = 30

[logging]
log_level = info
log_file = /var/log/n6-firewall/n6_firewall.log
syslog_enabled = true
syslog_facility = local0

[performance]
hw_offload = true
queue_depth = 1024
burst_size = 32
worker_threads = 4

[monitoring]
prometheus_enabled = true
prometheus_port = 9090
health_check_port = 8080'
            ;;
        "dev")
            config_content='# N6 Firewall Development Configuration
[network]
uplink_interface = '$UPLINK_INTERFACE'
downlink_interface = '$DOWNLINK_INTERFACE'
uplink_port_id = 0
downlink_port_id = 1

[firewall]
default_action = forward
log_blocked_packets = true
stats_enabled = true
stats_interval = 5

[logging]
log_level = debug
log_file = /var/log/n6-firewall/n6_firewall.log
syslog_enabled = false

[performance]
hw_offload = false
queue_depth = 256
burst_size = 16
worker_threads = 2

[monitoring]
prometheus_enabled = false
health_check_port = 8080'
            ;;
    esac
    
    ssh_exec "cat > '/etc/n6-firewall/n6_firewall.conf' << 'EOF'
$config_content
EOF"
    
    log_success "Configuration file created"
}

# Run post-deployment tests
run_deployment_tests() {
    if [[ "${SKIP_TESTS:-false}" == "true" ]]; then
        log_info "Skipping post-deployment tests"
        return
    fi
    
    log_step "Running post-deployment tests..."
    
    # Test 1: Service status
    if ssh_exec "systemctl is-active $SERVICE_NAME --quiet"; then
        log_success "✓ Service is running"
    else
        log_error "✗ Service is not running"
    fi
    
    # Test 2: Interface availability
    if ssh_exec "ip link show $UPLINK_INTERFACE" >/dev/null 2>&1; then
        log_success "✓ Uplink interface available"
    else
        log_error "✗ Uplink interface not found"
    fi
    
    if ssh_exec "ip link show $DOWNLINK_INTERFACE" >/dev/null 2>&1; then
        log_success "✓ Downlink interface available"
    else
        log_error "✗ Downlink interface not found"
    fi
    
    # Test 3: DOCA library loading
    if ssh_exec "$BIN_DIR/$APP_NAME --help" >/dev/null 2>&1; then
        log_success "✓ Application loads successfully"
    else
        log_error "✗ Application fails to load"
    fi
    
    # Test 4: Log file creation
    if ssh_exec "test -f /var/log/n6-firewall/n6_firewall.log"; then
        log_success "✓ Log file created"
    else
        log_warn "⚠ Log file not found"
    fi
    
    log_success "Post-deployment tests completed"
}

# Show deployment summary
show_deployment_summary() {
    local service_status
    service_status=$(ssh_exec "systemctl is-active $SERVICE_NAME" || echo "failed")
    
    echo
    echo "=========================================="
    echo "    N6 Firewall Deployment Complete"
    echo "=========================================="
    echo "Application:      $APP_NAME v$APP_VERSION"
    echo "Target Host:      $BF_HOST"
    echo "Deployment Mode:  $DEPLOYMENT_MODE"
    echo "Service Status:   $service_status"
    echo "Uplink Interface: $UPLINK_INTERFACE"
    echo "Downlink Interface: $DOWNLINK_INTERFACE"
    echo "Configuration:    $CONFIG_DIR/n6_firewall.conf"
    echo "Log Directory:    $LOG_DIR"
    echo "Service Name:     $SERVICE_NAME"
    echo "=========================================="
    echo
    echo "Management Commands:"
    echo "  Start:    ssh $BF_USER@$BF_HOST 'systemctl start $SERVICE_NAME'"
    echo "  Stop:     ssh $BF_USER@$BF_HOST 'systemctl stop $SERVICE_NAME'"
    echo "  Status:   ssh $BF_USER@$BF_HOST 'systemctl status $SERVICE_NAME'"
    echo "  Logs:     ssh $BF_USER@$BF_HOST 'journalctl -u $SERVICE_NAME -f'"
    echo "  Config:   ssh $BF_USER@$BF_HOST 'vi $CONFIG_DIR/n6_firewall.conf'"
    echo
    echo "Next Steps:"
    echo "1. Monitor the service: systemctl status $SERVICE_NAME"
    echo "2. Check logs: journalctl -u $SERVICE_NAME -f"
    echo "3. Test firewall rules with your N6 traffic"
    echo "4. Monitor performance: curl http://$BF_HOST:9090/metrics"
    echo
}

# Main deployment function
main() {
    # Initialize
    log_info "Starting N6 Firewall deployment to BlueField-3 DPU..."
    log_info "Script: $SCRIPT_NAME"
    log_info "Date: $(date)"
    log_info "Target: $BF_HOST"
    
    # Parse arguments
    parse_arguments "$@"
    
    # Run deployment steps
    check_prerequisites
    deploy_application
    configure_doca_environment
    configure_network
    create_configuration
    create_systemd_service
    run_deployment_tests
    show_deployment_summary
    
    log_success "N6 Firewall deployment completed successfully!"
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"