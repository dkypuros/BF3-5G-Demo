# NVIDIA BlueField-3 N6 Firewall Deployment Guide

## Production Deployment Guide for 5G N6 Interface Firewall

This comprehensive guide covers the deployment of the N6 Interface Firewall on NVIDIA BlueField-3 DPU in production environments.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Hardware Requirements](#hardware-requirements)
3. [DOCA SDK Installation](#doca-sdk-installation)
4. [Application Deployment](#application-deployment)
5. [Network Configuration](#network-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring & Logging](#monitoring--logging)
8. [Troubleshooting](#troubleshooting)
9. [Production Checklist](#production-checklist)

---

## Prerequisites

### Software Requirements

- **Operating System**: Ubuntu 22.04 LTS (recommended) or Ubuntu 20.04 LTS
- **NVIDIA DOCA SDK**: Version 2.6.0 or later
- **DPDK**: Version 23.11 (bundled with DOCA SDK)
- **Linux Kernel**: 5.15+ with IOMMU support
- **GCC**: Version 11.0 or later
- **CMake**: Version 3.18 or later

### Development Tools

```bash
# Essential build tools
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip

# DPDK dependencies
sudo apt install -y \
    libnuma-dev \
    libssl-dev \
    zlib1g-dev \
    libbsd-dev \
    libpcap-dev \
    libelf-dev
```

### Network Access Requirements

- **Internet Access**: For downloading DOCA SDK and dependencies
- **SSH Access**: To BlueField-3 DPU management interface
- **Management Network**: Dedicated management interface (tmfifo_net0)

---

## Hardware Requirements

### NVIDIA BlueField-3 DPU Specifications

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU Cores** | 16 ARM Cortex-A78AE | 16 ARM Cortex-A78AE | Fixed in BF3 |
| **Memory** | 16GB LPDDR5 | 32GB LPDDR5 | Configuration dependent |
| **Storage** | 64GB eMMC | 128GB eMMC | For OS and applications |
| **Network Ports** | 2x 200GbE | 2x 400GbE | ConnectX-7 based |
| **PCIe** | PCIe Gen4 x16 | PCIe Gen4 x16 | Host interface |

### Network Topology

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   5G UPF        │────▶│  BlueField-3    │────▶│  Data Network   │
│                 │     │  N6 Firewall    │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
      N6 Interface            DPU Ports              External Network
    (enp3s0f0)              Port 0 | Port 1         (enp3s0f1)
                               ▲       ▼
                        Hardware Flow Processing
                         (400 Gbps line rate)
```

### Port Assignment

- **Port 0 (Uplink)**: Connected to 5G UPF
- **Port 1 (Downlink)**: Connected to Data Network
- **Management**: tmfifo_net0 (PCIe-based management interface)

---

## DOCA SDK Installation

### Automated Installation

The recommended method is using our automated installation script:

```bash
# Clone the project
git clone <repository-url>
cd bf3-dpu-firewall

# Run DOCA SDK installer
sudo ./scripts/install_doca_sdk.sh \
    --version 2.6.0 \
    --docs \
    --dev-tools
```

### Manual Installation

If you prefer manual installation:

```bash
# 1. Add NVIDIA repository
wget https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-mellanox.gpg RPM-GPG-KEY-Mellanox

# 2. Configure repository
echo "deb [signed-by=/usr/share/keyrings/nvidia-mellanox.gpg] \
    https://linux.mellanox.com/public/repo/doca/2.6.0/ubuntu2204/amd64 ./" \
    | sudo tee /etc/apt/sources.list.d/doca-sdk.list

# 3. Install DOCA packages
sudo apt update && sudo apt install -y \
    doca-sdk \
    doca-runtime \
    doca-tools \
    libdoca-dev \
    doca-examples

# 4. Configure environment
source /opt/mellanox/doca/tools/set_env_vars.sh
```

### Verification

```bash
# Check DOCA installation
pkg-config --exists libdoca && echo "DOCA SDK installed successfully"

# Verify libraries
ls -la /opt/mellanox/doca/lib/x86_64-linux-gnu/libdoca_*.so

# Test compilation
echo '#include <doca_error.h>
int main() { return 0; }' > test.c && \
gcc $(pkg-config --cflags --libs libdoca) test.c -o test && \
echo "Compilation test passed" && rm test test.c
```

---

## Application Deployment

### Automated Deployment

Use the provided deployment script for BlueField-3 DPU:

```bash
# Production deployment
sudo ./scripts/deploy_bluefield.sh \
    --host <bluefield-ip> \
    --uplink enp3s0f0 \
    --downlink enp3s0f1 \
    --mode production \
    --uplink-ip 192.168.1.100/24 \
    --downlink-ip 192.168.2.100/24

# Development deployment
sudo ./scripts/deploy_bluefield.sh \
    --host 10.0.0.50 \
    --user ubuntu \
    --ssh-key ~/.ssh/bf3_key \
    --uplink enp3s0f0 \
    --downlink enp3s0f1 \
    --mode dev
```

### Manual Deployment Steps

#### 1. Build Application

```bash
# Clean build for production
make clean
make RELEASE=1

# Verify binary
ls -la build/bin/n6_firewall
file build/bin/n6_firewall
```

#### 2. Transfer to BlueField DPU

```bash
# Copy application binary
scp build/bin/n6_firewall root@<bluefield-ip>:/usr/local/bin/

# Copy configuration files
scp -r config/* root@<bluefield-ip>:/etc/n6-firewall/

# Set permissions
ssh root@<bluefield-ip> 'chmod +x /usr/local/bin/n6_firewall'
ssh root@<bluefield-ip> 'chmod 644 /etc/n6-firewall/*'
```

#### 3. Install as Systemd Service

```bash
# Create service file
sudo tee /etc/systemd/system/n6-firewall.service << 'EOF'
[Unit]
Description=N6 Interface Firewall for BlueField-3 DPU
After=network-online.target doca-services.target
Wants=network-online.target

[Service]
Type=exec
User=root
ExecStart=/usr/local/bin/n6_firewall --config /etc/n6-firewall/n6_firewall.conf
Restart=always
RestartSec=5
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable n6-firewall
sudo systemctl start n6-firewall
```

---

## Network Configuration

### Interface Configuration

#### Using Netplan (Ubuntu)

Create `/etc/netplan/99-n6-firewall.yaml`:

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    # Uplink interface (toward UPF)
    enp3s0f0:
      dhcp4: false
      dhcp6: false
      addresses: [192.168.1.100/24]
      
    # Downlink interface (toward Data Network)  
    enp3s0f1:
      dhcp4: false
      dhcp6: false
      addresses: [192.168.2.100/24]
      routes:
        - to: default
          via: 192.168.2.1
      nameservers:
        addresses: [8.8.8.8, 1.1.1.1]

    # Management interface
    tmfifo_net0:
      dhcp4: true
```

Apply configuration:
```bash
sudo netplan generate
sudo netplan apply
```

#### DPDK Interface Binding

```bash
# Check current interface status
sudo /opt/mellanox/doca/tools/dpdk-devbind.py --status

# Bind interfaces to DPDK (if needed)
sudo /opt/mellanox/doca/tools/dpdk-devbind.py \
    --bind=mlx5_core enp3s0f0 enp3s0f1
```

### Hugepage Configuration

```bash
# Configure 1GB hugepages
echo 'vm.nr_hugepages=1024' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Mount hugepages
sudo mkdir -p /mnt/huge
echo 'hugetlbfs /mnt/huge hugetlbfs defaults 0 0' | sudo tee -a /etc/fstab
sudo mount /mnt/huge

# Verify hugepage allocation
cat /proc/meminfo | grep -i huge
```

### IOMMU Configuration

Edit `/etc/default/grub`:
```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt intel_iommu=on"
```

Update GRUB and reboot:
```bash
sudo update-grub
sudo reboot
```

---

## Performance Tuning

### CPU Configuration

```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Isolate CPUs for DPDK (optional)
# Add to kernel command line: isolcpus=2-15 nohz_full=2-15 rcu_nocbs=2-15
```

### Memory Optimization

```bash
# Configure NUMA policy
echo 'vm.numa_balancing=0' | sudo tee -a /etc/sysctl.conf

# Set swappiness
echo 'vm.swappiness=1' | sudo tee -a /etc/sysctl.conf

sudo sysctl -p
```

### Network Stack Tuning

```bash
# Increase receive buffer sizes
echo 'net.core.rmem_max=268435456' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_default=67108864' | sudo tee -a /etc/sysctl.conf
echo 'net.core.netdev_max_backlog=5000' | sudo tee -a /etc/sysctl.conf

# Enable IP forwarding
echo 'net.ipv4.ip_forward=1' | sudo tee -a /etc/sysctl.conf

sudo sysctl -p
```

---

## Monitoring & Logging

### Systemd Service Monitoring

```bash
# Check service status
sudo systemctl status n6-firewall

# View real-time logs
sudo journalctl -u n6-firewall -f

# View recent logs
sudo journalctl -u n6-firewall --since "1 hour ago"
```

### Application Logs

```bash
# Application log location
tail -f /var/log/n6-firewall/n6_firewall.log

# Log rotation configuration
sudo tee /etc/logrotate.d/n6-firewall << 'EOF'
/var/log/n6-firewall/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    postrotate
        systemctl reload n6-firewall
    endscript
}
EOF
```

### Performance Metrics

#### Prometheus Metrics (if enabled)

```bash
# Check metrics endpoint
curl http://localhost:9090/metrics

# Key metrics to monitor:
# - n6_packets_processed_total
# - n6_packets_dropped_total
# - n6_rules_matched_total
# - n6_processing_latency_seconds
```

#### DOCA Flow Statistics

```bash
# Query flow statistics
sudo /usr/local/bin/n6_firewall --dump-stats

# Monitor hardware counters
watch -n 1 'sudo /usr/local/bin/n6_firewall --show-counters'
```

### Health Checks

```bash
# Create health check script
sudo tee /usr/local/bin/n6_firewall_healthcheck.sh << 'EOF'
#!/bin/bash
set -e

# Check if service is running
systemctl is-active --quiet n6-firewall

# Check if interfaces are up
ip link show enp3s0f0 | grep -q "state UP"
ip link show enp3s0f1 | grep -q "state UP"

# Check if application responds
timeout 5 /usr/local/bin/n6_firewall --health-check

echo "Health check passed"
EOF

sudo chmod +x /usr/local/bin/n6_firewall_healthcheck.sh
```

---

## Troubleshooting

### Common Issues

#### 1. DOCA Library Loading Errors

```bash
# Check library path
echo $LD_LIBRARY_PATH
ls -la /opt/mellanox/doca/lib/x86_64-linux-gnu/

# Fix library path
export LD_LIBRARY_PATH="/opt/mellanox/doca/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Make permanent
echo 'export LD_LIBRARY_PATH="/opt/mellanox/doca/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"' \
    >> ~/.bashrc
```

#### 2. Interface Not Found

```bash
# Check available interfaces
ip link show

# Check PCI devices
lspci | grep -i mellanox

# Check DPDK binding
sudo /opt/mellanox/doca/tools/dpdk-devbind.py --status
```

#### 3. Hugepage Allocation Failures

```bash
# Check current hugepage status
cat /proc/meminfo | grep -i huge

# Check mounted hugepages
mount | grep huge

# Manually allocate hugepages
echo 1024 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
```

#### 4. Permission Denied Errors

```bash
# Check VFIO permissions
ls -la /dev/vfio/

# Add user to vfio group
sudo usermod -a -G vfio $USER

# Check capabilities
sudo setcap cap_net_admin,cap_net_raw+ep /usr/local/bin/n6_firewall
```

### Debug Mode

Enable debug logging:

```bash
# Temporary debug mode
sudo DOCA_LOG_LEVEL=DEBUG /usr/local/bin/n6_firewall \
    --config /etc/n6-firewall/n6_firewall.conf \
    --verbose

# Enable debug in service
sudo systemctl edit n6-firewall

# Add to override file:
[Service]
Environment=DOCA_LOG_LEVEL=DEBUG
Environment=DOCA_LOG_STDOUT=1
```

### Log Analysis

```bash
# Extract error messages
sudo journalctl -u n6-firewall | grep -i error

# Check for segmentation faults
sudo dmesg | grep n6_firewall

# Monitor system resources
top -p $(pgrep n6_firewall)
```

---

## Production Checklist

### Pre-Deployment

- [ ] **Hardware Verification**
  - [ ] BlueField-3 DPU detected and functional
  - [ ] Network interfaces configured correctly
  - [ ] Sufficient memory and storage available
  
- [ ] **Software Prerequisites**
  - [ ] DOCA SDK 2.6.0+ installed
  - [ ] All dependencies resolved
  - [ ] Compilation successful
  
- [ ] **Network Configuration**
  - [ ] IP addresses assigned
  - [ ] Routing tables configured
  - [ ] Firewall rules (host-level) configured
  - [ ] DNS resolution working

### Deployment

- [ ] **Application Installation**
  - [ ] Binary deployed to correct location
  - [ ] Configuration files in place
  - [ ] Systemd service created and enabled
  - [ ] Log directories created with correct permissions
  
- [ ] **Security Configuration**
  - [ ] Service running with minimal privileges
  - [ ] Configuration files secured
  - [ ] Audit logging enabled

### Post-Deployment

- [ ] **Functional Testing**
  - [ ] Service starts successfully
  - [ ] Interfaces are operational
  - [ ] Basic traffic forwarding works
  - [ ] Firewall rules are effective
  
- [ ] **Performance Testing**
  - [ ] Latency measurements within acceptable range
  - [ ] Throughput meets requirements
  - [ ] CPU utilization reasonable
  - [ ] Memory usage stable
  
- [ ] **Monitoring Setup**
  - [ ] Log aggregation configured
  - [ ] Metrics collection active
  - [ ] Alerting rules defined
  - [ ] Health checks operational

### Ongoing Operations

- [ ] **Backup and Recovery**
  - [ ] Configuration backups automated
  - [ ] Recovery procedures documented
  - [ ] Rollback plan tested
  
- [ ] **Maintenance**
  - [ ] Update procedures defined
  - [ ] Monitoring dashboard configured
  - [ ] Documentation current
  - [ ] Team training completed

---

## Support and Resources

### NVIDIA Resources

- **DOCA Documentation**: https://docs.nvidia.com/doca/
- **BlueField DPU Guide**: https://docs.nvidia.com/bluefield-dpu/
- **Developer Forums**: https://forums.developer.nvidia.com/

### Monitoring Tools

- **Grafana Dashboard**: Available in `docs/grafana/`
- **Prometheus Config**: Available in `config/prometheus.yml`
- **Log Parser Scripts**: Available in `scripts/log_analysis/`

### Emergency Contacts

For production issues:

1. **Level 1**: Check systemd service status and logs
2. **Level 2**: Review application logs and metrics
3. **Level 3**: Contact NVIDIA support with diagnostic data

---

*Last Updated: August 2024*
*Version: 2.6.0*
*Copyright © 2024 NVIDIA Corporation*