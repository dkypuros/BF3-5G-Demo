#!/bin/bash
# 5G Emulator API Deployment Script
# Version: 2.0.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}5G Emulator API Deployment Script v2.0.0${NC}"
echo -e "${GREEN}==========================================${NC}"

# Check for required tools
check_requirements() {
    echo -e "\n${YELLOW}Checking requirements...${NC}"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 is not installed${NC}"
        exit 1
    fi

    if ! command -v pip3 &> /dev/null; then
        echo -e "${RED}Error: pip3 is not installed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Python3 found: $(python3 --version)${NC}"
}

# Create/activate virtual environment
setup_venv() {
    echo -e "\n${YELLOW}Setting up virtual environment...${NC}"

    cd "$PROJECT_DIR"

    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi

    source venv/bin/activate

    echo "Installing dependencies..."
    pip install -q -r requirements.txt

    echo -e "${GREEN}✓ Virtual environment ready${NC}"
}

# Run static analysis
run_checks() {
    echo -e "\n${YELLOW}Running code quality checks...${NC}"

    cd "$PROJECT_DIR"
    source venv/bin/activate

    python scripts/check_code_quality.py

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed${NC}"
    else
        echo -e "${RED}✗ Code quality checks failed${NC}"
        exit 1
    fi
}

# Test protocol implementations
test_protocols() {
    echo -e "\n${YELLOW}Testing protocol implementations...${NC}"

    cd "$PROJECT_DIR"
    source venv/bin/activate

    echo "Testing NAS protocol..."
    python3 -c "from protocols.nas.nas_5g import demo_nas_encoding; demo_nas_encoding()" 2>/dev/null && echo -e "${GREEN}✓ NAS OK${NC}" || echo -e "${RED}✗ NAS failed${NC}"

    echo "Testing 5G-AKA..."
    python3 -c "from protocols.crypto.aka_5g import demo_5g_aka; demo_5g_aka()" 2>/dev/null && echo -e "${GREEN}✓ 5G-AKA OK${NC}" || echo -e "${RED}✗ 5G-AKA failed${NC}"

    echo "Testing PFCP..."
    python3 -c "from protocols.pfcp.pfcp import demo_pfcp_encoding; demo_pfcp_encoding()" 2>/dev/null && echo -e "${GREEN}✓ PFCP OK${NC}" || echo -e "${RED}✗ PFCP failed${NC}"
}

# Start services locally
start_local() {
    echo -e "\n${YELLOW}Starting services locally...${NC}"

    cd "$PROJECT_DIR"
    source venv/bin/activate

    # Start NRF first
    echo "Starting NRF..."
    python core_network/nrf.py --host 0.0.0.0 --port 8000 &
    NRF_PID=$!
    sleep 2

    # Health check
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ NRF running (PID: $NRF_PID)${NC}"
    else
        echo -e "${RED}✗ NRF failed to start${NC}"
    fi

    echo -e "\n${YELLOW}Press Ctrl+C to stop services${NC}"
    wait
}

# Start with Docker Compose
start_docker() {
    echo -e "\n${YELLOW}Starting with Docker Compose...${NC}"

    cd "$PROJECT_DIR"

    if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: docker-compose not found${NC}"
        exit 1
    fi

    # Check if docker compose (v2) or docker-compose (v1)
    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    echo "Building images..."
    $COMPOSE_CMD build

    echo "Starting services..."
    $COMPOSE_CMD up -d

    echo -e "\n${GREEN}Services started!${NC}"
    echo "View logs: $COMPOSE_CMD logs -f"
    echo "Stop: $COMPOSE_CMD down"
}

# Deploy to remote VM via SSH
deploy_remote() {
    local VM_HOST="$1"
    local VM_USER="${2:-root}"
    local REMOTE_DIR="${3:-/opt/5g-emulator}"

    if [ -z "$VM_HOST" ]; then
        echo -e "${RED}Error: VM host not specified${NC}"
        echo "Usage: $0 remote <vm-host> [user] [remote-dir]"
        exit 1
    fi

    echo -e "\n${YELLOW}Deploying to $VM_USER@$VM_HOST:$REMOTE_DIR${NC}"

    # Sync files
    echo "Syncing files..."
    rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '*.pyc' \
        --exclude '.git' --exclude 'logs/*' \
        "$PROJECT_DIR/" "$VM_USER@$VM_HOST:$REMOTE_DIR/"

    # Remote setup
    echo "Setting up on remote..."
    ssh "$VM_USER@$VM_HOST" << EOF
        cd $REMOTE_DIR

        # Create venv if needed
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi

        # Install dependencies
        source venv/bin/activate
        pip install -q -r requirements.txt

        # Run checks
        python scripts/check_code_quality.py

        echo "Deployment complete!"
EOF

    echo -e "${GREEN}✓ Deployed to $VM_HOST${NC}"
}

# Show help
show_help() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  check       Run code quality checks"
    echo "  test        Test protocol implementations"
    echo "  local       Start services locally"
    echo "  docker      Start with Docker Compose"
    echo "  remote      Deploy to remote VM"
    echo "  help        Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 check"
    echo "  $0 test"
    echo "  $0 local"
    echo "  $0 docker"
    echo "  $0 remote 192.168.1.100 ubuntu /opt/5g-emulator"
}

# Main
case "${1:-help}" in
    check)
        check_requirements
        setup_venv
        run_checks
        ;;
    test)
        check_requirements
        setup_venv
        test_protocols
        ;;
    local)
        check_requirements
        setup_venv
        run_checks
        start_local
        ;;
    docker)
        start_docker
        ;;
    remote)
        check_requirements
        deploy_remote "${2}" "${3}" "${4}"
        ;;
    help|*)
        show_help
        ;;
esac
