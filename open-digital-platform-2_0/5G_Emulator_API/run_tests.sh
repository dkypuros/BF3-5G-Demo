#!/bin/bash
# =============================================================================
# 5G Emulator API - Test Suite Runner
# =============================================================================
# This script:
# 1. Checks Python and dependencies
# 2. Runs the pytest test suite
# 3. Reports results with summary
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}           5G Emulator API - Test Suite Runner                              ${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# =============================================================================
# Check Python
# =============================================================================
echo -e "${YELLOW}[1/4] Checking Python installation...${NC}"

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}Found: $PYTHON_VERSION${NC}"

# Check Python version is 3.8+
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8+ is required${NC}"
    exit 1
fi

echo ""

# =============================================================================
# Check/Activate Virtual Environment
# =============================================================================
echo -e "${YELLOW}[2/4] Checking virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${GREEN}Found virtual environment at ./venv${NC}"
    # Activate venv
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo -e "${GREEN}Virtual environment activated${NC}"
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        echo -e "${GREEN}Virtual environment activated (Windows)${NC}"
    fi
else
    echo -e "${YELLOW}No virtual environment found. Using system Python.${NC}"
fi

echo ""

# =============================================================================
# Check Dependencies
# =============================================================================
echo -e "${YELLOW}[3/4] Checking dependencies...${NC}"

MISSING_DEPS=()

# Check for pytest
if ! $PYTHON_CMD -c "import pytest" 2>/dev/null; then
    MISSING_DEPS+=("pytest")
fi

# Check for httpx (used in tests)
if ! $PYTHON_CMD -c "import httpx" 2>/dev/null; then
    MISSING_DEPS+=("httpx")
fi

# Check for fastapi (required for NFs)
if ! $PYTHON_CMD -c "import fastapi" 2>/dev/null; then
    MISSING_DEPS+=("fastapi")
fi

# Check for uvicorn (required for NFs)
if ! $PYTHON_CMD -c "import uvicorn" 2>/dev/null; then
    MISSING_DEPS+=("uvicorn")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing dependencies: ${MISSING_DEPS[*]}${NC}"
    echo -e "${YELLOW}Installing missing dependencies...${NC}"

    # Install pytest and httpx if missing
    $PYTHON_CMD -m pip install pytest httpx pytest-timeout 2>/dev/null || {
        echo -e "${RED}Failed to install dependencies. Please run:${NC}"
        echo -e "${RED}  pip install pytest httpx pytest-timeout${NC}"
        echo -e "${RED}  pip install -r requirements.txt${NC}"
        exit 1
    }

    echo -e "${GREEN}Dependencies installed${NC}"
else
    echo -e "${GREEN}All dependencies are installed${NC}"
fi

# Show installed versions
echo ""
echo "Installed versions:"
$PYTHON_CMD -c "import pytest; print(f'  pytest: {pytest.__version__}')" 2>/dev/null || true
$PYTHON_CMD -c "import httpx; print(f'  httpx: {httpx.__version__}')" 2>/dev/null || true
$PYTHON_CMD -c "import fastapi; print(f'  fastapi: {fastapi.__version__}')" 2>/dev/null || true

echo ""

# =============================================================================
# Run Tests
# =============================================================================
echo -e "${YELLOW}[4/4] Running tests...${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Parse command line arguments
PYTEST_ARGS=""
RUN_SPECIFIC=""
MARKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --5g)
            MARKERS="-m core_5g"
            shift
            ;;
        --4g)
            MARKERS="-m epc_4g"
            shift
            ;;
        --ims)
            MARKERS="-m ims"
            shift
            ;;
        --quick)
            # Run only health endpoint tests (faster)
            PYTEST_ARGS="-k 'health_endpoint'"
            shift
            ;;
        --nf)
            # Run tests for specific NF
            RUN_SPECIFIC="$2"
            shift 2
            ;;
        --parallel)
            # Enable parallel execution (requires pytest-xdist)
            PYTEST_ARGS="$PYTEST_ARGS -n auto"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --5g        Run only 5G Core tests"
            echo "  --4g        Run only 4G EPC tests"
            echo "  --ims       Run only IMS tests"
            echo "  --quick     Run only health endpoint tests (faster)"
            echo "  --nf NAME   Run tests for specific network function (e.g., --nf NRF)"
            echo "  --parallel  Enable parallel test execution (requires pytest-xdist)"
            echo "  --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all tests"
            echo "  $0 --5g               # Run 5G Core tests only"
            echo "  $0 --nf AMF           # Run AMF tests only"
            echo "  $0 --quick            # Quick health check for all NFs"
            exit 0
            ;;
        *)
            # Pass other args to pytest
            PYTEST_ARGS="$PYTEST_ARGS $1"
            shift
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="$PYTHON_CMD -m pytest tests/"

if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

if [ -n "$RUN_SPECIFIC" ]; then
    PYTEST_CMD="$PYTEST_CMD -k '$RUN_SPECIFIC'"
fi

if [ -n "$PYTEST_ARGS" ]; then
    PYTEST_CMD="$PYTEST_CMD $PYTEST_ARGS"
fi

# Add verbose output
PYTEST_CMD="$PYTEST_CMD -v"

echo "Running: $PYTEST_CMD"
echo ""

# Run pytest and capture exit code
START_TIME=$(date +%s)
eval $PYTEST_CMD
EXIT_CODE=$?
END_TIME=$(date +%s)

DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}=============================================================================${NC}"

# Report results
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    echo -e "${GREEN}Duration: ${DURATION}s${NC}"
else
    echo -e "${RED}Some tests failed (exit code: $EXIT_CODE)${NC}"
    echo -e "${YELLOW}Duration: ${DURATION}s${NC}"
fi

echo -e "${BLUE}=============================================================================${NC}"

exit $EXIT_CODE
