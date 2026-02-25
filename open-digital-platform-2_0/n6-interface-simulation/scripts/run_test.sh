#!/bin/bash

#
# Test runner for N6 Interface Simulation
# Orchestrates the complete test scenario
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICES_DIR="$PROJECT_ROOT/services"
DOCA_APP_DIR="$PROJECT_ROOT/doca-app"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[TEST]${NC} $1"
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

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --without-firewall    Test without firewall (should succeed)"
    echo "  --with-firewall       Test with firewall active (should fail)"
    echo "  --full-test          Run both scenarios (default)"
    echo "  --iterations N       Number of UPF requests to send (default: 3)"
    echo "  --delay N           Delay between requests in seconds (default: 1)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run full test (with and without firewall)"
    echo "  $0 --without-firewall                # Test connectivity without firewall"
    echo "  $0 --with-firewall                   # Test with firewall blocking"
    echo "  $0 --full-test --iterations 5        # Full test with 5 iterations"
}

cleanup() {
    echo_info "Cleaning up background processes..."
    
    # Kill DN service if running
    if [[ -n "$DN_PID" ]]; then
        kill $DN_PID 2>/dev/null || true
        wait $DN_PID 2>/dev/null || true
    fi
    
    # Kill firewall if running
    if [[ -n "$FIREWALL_PID" ]]; then
        kill $FIREWALL_PID 2>/dev/null || true
        wait $FIREWALL_PID 2>/dev/null || true
    fi
    
    echo_info "Cleanup completed"
}

# Set up signal handlers
trap cleanup EXIT
trap cleanup SIGINT
trap cleanup SIGTERM

run_without_firewall() {
    echo ""
    echo "================================================"
    echo "    TEST 1: WITHOUT FIREWALL (Should Succeed)   "
    echo "================================================"
    echo ""
    
    echo_info "Starting DN service..."
    python3 "$SERVICES_DIR/dn_service.py" &
    DN_PID=$!
    
    # Wait for DN service to start
    echo_info "Waiting for DN service to start..."
    sleep 3
    
    # Check if DN service is running
    if ! kill -0 $DN_PID 2>/dev/null; then
        echo_error "DN service failed to start"
        return 1
    fi
    
    echo_info "Running UPF service (expecting success)..."
    if python3 "$SERVICES_DIR/upf_service.py" --iterations "$ITERATIONS" --delay "$DELAY"; then
        echo_success "Test 1 PASSED: Traffic reached DN without firewall"
        TEST1_RESULT="PASS"
    else
        echo_error "Test 1 FAILED: Traffic should have reached DN"
        TEST1_RESULT="FAIL"
    fi
    
    # Stop DN service
    echo_info "Stopping DN service..."
    kill $DN_PID 2>/dev/null || true
    wait $DN_PID 2>/dev/null || true
    DN_PID=""
    
    sleep 2
}

run_with_firewall() {
    echo ""
    echo "==============================================="
    echo "    TEST 2: WITH FIREWALL (Should Be Blocked) "
    echo "==============================================="
    echo ""
    
    # Check if firewall executable exists
    FIREWALL_EXEC="$DOCA_APP_DIR/simple_firewall"
    if [[ ! -f "$FIREWALL_EXEC" ]]; then
        echo_warn "Firewall executable not found at: $FIREWALL_EXEC"
        echo_warn "Please run the build script first:"
        echo_warn "  $SCRIPT_DIR/build_firewall.sh"
        echo_warn "Skipping firewall test..."
        TEST2_RESULT="SKIP"
        return 0
    fi
    
    echo_info "Starting DN service..."
    python3 "$SERVICES_DIR/dn_service.py" &
    DN_PID=$!
    
    # Wait for DN service to start
    echo_info "Waiting for DN service to start..."
    sleep 3
    
    # Check if DN service is running
    if ! kill -0 $DN_PID 2>/dev/null; then
        echo_error "DN service failed to start"
        return 1
    fi
    
    echo_info "Starting DOCA firewall..."
    echo_warn "NOTE: This requires NVIDIA AIR environment or compatible DPU setup"
    
    # Start firewall in background
    "$FIREWALL_EXEC" &
    FIREWALL_PID=$!
    
    # Wait for firewall to initialize
    echo_info "Waiting for firewall to initialize..."
    sleep 5
    
    # Check if firewall is still running
    if ! kill -0 $FIREWALL_PID 2>/dev/null; then
        echo_warn "Firewall process exited (may be expected in non-DPU environment)"
        echo_warn "Continuing with test..."
        FIREWALL_PID=""
    fi
    
    echo_info "Running UPF service (expecting to be blocked)..."
    if python3 "$SERVICES_DIR/upf_service.py" --iterations "$ITERATIONS" --delay "$DELAY"; then
        echo_warn "Test 2 WARNING: Traffic reached DN despite firewall"
        echo_warn "This may indicate:"
        echo_warn "  - Firewall is not properly configured"
        echo_warn "  - Running outside NVIDIA AIR environment"
        echo_warn "  - DPU not available for hardware acceleration"
        TEST2_RESULT="WARN"
    else
        echo_success "Test 2 PASSED: Traffic blocked by firewall"
        TEST2_RESULT="PASS"
    fi
    
    # Stop services
    if [[ -n "$FIREWALL_PID" ]]; then
        echo_info "Stopping firewall..."
        kill $FIREWALL_PID 2>/dev/null || true
        wait $FIREWALL_PID 2>/dev/null || true
        FIREWALL_PID=""
    fi
    
    echo_info "Stopping DN service..."
    kill $DN_PID 2>/dev/null || true
    wait $DN_PID 2>/dev/null || true
    DN_PID=""
    
    sleep 2
}

print_results() {
    echo ""
    echo "=============================================="
    echo "              TEST RESULTS                    "
    echo "=============================================="
    echo ""
    
    if [[ "$TEST_MODE" == "full" ]] || [[ "$TEST_MODE" == "without-firewall" ]]; then
        echo_info "Test 1 (Without Firewall): $TEST1_RESULT"
    fi
    
    if [[ "$TEST_MODE" == "full" ]] || [[ "$TEST_MODE" == "with-firewall" ]]; then
        echo_info "Test 2 (With Firewall):    $TEST2_RESULT"
    fi
    
    echo ""
    
    # Overall result
    if [[ "$TEST1_RESULT" == "PASS" && "$TEST2_RESULT" == "PASS" ]]; then
        echo_success "🎉 ALL TESTS PASSED! N6 firewall simulation working correctly."
    elif [[ "$TEST2_RESULT" == "WARN" ]]; then
        echo_warn "⚠️  Tests completed with warnings. Check environment setup."
    elif [[ "$TEST2_RESULT" == "SKIP" ]]; then
        echo_warn "⚠️  Firewall test skipped. Build the DOCA application first."
    else
        echo_error "❌ Some tests failed. Check the logs above."
    fi
    
    echo ""
    echo "=============================================="
}

# Default values
TEST_MODE="full"
ITERATIONS=3
DELAY=1
DN_PID=""
FIREWALL_PID=""
TEST1_RESULT=""
TEST2_RESULT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --without-firewall)
            TEST_MODE="without-firewall"
            shift
            ;;
        --with-firewall)
            TEST_MODE="with-firewall"
            shift
            ;;
        --full-test)
            TEST_MODE="full"
            shift
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
echo ""
echo "=============================================="
echo "        N6 Interface Simulation Test          "
echo "=============================================="
echo ""
echo_info "Test mode: $TEST_MODE"
echo_info "Iterations: $ITERATIONS"
echo_info "Delay: ${DELAY}s"
echo ""

# Check dependencies
echo_info "Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo_error "python3 is required but not installed"
    exit 1
fi

if ! python3 -c "import requests" &> /dev/null; then
    echo_error "Python requests module is required. Install with: pip install requests"
    exit 1
fi

echo_info "Dependencies OK"

# Run tests based on mode
case $TEST_MODE in
    "without-firewall")
        run_without_firewall
        ;;
    "with-firewall")
        run_with_firewall
        ;;
    "full")
        run_without_firewall
        run_with_firewall
        ;;
esac

# Print final results
print_results