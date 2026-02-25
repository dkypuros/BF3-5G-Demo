"""
Pytest Configuration and Shared Fixtures for 5G Emulator API Test Suite

This module provides fixtures for:
- Starting/stopping network function subprocesses
- Managing unique ports for each NF
- Handling timeouts gracefully
- Shared HTTP client utilities
"""

import pytest
import subprocess
import time
import socket
import signal
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Generator, Tuple
import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Network Function Configuration
# =============================================================================

# Base port for dynamic port allocation during tests
BASE_TEST_PORT = 18000

# Network Function definitions with their script paths and default ports
# Format: nf_name -> (script_path, default_port, compliance_spec)
NETWORK_FUNCTIONS: Dict[str, Tuple[str, int, str]] = {
    # 5G Core Network Functions
    "NRF": ("core_network/nrf.py", 9000, "3GPP TS 29.510"),
    "AMF": ("core_network/amf.py", 9001, "3GPP TS 29.518, TS 38.413"),
    "SMF": ("core_network/smf.py", 9002, "3GPP TS 29.502"),
    "UPF": ("core_network/upf.py", 9003, "3GPP TS 29.244"),
    "AUSF": ("core_network/ausf.py", 9004, "3GPP TS 29.509"),
    "UDM": ("core_network/udm.py", 9005, "3GPP TS 29.503"),
    "UDR": ("core_network/udr.py", 9007, "3GPP TS 29.504"),
    "PCF": ("core_network/pcf.py", 9006, "3GPP TS 29.507, TS 29.512, TS 29.514"),
    "NSSF": ("core_network/nssf.py", 9010, "3GPP TS 29.531"),
    "BSF": ("core_network/bsf.py", 9011, "3GPP TS 29.521"),
    "SCP": ("core_network/scp.py", 9012, "3GPP TS 29.500"),
    "CHF": ("core_network/chf.py", 9013, "3GPP TS 32.290/32.291"),
    "NEF": ("core_network/nef.py", 9016, "3GPP TS 29.522"),
    "SEPP": ("core_network/sepp.py", 9014, "3GPP TS 29.573"),
    "N3IWF": ("core_network/n3iwf.py", 9015, "3GPP TS 29.502/24.502"),
    # 4G EPC Network Functions
    "MME": ("core_network/mme.py", 9020, "3GPP TS 23.401"),
    "SGW": ("core_network/sgw.py", 9021, "3GPP TS 23.401"),
    "PGW": ("core_network/pgw.py", 9022, "3GPP TS 23.401"),
    "HSS": ("core_network/hss.py", 9023, "3GPP TS 29.272"),
    # IMS Core Network Functions
    "P-CSCF": ("core_network/pcscf.py", 9030, "3GPP TS 24.229"),
    "I-CSCF": ("core_network/icscf.py", 9031, "3GPP TS 24.229"),
    "S-CSCF": ("core_network/scscf.py", 9032, "3GPP TS 24.229"),
    "MRF": ("core_network/mrf.py", 9033, "3GPP TS 23.228"),
    "IMS-HSS": ("core_network/ims_hss.py", 9040, "3GPP TS 29.228"),
}

# Timeout settings
STARTUP_TIMEOUT = 15  # seconds to wait for NF to start
SHUTDOWN_TIMEOUT = 5  # seconds to wait for NF to stop
REQUEST_TIMEOUT = 10  # seconds for HTTP requests


# =============================================================================
# Utility Functions
# =============================================================================

def find_free_port(start_port: int = BASE_TEST_PORT) -> int:
    """Find a free port starting from start_port."""
    port = start_port
    while port < start_port + 1000:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            port += 1
    raise RuntimeError(f"Could not find free port starting from {start_port}")


def is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def wait_for_port(port: int, timeout: float = STARTUP_TIMEOUT) -> bool:
    """Wait for a port to become available (service started)."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(0.2)
    return False


def wait_for_health(base_url: str, timeout: float = STARTUP_TIMEOUT) -> bool:
    """Wait for health endpoint to respond with healthy status."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        return True
        except (httpx.RequestError, Exception):
            pass
        time.sleep(0.3)
    return False


# =============================================================================
# Port Allocation Fixture
# =============================================================================

# Global port counter to ensure unique ports across test session
_port_counter = BASE_TEST_PORT


@pytest.fixture(scope="session")
def port_allocator():
    """Provides a function to allocate unique ports for tests."""
    allocated_ports = {}
    current_port = BASE_TEST_PORT

    def allocate(nf_name: str) -> int:
        nonlocal current_port
        if nf_name in allocated_ports:
            return allocated_ports[nf_name]

        port = find_free_port(current_port)
        allocated_ports[nf_name] = port
        current_port = port + 1
        return port

    return allocate


# =============================================================================
# Network Function Process Management
# =============================================================================

class NFProcess:
    """Manages a network function subprocess."""

    def __init__(self, nf_name: str, script_path: str, port: int, project_root: Path):
        self.nf_name = nf_name
        self.script_path = project_root / script_path
        self.port = port
        self.project_root = project_root
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://127.0.0.1:{port}"

    def start(self, timeout: float = STARTUP_TIMEOUT) -> bool:
        """Start the network function subprocess."""
        if not self.script_path.exists():
            raise FileNotFoundError(f"Script not found: {self.script_path}")

        # Prepare environment
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Start the process
        cmd = [
            sys.executable,
            str(self.script_path),
            "--host", "127.0.0.1",
            "--port", str(self.port)
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(self.project_root)
        )

        # Wait for the service to be ready
        if wait_for_health(self.base_url, timeout):
            return True

        # If health check fails, try just port check
        if wait_for_port(self.port, timeout):
            # Give it a moment more for health endpoint
            time.sleep(0.5)
            return True

        return False

    def stop(self, timeout: float = SHUTDOWN_TIMEOUT):
        """Stop the network function subprocess."""
        if self.process is None:
            return

        # Try graceful shutdown first
        self.process.terminate()

        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown fails
            self.process.kill()
            self.process.wait(timeout=2)

        self.process = None

    def is_running(self) -> bool:
        """Check if the process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None


@pytest.fixture
def nf_process_factory(port_allocator):
    """
    Factory fixture to create NF processes.

    Usage:
        def test_example(nf_process_factory):
            nf = nf_process_factory("NRF")
            # nf.base_url contains the URL
            # nf is automatically cleaned up after the test
    """
    processes = []

    def create(nf_name: str, timeout: float = STARTUP_TIMEOUT) -> NFProcess:
        if nf_name not in NETWORK_FUNCTIONS:
            raise ValueError(f"Unknown network function: {nf_name}")

        script_path, _, _ = NETWORK_FUNCTIONS[nf_name]
        port = port_allocator(nf_name)

        nf = NFProcess(nf_name, script_path, port, PROJECT_ROOT)

        if not nf.start(timeout):
            nf.stop()
            pytest.skip(f"Failed to start {nf_name} within {timeout}s")

        processes.append(nf)
        return nf

    yield create

    # Cleanup all processes
    for nf in processes:
        nf.stop()


# =============================================================================
# Individual NF Fixtures (for convenience)
# =============================================================================

def make_nf_fixture(nf_name: str):
    """Factory to create individual NF fixtures."""
    @pytest.fixture
    def nf_fixture(nf_process_factory) -> NFProcess:
        return nf_process_factory(nf_name)
    return nf_fixture


# Create individual fixtures for each NF
nrf_service = make_nf_fixture("NRF")
amf_service = make_nf_fixture("AMF")
smf_service = make_nf_fixture("SMF")
upf_service = make_nf_fixture("UPF")
ausf_service = make_nf_fixture("AUSF")
udm_service = make_nf_fixture("UDM")
udr_service = make_nf_fixture("UDR")
pcf_service = make_nf_fixture("PCF")
nssf_service = make_nf_fixture("NSSF")
bsf_service = make_nf_fixture("BSF")
scp_service = make_nf_fixture("SCP")
chf_service = make_nf_fixture("CHF")
nef_service = make_nf_fixture("NEF")
sepp_service = make_nf_fixture("SEPP")
n3iwf_service = make_nf_fixture("N3IWF")
mme_service = make_nf_fixture("MME")
sgw_service = make_nf_fixture("SGW")
pgw_service = make_nf_fixture("PGW")
hss_service = make_nf_fixture("HSS")
pcscf_service = make_nf_fixture("P-CSCF")
icscf_service = make_nf_fixture("I-CSCF")
scscf_service = make_nf_fixture("S-CSCF")
mrf_service = make_nf_fixture("MRF")
ims_hss_service = make_nf_fixture("IMS-HSS")


# =============================================================================
# HTTP Client Fixtures
# =============================================================================

@pytest.fixture
def http_client():
    """Provides an HTTP client for making requests."""
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        yield client


@pytest.fixture
def async_http_client():
    """Provides an async HTTP client for making requests."""
    async def get_client():
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            yield client
    return get_client


# =============================================================================
# Test Markers Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "core_5g: Tests for 5G Core Network Functions"
    )
    config.addinivalue_line(
        "markers", "epc_4g: Tests for 4G EPC Network Functions"
    )
    config.addinivalue_line(
        "markers", "ims: Tests for IMS Core Network Functions"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that may take longer to run"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring multiple NFs"
    )


# =============================================================================
# Session-Level Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def nf_definitions() -> Dict[str, Tuple[str, int, str]]:
    """Returns the network function definitions."""
    return NETWORK_FUNCTIONS.copy()
