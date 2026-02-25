# 5G Emulator API

A comprehensive 5G/LTE/IMS network emulator implementing 3GPP-compliant network functions for development, testing, and educational purposes.

## Project Overview

This project provides a software-based emulation of a complete mobile network stack including:

- **5G Core Network (5GC)**: Full implementation of 5G Service-Based Architecture (SBA) network functions
- **4G/LTE Evolved Packet Core (EPC)**: Legacy LTE core network components for interworking
- **IP Multimedia Subsystem (IMS)**: Voice over LTE (VoLTE) and Voice over NR (VoNR) support
- **Radio Access Network (RAN)**: gNodeB and disaggregated O-RAN components (CU/DU/RRU)
- **Service Assurance**: Real-time monitoring, KQI calculation, and anomaly detection

Each network function exposes RESTful APIs built with FastAPI and includes OpenTelemetry instrumentation for distributed tracing and Prometheus metrics for observability.

## Architecture Overview

### 5G Core Network Functions

| Category | Network Function | Description |
|----------|-----------------|-------------|
| **Service Discovery** | NRF | Network Repository Function - Service registry and discovery |
| **Access & Mobility** | AMF | Access and Mobility Management Function - UE registration, mobility |
| **Session Management** | SMF | Session Management Function - PDU session control |
| **User Plane** | UPF | User Plane Function - Packet forwarding, QoS enforcement |
| **Authentication** | AUSF | Authentication Server Function - 5G-AKA authentication |
| **Data Management** | UDM | Unified Data Management - Subscription data handling |
| **Data Repository** | UDR | Unified Data Repository - Subscriber data storage |
| **Policy Control** | PCF | Policy Control Function - QoS and charging policies |
| **Slice Selection** | NSSF | Network Slice Selection Function - Slice routing |
| **Binding Support** | BSF | Binding Support Function - Session binding management |
| **Service Proxy** | SCP | Service Communication Proxy - Inter-NF routing |
| **Charging** | CHF | Charging Function - Usage reporting and billing |
| **Network Exposure** | NEF | Network Exposure Function - API exposure to external AFs |
| **Roaming Security** | SEPP | Security Edge Protection Proxy - Inter-PLMN security |
| **Non-3GPP Access** | N3IWF | Non-3GPP Interworking Function - WiFi access gateway |

### 4G/LTE EPC Functions

| Network Function | Description |
|-----------------|-------------|
| MME | Mobility Management Entity - EPC control plane |
| SGW | Serving Gateway - User plane anchor for eNB |
| PGW | PDN Gateway - Connection to external networks |
| HSS | Home Subscriber Server - 4G subscriber database |

### IMS Core Functions

| Network Function | Description |
|-----------------|-------------|
| P-CSCF | Proxy Call Session Control Function - IMS entry point |
| I-CSCF | Interrogating Call Session Control Function - Routing queries |
| S-CSCF | Serving Call Session Control Function - Call control |
| MRF | Media Resource Function - Media handling |
| IMS-HSS | IMS Home Subscriber Server - IMS user profiles |

### Radio Access Network (RAN)

| Component | Description |
|-----------|-------------|
| gNB | 5G New Radio Base Station (integrated) |
| CU | Central Unit - Higher layer processing (PDCP, RRC, SDAP) |
| DU | Distributed Unit - Lower layer processing (RLC, MAC, PHY) |
| RRU | Remote Radio Unit - RF frontend |

### Supporting Components

| Component | Description |
|-----------|-------------|
| PTP | Precision Time Protocol - Network synchronization |
| Service Assurance | Real-time monitoring and analytics |

### ETSI Standards Integration

Beyond 3GPP, this platform implements ETSI standards for edge computing, NFV, and automation:

| Category | Component | Specification | Description |
|----------|-----------|---------------|-------------|
| **MEC** | MEC Platform | GS MEC 011 | Edge computing platform with Mp1 interface |
| **MEC** | Location API | GS MEC 013 | UE location service for edge apps |
| **MEC** | RNIS API | GS MEC 012 | Radio network information service |
| **NFV** | VNFM | GS NFV-SOL 003 | VNF lifecycle management |
| **ZSM** | Closed Loop | GS ZSM 002 | Zero-touch automation engine |

For detailed ETSI documentation, see [etsi/README.md](etsi/README.md).

## Prerequisites

- **Python 3.11+** (tested with Python 3.14)
- **pip** package manager
- **MongoDB** (optional, for persistent storage)
- **Prometheus** (optional, for metrics collection)
- **Jaeger** (optional, for distributed tracing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 5G_Emulator_API
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
.\venv\Scripts\activate   # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### Option 1: Start All Network Functions (Root Launcher)

Start all basic network functions using the root-level launcher:

```bash
python main.py
```

This starts: NRF, AMF, SMF, UPF, AUSF, UDM, UDR, UDSF, CU, DU, RRU, PTP, and Service Assurance.

### Option 2: Start Complete 5G/4G/IMS Stack

Start the full network stack including 4G EPC and IMS:

```bash
python core_network/main.py
```

This provides a phased startup of all network functions in the correct order.

### Option 3: Start Individual Network Functions

Run specific network functions as needed:

```bash
# 5G Core
python core_network/nrf.py
python core_network/amf.py
python core_network/smf.py

# 4G EPC
python core_network/mme.py
python core_network/hss.py

# IMS
python core_network/pcscf.py
python core_network/scscf.py

# RAN
python ran/gnb.py
```

### Command-Line Arguments

All network functions support the following CLI arguments for flexible deployment:

```bash
python core_network/<nf>.py --host <host> --port <port>
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Host address to bind to |
| `--port` | varies | Port to bind to (see Service Ports table) |

**Examples:**

```bash
# Start NRF on custom port
python core_network/nrf.py --port 8080

# Start MME on specific interface
python core_network/mme.py --host 192.168.1.100 --port 9020

# Start multiple instances for testing
python core_network/amf.py --port 9000 &
python core_network/amf.py --port 9001 &
```

## Service Ports

**Note:** All service ports are centrally defined in `config/ports.py`. To change a port assignment, modify the `NF_PORTS` dictionary in that file.

### 5G Core Network Functions

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| NRF | 8000 | http://localhost:8000 | Service Discovery |
| AMF | 9000 | http://localhost:9000 | Access & Mobility |
| SMF | 9001 | http://localhost:9001 | Session Management |
| UPF | 9002 | http://localhost:9002 | User Plane |
| AUSF | 9003 | http://localhost:9003 | Authentication |
| UDM | 9004 | http://localhost:9004 | Data Management |
| UDR | 9005 | http://localhost:9005 | Data Repository |
| UDSF | 9006 | http://localhost:9006 | Unstructured Data Storage |
| PCF | 9007 | http://localhost:9007 | Policy Control |
| NSSF | 9010 | http://localhost:9010 | Slice Selection |
| BSF | 9011 | http://localhost:9011 | Binding Support |
| SCP | 9012 | http://localhost:9012 | Service Proxy |
| CHF | 9013 | http://localhost:9013 | Charging |
| SEPP | 9014 | http://localhost:9014 | Roaming Security |
| N3IWF | 9015 | http://localhost:9015 | Non-3GPP Access |
| NEF | 9016 | http://localhost:9016 | Network Exposure |

### 4G/LTE EPC Functions

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| MME | 9020 | http://localhost:9020 | Mobility Management |
| SGW | 9021 | http://localhost:9021 | Serving Gateway |
| PGW | 9022 | http://localhost:9022 | PDN Gateway |
| HSS | 9023 | http://localhost:9023 | Subscriber Server (4G) |

### IMS Core Functions

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| P-CSCF | 9030 | http://localhost:9030 | Proxy CSCF (Entry Point) |
| I-CSCF | 9031 | http://localhost:9031 | Interrogating CSCF |
| S-CSCF | 9032 | http://localhost:9032 | Serving CSCF |
| MRF | 9033 | http://localhost:9033 | Media Resource Function |
| IMS-HSS | 9040 | http://localhost:9040 | IMS Home Subscriber Server |

### Radio Access Network

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| gNB | 9100 | http://localhost:9100 | 5G Base Station |
| CU | 9101 | http://localhost:9101 | Central Unit |
| DU | 9102 | http://localhost:9102 | Distributed Unit |
| RRU | 9103 | http://localhost:9103 | Remote Radio Unit |

## Health Endpoint Documentation

All network functions expose standard health check endpoints for monitoring and orchestration.

### GET /health

Returns the health status of the network function with 3GPP compliance information.

**Standard Response Format:**
```json
{
  "status": "healthy",
  "service": "<nf-name>",
  "compliance": "3GPP TS XX.XXX",
  "version": "1.0.0"
}
```

**Extended Response (some NFs include additional fields):**
```json
{
  "status": "healthy",
  "service": "MME",
  "compliance": "3GPP TS 23.401",
  "version": "1.0.0",
  "mme_id": "2f79103c",
  "mme_name": "MME-01",
  "connected_ues": 0,
  "connected_enbs": 0
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Health status: "healthy" or "unhealthy" |
| `service` | string | Network function name (e.g., "AMF", "SMF", "MME") |
| `compliance` | string | Primary 3GPP specification reference |
| `version` | string | Implementation version |

**Status Codes:**
- `200 OK`: Service is healthy and operational
- `503 Service Unavailable`: Service is unhealthy or starting up

### GET /metrics

Returns Prometheus-formatted metrics for monitoring.

**Response:** Prometheus text format containing:
- Request counts and latencies
- Active sessions/connections
- NF-specific operational metrics

### GET /docs

Returns interactive Swagger/OpenAPI documentation for the network function's API.

## 3GPP Specification Compliance

Each network function is implemented according to 3GPP Release 19 specifications:

### 5G Core Network Functions

| NF | Primary Specification | SBI Service | YAML API |
|----|----------------------|-------------|----------|
| NRF | TS 29.510 | Nnrf | TS29510_Nnrf_NFManagement.yaml |
| AMF | TS 29.518 | Namf | TS29518_Namf_Communication.yaml |
| SMF | TS 29.502 | Nsmf | TS29502_Nsmf_PDUSession.yaml |
| UPF | TS 29.244 | N4 (PFCP) | - |
| AUSF | TS 29.509 | Nausf | TS29509_Nausf_UEAuthentication.yaml |
| UDM | TS 29.503 | Nudm | TS29503_Nudm_SDM.yaml |
| UDR | TS 29.504 | Nudr | TS29504_Nudr_DR.yaml |
| PCF | TS 29.512 | Npcf | TS29512_Npcf_SMPolicyControl.yaml |
| NSSF | TS 29.531 | Nnssf | TS29531_Nnssf_NSSelection.yaml |
| BSF | TS 29.521 | Nbsf | TS29521_Nbsf_Management.yaml |
| SCP | TS 29.500 | - | - |
| CHF | TS 32.290/32.291 | Nchf | TS32291_Nchf_SpendingLimit.yaml |
| NEF | TS 29.522 | Nnef | TS29522_Nnef_EventExposure.yaml |
| SEPP | TS 29.573 | N32 | TS29573_N32_Handshake.yaml |
| N3IWF | TS 29.502/24.502 | - | - |

### 4G/LTE EPC Functions

| NF | Primary Specification | Interface |
|----|----------------------|-----------|
| MME | TS 23.401, TS 24.301 | S1AP (TS 36.413) |
| SGW | TS 23.401, TS 29.274 | GTPv2-C (S11/S5) |
| PGW | TS 23.401, TS 29.274 | GTPv2-C (S5/S8) |
| HSS | TS 23.002, TS 29.272 | Diameter (S6a) |

### IMS Core Functions

| NF | Primary Specification | Protocol |
|----|----------------------|----------|
| P-CSCF | TS 24.229 | SIP, Diameter (Rx) |
| I-CSCF | TS 24.229 | SIP, Diameter (Cx) |
| S-CSCF | TS 24.229 | SIP, Diameter (Cx) |
| MRF | TS 23.228 | SIP, RTP/RTCP |
| IMS-HSS | TS 29.228/29.229 | Diameter (Cx/Dx) |

### RAN Components

| Component | Primary Specification | Interface |
|-----------|----------------------|-----------|
| gNB | TS 38.413 | NGAP (N2) |
| CU | TS 38.401 | F1AP (F1) |
| DU | TS 38.401 | F1AP (F1) |

## Detailed Code-to-Spec Mapping

Each network function has an accompanying `.spec.txt` file that provides line-by-line mapping between the implementation and 3GPP specifications:

```
core_network/
  nrf.py              # Implementation
  nrf.py.spec.txt     # Specification mapping
  amf.py
  amf.py.spec.txt
  smf.py
  smf.py.spec.txt
  ...
ran/
  gnb.py
  gnb.py.spec.txt
```

These specification files document:
- Primary governing 3GPP specifications
- Line-by-line specification mapping
- Data model references to OpenAPI YAML schemas
- Implementation gaps vs. full specification compliance
- Related IETF RFCs (OAuth2, JWT, HTTP/2, etc.)

## Project Structure

```
5G_Emulator_API/
+-- main.py                    # Root launcher (basic NFs)
+-- requirements.txt           # Python dependencies
+-- README.md                  # This file
+-- core_network/
|   +-- main.py               # Full stack launcher
|   +-- nrf.py                # Network Repository Function
|   +-- amf.py                # Access and Mobility Management
|   +-- smf.py                # Session Management
|   +-- upf.py                # User Plane Function
|   +-- upf_enhanced.py       # Enhanced UPF with full PFCP
|   +-- ausf.py               # Authentication Server
|   +-- udm.py                # Unified Data Management
|   +-- udr.py                # Unified Data Repository
|   +-- pcf.py                # Policy Control
|   +-- nssf.py               # Slice Selection
|   +-- bsf.py                # Binding Support
|   +-- scp.py                # Service Communication Proxy
|   +-- chf.py                # Charging Function
|   +-- nef.py                # Network Exposure
|   +-- sepp.py               # Security Edge Protection
|   +-- n3iwf.py              # Non-3GPP Interworking
|   +-- mme.py                # 4G Mobility Management
|   +-- sgw.py                # 4G Serving Gateway
|   +-- pgw.py                # 4G PDN Gateway
|   +-- hss.py                # 4G Home Subscriber Server
|   +-- pcscf.py              # IMS Proxy CSCF
|   +-- icscf.py              # IMS Interrogating CSCF
|   +-- scscf.py              # IMS Serving CSCF
|   +-- mrf.py                # IMS Media Resource
|   +-- ims_hss.py            # IMS Home Subscriber Server
|   +-- db.py                 # Database utilities
|   +-- *.py.spec.txt         # Specification mapping files
+-- ran/
|   +-- gnb.py                # Integrated gNodeB
|   +-- gnb.py.spec.txt       # gNB spec mapping
|   +-- cu/                   # Central Unit
|   |   +-- cu.py
|   |   +-- pdcp.py
|   |   +-- rrc.py
|   |   +-- sdap.py
|   +-- du/                   # Distributed Unit
|   |   +-- du.py
|   |   +-- rlc.py
|   |   +-- mac.py
|   |   +-- phy.py
|   +-- rru/                  # Remote Radio Unit
|       +-- rru.py
+-- ptp/
|   +-- ptp.py                # Precision Time Protocol
+-- service_assurance/
|   +-- assurance_api.py      # Assurance REST API
|   +-- collector.py          # Metrics collector
|   +-- kqi_calculator.py     # KQI computation
|   +-- anomaly_detector.py   # Anomaly detection
|   +-- rca_engine.py         # Root cause analysis
|   +-- sla_manager.py        # SLA management
|   +-- models.py             # Data models
+-- protocols/                # Real 3GPP protocol implementations
|   +-- nas/                  # 5G NAS (TS 24.501)
|   |   +-- nas_5g.py         # Binary NAS encoding/decoding
|   +-- crypto/               # 5G-AKA (TS 33.501)
|   |   +-- aka_5g.py         # Milenage, key derivation
|   +-- pfcp/                 # PFCP (TS 29.244)
|   |   +-- pfcp.py           # N4 interface protocol
|   +-- sctp/                 # SCTP Transport (TS 38.412)
|       +-- sctp_transport.py # NGAP transport layer
+-- config/
|   +-- ports.py              # Centralized port configuration
+-- tests/
|   +-- conftest.py           # Pytest fixtures
|   +-- test_network_functions.py  # Comprehensive test suite
+-- scripts/
|   +-- check_code_quality.py # Static analysis tool
+-- docs/
|   +-- COMPLIANCE_REPORT.md  # 3GPP compliance assessment
|   +-- BEFORE_STATE_*.md     # Baseline documentation
|   +-- IMPLEMENTATION_CHANGES_*.md  # Implementation docs
+-- logs/                     # Runtime logs
```

## Real Protocol Implementations

The `protocols/` directory contains real 3GPP protocol implementations for enhanced realism:

### NAS Protocol (protocols/nas/)
**Reference:** TS 24.501

Binary encoding/decoding for 5G Non-Access Stratum messages:
- Registration Request/Accept
- Authentication Request/Response
- Security Mode Command/Complete
- PDU Session Establishment Request/Accept

```python
from protocols import NASCodec, PLMN, SNSSAI, RegistrationType

codec = NASCodec()
reg_req = codec.encode_registration_request(
    registration_type=RegistrationType.INITIAL,
    supi="imsi-001010000000001",
    plmn=PLMN("001", "01"),
    requested_nssai=[SNSSAI(sst=1)]
)
# Returns binary NAS message bytes
```

### 5G-AKA Authentication (protocols/crypto/)
**Reference:** TS 33.501, TS 35.206

Complete Milenage-based authentication:
- Real Milenage algorithm (f1-f5)
- Key derivation (KAUSF → KSEAF → KAMF → KgNB)
- RES*/HRES* computation

```python
from protocols import AKA5GHandler, SubscriberData

handler = AKA5GHandler()
av = handler.generate_auth_vector(subscriber)
# Returns: rand, autn, xres_star, hxres_star, kausf, kseaf
```

### PFCP Protocol (protocols/pfcp/)
**Reference:** TS 29.244

Binary PFCP for SMF↔UPF (N4 interface):
- Session Establishment/Modification/Deletion
- Create PDR/FAR/QER
- Heartbeat, Association Setup

```python
from protocols import PFCPNode

smf = PFCPNode("SMF", "10.0.0.1")
session_req = smf.create_session_establishment_request(
    seid=0x123456789ABCDEF0,
    ue_ip="10.45.0.1",
    gnb_teid=0x12345678,
    gnb_ip="192.168.1.100",
    upf_teid=0x87654321
)
```

### SCTP Transport (protocols/sctp/)
**Reference:** TS 38.412

SCTP transport for NGAP signaling:
- Multi-homing support
- Multi-streaming (UE-associated on separate streams)
- TCP fallback when SCTP unavailable

```python
from protocols import NGAPSCTPHandler

amf = NGAPSCTPHandler("AMF", ["10.0.0.1"], 38412)
await amf.start()

gnb = NGAPSCTPHandler("gNB", ["192.168.1.1"])
await gnb.connect(["10.0.0.1"], 38412)
await gnb.send_non_ue_message(ng_setup_bytes)
```

## Deployment

### Local Development

```bash
# Run code quality checks
./scripts/deploy.sh check

# Test protocol implementations
./scripts/deploy.sh test

# Start services locally
./scripts/deploy.sh local
```

### Docker Deployment

```bash
# Start with Docker Compose
./scripts/deploy.sh docker

# Or directly:
docker-compose up -d
```

### Remote VM Deployment

```bash
# Deploy to remote VM via SSH
./scripts/deploy.sh remote <vm-host> [user] [remote-dir]

# Example:
./scripts/deploy.sh remote 192.168.1.100 ubuntu /opt/5g-emulator
```

The deployment script:
1. Syncs files via rsync (excluding venv, __pycache__, .git)
2. Creates virtual environment on remote
3. Installs dependencies
4. Runs code quality checks

## Testing

Run the test scripts to verify network function operation:

```bash
# Test 5G network procedures
python test_5g_network.py

# Test CU/DU disaggregated RAN
python test_cu_du.py

# Test Service Assurance
python test_service_assurance.py

# Run protocol implementation tests
./scripts/deploy.sh test
```

## Docker Support

Build and run using Docker:

```bash
# Build the image
docker build -t 5g-emulator .

# Run with docker-compose
docker-compose up -d
```

## Observability

### Metrics

All network functions expose Prometheus metrics at `/metrics`. Configure your Prometheus instance to scrape these endpoints using the provided `prometheus.yml`.

### Distributed Tracing

OpenTelemetry instrumentation is enabled for all network functions. Traces are exported to:
- Jaeger (via OTLP exporter)
- Console (for development)
- JSON files (in `logs/` directory)

## Contributing

Contributions are welcome. Please ensure:
1. New network functions follow existing patterns
2. Include corresponding `.spec.txt` specification mapping
3. Add OpenTelemetry instrumentation
4. Update this README with new ports and specifications

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed list of changes between versions.

## License

This project is for educational and development purposes.

## References

- [3GPP Specifications](https://www.3gpp.org/specifications)
- [OpenTelemetry](https://opentelemetry.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Prometheus](https://prometheus.io/)

---

*Last updated: 2026-01-10 | Version 2.0.0*
