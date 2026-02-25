# Changelog

All notable changes to the 5G Emulator API project are documented in this file.

## [2.0.0] - 2026-01-10

### Added

#### Real Protocol Implementations (~4,200 lines of new code)

**NAS Protocol (protocols/nas/nas_5g.py) - TS 24.501:**
- Binary encoding/decoding for 5G NAS messages
- Registration Request/Accept messages
- Authentication Request/Response messages
- Security Mode Command/Complete messages
- PDU Session Establishment Request/Accept messages
- All 5GMM and 5GSM message types
- PLMN, S-NSSAI, 5G-GUTI, SUCI data structures
- UE Security Capability encoding

**5G-AKA Authentication (protocols/crypto/aka_5g.py) - TS 33.501:**
- Complete Milenage algorithm (TS 35.206)
  - f1 (MAC-A), f1* (MAC-S)
  - f2 (RES), f3 (CK), f4 (IK), f5 (AK), f5*
- Full key derivation hierarchy:
  - CK'/IK' derivation
  - KAUSF, KSEAF, KAMF derivation
  - KgNB derivation
  - NAS encryption/integrity keys
- RES* and HRES* computation
- SUCI/SUPI conversion handler
- Authentication vector generation

**PFCP Protocol (protocols/pfcp/pfcp.py) - TS 29.244:**
- Binary PFCP message encoding/decoding
- Node messages: Heartbeat, Association Setup/Update/Release
- Session messages: Establishment, Modification, Deletion, Report
- Information Elements: Node ID, F-SEID, F-TEID, PDI, FAR, PDR, QER
- Grouped IEs: Create PDR, Create FAR, Create QER
- Async UDP transport for PFCP
- PDU session setup with uplink/downlink PDRs and FARs

**SCTP Transport (protocols/sctp/sctp_transport.py) - TS 38.412:**
- Real SCTP support via pysctp (when available)
- TCP fallback with message framing
- Multi-homing support (multiple IP addresses)
- Multi-streaming for NGAP (stream 0 for non-UE, 1+ for UE-associated)
- NGAP-specific handler class
- Async transport implementation

#### Documentation
- `docs/BEFORE_STATE_2026-01-10.md` - Baseline state documentation
- `docs/IMPLEMENTATION_CHANGES_2026-01-10.md` - Implementation details
- Updated README.md with protocol usage examples

---

## [1.1.1] - 2026-01-10

### Fixed

#### Pydantic v2 Compatibility (3 files)
Fixed deprecated `regex=` parameter in Field definitions, changed to `pattern=` for Pydantic v2:
- `core_network/nrf.py` - PlmnId and Snssai models
- `core_network/amf_nas.py` - PlmnId and Snssai models
- `ran/gnb.py` - PlmnIdentity, NrCgi, and Tai models

#### Test Suite Import (1 file)
Fixed import path in `tests/test_network_functions.py` to correctly import from conftest.

### Added

#### Quality Assurance Infrastructure
- `scripts/check_code_quality.py` - Static analysis tool with 5 validation checks
- `config/ports.py` - Centralized port configuration for all 29 NFs
- `tests/conftest.py` - Pytest fixtures for all 24 testable NFs
- `tests/test_network_functions.py` - Comprehensive test suite (205 tests)
- `docs/COMPLIANCE_REPORT.md` - 3GPP protocol compliance assessment
- `pytest.ini` - Pytest configuration with custom markers
- `run_tests.sh` - Test runner helper script

---

## [1.1.0] - 2026-01-10

### Added

#### Specification Mapping Files
Created comprehensive `.spec.txt` files for all 25 network functions, mapping each code section to its governing 3GPP specifications:

**5G Core Network (15 files):**
- `nrf.py.spec.txt` - TS 29.510 (Nnrf_NFManagement, Nnrf_NFDiscovery)
- `amf.py.spec.txt` - TS 29.518, TS 38.413 (Namf, NGAP)
- `smf.py.spec.txt` - TS 29.502 (Nsmf_PDUSession)
- `upf_enhanced.py.spec.txt` - TS 29.244 (PFCP N4 interface)
- `ausf.py.spec.txt` - TS 29.509 (Nausf_UEAuthentication)
- `udm.py.spec.txt` - TS 29.503 (Nudm services)
- `udr.py.spec.txt` - TS 29.504 (Nudr_DataRepository)
- `pcf.py.spec.txt` - TS 29.512, TS 29.507 (Npcf services)
- `nssf.py.spec.txt` - TS 29.531 (Nnssf_NSSelection)
- `bsf.py.spec.txt` - TS 29.521 (Nbsf_Management)
- `scp.py.spec.txt` - TS 29.500 (SBI routing)
- `chf.py.spec.txt` - TS 32.291 (Nchf_SpendingLimit)
- `nef.py.spec.txt` - TS 29.522 (NEF northbound APIs)
- `sepp.py.spec.txt` - TS 29.573 (N32 interface)
- `n3iwf.py.spec.txt` - TS 24.502 (Non-3GPP access)

**4G/LTE EPC (4 files):**
- `mme.py.spec.txt` - TS 23.401, TS 24.301, TS 36.413
- `sgw.py.spec.txt` - TS 23.401, TS 29.274
- `pgw.py.spec.txt` - TS 23.401, TS 29.274, TS 23.203
- `hss.py.spec.txt` - TS 29.272, TS 33.401, TS 35.206

**IMS Core (5 files):**
- `pcscf.py.spec.txt` - TS 24.229 (SIP, Diameter Rx)
- `icscf.py.spec.txt` - TS 24.229 (SIP, Diameter Cx)
- `scscf.py.spec.txt` - TS 24.229 (SIP, Diameter Cx)
- `mrf.py.spec.txt` - TS 23.228 (Media handling)
- `ims_hss.py.spec.txt` - TS 29.228, TS 29.229 (Diameter Cx/Dx)

**RAN (1 file):**
- `gnb.py.spec.txt` - TS 38.413, TS 38.401 (NGAP, F1AP)

#### Project Documentation
- Created comprehensive `README.md` (433 lines) documenting:
  - Project architecture overview
  - All network functions with descriptions
  - Service port assignments
  - 3GPP specification compliance table
  - Installation and running instructions
  - Health endpoint documentation
  - Project structure

### Changed

#### Port Binding (27 files)
All network function files now properly handle `--host` and `--port` command-line arguments:

```python
# Before (hardcoded)
uvicorn.run(app, host="127.0.0.1", port=9000)

# After (configurable)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NF Description")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
```

**Files modified:**
- 5G Core: nrf.py, amf.py, smf.py, upf.py, ausf.py, udm.py, udr.py, udsf.py, pcf.py, nssf.py, bsf.py, scp.py, chf.py, sepp.py, n3iwf.py, nef.py
- 4G EPC: mme.py, sgw.py, pgw.py, hss.py
- IMS: pcscf.py, icscf.py, scscf.py, mrf.py, ims_hss.py
- Other: upf_enhanced.py, amf-orig.py, amf_nas.py

#### Health Endpoints (25 files)
Standardized health check responses across all network functions:

```json
{
  "status": "healthy",
  "service": "SERVICE_NAME",
  "compliance": "3GPP TS XX.XXX",
  "version": "1.0.0"
}
```

**New health endpoints added:**
- smf.py, upf.py, udr.py, udsf.py

**Compliance field added to existing endpoints:**
- mme.py, sgw.py, pgw.py, hss.py
- pcscf.py, icscf.py, scscf.py, mrf.py, ims_hss.py

#### Dependencies (requirements.txt)
Updated with organized sections and missing packages:

**Added packages:**
- `deprecated>=1.2.14` - Required by opentelemetry-exporter-jaeger
- `cryptography>=41.0.0` - Used by SEPP for TLS/crypto operations
- `PyJWT>=2.8.0` - Used by NRF for OAuth2 tokens
- `tabulate>=0.9.0` - Used for metrics formatting

**Organized sections:**
- Core Web Framework (fastapi, uvicorn, pydantic)
- HTTP Clients (requests, httpx)
- OpenTelemetry (tracing and metrics)
- Prometheus Metrics
- Database (pymongo)
- Security and Cryptography
- Utilities

### Fixed

#### Missing Imports (5 files)
Added missing `import uvicorn` to IMS files:
- pcscf.py
- icscf.py
- scscf.py
- mrf.py
- ims_hss.py

#### Dependency Conflicts
Resolved OpenTelemetry package version conflicts by removing overly strict version pins.

---

## [1.0.0] - Initial Release

### Features
- 5G Core Network Functions (NRF, AMF, SMF, UPF, AUSF, UDM, UDR, PCF, NSSF, BSF, SCP, CHF, NEF, SEPP, N3IWF)
- 4G/LTE EPC Functions (MME, SGW, PGW, HSS)
- IMS Core Functions (P-CSCF, I-CSCF, S-CSCF, MRF, IMS-HSS)
- RAN Components (gNB, CU, DU, RRU)
- Service Assurance (metrics, KQI, anomaly detection)
- OpenTelemetry instrumentation
- Prometheus metrics export
