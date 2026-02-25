# 5G Emulator API - State Snapshot (Before Protocol Realism Implementation)

**Date:** 2026-01-10
**Version:** 1.1.1
**Purpose:** Baseline documentation before implementing real protocol support

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Total Python files | 71 |
| Lines of code | 28,411 |
| Core Network Functions | 27 files |
| Largest NF | scscf.py (1,096 lines) |
| Project size | ~149 MB (including venv) |

## Architecture Overview

### Current Implementation Style

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT: HTTP/JSON over TCP                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   UE Simulator ──HTTP──▶ AMF ──HTTP──▶ SMF ──HTTP──▶ UPF       │
│        │                  │             │                       │
│        │                  ▼             ▼                       │
│        │               AUSF           PCF                       │
│        │                  │             │                       │
│        │                  ▼             │                       │
│        │                UDM ◀──────────┘                       │
│        │                  │                                     │
│        │                  ▼                                     │
│        └────────────▶   NRF (Discovery)                        │
│                                                                 │
│   Transport: TCP (HTTP/1.1)                                    │
│   Encoding:  JSON                                              │
│   Security:  None / Basic Auth                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Target Implementation Style (3GPP Compliant)

```
┌─────────────────────────────────────────────────────────────────┐
│                    TARGET: Real 3GPP Protocols                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   UE ──NAS/N1──▶ gNB ──NGAP/N2──▶ AMF ──HTTP/SBI──▶ SMF       │
│   │   (binary)    │    (ASN.1/     │                │          │
│   │               │     SCTP)      │                ▼          │
│   │               │                │            PFCP/N4        │
│   │               │                │            (binary/       │
│   │               │                │             UDP)          │
│   │               │                │                │          │
│   │               │                ▼                ▼          │
│   │               │             5G-AKA            UPF          │
│   │               │            (crypto)                        │
│   │               │                                            │
│   Transport: SCTP (signaling), UDP (user plane), TCP (SBI)    │
│   Encoding:  ASN.1 PER, Binary TLV, JSON (SBI only)           │
│   Security:  5G-AKA, SUCI encryption, NAS ciphering           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Current Protocol Implementation Analysis

### Protocols NOT Implemented (All Simulated via HTTP/JSON)

| Protocol | Interface | 3GPP Spec | Current State |
|----------|-----------|-----------|---------------|
| **NAS** | N1 (UE↔AMF) | TS 24.501 | JSON messages, no binary encoding |
| **NGAP** | N2 (gNB↔AMF) | TS 38.413 | HTTP endpoints, no ASN.1 |
| **PFCP** | N4 (SMF↔UPF) | TS 29.244 | HTTP REST, no binary protocol |
| **GTP-U** | N3 (gNB↔UPF) | TS 29.281 | Not implemented |
| **Diameter** | Various | TS 29.272 | HTTP simulation |
| **SIP** | IMS | TS 24.229 | Partial text parsing |

### Transport Layer

| Required | Current | Gap |
|----------|---------|-----|
| SCTP | Not used | Full implementation needed |
| UDP (PFCP) | Not used | Full implementation needed |
| TCP (SBI) | HTTP/JSON | Correct for SBI layer |

### Security Implementation

| Feature | 3GPP Spec | Current State |
|---------|-----------|---------------|
| **5G-AKA** | TS 33.501 | Simulated with SHA256 hashes |
| **SUPI→SUCI** | TS 33.501 | Not implemented |
| **NAS Security** | TS 24.501 | Not implemented |
| **IPsec (N3IWF)** | TS 33.501 | Stub implementation |

### What Currently Works

1. **Service Discovery (NRF)** - Real OAuth2 tokens, NF registration
2. **Health Endpoints** - All 27 NFs have `/health`
3. **Swagger/OpenAPI** - All NFs have `/docs`
4. **Basic Flow Simulation** - Registration, PDU session (JSON-based)
5. **Metrics** - Prometheus endpoints on most NFs
6. **IMS SIP Parsing** - Basic REGISTER/INVITE handling

### Crypto Libraries Present

```python
# Currently used for:
- hashlib.sha256()     # Token generation, simple hashing
- hmac                 # IPsec stub, SIP auth
- cryptography         # SEPP TLS, basic EC operations
```

## Network Function Inventory

### 5G Core (16 NFs)
| NF | File | Lines | Compliance |
|----|------|-------|------------|
| NRF | nrf.py | 722 | 85% (best) |
| AMF | amf.py | 882 | 75% |
| SMF | smf.py | 548 | 70% |
| UPF | upf.py | 428 | 40% |
| UPF Enhanced | upf_enhanced.py | 1,051 | 60% |
| AUSF | ausf.py | 506 | 65% |
| UDM | udm.py | 584 | 70% |
| UDR | udr.py | 457 | 65% |
| UDSF | udsf.py | 375 | 50% |
| PCF | pcf.py | 762 | 60% |
| NSSF | nssf.py | 882 | 70% |
| BSF | bsf.py | 512 | 55% |
| SCP | scp.py | 498 | 50% |
| CHF | chf.py | 766 | 55% |
| NEF | nef.py | 624 | 50% |
| SEPP | sepp.py | 716 | 60% |
| N3IWF | n3iwf.py | 736 | 45% |

### 4G EPC (4 NFs)
| NF | File | Lines | Compliance |
|----|------|-------|------------|
| MME | mme.py | 832 | 75% |
| SGW | sgw.py | 547 | 60% |
| PGW | pgw.py | 829 | 65% |
| HSS | hss.py | 786 | 80% |

### IMS Core (5 NFs)
| NF | File | Lines | Compliance |
|----|------|-------|------------|
| P-CSCF | pcscf.py | 648 | 60% |
| I-CSCF | icscf.py | 509 | 55% |
| S-CSCF | scscf.py | 1,096 | 65% |
| MRF | mrf.py | 863 | 50% |
| IMS-HSS | ims_hss.py | 893 | 60% |

### RAN (4 components)
| Component | File | Lines |
|-----------|------|-------|
| gNB | gnb.py | 687 |
| CU | cu.py | 312 |
| DU | du.py | 298 |
| RRU | rru.py | 276 |

## Test Coverage

- **Test files:** 2 (conftest.py, test_network_functions.py)
- **Total tests:** 205
- **Passing:** ~80% (some skip due to port conflicts)
- **Static analysis:** All 27 core files pass

## Dependencies (Current)

```
# Core
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0

# No protocol-specific libraries:
# - No pycrate (ASN.1)
# - No sctp (pysctp)
# - No dpkt/scapy (packet crafting)
```

## Key Gaps to Address

### Priority 1: NAS Protocol (N1)
- **Impact:** UE↔AMF communication
- **Work:** Binary message encoding per TS 24.501
- **Files affected:** amf.py, amf_nas.py, new nas_codec.py

### Priority 2: 5G-AKA Authentication
- **Impact:** Security anchor for entire system
- **Work:** Real Milenage/TUAK algorithms, key derivation
- **Files affected:** ausf.py, udm.py, new crypto_5g.py

### Priority 3: PFCP Protocol (N4)
- **Impact:** SMF↔UPF session management
- **Work:** Binary PFCP messages over UDP
- **Files affected:** smf.py, upf.py, new pfcp_codec.py

### Priority 4: SCTP Transport (N2)
- **Impact:** gNB↔AMF signaling reliability
- **Work:** SCTP socket handling, multi-homing
- **Files affected:** amf.py, gnb.py, new sctp_transport.py

---

## Summary

**Current Reality:** HTTP/JSON emulator simulating 3GPP message flows
**Target Reality:** Real protocol implementations with binary encoding, proper transport, and cryptographic security

**Estimated effort:** Significant (weeks of focused development)

**Next steps:** Begin implementing real NAS protocol encoding as the foundation for other improvements.
