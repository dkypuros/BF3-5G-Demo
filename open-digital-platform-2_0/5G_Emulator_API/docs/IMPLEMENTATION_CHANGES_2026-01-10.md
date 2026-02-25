# Protocol Realism Implementation - Changes Documentation

**Date:** 2026-01-10
**Version:** 2.0.0
**Author:** Protocol Implementation Sprint

---

## Overview

This document describes the implementation of real 3GPP protocol support added to the 5G Emulator API. The goal was to move from HTTP/JSON-only emulation to real binary protocol implementations.

## New Components Created

### 1. NAS Protocol (protocols/nas/nas_5g.py)
**Lines of Code:** ~1,200
**3GPP Reference:** TS 24.501

#### What It Does
Implements binary encoding/decoding for 5G Non-Access Stratum messages between UE and AMF.

#### Key Classes
| Class | Purpose |
|-------|---------|
| `NASCodec` | Main codec for encoding/decoding NAS messages |
| `RegistrationRequest` | UE initial/mobility registration |
| `RegistrationAccept` | Network acceptance with GUTI allocation |
| `AuthenticationRequest` | Network authentication challenge |
| `AuthenticationResponse` | UE authentication response (RES*) |
| `SecurityModeCommand` | NAS security activation |
| `PDUSessionEstablishmentRequest` | PDU session setup (UE) |
| `PDUSessionEstablishmentAccept` | PDU session acceptance (SMF) |

#### Data Structures
| Class | Purpose |
|-------|---------|
| `PLMN` | MCC/MNC encoding per TS 24.008 |
| `SNSSAI` | Network slice identifier (SST + SD) |
| `GUTI5G` | 5G Globally Unique Temporary ID |
| `SUCI` | Subscription Concealed Identifier |
| `UESecurityCapability` | UE crypto capabilities |

#### Protocol Constants
- All 5GMM message types (0x41-0x68)
- All 5GSM message types (0xC1-0xD6)
- Registration types (initial, mobility, periodic, emergency)
- Cause values (illegal UE, services not allowed, etc.)
- Information Element identifiers

#### Example Usage
```python
from protocols import NASCodec, PLMN, SNSSAI, RegistrationType

codec = NASCodec()

# Create Registration Request
reg_req = codec.encode_registration_request(
    registration_type=RegistrationType.INITIAL,
    supi="imsi-001010000000001",
    plmn=PLMN("001", "01"),
    requested_nssai=[SNSSAI(sst=1, sd=bytes.fromhex("010203"))]
)
# Returns: bytes (binary NAS message)

# Decode any NAS message
decoded = codec.decode_message(reg_req)
# Returns: {'message_type': 65, 'message_type_name': 'REGISTRATION_REQUEST', ...}
```

---

### 2. 5G-AKA Authentication (protocols/crypto/aka_5g.py)
**Lines of Code:** ~800
**3GPP Reference:** TS 33.501, TS 35.206

#### What It Does
Implements complete 5G Authentication and Key Agreement with real Milenage cryptography.

#### Key Classes
| Class | Purpose |
|-------|---------|
| `Milenage` | AES-based auth algorithm (TS 35.206) |
| `KeyDerivation5G` | Key derivation functions per TS 33.501 |
| `AKA5GHandler` | Complete 5G-AKA flow management |
| `SubscriberData` | Subscriber authentication data (K, OPc, SQN) |
| `SUCIHandler` | SUPI↔SUCI conversion |

#### Milenage Functions Implemented
- `f1` - Network authentication (MAC-A)
- `f1*` - Re-synchronization (MAC-S)
- `f2` - User authentication (RES)
- `f3` - Cipher key derivation (CK)
- `f4` - Integrity key derivation (IK)
- `f5` - Anonymity key (AK)
- `f5*` - Re-sync anonymity key

#### Key Derivation Functions
- `derive_ck_ik_prime()` - CK'/IK' from CK/IK
- `derive_kausf()` - KAUSF from CK'/IK'
- `derive_kseaf()` - KSEAF from KAUSF
- `derive_kamf()` - KAMF from KSEAF
- `derive_kgnb()` - KgNB from KAMF
- `derive_nas_keys()` - KNASenc/KNASint from KAMF
- `compute_res_star()` - RES* computation
- `compute_hres_star()` - HRES* for AUSF verification

#### Example Usage
```python
from protocols import AKA5GHandler, SubscriberData

# Subscriber data (from UDM/ARPF)
subscriber = SubscriberData(
    supi="imsi-001010000000001",
    k=bytes.fromhex("465B5CE8B199B49FAA5F0A2EE238A6BC"),
    opc=bytes.fromhex("E8ED289DEBA952E4283B54E88E6183CA"),
    sqn=bytes.fromhex("000000000001"),
    amf=bytes.fromhex("8000")
)

handler = AKA5GHandler()

# Network: Generate authentication vector
av = handler.generate_auth_vector(subscriber)
# av.rand, av.autn, av.xres_star, av.hxres_star, av.kausf, av.kseaf

# UE: Compute response
res_star, ck, ik = handler.ue_compute_response(
    subscriber.k, subscriber.opc, av.rand, av.autn
)

# Network: Verify
success = handler.verify_auth_response(av, res_star)
```

---

### 3. PFCP Protocol (protocols/pfcp/pfcp.py)
**Lines of Code:** ~1,400
**3GPP Reference:** TS 29.244

#### What It Does
Implements binary PFCP messages for SMF↔UPF communication (N4 interface).

#### Key Classes
| Class | Purpose |
|-------|---------|
| `PFCPCodec` | Message encoding/decoding |
| `PFCPNode` | SMF/UPF node abstraction |
| `PFCPTransport` | Async UDP transport |
| `SessionEstablishmentRequest` | PDU session setup on UPF |
| `SessionEstablishmentResponse` | UPF confirmation |
| `CreatePDR` | Packet Detection Rule |
| `CreateFAR` | Forwarding Action Rule |
| `CreateQER` | QoS Enforcement Rule |

#### Information Elements
| IE | Purpose |
|----|---------|
| `NodeId` | SMF/UPF identity (IPv4/IPv6/FQDN) |
| `FSEID` | Fully-qualified SEID |
| `FTEID` | Fully-qualified TEID |
| `PDI` | Packet Detection Information |
| `ForwardingParameters` | Forwarding config |
| `OuterHeaderCreation` | GTP-U encapsulation |
| `MBR/GBR` | Bit rate limits |
| `GateStatus` | UL/DL gate control |

#### Message Types
- Node: Heartbeat, Association Setup/Update/Release
- Session: Establishment, Modification, Deletion, Report

#### Example Usage
```python
from protocols import PFCPNode

# Create SMF node
smf = PFCPNode("SMF", "10.0.0.1")

# Create session establishment request
session_req = smf.create_session_establishment_request(
    seid=0x123456789ABCDEF0,
    ue_ip="10.45.0.1",
    gnb_teid=0x12345678,
    gnb_ip="192.168.1.100",
    upf_teid=0x87654321,
    qfi=1
)
# Returns: bytes (PFCP binary message)
```

---

### 4. SCTP Transport (protocols/sctp/sctp_transport.py)
**Lines of Code:** ~800
**3GPP Reference:** TS 38.412

#### What It Does
Provides SCTP transport for NGAP signaling with multi-homing support.

#### Key Classes
| Class | Purpose |
|-------|---------|
| `SCTPTransportBase` | Abstract base class |
| `RealSCTPTransport` | Native SCTP (pysctp) |
| `TCPFallbackTransport` | TCP fallback with framing |
| `NGAPSCTPHandler` | NGAP-specific handler |
| `SCTPMessage` | Message with stream metadata |
| `SCTPAssociation` | Association state |

#### Features
- **Multi-homing**: Multiple IP addresses per endpoint
- **Multi-streaming**: Stream 0 for non-UE, Stream 1+ for UE-associated
- **Message boundaries**: Preserved (unlike TCP)
- **Automatic fallback**: Uses TCP if SCTP unavailable

#### Stream Allocation (per TS 38.412)
| Stream | Usage |
|--------|-------|
| 0 | Non-UE-associated signaling (NG Setup, Reset) |
| 1+ | UE-associated signaling (per UE) |

#### Example Usage
```python
from protocols import NGAPSCTPHandler

# Create AMF handler
amf = NGAPSCTPHandler("AMF", ["10.0.0.1", "10.0.0.2"], 38412)

# Start listening
await amf.start()

# Handle incoming message
def message_handler(data: bytes, stream_id: int):
    if stream_id == 0:
        # Non-UE message (NG Setup, etc.)
        pass
    else:
        # UE-associated message
        pass

amf.set_message_handler(message_handler)

# gNB connects
gnb = NGAPSCTPHandler("gNB", ["192.168.1.1"], 0)
await gnb.start()
await gnb.connect(["10.0.0.1"], 38412)

# Send NG Setup Request
await gnb.send_non_ue_message(ng_setup_bytes)
```

---

## Directory Structure

```
protocols/
├── __init__.py          # Package exports
├── nas/
│   ├── __init__.py
│   └── nas_5g.py        # NAS protocol (1,200 lines)
├── crypto/
│   ├── __init__.py
│   └── aka_5g.py        # 5G-AKA auth (800 lines)
├── pfcp/
│   ├── __init__.py
│   └── pfcp.py          # PFCP protocol (1,400 lines)
└── sctp/
    ├── __init__.py
    └── sctp_transport.py # SCTP transport (800 lines)
```

**Total new code:** ~4,200 lines

---

## Integration Points

### AMF Integration
```python
# In core_network/amf.py
from protocols import NASCodec, AKA5GHandler

# Use NAS codec for real message encoding
codec = NASCodec()
auth_req = codec.encode_authentication_request(ngksi, abba, rand, autn)

# Use AKA handler for authentication
aka = AKA5GHandler()
av = aka.generate_auth_vector(subscriber)
```

### SMF Integration
```python
# In core_network/smf.py
from protocols import PFCPNode

# Create PFCP node for N4 communication
pfcp = PFCPNode("SMF", smf_ip)
session_req = pfcp.create_session_establishment_request(...)
```

### gNB Integration
```python
# In ran/gnb.py
from protocols import NGAPSCTPHandler

# Use SCTP for NGAP signaling
ngap = NGAPSCTPHandler("gNB", [gnb_ip])
await ngap.connect([amf_ip], 38412)
await ngap.send_non_ue_message(ng_setup_request)
```

---

## Verification Results

### NAS Protocol Test
```
=== Registration Request ===
Encoded: 7e004171121000f11000000000303030303030303030312f050401010203

=== Authentication Request ===
Encoded: 7e0056000200002100000000000000000000000000000000201000...

=== PDU Session Establishment Accept ===
Encoded: 2e01c20111000a0100062111100101ff01060603e80603e82905010a0000012508696e7465726e6574
```

### 5G-AKA Test
```
[Network] Generating authentication vector...
  RAND: 60352178a7a269f8a6b990c2eef8c0ce
  AUTN: b0a54478b36a8000c75dbc47798b7058
  XRES*: a4f38d54dd41f645167a1a70b50aae44

[UE] Computing authentication response...
  RES*: a4f38d54dd41f645167a1a70b50aae44

[Network] Verifying authentication response...
  Authentication SUCCESSFUL!

[UE] Deriving session keys...
  KAUSF: 31649429e3ca57b99025cadaadd47673c1800b5740dd3b9b27ab99a065e5a166
  KSEAF: 3e04ddc8c16d07ec64d17e01b273e51224b34b21b011440e563a2bd36f769f4c
  KAMF: 70229b4cd033c3448c524ea8a6ca0c1c82d1a61d29c10894ddd01a9326c98e28
```

### PFCP Test
```
=== Association Setup Request ===
Encoded (33 bytes): 20050017000000010000003c0005000a00000100600004ed0d0d25005900020001

=== Session Establishment Request ===
Encoded (267 bytes): 213200f9123456789abcdef0000000030000003c0005000a0000010039000d02...
```

---

## What's Next

### Immediate Integration (Low Effort)
1. Replace JSON NAS messages in AMF with binary encoding
2. Add PFCP to SMF↔UPF communication
3. Use real 5G-AKA in AUSF

### Future Enhancements
1. Add ASN.1 PER encoding for NGAP (requires pycrate)
2. Implement full UE state machines
3. Add GTP-U user plane handling
4. Support ECIES for SUCI encryption

---

## Dependencies Added

```
# requirements.txt additions
# Protocol Libraries (optional - for enhanced features)
# pysctp>=0.7.0      # Real SCTP support
# pycrate>=0.6.0     # ASN.1 PER encoding
```

---

## Compliance Improvement

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| NAS Messages | JSON | Binary (TS 24.501) | Real protocol |
| Authentication | SHA256 mock | Milenage (TS 35.206) | Cryptographically correct |
| Key Derivation | Simulated | TS 33.501 compliant | Real key hierarchy |
| PFCP | HTTP/REST | Binary/UDP (TS 29.244) | Real protocol |
| NGAP Transport | TCP | SCTP (TS 38.412) | Multi-homing support |
