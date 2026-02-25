# RAN Intelligent Controller (RIC) Architecture

## Overview

This document describes the O-RAN RAN Intelligent Controller (RIC) implementation in the 5G Network Simulator. The RIC components enable intelligent RAN control and optimization through standardized E2 and A1 interfaces.

## O-RAN Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │                 Service Management &                 │
                        │                   Orchestration (SMO)                │
                        │                                                      │
                        │  ┌───────────┐   ┌───────────┐   ┌───────────┐      │
                        │  │    ZSM    │   │   VNFM    │   │   RNIS    │      │
                        │  │  :8094    │   │  :8093    │   │  :8092    │      │
                        │  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘      │
                        └────────┼───────────────┼───────────────┼────────────┘
                                 │       O1      │               │
                                 └───────┬───────┘               │
                                         │                       │
                        ┌────────────────▼───────────────────────▼────────────┐
                        │                   Non-RT RIC                        │
                        │                    Port 8096                        │
                        │                                                     │
                        │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
                        │  │   A1-P      │  │   A1-EI     │  │   rApps     │  │
                        │  │  (Policy)   │  │ (Enrichment)│  │  Framework  │  │
                        │  └─────────────┘  └─────────────┘  └─────────────┘  │
                        └─────────────────────────┬───────────────────────────┘
                                                  │
                                                 A1
                                                  │
                        ┌─────────────────────────▼───────────────────────────┐
                        │                    Near-RT RIC                      │
                        │                     Port 8095                       │
                        │                                                     │
                        │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
                        │  │    E2AP     │  │    xApps    │  │Subscription │  │
                        │  │  Interface  │  │  Framework  │  │  Manager    │  │
                        │  └─────────────┘  └─────────────┘  └─────────────┘  │
                        └─────────────────────────┬───────────────────────────┘
                                                  │
                                                 E2
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
             ┌──────▼──────┐              ┌───────▼──────┐              ┌───────▼──────┐
             │     CU      │              │     DU       │              │     gNB      │
             │   :38472    │              │   :38473     │              │   :38412     │
             │   E2 Agent  │              │   E2 Agent   │              │   E2 Agent   │
             └─────────────┘              └──────────────┘              └──────────────┘
```

## RIC Components

### Near-RT RIC (Near-Real-Time RIC)

**File Location:** `5G_Emulator_API/ran/ric/near_rt_ric.py`
**Port:** 8095
**Control Loop Latency:** 10ms - 1 second
**Primary Interfaces:** E2 (to E2 Nodes), A1 receiver (from Non-RT RIC)

The Near-RT RIC provides near-real-time control of the RAN via the E2 interface. It hosts xApps that perform intelligent RAN optimization functions.

#### Key Features

| Feature | Description | Specification |
|---------|-------------|---------------|
| E2 Node Management | Register and manage E2 nodes (CU, DU, gNB) | ETSI TS 104038 |
| RIC Subscription | Subscribe to RAN events and measurements | ETSI TS 104039 |
| RIC Indication | Receive reports and decision points from E2 nodes | ETSI TS 104039 |
| RIC Control | Send control commands to E2 nodes | ETSI TS 104039 |
| xApp Hosting | Host and manage xApp instances | O-RAN WG3 |
| A1 Policy Receiver | Receive policies from Non-RT RIC | ETSI TS 103983 |

#### E2 Service Models Supported

| Service Model | OID | Actions | Use Case |
|---------------|-----|---------|----------|
| E2SM-KPM | 1.3.6.1.4.1.53148.1.2.2.2 | REPORT | Cell/UE/Bearer level measurements |
| E2SM-RC | 1.3.6.1.4.1.53148.1.2.2.3 | REPORT, INSERT, CONTROL, POLICY | RAN configuration and control |

#### RIC Services

| Service | Description | Direction |
|---------|-------------|-----------|
| REPORT | E2 Node sends measurement reports to RIC | E2 Node → RIC |
| INSERT | E2 Node sends decision point to RIC for action | E2 Node → RIC |
| CONTROL | RIC sends control commands to E2 Node | RIC → E2 Node |
| POLICY | RIC configures autonomous policies in E2 Node | RIC → E2 Node |
| QUERY | RIC queries E2 Node for specific information | RIC ↔ E2 Node |

---

### Non-RT RIC (Non-Real-Time RIC)

**File Location:** `5G_Emulator_API/ran/ric/non_rt_ric.py`
**Port:** 8096
**Control Loop Latency:** > 1 second
**Primary Interfaces:** A1 (to Near-RT RIC), O1 (to SMO)

The Non-RT RIC provides non-real-time control and optimization functions. It hosts rApps that analyze RAN data and create policies for the Near-RT RIC.

#### Key Features

| Feature | Description | Specification |
|---------|-------------|---------------|
| A1 Policy Management | Create and manage A1 policies | ETSI TS 103983 |
| Enrichment Information | Deliver enrichment data to Near-RT RIC | ETSI TS 103983 |
| rApp Hosting | Host and manage rApp instances | O-RAN WG2 |
| O1 Integration | Interface with ZSM and VNFM | ETSI TS 104043 |
| Analytics | Collect and analyze RAN data | O-RAN WG2 |

#### Default Policy Types

| Policy Type | Description |
|-------------|-------------|
| ORAN_QoSTarget_1.0.0 | QoS optimization policies |
| ORAN_TrafficSteering_1.0.0 | Traffic steering and load balancing |
| ORAN_LoadBalancing_1.0.0 | Cell load balancing policies |

---

## E2 Interface

The E2 interface connects the Near-RT RIC to E2 Nodes (CU, DU, gNB). It is defined by ETSI TS 104039 (E2AP).

### E2AP Procedures

**File Location:** `5G_Emulator_API/ran/ric/e2ap.py`

| Procedure | Initiator | Description |
|-----------|-----------|-------------|
| E2 Setup | E2 Node | E2 Node connects and registers with Near-RT RIC |
| RIC Subscription | Near-RT RIC | Subscribe to events/measurements from E2 Node |
| RIC Subscription Delete | Near-RT RIC | Remove existing subscription |
| RIC Indication | E2 Node | E2 Node sends report or decision point |
| RIC Control | Near-RT RIC | Send control command to E2 Node |
| RIC Query | Near-RT RIC | Query E2 Node for information |
| E2 Node Configuration Update | E2 Node | Update E2 Node configuration |
| Error Indication | Either | Report error condition |

### E2 Node Types

| Type | Description |
|------|-------------|
| gNB | Integrated gNodeB |
| gNB-CU | gNodeB Central Unit |
| gNB-DU | gNodeB Distributed Unit |
| gNB-CU-CP | gNodeB Central Unit Control Plane |
| gNB-CU-UP | gNodeB Central Unit User Plane |
| en-gNB | EN-DC gNodeB |
| ng-eNB | NG-RAN eNodeB |

---

## A1 Interface

The A1 interface connects the Non-RT RIC to the Near-RT RIC. It is defined by ETSI TS 103983.

### A1 Services

**File Location:** `5G_Emulator_API/ran/ric/a1_interface.py`

| Service | Description |
|---------|-------------|
| A1-P (Policy) | Policy management - create, update, delete policies |
| A1-EI (Enrichment Information) | Deliver enrichment data to Near-RT RIC |
| A1-ML (ML Models) | Deploy ML models (future) |

### Policy Lifecycle

```
Non-RT RIC                                Near-RT RIC
     │                                          │
     │  ──── PUT /a1/policies/{id} ────────►   │  Policy Create
     │                                          │
     │  ◄─── 201 Created ──────────────────    │
     │                                          │
     │  ──── GET /a1/policies/{id}/status ──►  │  Policy Status
     │                                          │
     │  ◄─── {"enforceStatus": "ENFORCED"} ──  │
     │                                          │
     │  ──── DELETE /a1/policies/{id} ─────►   │  Policy Delete
     │                                          │
     │  ◄─── 204 No Content ────────────────   │
```

---

## xApp Framework

xApps are applications that run on the Near-RT RIC and implement near-real-time RAN intelligence.

### xApp SDK

**File Location:** `5G_Emulator_API/ran/ric/xapp_sdk.py`

```python
from ran.ric.xapp_sdk import XAppBase

class MyXApp(XAppBase):
    async def on_indication(self, indication):
        # Process RIC Indication
        if indication.get("load") > 80:
            await self.send_control(
                e2_node_id=indication["e2NodeId"],
                control_header={"action": "load_balance"},
                control_message={"targetLoad": 60}
            )

xapp = MyXApp("my-xapp", "http://127.0.0.1:8095")
await xapp.start()
```

### Example xApps

| xApp | Description |
|------|-------------|
| LoadBalancingXApp | Monitor cell load, trigger handovers |
| QoSOptimizationXApp | Adjust scheduler based on QoS requirements |

---

## rApp Framework

rApps are applications that run on the Non-RT RIC and implement non-real-time RAN intelligence.

### rApp SDK

**File Location:** `5G_Emulator_API/ran/ric/rapp_sdk.py`

```python
from ran.ric.rapp_sdk import RAppBase

class MyRApp(RAppBase):
    async def on_analytics(self, analytics):
        # Analyze RAN data and create policies
        if analytics.get("avgLoad") > 70:
            await self.create_policy(
                policy_type_id="ORAN_LoadBalancing_1.0.0",
                policy_data={"targetLoad": 50}
            )

rapp = MyRApp("my-rapp", "http://127.0.0.1:8096")
await rapp.start()
```

### Example rApps

| rApp | Description |
|------|-------------|
| TrafficSteeringRApp | Analyze traffic and create steering policies |
| EnergyEfficiencyRApp | Optimize energy consumption during low-load periods |

---

## E2 Agent Integration

CU and DU components include E2 agents for RIC connectivity.

### CU E2 Agent

**File Location:** `5G_Emulator_API/ran/cu.py` (E2 agent section)

Supported RAN Functions:
- E2SM-KPM-CU: PDCP throughput, RLC retransmissions, RRC state
- E2SM-RC-CU: Handover control, bearer management

### DU E2 Agent

**File Location:** `5G_Emulator_API/ran/du.py` (E2 agent section)

Supported RAN Functions:
- E2SM-KPM-DU: PRB utilization, CQI, MCS, HARQ
- E2SM-RC-DU: Scheduler control, power management

---

## O1 Integration

The Non-RT RIC integrates with SMO components via the O1 interface.

### ZSM Integration

```python
# Create ZSM intent for closed-loop automation
await non_rt_ric.create_zsm_intent(
    intent_type="SCALING",
    target="amf",
    objective="Scale AMF to handle increased load"
)
```

### VNFM Integration

```python
# Request VNF scaling
await non_rt_ric.request_vnf_scaling(
    vnf_id="amf-001",
    scale_type="SCALE_OUT",
    aspect_id="processing"
)
```

---

## Port Configuration

| Component | Port | Protocol |
|-----------|------|----------|
| Near-RT RIC | 8095 | HTTP/REST |
| Non-RT RIC | 8096 | HTTP/REST |
| CU (E2 Agent) | 38472 | HTTP/REST |
| DU (E2 Agent) | 38473 | HTTP/REST |
| gNB (E2 Agent) | 38412 | HTTP/REST |
| RNIS | 8092 | HTTP/REST |
| ZSM | 8094 | HTTP/REST |
| VNFM | 8093 | HTTP/REST |

---

## Specification Compliance

### Primary Specifications

| Specification | Coverage |
|---------------|----------|
| ETSI TS 104038 | E2 General Aspects and Principles |
| ETSI TS 104039 | E2 Application Protocol (E2AP) |
| ETSI TS 104040 | E2 Service Model (E2SM) |
| ETSI TS 103983 | A1 Interface |
| ETSI TS 104043 | O1 Interface |
| O-RAN WG3 | xApp framework guidelines |
| O-RAN WG2 | Non-RT RIC use cases |

### Timing Requirements

| Requirement | Value | Source |
|-------------|-------|--------|
| Near-RT control loop | 10ms - 1s | ETSI TS 104038 |
| E2AP response timeout | 1s (configurable) | Implementation |
| Indication delivery | <100ms typical | Implementation |
| Non-RT control loop | >1s | ETSI TS 103983 |

---

## Testing

### Start RIC Components

```bash
# Start Near-RT RIC
python -m ran.ric.near_rt_ric

# Start Non-RT RIC
python -m ran.ric.non_rt_ric
```

### Verify E2 Setup

```bash
# Check connected E2 Nodes
curl http://localhost:8095/ric/e2-nodes

# Check active subscriptions
curl http://localhost:8095/ric/subscriptions

# Check registered xApps
curl http://localhost:8095/ric/xapps
```

### Verify A1 Policies

```bash
# List policy types
curl http://localhost:8096/a1-p/policytypes

# Create a policy
curl -X PUT http://localhost:8096/a1-p/policytypes/ORAN_QoSTarget_1.0.0/policies/policy-001 \
  -H "Content-Type: application/json" \
  -d '{"qosObjective": "maximize_throughput"}'

# Check policy status
curl http://localhost:8096/a1-p/policytypes/ORAN_QoSTarget_1.0.0/policies/policy-001/status
```

---

## Files Reference

| File | Description | Lines |
|------|-------------|-------|
| `ran/ric/__init__.py` | Package initialization | ~20 |
| `ran/ric/near_rt_ric.py` | Near-RT RIC core | ~700 |
| `ran/ric/non_rt_ric.py` | Non-RT RIC core | ~600 |
| `ran/ric/e2ap.py` | E2AP protocol messages | ~500 |
| `ran/ric/a1_interface.py` | A1 client/server | ~300 |
| `ran/ric/xapp_sdk.py` | xApp development framework | ~400 |
| `ran/ric/rapp_sdk.py` | rApp development framework | ~350 |
| `ran/ric/near_rt_ric.py.spec.txt` | Specification mapping | ~280 |

---

## Conclusion

The RIC implementation provides comprehensive O-RAN RAN Intelligent Controller functionality with:

- **Near-RT RIC** for near-real-time control via E2 interface
- **Non-RT RIC** for non-real-time policy management via A1 interface
- **xApp/rApp SDKs** for extensible RAN intelligence applications
- **E2 agents** integrated into CU and DU components
- **O1 integration** with existing SMO components (ZSM, VNFM)
- **Full specification compliance** with ETSI O-RAN standards

This enables advanced RAN optimization use cases including:
- Traffic steering and load balancing
- QoS optimization
- Energy efficiency
- Interference management
- Mobility optimization
