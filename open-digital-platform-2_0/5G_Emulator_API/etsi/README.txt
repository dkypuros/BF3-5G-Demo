================================================================================
                        ETSI STANDARDS INTEGRATION
================================================================================

This directory contains implementations of ETSI (European Telecommunications
Standards Institute) specifications that complement the 3GPP 5G Core network
emulation.


================================================================================
OVERVIEW
================================================================================

While 3GPP defines the core 5G network functions and protocols, ETSI provides
standards for:

  - NFV (Network Functions Virtualisation): Lifecycle management of virtualized
    network functions

  - MEC (Multi-access Edge Computing): Edge computing services and APIs

  - ZSM (Zero-touch Service Management): Automated network management


================================================================================
ARCHITECTURE INTEGRATION
================================================================================

+-------------------------------------------------------------------------+
|                     ZSM - Zero-touch Management                         |
|                   (Closed Loop Automation Layer)                        |
+-------------------------------------------------------------------------+
|                        NFV MANO Layer                                   |
|          +---------+              +---------+                           |
|          |  NFVO   |--------------+  VNFM   |                           |
|          +----+----+              +----+----+                           |
+---------------+---------------------------+-----------------------------+
|               |                           |                             |
|   +-----------+----------+----------------+----------+                  |
|   |                      |                           |                  |
| +-+-+   +-----+   +------+-+   +-----+   +------+    |   3GPP 5G Core  |
| |AMF|   | SMF |   | UPF    |   | NRF |   | PCF  |    |                  |
| +-+-+   +--+--+   +---+----+   +-----+   +------+    |                  |
|   |        |          |                              |                  |
+---+--------+----------+------------------------------+------------------+
|   |        |          |                                                 |
|   |        |     +----+-----+        MEC Platform                       |
|   |        |     |   N6     |   +---------------------------+           |
|   |        |     | Interface|   |  MEC Platform Manager     |           |
|   |        |     |  (DPU)   |   |  +---------------------+  |           |
|   |        |     +----+-----+   |  | Location API        |  |           |
|   |        |          |         |  | RNIS API            |  |           |
|   |        |          |         |  | Traffic Rules       |  |           |
|   |        |          |         |  +---------------------+  |           |
|   |        |          |         +---------------------------+           |
+---+--------+----------+-------------------------------------------------+


================================================================================
DIRECTORY STRUCTURE
================================================================================

etsi/
    README.txt                      This file
    __init__.py

    mec/                            Multi-access Edge Computing
        __init__.py
        mec_platform.py             MEC Platform Manager (GS MEC 011)
        mec_platform.py.spec.txt    Spec reference map
        location_api.py             Location Service (GS MEC 013)
        location_api.py.spec.txt
        rnis_api.py                 Radio Network Info Service (GS MEC 012)
        openapi/                    Symlinks to OpenAPI specs
            gs011-app-enablement-api -> ETSI/forge/...
            gs012-rnis-api -> ETSI/forge/...
            gs013-location-api -> ETSI/forge/...

    nfv/                            Network Functions Virtualisation
        __init__.py
        vnfm.py                     VNF Manager (GS NFV-SOL 003)
        descriptors/                Symlinks to NFV specs
            SOL001 -> ETSI/forge/... (VNFD/TOSCA)
            SOL002-SOL003 -> ...     (VNF LCM APIs)
            SOL005 -> ...            (NFVO APIs)
            SOL006 -> ...            (YANG models)

    zsm/                            Zero-touch Service Management
        __init__.py
        closed_loop.py              Closed Loop Automation (GS ZSM 002)


================================================================================
COMPONENTS
================================================================================


MEC - MULTI-ACCESS EDGE COMPUTING
--------------------------------------------------------------------------------

MEC Platform Manager (mec_platform.py)
  Spec:     ETSI GS MEC 003, GS MEC 011
  Port:     8090
  Purpose:  Service registry and lifecycle management for MEC applications

  Key Features:
    - Service registration and discovery (Mp1 interface)
    - DNS rule management
    - Traffic rule management (integrates with UPF/N6)
    - Application lifecycle management

  Start Command:
    python etsi/mec/mec_platform.py

  API Docs:
    http://localhost:8090/mp1/docs


Location API (location_api.py)
  Spec:     ETSI GS MEC 013
  Port:     8091
  Purpose:  Provides UE location information to MEC applications
  Integration: Queries AMF for UE context and location

  Start Command:
    python etsi/mec/location_api.py

  API Docs:
    http://localhost:8091/location/docs


RNIS API (rnis_api.py)
  Spec:     ETSI GS MEC 012
  Port:     8092
  Purpose:  Provides radio network information (RSRP, RSRQ, cell info)
  Integration: Queries gNodeB/CU for radio measurements

  Start Command:
    python etsi/mec/rnis_api.py

  API Docs:
    http://localhost:8092/rni/docs


NFV - NETWORK FUNCTIONS VIRTUALISATION
--------------------------------------------------------------------------------

VNFM (vnfm.py)
  Spec:     ETSI GS NFV-SOL 003
  Port:     8093
  Purpose:  VNF lifecycle management (instantiate, scale, terminate)

  Key Features:
    - VNF instance creation
    - LCM operations (instantiate, terminate, scale)
    - Operation occurrence tracking
    - Pre-configured VNFDs for 5G Core NFs

  Start Command:
    python etsi/nfv/vnfm.py

  API Docs:
    http://localhost:8093/vnflcm/docs


Example: Instantiate AMF as VNF

  Step 1 - Create VNF instance:
    curl -X POST http://localhost:8093/vnflcm/v1/vnf_instances \
      -H "Content-Type: application/json" \
      -d '{"vnfdId": "vnfd-amf-v1", "vnfInstanceName": "amf-001"}'

  Step 2 - Instantiate:
    curl -X POST http://localhost:8093/vnflcm/v1/vnf_instances/{id}/instantiate \
      -H "Content-Type: application/json" \
      -d '{"flavourId": "default"}'


ZSM - ZERO-TOUCH SERVICE MANAGEMENT
--------------------------------------------------------------------------------

Closed Loop Engine (closed_loop.py)
  Spec:     ETSI GS ZSM 002
  Port:     8094
  Purpose:  Automated observe-analyze-act loop for network management

  Key Features:
    - Intent-based management
    - Prometheus integration for metrics
    - Automated remediation actions
    - VNFM integration for scaling

  Start Command:
    python etsi/zsm/closed_loop.py

  API Docs:
    http://localhost:8094/zsm/docs


Example: Create Scaling Intent

  Step 1 - Create intent:
    curl -X POST http://localhost:8094/zsm/v1/intents \
      -H "Content-Type: application/json" \
      -d '{
        "intentId": "intent-001",
        "intentType": "SCALING",
        "target": "upf",
        "objective": "Maintain UPF throughput",
        "constraints": {"max_value": 80},
        "priority": 3
      }'

  Step 2 - Create closed loop:
    curl -X POST http://localhost:8094/zsm/v1/closed-loops \
      -H "Content-Type: application/json" \
      -d '{
        "loopId": "loop-001",
        "name": "UPF Auto-Scaler",
        "intents": ["intent-001"],
        "observeInterval": 30
      }'

  Step 3 - Start closed loop:
    curl -X POST http://localhost:8094/zsm/v1/closed-loops/loop-001/start


================================================================================
INTEGRATION POINTS
================================================================================

MEC Platform <-> UPF (N6 Interface)
  The MEC Platform integrates with the UPF-Enhanced at the N6 interface.
  Traffic rules define which packets are steered to MEC applications. The
  BlueField-3 DPU can accelerate this traffic steering at line rate.

Location API <-> AMF
  Location Service queries the AMF for UE context including current serving
  cell, UE geographic location (if available), and mobility events.

RNIS API <-> gNodeB/CU
  RNIS queries the RAN for radio measurements including RSRP/RSRQ values,
  cell load information, and connected UE count.

VNFM <-> 5G Core NFs
  VNFM manages lifecycle of 5G Core NFs as VNFs. Each NF (AMF, SMF, UPF,
  etc.) can be instantiated as a VNF. Supports scaling, healing, and
  termination.

ZSM <-> All Components
  ZSM provides closed-loop automation. It observes metrics from
  Prometheus/OpenTelemetry, analyzes against intent objectives, and acts
  via VNFM for scaling/healing.


================================================================================
OPENAPI SPECIFICATIONS
================================================================================

The openapi/ and descriptors/ directories contain symlinks to the machine-
readable specifications downloaded from ETSI Forge (forge.etsi.org):

  Spec          Location                              Contents
  ------------  ------------------------------------  ------------------------
  MEC 011       mec/openapi/gs011-app-enablement-api  App Support & Service Mgmt
  MEC 012       mec/openapi/gs012-rnis-api            Radio Network Info API
  MEC 013       mec/openapi/gs013-location-api        Location API
  SOL001        nfv/descriptors/SOL001                VNFD TOSCA models
  SOL002-003    nfv/descriptors/SOL002-SOL003         VNF LCM APIs
  SOL005        nfv/descriptors/SOL005                NFVO APIs
  SOL006        nfv/descriptors/SOL006                NFV YANG models


================================================================================
QUICK START
================================================================================

Start all ETSI components:

  cd 5G_Emulator_API

  # Terminal 1: MEC Platform
  python etsi/mec/mec_platform.py &

  # Terminal 2: Location API
  python etsi/mec/location_api.py &

  # Terminal 3: RNIS API
  python etsi/mec/rnis_api.py &

  # Terminal 4: VNFM
  python etsi/nfv/vnfm.py &

  # Terminal 5: ZSM
  python etsi/zsm/closed_loop.py &


================================================================================
SPECIFICATION REFERENCES
================================================================================

  Spec            Title                                   PDF Location
  --------------  --------------------------------------  ---------------------
  GS MEC 003      MEC Framework and Reference Arch        ETSI/MEC/gs_mec003*.pdf
  GS MEC 011      MEC Platform Application Enablement     ETSI/MEC/gs_MEC011*.pdf
  GS MEC 012      Radio Network Information API           ETSI/MEC/gs_MEC012*.pdf
  GS MEC 013      Location API                            ETSI/MEC/gs_MEC013*.pdf
  GS NFV-SOL 003  VNF LCM Interface                       ETSI/NFV/*.pdf
  GS ZSM 002      ZSM Reference Architecture              ETSI/ZSM/*.pdf


================================================================================
VERSION INFORMATION
================================================================================

  MEC APIs:           v2.2.1 (aligned with MEC Phase 3)
  NFV SOL APIs:       v1.4.1
  Implementation:     January 2026


================================================================================
                              END OF DOCUMENT
================================================================================
