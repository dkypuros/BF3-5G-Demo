# File location: 5G_Emulator_API/core_network/main.py
# 5G Core Network Function Launcher
# Starts all 5G core network functions as separate processes

import subprocess
import time
import sys
import os

# Configuration
STARTUP_DELAY = 0.5  # Seconds between NF launches
NRF_STARTUP_DELAY = 2.0  # Extra time for NRF to be ready

def start_nf(file_path, description=""):
    """Start a network function as a subprocess."""
    try:
        process = subprocess.Popen(
            ["python", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"[STARTED] {description or file_path} (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"[ERROR] Failed to start {file_path}: {e}")
        return None

def main():
    """Start all 5G core network functions in proper order."""
    print("=" * 60)
    print("5G Core Network Emulator - Starting All Network Functions")
    print("=" * 60)

    processes = []

    # Phase 1: Start NRF first (service discovery)
    print("\n[Phase 1] Starting Service Discovery...")
    p = start_nf("core_network/nrf.py", "NRF - Network Repository Function (Port 9000)")
    if p: processes.append(p)
    time.sleep(NRF_STARTUP_DELAY)

    # Phase 2: Start core data functions
    print("\n[Phase 2] Starting Data Management Functions...")
    p = start_nf("core_network/udr.py", "UDR - Unified Data Repository (Port 9007)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/udm.py", "UDM - Unified Data Management (Port 9005)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    # Phase 3: Start security functions
    print("\n[Phase 3] Starting Security Functions...")
    p = start_nf("core_network/ausf.py", "AUSF - Authentication Server Function (Port 9004)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    # Phase 4: Start policy and charging functions
    print("\n[Phase 4] Starting Policy & Charging Functions...")
    p = start_nf("core_network/pcf.py", "PCF - Policy Control Function (Port 9006)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/chf.py", "CHF - Charging Function (Port 9013)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    # Phase 5: Start slice and binding functions
    print("\n[Phase 5] Starting Slice & Binding Functions...")
    p = start_nf("core_network/nssf.py", "NSSF - Network Slice Selection Function (Port 9010)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/bsf.py", "BSF - Binding Support Function (Port 9011)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    # Phase 6: Start session management functions
    print("\n[Phase 6] Starting Session Management Functions...")
    p = start_nf("core_network/amf.py", "AMF - Access and Mobility Management Function (Port 9001)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/smf.py", "SMF - Session Management Function (Port 9002)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/upf.py", "UPF - User Plane Function (Port 9003)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    # Phase 7: Start proxy and exposure functions
    print("\n[Phase 7] Starting Proxy & Exposure Functions...")
    p = start_nf("core_network/scp.py", "SCP - Service Communication Proxy (Port 9012)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/nef.py", "NEF - Network Exposure Function (Port 9016)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    # Phase 8: Start interworking and roaming functions
    print("\n[Phase 8] Starting Interworking & Roaming Functions...")
    p = start_nf("core_network/n3iwf.py", "N3IWF - Non-3GPP Interworking Function (Port 9015)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/sepp.py", "SEPP - Security Edge Protection Proxy (Port 9014)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    # Phase 9: Start RAN
    print("\n[Phase 9] Starting Radio Access Network...")
    p = start_nf("ran/gnb.py", "gNodeB - 5G Base Station (Port 9100)")
    if p: processes.append(p)

    # Phase 10: Start 4G/LTE EPC components (optional)
    print("\n[Phase 10] Starting 4G/LTE EPC Components...")
    p = start_nf("core_network/hss.py", "HSS - Home Subscriber Server (Port 9023)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/mme.py", "MME - Mobility Management Entity (Port 9020)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/sgw.py", "SGW - Serving Gateway (Port 9021)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/pgw.py", "PGW - PDN Gateway (Port 9022)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    # Phase 11: Start IMS Core Components
    print("\n[Phase 11] Starting IMS Core Components...")
    p = start_nf("core_network/ims_hss.py", "IMS-HSS - IMS Home Subscriber Server (Port 9040)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/pcscf.py", "P-CSCF - Proxy Call Session Control Function (Port 9030)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/icscf.py", "I-CSCF - Interrogating Call Session Control Function (Port 9031)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/scscf.py", "S-CSCF - Serving Call Session Control Function (Port 9032)")
    if p: processes.append(p)
    time.sleep(STARTUP_DELAY)

    p = start_nf("core_network/mrf.py", "MRF - Media Resource Function (Port 9033)")
    if p: processes.append(p)

    print("\n" + "=" * 60)
    print(f"5G/4G/IMS Core Network Started - {len(processes)} functions running")
    print("=" * 60)
    print("\n5G Core Network Functions:")
    print("  NRF:   http://localhost:9000  - Service Discovery")
    print("  AMF:   http://localhost:9001  - Access & Mobility")
    print("  SMF:   http://localhost:9002  - Session Management")
    print("  UPF:   http://localhost:9003  - User Plane")
    print("  AUSF:  http://localhost:9004  - Authentication")
    print("  UDM:   http://localhost:9005  - Data Management")
    print("  PCF:   http://localhost:9006  - Policy Control")
    print("  UDR:   http://localhost:9007  - Data Repository")
    print("  NSSF:  http://localhost:9010  - Slice Selection")
    print("  BSF:   http://localhost:9011  - Binding Support")
    print("  SCP:   http://localhost:9012  - Service Proxy")
    print("  CHF:   http://localhost:9013  - Charging")
    print("  SEPP:  http://localhost:9014  - Roaming Security")
    print("  N3IWF: http://localhost:9015  - Non-3GPP Access")
    print("  NEF:   http://localhost:9016  - Network Exposure")
    print("\n4G/LTE EPC Functions:")
    print("  MME:   http://localhost:9020  - Mobility Management")
    print("  SGW:   http://localhost:9021  - Serving Gateway")
    print("  PGW:   http://localhost:9022  - PDN Gateway")
    print("  HSS:   http://localhost:9023  - Subscriber Server (4G)")
    print("\nIMS Core Functions:")
    print("  P-CSCF:  http://localhost:9030  - Proxy CSCF (Entry Point)")
    print("  I-CSCF:  http://localhost:9031  - Interrogating CSCF")
    print("  S-CSCF:  http://localhost:9032  - Serving CSCF")
    print("  MRF:     http://localhost:9033  - Media Resource Function")
    print("  IMS-HSS: http://localhost:9040  - IMS Home Subscriber Server")
    print("\nRadio Access Network:")
    print("  gNB:   http://localhost:9100  - 5G Base Station")
    print("\nPress Ctrl+C to stop all services...")

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down all network functions...")
        for p in processes:
            if p and p.poll() is None:
                p.terminate()
        print("All services stopped.")

if __name__ == "__main__":
    main()
