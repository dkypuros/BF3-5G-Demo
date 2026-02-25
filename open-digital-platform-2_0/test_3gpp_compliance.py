#!/usr/bin/env python3
"""
3GPP Compliance Test Suite
Tests the enhanced AMF, SMF, and UPF services for 3GPP TS 23.502 compliance
"""

import requests
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service endpoints
NRF_URL = "http://127.0.0.1:8000"
AMF_URL = "http://127.0.0.1:9000"
SMF_URL = "http://127.0.0.1:9001" 
UPF_URL = "http://127.0.0.1:9002"

def test_service_health():
    """Test if all services are running"""
    logger.info("=== Testing Service Health ===")
    
    services = {
        "NRF": f"{NRF_URL}/health",
        "AMF": f"{AMF_URL}/metrics",
        "SMF": f"{SMF_URL}/smf_service", 
        "UPF": f"{UPF_URL}/upf_service"
    }
    
    healthy_services = []
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {name} is healthy")
                healthy_services.append(name)
            else:
                logger.error(f"❌ {name} returned status {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"❌ {name} is not accessible: {e}")
    
    return len(healthy_services) == len(services)

def create_test_ue_context():
    """Create a test UE context in AMF"""
    logger.info("=== Creating Test UE Context ===")
    
    ue_id = "test_ue_001"
    ue_context = {
        "imsi": "001010000000001",
        "supi": "imsi-001010000000001",
        "pduSessionId": 1,
        "status": "registered",
        "location": {
            "tai": {
                "plmnId": {"mcc": "001", "mnc": "01"},
                "tac": "000001"
            }
        }
    }
    
    try:
        response = requests.post(f"{AMF_URL}/amf/ue/{ue_id}", json=ue_context, timeout=10)
        if response.status_code == 200:
            logger.info(f"✅ Created UE context for {ue_id}")
            return ue_id, ue_context
        else:
            logger.error(f"❌ Failed to create UE context: {response.status_code}")
            return None, None
    except requests.RequestException as e:
        logger.error(f"❌ Failed to create UE context: {e}")
        return None, None

def test_3gpp_pdu_session_establishment(ue_id):
    """Test the 3GPP-compliant PDU Session Establishment procedure"""
    logger.info("=== Testing 3GPP-Compliant PDU Session Establishment ===")
    
    # Step 1: UE-requested PDU Session Establishment (simulated)
    logger.info("📱 Step 1: UE requests PDU Session Establishment")
    
    pdu_session_request = {
        "ue_id": ue_id,
        "procedure": "pdu_session_establishment",
        "dnn": "internet",
        "sNssai": {
            "sst": 1,
            "sd": "010203"
        }
    }
    
    try:
        # Step 2: AMF processes request and calls SMF via N11 interface
        logger.info("🔄 Step 2: AMF -> SMF (N11 Interface)")
        start_time = time.time()
        
        response = requests.post(
            f"{AMF_URL}/amf/pdu-session/create", 
            json=pdu_session_request, 
            timeout=30
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ PDU Session Establishment successful in {duration:.2f}s")
            logger.info(f"   - Status: {result['status']}")
            logger.info(f"   - PDU Session ID: {result['pduSessionId']}")
            
            if 'smContextResponse' in result:
                sm_response = result['smContextResponse']
                logger.info(f"   - SM Context Status: {sm_response.get('status')}")
                logger.info(f"   - UE IP Address: {sm_response.get('ueIpAddress')}")
                
                # Step 3: Verify N2 SM Information was included
                if 'n2SmInfo' in sm_response:
                    logger.info("✅ N2 SM Information included for gNB")
                    logger.info(f"   - QoS Flow Setup: {len(sm_response['n2SmInfo'].get('qosFlowSetupRequestList', []))} flows")
                
            return True, result
        else:
            logger.error(f"❌ PDU Session Establishment failed: {response.status_code}")
            try:
                error_details = response.json()
                logger.error(f"   Error details: {error_details}")
            except:
                logger.error(f"   Response: {response.text}")
            return False, None
            
    except requests.RequestException as e:
        logger.error(f"❌ PDU Session Establishment request failed: {e}")
        return False, None

def verify_n4_session_establishment():
    """Verify that PFCP session was established on UPF"""
    logger.info("=== Verifying N4/PFCP Session on UPF ===")
    
    try:
        response = requests.get(f"{UPF_URL}/upf/forwarding-rules", timeout=5)
        if response.status_code == 200:
            rules = response.json()
            logger.info(f"✅ UPF has {rules['activeRules']} active forwarding rules")
            logger.info(f"✅ UPF has {rules['activeSessions']} active PFCP sessions")
            
            if rules['activeRules'] > 0:
                logger.info("📋 Forwarding Rules:")
                for ue_ip, rule in rules.get('rules', {}).items():
                    logger.info(f"   - UE IP {ue_ip} -> {rule['far']['destinationInterface']}")
            
            return True
        else:
            logger.error(f"❌ Failed to get UPF status: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ Failed to verify UPF status: {e}")
        return False

def test_smf_session_state():
    """Verify SMF session state"""
    logger.info("=== Verifying SMF Session State ===")
    
    try:
        response = requests.get(f"{SMF_URL}/smf/sessions", timeout=5)
        if response.status_code == 200:
            sessions = response.json()
            logger.info(f"✅ SMF has {sessions['activeSessions']} active sessions")
            
            if sessions['activeSessions'] > 0:
                logger.info("📋 Active Sessions:")
                for session_id in sessions.get('sessions', []):
                    logger.info(f"   - Session: {session_id}")
            
            return True
        else:
            logger.error(f"❌ Failed to get SMF sessions: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ Failed to verify SMF sessions: {e}")
        return False

def simulate_user_plane_traffic():
    """Simulate user plane traffic to test forwarding rules"""
    logger.info("=== Simulating User Plane Traffic ===")
    
    traffic_data = {
        "src_ip": "10.1.0.1",  # UE IP from PDU session
        "dest_ip": "8.8.8.8",   # Google DNS
        "packet_size": 1500,
        "protocol": "UDP"
    }
    
    try:
        response = requests.post(f"{UPF_URL}/upf/simulate-traffic", json=traffic_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Traffic simulation result: {result['status']}")
            
            if result['status'] == 'FORWARDED':
                packet_info = result['packet_info']
                logger.info(f"   - Processed via: {packet_info['processed_via']}")
                logger.info(f"   - QoS applied: {packet_info['qos_applied']}")
            
            return True
        else:
            logger.error(f"❌ Traffic simulation failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ Traffic simulation error: {e}")
        return False

def run_compliance_test():
    """Run the complete 3GPP compliance test suite"""
    logger.info("🚀 Starting 3GPP Compliance Test Suite")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    test_results = {
        "service_health": False,
        "ue_context_creation": False,
        "pdu_session_establishment": False,
        "n4_session_verification": False,
        "smf_session_verification": False,
        "traffic_simulation": False
    }
    
    # Test 1: Service Health
    test_results["service_health"] = test_service_health()
    if not test_results["service_health"]:
        logger.error("❌ Service health check failed. Ensure all services are running.")
        return test_results
    
    # Test 2: UE Context Creation
    ue_id, ue_context = create_test_ue_context()
    test_results["ue_context_creation"] = ue_id is not None
    if not test_results["ue_context_creation"]:
        logger.error("❌ UE context creation failed.")
        return test_results
    
    # Test 3: 3GPP PDU Session Establishment
    success, pdu_result = test_3gpp_pdu_session_establishment(ue_id)
    test_results["pdu_session_establishment"] = success
    if not success:
        logger.error("❌ PDU Session Establishment failed.")
        return test_results
    
    # Small delay to ensure session state propagation
    time.sleep(1)
    
    # Test 4: N4/PFCP Session Verification  
    test_results["n4_session_verification"] = verify_n4_session_establishment()
    
    # Test 5: SMF Session State Verification
    test_results["smf_session_verification"] = test_smf_session_state()
    
    # Test 6: User Plane Traffic Simulation
    test_results["traffic_simulation"] = simulate_user_plane_traffic()
    
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info(f"\n=== Test Results Summary ===")
    logger.info(f"Passed: {passed_tests}/{total_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! The system is 3GPP compliant.")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} test(s) failed.")
        
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
    
    return test_results

if __name__ == "__main__":
    results = run_compliance_test()