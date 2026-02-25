# File location: 5G_Emulator_API/test_5g_network.py
# File location: 5G_Emulator_API/test_5g_network.py
import requests
import time

def test_5g_network():
    gnb_url = "http://127.0.0.1:38412"
    amf_url = "http://127.0.0.1:9000"

    # Step 1: Check gNodeB status
    response = requests.get(f"{gnb_url}/gnb_status")
    print("gNodeB Status:", response.json())

    # Step 2: Simulate Initial UE Message
    ue_data = {
        "imsi": "123456789012345",
        "initial_message": "RRC Connection Request"
    }
    response = requests.post(f"{gnb_url}/initial_ue_message", json=ue_data)
    print("Initial UE Message Response:", response.json())

    # Step 3: Registration Request
    registration_data = {
        "imsi": "123456789012345",
        "registration_type": "initial_registration"
    }
    response = requests.post(f"{amf_url}/registration_request", json=registration_data)
    print("Registration Request Response:", response.json())

    # Step 4: Authentication Response (simulating successful authentication)
    auth_data = {
        "imsi": "123456789012345",
        "auth_result": "success"
    }
    response = requests.post(f"{amf_url}/authentication_response", json=auth_data)
    print("Authentication Response:", response.json())

    # Step 5: Security Mode Complete
    security_data = {
        "imsi": "123456789012345",
        "security_capabilities": "5G-EA1, 5G-IA1"
    }
    response = requests.post(f"{amf_url}/security_mode_complete", json=security_data)
    print("Security Mode Complete Response:", response.json())

    # Step 6: PDU Session Establishment Request
    session_data = {
        "imsi": "123456789012345",
        "pdu_session_type": "IPv4",
        "sst": 1,
        "sd": "010203"
    }
    response = requests.post(f"{amf_url}/pdu_session_establishment_request", json=session_data)
    print("PDU Session Establishment Response:", response.json())

if __name__ == "__main__":
    test_5g_network()