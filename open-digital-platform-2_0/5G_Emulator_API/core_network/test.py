# File location: 5G_Emulator_API/core_network/test.py
# File location: 5G_Emulator_API/core_network/test.py
# File location: 5G_Emulator_API/core_network/test.py
import requests

nrf_url = "http://127.0.0.1:8000"
udr_url = "http://127.0.0.1:9005"
ausf_url = "http://127.0.0.1:9003"

# Register Network Functions with NRF
def register_nfs():
    nfs = [
        {"nf_type": "AMF", "ip": "127.0.0.1", "port": 9000},
        {"nf_type": "SMF", "ip": "127.0.0.1", "port": 9001},
        {"nf_type": "UPF", "ip": "127.0.0.1", "port": 9002},
        {"nf_type": "AUSF", "ip": "127.0.0.1", "port": 9003},
        {"nf_type": "UDM", "ip": "127.0.0.1", "port": 9004},
        {"nf_type": "UDR", "ip": "127.0.0.1", "port": 9005},
        {"nf_type": "UDSF", "ip": "127.0.0.1", "port": 9006},
        {"nf_type": "gNodeB", "ip": "127.0.0.1", "port": 38412},
    ]

    for nf in nfs:
        response = requests.post(f"{nrf_url}/register", json=nf)
        print(response.json())

# Register a user with UDR
def register_user():
    user_data = {"imsi": "123456789012345", "key": "secret_key"}
    response = requests.post(f"{udr_url}/register_user", json=user_data)
    print(response.json())

# Simulate AUSF authentication request
def authenticate_user():
    imsi = "123456789012345"
    response = requests.get(f"{udr_url}/get_user/{imsi}")
    if response.status_code == 200:
        user_data = response.json()
        print(f"AUSF authenticated user: {user_data}")
    else:
        print(f"AUSF failed to authenticate user: {response.json()}")

if __name__ == "__main__":
    register_nfs()
    register_user()
    authenticate_user()
