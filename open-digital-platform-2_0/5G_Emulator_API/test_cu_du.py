# File location: 5G_Emulator_API/test_cu_du.py
# File location: 5G_Emulator_API/test_cu_du.py
# File location: 5G_Emulator_API/test_cu_du.py
import requests

cu_url = "http://127.0.0.1:9008"
du_url = "http://127.0.0.1:9007"

# Simulate CU sending data to DU
def cu_to_du():
    data = {"message": "Data from CU to DU"}
    response = requests.post(f"{cu_url}/cu_to_du", json=data)
    print(response.json())

# Simulate DU sending data to CU
def du_to_cu():
    data = {"message": "Data from DU to CU"}
    response = requests.post(f"{du_url}/du_service", json=data)
    print(response.json())

if __name__ == "__main__":
    cu_to_du()
    du_to_cu()
