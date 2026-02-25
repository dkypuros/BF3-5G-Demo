# File location: 5G_Emulator_API/ran/cu/cu.py
# File location: 5G_Emulator_API/ran/cu/cu.py
from fastapi import FastAPI
import uvicorn
import requests
from contextlib import asynccontextmanager

nrf_url = "http://127.0.0.1:8000"
du_url = "http://127.0.0.1:9007"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    nf_registration = {
        "nf_type": "CU",
        "ip": "127.0.0.1",
        "port": 9008
    }
    try:
        response = requests.post(f"{nrf_url}/register", json=nf_registration)
        response.raise_for_status()
        print("CU registered with NRF")
    except requests.RequestException as e:
        print(f"Failed to register CU with NRF: {str(e)}")
    
    yield
    # Shutdown
    # Add any cleanup code here if needed

app = FastAPI(lifespan=lifespan)

@app.post("/cu_to_du")
def cu_to_du(data: dict):
    response = requests.post(f"{du_url}/du_service", json=data)
    return response.json()

@app.get("/cu_service")
def cu_service():
    return {"message": "CU service response"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9008)