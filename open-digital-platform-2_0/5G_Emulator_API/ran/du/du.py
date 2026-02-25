# File location: 5G_Emulator_API/ran/du/du.py
# File location: 5G_Emulator_API/ran/du/du.py
from fastapi import FastAPI
import uvicorn
import requests
from contextlib import asynccontextmanager

nrf_url = "http://127.0.0.1:8000"
cu_url = "http://127.0.0.1:9008"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    nf_registration = {
        "nf_type": "DU",
        "ip": "127.0.0.1",
        "port": 9007
    }
    try:
        response = requests.post(f"{nrf_url}/register", json=nf_registration)
        response.raise_for_status()
        print("DU registered with NRF")
    except requests.RequestException as e:
        print(f"Failed to register DU with NRF: {str(e)}")
    
    yield
    # Shutdown
    # Add any cleanup code here if needed

app = FastAPI(lifespan=lifespan)

@app.post("/du_service")
def du_service(data: dict):
    print(f"Received data at DU: {data}")
    return {"message": "DU received data"}

@app.get("/du_to_cu")
def du_to_cu(data: dict):
    response = requests.post(f"{cu_url}/cu_service", json=data)
    return response.json()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9007)