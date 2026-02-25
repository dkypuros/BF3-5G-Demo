# File location: 5G_Emulator_API/core_network/amf.py
# File location: 5G_Emulator_API/core_network/amf.py
from fastapi import FastAPI
import uvicorn
import requests
from contextlib import asynccontextmanager

nrf_url = "http://127.0.0.1:8000"
smf_url = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global smf_url
    nf_registration = {
        "nf_type": "AMF",
        "ip": "127.0.0.1",
        "port": 9000
    }
    try:
        response = requests.post(f"{nrf_url}/register", json=nf_registration)
        response.raise_for_status()
        
        # Discover SMF
        smf_info = requests.get(f"{nrf_url}/discover/SMF").json()
        if 'message' in smf_info:
            print(f"SMF discovery failed: {smf_info['message']}")
        else:
            smf_url = f"http://{smf_info.get('ip')}:{smf_info.get('port')}"
            print(f"SMF discovered at {smf_url}")
    except requests.RequestException as e:
        print(f"Failed to register with NRF or discover SMF: {str(e)}")
    
    yield
    # Shutdown
    # Add any cleanup code here if needed

app = FastAPI(lifespan=lifespan)

@app.get("/amf_service")
def amf_service():
    return {"message": "AMF service response"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AMF Original")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)