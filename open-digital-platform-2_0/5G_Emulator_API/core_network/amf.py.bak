from fastapi import FastAPI
import uvicorn
import requests
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, start_http_server

nrf_url = "http://127.0.0.1:8000"
smf_url = None

# Define Prometheus metrics
REQUEST_COUNT = Counter('amf_requests_total', 'Total AMF requests')
REQUEST_LATENCY = Histogram('amf_request_latency_seconds', 'AMF request latency')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global smf_url
    nf_registration = {
        "nf_type": "AMF",
        "ip": "0.0.0.0",  # Changed to 0.0.0.0 to bind to all interfaces
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
    
    # Start Prometheus metrics server
    start_http_server(9100, addr='0.0.0.0')  # Changed to bind to all interfaces
    
    yield
    # Shutdown
    # Add any cleanup code here if needed

app = FastAPI(lifespan=lifespan)

@app.get("/amf_service")
@REQUEST_LATENCY.time()
def amf_service():
    REQUEST_COUNT.inc()
    return {"message": "AMF service response"}

@app.get("/metrics")
def metrics():
    return {"message": "Metrics are exposed on port 9100"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)  # Changed to bind to all interfaces