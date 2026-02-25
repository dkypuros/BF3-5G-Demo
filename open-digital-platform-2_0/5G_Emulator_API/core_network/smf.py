# File location: 5G_Emulator_API/core_network/smf.py
# Enhanced with 3GPP TS 23.502 compliance for PDU Session Establishment
from fastapi import FastAPI, Request, HTTPException
import uvicorn
import requests
import logging
import json
from contextlib import asynccontextmanager
from typing import Dict
from datetime import datetime
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

# Create logs directory if it doesn't exist
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{logs_dir}/smf_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up tracing
resource = Resource.create({"service.name": "smf-service"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

nrf_url = "http://127.0.0.1:8000"
upf_url = None  # Will be discovered from NRF

# SMF Session contexts - stores PDU session state
session_contexts: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global upf_url
    # Startup
    nf_registration = {
        "nf_type": "SMF",
        "ip": "127.0.0.1",
        "port": 9001
    }
    try:
        response = requests.post(f"{nrf_url}/register", json=nf_registration)
        response.raise_for_status()
        logger.info("SMF registered with NRF")
        
        # Discover UPF for N4 interface
        upf_info = requests.get(f"{nrf_url}/discover/UPF").json()
        if 'message' in upf_info:
            logger.error(f"UPF discovery failed: {upf_info['message']}")
        else:
            upf_url = f"http://{upf_info.get('ip')}:{upf_info.get('port')}"
            logger.info(f"UPF discovered at {upf_url}")
    except requests.RequestException as e:
        logger.error(f"Failed to register SMF with NRF or discover UPF: {str(e)}")
    
    yield
    # Shutdown
    # Add any cleanup code here if needed

app = FastAPI(lifespan=lifespan)

def _send_pfcp_establishment_request(pdu_session: dict) -> dict:
    """
    Models sending a PFCP Session Establishment Request to the UPF over N4.
    Reference: 3GPP TS 29.244 - PFCP Protocol
    """
    global upf_url
    if not upf_url:
        logger.error("UPF URL not available - service discovery failed")
        raise HTTPException(status_code=502, detail="UPF service not available")
    
    n4_endpoint = f"{upf_url}/n4/sessions"
    
    # Construct a payload that models a PFCP message
    pfcp_request = {
        "messageType": "PFCP_SESSION_ESTABLISHMENT_REQUEST",
        "seid": f"smf-seid-{pdu_session['pduSessionId']}", # SMF's Session Endpoint ID
        "createPDR": [{
            "pdrId": 1,
            "precedence": 200,
            "pdi": { # Packet Detection Information
                "sourceInterface": "ACCESS",
                "ueIpAddress": "10.0.0.1", # This would be allocated by the SMF
                "networkInstance": pdu_session.get('dnn', 'internet')
            },
            "farId": 1 # Forwarding Action Rule ID
        }],
        "createFAR": [{
            "farId": 1,
            "applyAction": "FORWARD",
            "forwardingParameters": {
                "destinationInterface": "CORE", # CORE means the N6 interface
                "outerHeaderCreation": {
                    "description": "GTP-U/UDP/IPv4",
                    "teid": "1001" # Tunnel Endpoint ID for the N3 tunnel
                }
            }
        }],
        "createQER": [{ # Quality Enforcement Rule
            "qerId": 1,
            "qfi": 9, # QoS Flow Identifier
            "uplinkMBR": 100000000, # 100 Mbps
            "downlinkMBR": 100000000
        }]
    }
    
    logger.info(f"SMF -> UPF: Sending PFCP Session Establishment Request for PDU Session {pdu_session['pduSessionId']}")
    
    with tracer.start_as_current_span("smf_pfcp_session_establishment") as span:
        span.set_attribute("3gpp.interface", "N4")
        span.set_attribute("3gpp.protocol", "PFCP")
        span.set_attribute("pdu.session.id", str(pdu_session['pduSessionId']))
        span.set_attribute("pfcp.seid", pfcp_request['seid'])
        
        try:
            response = requests.post(n4_endpoint, json=pfcp_request, timeout=5)
            response.raise_for_status()
            n4_response = response.json()
            
            logger.info(f"SMF <- UPF: PFCP Session Establishment Response received")
            span.add_event("pfcp_session_established", {
                "upf.response.status": n4_response.get("status"),
                "n3.endpoint": n4_response.get("n3_endpoint")
            })
            
            return n4_response
        except requests.RequestException as e:
            logger.error(f"SMF -> UPF: N4 interface error: {e}")
            span.record_exception(e)
            raise HTTPException(status_code=502, detail=f"N4 interface error: {e}")

@app.post("/nsmf-pdusession/v1/sm-contexts")
async def create_sm_context(request: Request):
    """
    3GPP-compliant Nsmf_PDUSession Create SM Context operation.
    Reference: 3GPP TS 29.502 Section 5.2.2.2.1
    """
    pdu_session_data = await request.json()
    supi = pdu_session_data.get('supi')
    pdu_session_id = pdu_session_data.get('pduSessionId')
    
    logger.info(f"SMF <- AMF: Received Create SM Context Request for SUPI {supi}, PDU Session ID {pdu_session_id}")
    
    # Validate 3GPP mandatory parameters
    required_fields = ['supi', 'pduSessionId', 'dnn', 'sNssai', 'anType']
    missing_fields = [field for field in required_fields if not pdu_session_data.get(field)]
    if missing_fields:
        logger.error(f"Missing required 3GPP fields: {missing_fields}")
        raise HTTPException(status_code=400, detail=f"Missing required fields: {missing_fields}")
    
    try:
        with tracer.start_as_current_span("smf_create_sm_context") as span:
            span.set_attribute("3gpp.procedure", "pdu_session_establishment")
            span.set_attribute("3gpp.interface", "N11")
            span.set_attribute("3gpp.service", "Nsmf_PDUSession")
            span.set_attribute("ue.supi", supi)
            span.set_attribute("pdu.session.id", str(pdu_session_id))
            span.set_attribute("dnn", pdu_session_data.get('dnn'))
            
            # 1. UE IP Address Allocation (simplified)
            ue_ip = f"10.{(pdu_session_id % 254) + 1}.0.1"  # Simple IP allocation
            logger.info(f"Allocated UE IP address: {ue_ip}")
            
            # 2. Select UPF and establish N4 session
            session_context = {
                **pdu_session_data,
                "ueIpAddress": ue_ip,
                "sessionState": "ESTABLISHING"
            }
            
            n4_response = _send_pfcp_establishment_request(session_context)
            
            # 3. Store session context
            session_key = f"{supi}:{pdu_session_id}"
            session_contexts[session_key] = {
                **session_context,
                "sessionState": "ACTIVE",
                "pfcpSeid": n4_response.get('upfSeid', 'upf-seid-unknown'),
                "n3TunnelInfo": n4_response.get('n3_endpoint')
            }
            
            # 4. Respond to AMF with N2 SM Information
            # This response would contain N2 SM Information for the gNB
            amf_response = {
                "status": "CREATED",
                "cause": "PDU_SESSION_ESTABLISHMENT_ACCEPTED",
                "pduSessionId": pdu_session_id,
                "ueIpAddress": ue_ip,
                "n2SmInfo": {
                    "pduSessionId": pdu_session_id,
                    "qosFlowSetupRequestList": [{
                        "qfi": 9,
                        "qosCharacteristics": {
                            "5qi": 9,  # 5G QoS Identifier for best effort
                            "priority": 80
                        }
                    }],
                    "n2InfoContent": "base64-encoded-ngap-pdu-session-resource-setup-request"
                },
                "smContext": {
                    "contextId": session_key,
                    "ueIpAddress": ue_ip
                }
            }
            
            logger.info(f"SMF -> AMF: SM Context created successfully for {supi}")
            span.add_event("sm_context_created", {
                "context.id": session_key,
                "ue.ip.address": ue_ip,
                "session.state": "ACTIVE"
            })
            
            return amf_response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create SM context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SM Context creation failed: {str(e)}")

@app.get("/smf/sessions")
async def get_sessions():
    """Get all active PDU sessions - for debugging/monitoring"""
    return {
        "activeSessions": len(session_contexts),
        "sessions": list(session_contexts.keys())
    }

@app.get("/smf_service")
def smf_service():
    return {"message": "SMF service response"}

@app.get("/health")
def health_check():
    """Health check endpoint for SMF - TS 29.502"""
    return {
        "status": "healthy",
        "service": "SMF",
        "compliance": "3GPP TS 29.502",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="SMF - Session Management Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("smf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)