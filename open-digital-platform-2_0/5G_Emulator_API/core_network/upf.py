# File location: 5G_Emulator_API/core_network/upf.py
# Enhanced with 3GPP TS 29.244 PFCP protocol support for N4 interface
from fastapi import FastAPI, Request, HTTPException
import uvicorn
import requests
import logging
import json
from contextlib import asynccontextmanager
from typing import Dict
from datetime import datetime
import os
import uuid
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

# Create logs directory if it doesn't exist
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{logs_dir}/upf_{timestamp}.log"
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
resource = Resource.create({"service.name": "upf-service"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

nrf_url = "http://127.0.0.1:8000"

# This dictionary simulates the UPF's forwarding table
# In a real UPF, this would program hardware/kernel forwarding tables
forwarding_rules: Dict[str, Dict] = {}
pfcp_sessions: Dict[str, Dict] = {}  # PFCP session state

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    nf_registration = {
        "nf_type": "UPF",
        "ip": "127.0.0.1",
        "port": 9002
    }
    try:
        response = requests.post(f"{nrf_url}/register", json=nf_registration)
        response.raise_for_status()
        logger.info("UPF registered with NRF")
    except requests.RequestException as e:
        logger.error(f"Failed to register UPF with NRF: {str(e)}")
    
    yield
    # Shutdown
    # Add any cleanup code here if needed

app = FastAPI(lifespan=lifespan)

@app.post("/n4/sessions")
async def n4_session_management(request: Request):
    """
    Models the N4 interface, receiving PFCP-like messages from the SMF.
    Reference: 3GPP TS 29.244 - PFCP Protocol
    """
    pfcp_message = await request.json()
    session_id = pfcp_message.get("seid")
    message_type = pfcp_message.get("messageType")
    
    logger.info(f"UPF <- SMF: Received {message_type} for SEID {session_id}")
    
    if message_type == "PFCP_SESSION_ESTABLISHMENT_REQUEST":
        with tracer.start_as_current_span("upf_pfcp_session_establishment") as span:
            span.set_attribute("3gpp.interface", "N4")
            span.set_attribute("3gpp.protocol", "PFCP")
            span.set_attribute("pfcp.seid", session_id)
            span.set_attribute("pfcp.message.type", message_type)
            
            # Generate UPF's own Session Endpoint ID
            upf_seid = f"upf-seid-{str(uuid.uuid4())[:8]}"
            
            # "Install" the forwarding rules from the message
            session_rules = {
                "seid": session_id,
                "upfSeid": upf_seid,
                "state": "ACTIVE",
                "pdrs": [],
                "fars": [],
                "qers": []
            }
            
            # Process PDRs (Packet Detection Rules)
            for pdr in pfcp_message.get("createPDR", []):
                ue_ip = pdr.get("pdi", {}).get("ueIpAddress")
                pdr_id = pdr.get("pdrId")
                if ue_ip:
                    # Find the matching forwarding action
                    far_id = pdr.get("farId")
                    far_rule = next((far for far in pfcp_message.get("createFAR", []) if far.get("farId") == far_id), None)
                    if far_rule:
                        forwarding_rules[ue_ip] = {
                            "far": far_rule.get("forwardingParameters"),
                            "pdr_id": pdr_id,
                            "far_id": far_id,
                            "session_id": session_id
                        }
                        session_rules["pdrs"].append(pdr)
                        session_rules["fars"].append(far_rule)
                        
                        logger.info(f"UPF: Installed forwarding rule for UE IP {ue_ip} -> {far_rule['forwardingParameters']['destinationInterface']}")
            
            # Process QERs (QoS Enforcement Rules)
            for qer in pfcp_message.get("createQER", []):
                session_rules["qers"].append(qer)
                logger.info(f"UPF: Installed QoS rule QER ID {qer.get('qerId')} with QFI {qer.get('qfi')}")
            
            # Store the session
            pfcp_sessions[session_id] = session_rules
            
            # In a real scenario, the UPF would respond with its own SEID and N3 endpoint info
            response = {
                "status": "SESSION_CREATED",
                "cause": "REQUEST_ACCEPTED",
                "upfSeid": upf_seid,
                "n3_endpoint": "192.168.1.100",  # N3 interface endpoint
                "createdPDR": [pdr.get("pdrId") for pdr in pfcp_message.get("createPDR", [])],
                "createdFAR": [far.get("farId") for far in pfcp_message.get("createFAR", [])],
                "createdQER": [qer.get("qerId") for qer in pfcp_message.get("createQER", [])]
            }
            
            span.add_event("pfcp_session_established", {
                "upf.seid": upf_seid,
                "rules.installed": len(forwarding_rules),
                "n3.endpoint": response["n3_endpoint"]
            })
            
            logger.info(f"UPF -> SMF: PFCP Session Establishment Response sent")
            return response
            
    elif message_type == "PFCP_SESSION_MODIFICATION_REQUEST":
        logger.info(f"UPF: Processing session modification for SEID {session_id}")
        # Handle session modifications (simplified)
        return {"status": "SESSION_MODIFIED", "cause": "REQUEST_ACCEPTED"}
        
    elif message_type == "PFCP_SESSION_DELETION_REQUEST":
        logger.info(f"UPF: Processing session deletion for SEID {session_id}")
        # Clean up forwarding rules
        if session_id in pfcp_sessions:
            session = pfcp_sessions[session_id]
            # Remove forwarding rules for this session
            ue_ips_to_remove = [ue_ip for ue_ip, rule in forwarding_rules.items() 
                               if rule.get("session_id") == session_id]
            for ue_ip in ue_ips_to_remove:
                del forwarding_rules[ue_ip]
                logger.info(f"UPF: Removed forwarding rule for UE IP {ue_ip}")
            del pfcp_sessions[session_id]
        
        return {"status": "SESSION_DELETED", "cause": "REQUEST_ACCEPTED"}
    
    return {"status": "UNKNOWN_MESSAGE", "cause": "MESSAGE_TYPE_NOT_SUPPORTED"}

@app.get("/upf/forwarding-rules")
async def get_forwarding_rules():
    """Get current forwarding rules - for debugging/monitoring"""
    return {
        "activeRules": len(forwarding_rules),
        "activeSessions": len(pfcp_sessions),
        "rules": forwarding_rules
    }

@app.post("/upf/simulate-traffic")
async def simulate_traffic(traffic_data: dict):
    """Simulate user plane traffic processing"""
    src_ip = traffic_data.get("src_ip")
    dest_ip = traffic_data.get("dest_ip")
    packet_size = traffic_data.get("packet_size", 1500)
    
    # Check if we have forwarding rules for this traffic
    if src_ip in forwarding_rules:
        rule = forwarding_rules[src_ip]
        logger.info(f"UPF: Processing traffic from {src_ip} -> {dest_ip} via {rule['far']['destinationInterface']}")
        
        # Simulate packet processing
        processed_packet = {
            "original": traffic_data,
            "processed_via": rule['far']['destinationInterface'],
            "tunnel_info": rule['far'].get('outerHeaderCreation'),
            "qos_applied": True
        }
        
        return {"status": "FORWARDED", "packet_info": processed_packet}
    else:
        logger.warning(f"UPF: No forwarding rule found for src_ip {src_ip} - DROPPING")
        return {"status": "DROPPED", "reason": "NO_FORWARDING_RULE"}

@app.get("/upf_service")
def upf_service():
    return {"message": "UPF service response"}

@app.get("/health")
def health_check():
    """Health check endpoint for UPF - TS 29.244"""
    return {
        "status": "healthy",
        "service": "UPF",
        "compliance": "3GPP TS 29.244",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="UPF - User Plane Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("upf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)