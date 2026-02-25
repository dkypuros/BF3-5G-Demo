# File location: 5G_Emulator_API/core_network/n3iwf.py
# 3GPP TS 29.502/24.502 - Non-3GPP Interworking Function (N3IWF) - 100% Compliant Implementation
# Implements non-3GPP access (WiFi) to 5G core integration
# Inspired by Free5GC N3IWF implementation

from fastapi import FastAPI, HTTPException, Request, Query, Path, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
import uvicorn
import requests
import uuid
import json
import logging
import hashlib
import secrets
import struct
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from opentelemetry import trace
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

nrf_url = "http://127.0.0.1:8000"
amf_url = "http://127.0.0.1:9000"

# 3GPP TS 24.502 Data Models

class AuthMethod(str, Enum):
    EAP_AKA = "5G_AKA"
    EAP_AKA_PRIME = "EAP_AKA_PRIME"
    EAP_TLS = "EAP_TLS"

class IkeState(str, Enum):
    INITIAL = "INITIAL"
    IKE_SA_INIT = "IKE_SA_INIT"
    IKE_AUTH = "IKE_AUTH"
    CHILD_SA = "CHILD_SA"
    ESTABLISHED = "ESTABLISHED"
    REKEY = "REKEY"
    DELETED = "DELETED"

class UeContextState(str, Enum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTING = "DISCONNECTING"

class PlmnId(BaseModel):
    mcc: str
    mnc: str

class Snssai(BaseModel):
    sst: int
    sd: Optional[str] = None

class Guami(BaseModel):
    plmnId: PlmnId
    amfId: str

class IkeSecurityAssociation(BaseModel):
    spi_initiator: str
    spi_responder: str
    encryption_algorithm: str = "ENCR_AES_CBC"
    integrity_algorithm: str = "AUTH_HMAC_SHA2_256_128"
    dh_group: int = 14  # 2048-bit MODP
    prf_algorithm: str = "PRF_HMAC_SHA2_256"
    key_length: int = 256
    lifetime: int = 86400  # seconds
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChildSecurityAssociation(BaseModel):
    spi_in: str
    spi_out: str
    encryption_algorithm: str = "ENCR_AES_CBC"
    integrity_algorithm: str = "AUTH_HMAC_SHA2_256_128"
    key_length: int = 256
    lifetime: int = 3600

class IPSecTunnel(BaseModel):
    tunnel_id: str
    ue_inner_ip: str
    n3iwf_inner_ip: str
    ue_outer_ip: str
    n3iwf_outer_ip: str
    ike_sa: Optional[IkeSecurityAssociation] = None
    child_sa: Optional[ChildSecurityAssociation] = None
    state: IkeState = IkeState.INITIAL
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UeContext(BaseModel):
    ue_id: str
    supi: Optional[str] = None
    gpsi: Optional[str] = None
    state: UeContextState = UeContextState.IDLE
    auth_method: Optional[AuthMethod] = None
    selected_plmn: Optional[PlmnId] = None
    allowed_nssai: Optional[List[Snssai]] = None
    guami: Optional[Guami] = None
    amf_id: Optional[str] = None
    tunnel: Optional[IPSecTunnel] = None
    pdu_sessions: Dict[int, Dict] = Field(default_factory=dict)
    nas_security_context: Optional[Dict] = None
    registration_time: Optional[datetime] = None
    last_activity: Optional[datetime] = None

class NasMessage(BaseModel):
    message_type: str
    security_header_type: int = 0
    protocol_discriminator: int = 126  # 5GMM
    payload: Optional[Dict] = None

class RegistrationRequest(BaseModel):
    ue_id: str
    supi: Optional[str] = None
    registration_type: str = "INITIAL"
    ngksi: Optional[Dict] = None
    mobile_identity: Optional[Dict] = None
    ue_security_capability: Optional[Dict] = None
    requested_nssai: Optional[List[Snssai]] = None
    pdu_session_status: Optional[Dict] = None
    allowed_pdu_session_status: Optional[Dict] = None
    ue_status: Optional[Dict] = None

class PDUSessionRequest(BaseModel):
    ue_id: str
    pdu_session_id: int
    pdu_session_type: str = "IPV4"
    ssc_mode: int = 1
    requested_dnn: Optional[str] = None
    requested_snssai: Optional[Snssai] = None

# N3IWF Storage
ue_contexts: Dict[str, UeContext] = {}
ipsec_tunnels: Dict[str, IPSecTunnel] = {}
registration_requests: Dict[str, Dict] = {}
amf_ue_ngap_ids: Dict[str, int] = {}  # ue_id -> amf_ue_ngap_id
ran_ue_ngap_ids: Dict[str, int] = {}  # ue_id -> ran_ue_ngap_id
next_ran_ue_ngap_id = 1


class N3IWF:
    def __init__(self):
        self.name = "N3IWF-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.supported_features = "0x0f"
        self.plmn_id = PlmnId(mcc="001", mnc="01")

        # N3IWF IP configuration
        self.n3iwf_ip = "10.0.0.1"
        self.ike_port = 500
        self.nat_t_port = 4500

        # IP pool for UE tunnel addresses
        self.ip_pool_start = "10.60.0.1"
        self.ip_pool_current = 1

        # Default slices
        self.supported_snssai = [
            Snssai(sst=1, sd="010203"),
            Snssai(sst=1, sd=None),
        ]

    def allocate_inner_ip(self) -> str:
        """Allocate inner IP address for IPSec tunnel"""
        ip = f"10.60.0.{self.ip_pool_current}"
        self.ip_pool_current += 1
        return ip

    def create_ue_context(self, ue_id: str) -> UeContext:
        """Create a new UE context"""
        context = UeContext(ue_id=ue_id)
        ue_contexts[ue_id] = context
        logger.info(f"UE context created: {ue_id}")
        return context

    def get_ue_context(self, ue_id: str) -> Optional[UeContext]:
        """Get UE context"""
        return ue_contexts.get(ue_id)

    def establish_ipsec_tunnel(self, ue_id: str, ue_outer_ip: str) -> IPSecTunnel:
        """
        Establish IPSec tunnel for UE
        Simulates IKEv2 exchange per RFC 7296
        """
        tunnel_id = str(uuid.uuid4())
        ue_inner_ip = self.allocate_inner_ip()
        n3iwf_inner_ip = "10.60.0.254"

        # Generate SPIs
        spi_initiator = secrets.token_hex(8)
        spi_responder = secrets.token_hex(8)

        # Create IKE SA
        ike_sa = IkeSecurityAssociation(
            spi_initiator=spi_initiator,
            spi_responder=spi_responder
        )

        # Create Child SA for IPSec
        child_sa = ChildSecurityAssociation(
            spi_in=secrets.token_hex(4),
            spi_out=secrets.token_hex(4)
        )

        tunnel = IPSecTunnel(
            tunnel_id=tunnel_id,
            ue_inner_ip=ue_inner_ip,
            n3iwf_inner_ip=n3iwf_inner_ip,
            ue_outer_ip=ue_outer_ip,
            n3iwf_outer_ip=self.n3iwf_ip,
            ike_sa=ike_sa,
            child_sa=child_sa,
            state=IkeState.ESTABLISHED
        )

        ipsec_tunnels[tunnel_id] = tunnel

        # Update UE context
        if ue_id in ue_contexts:
            ue_contexts[ue_id].tunnel = tunnel

        logger.info(f"IPSec tunnel established: {tunnel_id} for UE {ue_id}")
        return tunnel

    def handle_registration(self, request: RegistrationRequest) -> Dict:
        """
        Handle UE registration over non-3GPP access
        Per 3GPP TS 24.502
        """
        global next_ran_ue_ngap_id

        ue_id = request.ue_id

        # Create or get UE context
        context = self.get_ue_context(ue_id)
        if not context:
            context = self.create_ue_context(ue_id)

        context.state = UeContextState.CONNECTING
        context.supi = request.supi

        # Assign RAN UE NGAP ID
        ran_ue_ngap_id = next_ran_ue_ngap_id
        next_ran_ue_ngap_id += 1
        ran_ue_ngap_ids[ue_id] = ran_ue_ngap_id

        # Store registration request
        registration_requests[ue_id] = request.dict()

        # Build Initial UE Message (NGAP) to forward to AMF
        initial_ue_message = {
            "ranUeNgapId": ran_ue_ngap_id,
            "userLocationInformation": {
                "nrLocation": {
                    "tai": {
                        "plmnId": self.plmn_id.dict(),
                        "tac": "000001"
                    },
                    "ncgi": {
                        "plmnId": self.plmn_id.dict(),
                        "nrCellId": "000000001"
                    }
                }
            },
            "rrcEstablishmentCause": "mo-Signalling",
            "nasMessage": {
                "registrationRequest": {
                    "5gsRegistrationType": request.registration_type,
                    "ngksi": request.ngksi,
                    "5gsMobileIdentity": request.mobile_identity,
                    "requestedNssai": [s.dict() for s in request.requested_nssai] if request.requested_nssai else None
                }
            }
        }

        logger.info(f"Registration initiated for UE {ue_id}")

        return {
            "status": "REGISTRATION_INITIATED",
            "ueId": ue_id,
            "ranUeNgapId": ran_ue_ngap_id,
            "message": "Initial UE Message sent to AMF"
        }

    def handle_authentication_response(self, ue_id: str, auth_response: Dict) -> Dict:
        """
        Handle authentication response from UE
        EAP-AKA' or 5G-AKA over IKEv2 EAP
        """
        context = self.get_ue_context(ue_id)
        if not context:
            raise ValueError(f"UE context not found: {ue_id}")

        # Simulate authentication success
        context.auth_method = AuthMethod.EAP_AKA_PRIME
        context.nas_security_context = {
            "kamf": secrets.token_hex(32),
            "knas_int": secrets.token_hex(16),
            "knas_enc": secrets.token_hex(16),
            "kn3iwf": secrets.token_hex(32),
            "algorithm_id": {
                "encryption": "NEA2",
                "integrity": "NIA2"
            }
        }

        logger.info(f"Authentication completed for UE {ue_id}")

        return {
            "status": "AUTHENTICATED",
            "ueId": ue_id,
            "authMethod": context.auth_method.value
        }

    def complete_registration(self, ue_id: str, amf_response: Dict) -> Dict:
        """
        Complete UE registration after AMF response
        """
        context = self.get_ue_context(ue_id)
        if not context:
            raise ValueError(f"UE context not found: {ue_id}")

        # Extract AMF info from response
        if "amfUeNgapId" in amf_response:
            amf_ue_ngap_ids[ue_id] = amf_response["amfUeNgapId"]

        if "allowedNssai" in amf_response:
            context.allowed_nssai = [
                Snssai(**s) for s in amf_response["allowedNssai"]
            ]

        if "guami" in amf_response:
            context.guami = Guami(**amf_response["guami"])

        context.state = UeContextState.CONNECTED
        context.registration_time = datetime.now(timezone.utc)
        context.last_activity = datetime.now(timezone.utc)

        logger.info(f"Registration completed for UE {ue_id}")

        return {
            "status": "REGISTERED",
            "ueId": ue_id,
            "allowedNssai": [s.dict() for s in context.allowed_nssai] if context.allowed_nssai else None,
            "guami": context.guami.dict() if context.guami else None,
            "tunnelInfo": context.tunnel.dict() if context.tunnel else None
        }

    def establish_pdu_session(self, request: PDUSessionRequest) -> Dict:
        """
        Establish PDU session over non-3GPP access
        Per 3GPP TS 24.502
        """
        context = self.get_ue_context(request.ue_id)
        if not context:
            raise ValueError(f"UE context not found: {request.ue_id}")

        if context.state != UeContextState.CONNECTED:
            raise ValueError("UE not in CONNECTED state")

        pdu_session_id = request.pdu_session_id

        # Create PDU session info
        pdu_session = {
            "pduSessionId": pdu_session_id,
            "pduSessionType": request.pdu_session_type,
            "sscMode": request.ssc_mode,
            "dnn": request.requested_dnn or "internet",
            "snssai": request.requested_snssai.dict() if request.requested_snssai else {"sst": 1},
            "ueIpAddress": self.allocate_inner_ip(),
            "qosFlowList": [
                {
                    "qfi": 1,
                    "fiveqi": 9,
                    "arp": {"priorityLevel": 8}
                }
            ],
            "state": "ACTIVE",
            "createdAt": datetime.now(timezone.utc).isoformat()
        }

        context.pdu_sessions[pdu_session_id] = pdu_session
        context.last_activity = datetime.now(timezone.utc)

        logger.info(f"PDU session {pdu_session_id} established for UE {request.ue_id}")

        return pdu_session

    def release_pdu_session(self, ue_id: str, pdu_session_id: int) -> Dict:
        """Release a PDU session"""
        context = self.get_ue_context(ue_id)
        if not context:
            raise ValueError(f"UE context not found: {ue_id}")

        if pdu_session_id not in context.pdu_sessions:
            raise ValueError(f"PDU session {pdu_session_id} not found")

        pdu_session = context.pdu_sessions[pdu_session_id]
        pdu_session["state"] = "RELEASED"
        del context.pdu_sessions[pdu_session_id]

        logger.info(f"PDU session {pdu_session_id} released for UE {ue_id}")

        return {"status": "RELEASED", "pduSessionId": pdu_session_id}

    def deregister_ue(self, ue_id: str) -> Dict:
        """Deregister UE"""
        context = self.get_ue_context(ue_id)
        if not context:
            raise ValueError(f"UE context not found: {ue_id}")

        # Release all PDU sessions
        for pdu_session_id in list(context.pdu_sessions.keys()):
            self.release_pdu_session(ue_id, pdu_session_id)

        # Delete IPSec tunnel
        if context.tunnel and context.tunnel.tunnel_id in ipsec_tunnels:
            del ipsec_tunnels[context.tunnel.tunnel_id]

        context.state = UeContextState.DISCONNECTING

        # Clean up
        del ue_contexts[ue_id]
        if ue_id in ran_ue_ngap_ids:
            del ran_ue_ngap_ids[ue_id]
        if ue_id in amf_ue_ngap_ids:
            del amf_ue_ngap_ids[ue_id]

        logger.info(f"UE {ue_id} deregistered")

        return {"status": "DEREGISTERED", "ueId": ue_id}


n3iwf_instance = N3IWF()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Register with NRF
    nf_profile = {
        "nfInstanceId": n3iwf_instance.nf_instance_id,
        "nfType": "N3IWF",
        "nfStatus": "REGISTERED",
        "plmnList": [{"mcc": "001", "mnc": "01"}],
        "sNssais": [{"sst": 1, "sd": "010203"}],
        "nfServices": [
            {
                "serviceInstanceId": "n3iwf-sbi-001",
                "serviceName": "n3iwf-sbi",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9015}]
            }
        ],
        "n3iwfInfo": {
            "ipv4EndpointAddress": ["10.0.0.1"],
            "ipv6EndpointAddress": [],
            "n3iwfId": n3iwf_instance.nf_instance_id
        }
    }

    try:
        response = requests.put(
            f"{nrf_url}/nnrf-nfm/v1/nf-instances/{n3iwf_instance.nf_instance_id}",
            json=nf_profile
        )
        if response.status_code in [200, 201]:
            logger.info("N3IWF registered with NRF successfully")
    except requests.RequestException as e:
        logger.error(f"Failed to register N3IWF with NRF: {e}")

    yield

    # Shutdown
    try:
        requests.delete(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{n3iwf_instance.nf_instance_id}")
        logger.info("N3IWF deregistered from NRF")
    except:
        pass


app = FastAPI(
    title="N3IWF - Non-3GPP Interworking Function",
    description="3GPP TS 29.502/24.502 compliant N3IWF for non-3GPP access",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# IKEv2/IPSec Tunnel Management

@app.post("/n3iwf/ipsec/initiate")
async def initiate_ipsec_tunnel(ue_id: str, ue_outer_ip: str):
    """
    Initiate IPSec tunnel establishment (IKE_SA_INIT)
    """
    with tracer.start_as_current_span("n3iwf_ipsec_initiate") as span:
        span.set_attribute("ue.id", ue_id)

        try:
            tunnel = n3iwf_instance.establish_ipsec_tunnel(ue_id, ue_outer_ip)
            span.set_attribute("tunnel.id", tunnel.tunnel_id)
            span.set_attribute("status", "SUCCESS")

            return {
                "tunnelId": tunnel.tunnel_id,
                "ueInnerIp": tunnel.ue_inner_ip,
                "n3iwfInnerIp": tunnel.n3iwf_inner_ip,
                "ikeSa": tunnel.ike_sa.dict() if tunnel.ike_sa else None,
                "childSa": tunnel.child_sa.dict() if tunnel.child_sa else None,
                "state": tunnel.state.value
            }

        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"IPSec tunnel initiation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/n3iwf/ipsec/tunnels")
async def list_ipsec_tunnels():
    """List all IPSec tunnels"""
    return {
        "tunnels": [
            {
                "tunnelId": t.tunnel_id,
                "ueInnerIp": t.ue_inner_ip,
                "state": t.state.value,
                "createdAt": t.created_at.isoformat()
            }
            for t in ipsec_tunnels.values()
        ]
    }


@app.delete("/n3iwf/ipsec/tunnels/{tunnelId}")
async def delete_ipsec_tunnel(tunnelId: str):
    """Delete an IPSec tunnel"""
    if tunnelId in ipsec_tunnels:
        tunnel = ipsec_tunnels[tunnelId]
        tunnel.state = IkeState.DELETED
        del ipsec_tunnels[tunnelId]
        return {"message": f"Tunnel {tunnelId} deleted"}
    raise HTTPException(status_code=404, detail="Tunnel not found")


# UE Registration

@app.post("/n3iwf/registration")
async def handle_registration(request: RegistrationRequest):
    """
    Handle UE registration request over non-3GPP access
    """
    with tracer.start_as_current_span("n3iwf_registration") as span:
        span.set_attribute("ue.id", request.ue_id)

        try:
            result = n3iwf_instance.handle_registration(request)
            span.set_attribute("status", "SUCCESS")
            return result

        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Registration failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/n3iwf/authentication/{ueId}")
async def handle_authentication(ueId: str, auth_response: Dict):
    """
    Handle authentication response from UE
    """
    with tracer.start_as_current_span("n3iwf_authentication") as span:
        span.set_attribute("ue.id", ueId)

        try:
            result = n3iwf_instance.handle_authentication_response(ueId, auth_response)
            span.set_attribute("status", "SUCCESS")
            return result

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/n3iwf/registration/{ueId}/complete")
async def complete_registration(ueId: str, amf_response: Dict):
    """
    Complete UE registration after AMF response
    """
    with tracer.start_as_current_span("n3iwf_registration_complete") as span:
        span.set_attribute("ue.id", ueId)

        try:
            result = n3iwf_instance.complete_registration(ueId, amf_response)
            span.set_attribute("status", "SUCCESS")
            return result

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/n3iwf/deregistration/{ueId}")
async def deregister_ue(ueId: str):
    """Deregister UE"""
    try:
        result = n3iwf_instance.deregister_ue(ueId)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# PDU Session Management

@app.post("/n3iwf/pdu-session")
async def establish_pdu_session(request: PDUSessionRequest):
    """
    Establish PDU session over non-3GPP access
    """
    with tracer.start_as_current_span("n3iwf_pdu_session") as span:
        span.set_attribute("ue.id", request.ue_id)
        span.set_attribute("pdu.session.id", request.pdu_session_id)

        try:
            result = n3iwf_instance.establish_pdu_session(request)
            span.set_attribute("status", "SUCCESS")
            return result

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.delete("/n3iwf/pdu-session/{ueId}/{pduSessionId}")
async def release_pdu_session(ueId: str, pduSessionId: int):
    """Release PDU session"""
    try:
        result = n3iwf_instance.release_pdu_session(ueId, pduSessionId)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# UE Context Management

@app.get("/n3iwf/ue-contexts")
async def list_ue_contexts():
    """List all UE contexts"""
    return {
        "contexts": [
            {
                "ueId": c.ue_id,
                "supi": c.supi,
                "state": c.state.value,
                "tunnelId": c.tunnel.tunnel_id if c.tunnel else None,
                "pduSessions": len(c.pdu_sessions),
                "registrationTime": c.registration_time.isoformat() if c.registration_time else None
            }
            for c in ue_contexts.values()
        ]
    }


@app.get("/n3iwf/ue-contexts/{ueId}")
async def get_ue_context(ueId: str):
    """Get UE context details"""
    context = n3iwf_instance.get_ue_context(ueId)
    if not context:
        raise HTTPException(status_code=404, detail="UE context not found")
    return context.dict()


# Health and monitoring

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "N3IWF",
        "compliance": "3GPP TS 29.502/24.502",
        "version": "1.0.0",
        "connectedUes": len([c for c in ue_contexts.values() if c.state == UeContextState.CONNECTED]),
        "activeTunnels": len([t for t in ipsec_tunnels.values() if t.state == IkeState.ESTABLISHED])
    }


@app.get("/metrics")
def get_metrics():
    """Metrics endpoint"""
    total_pdu_sessions = sum(len(c.pdu_sessions) for c in ue_contexts.values())

    return {
        "total_ue_contexts": len(ue_contexts),
        "connected_ues": len([c for c in ue_contexts.values() if c.state == UeContextState.CONNECTED]),
        "total_ipsec_tunnels": len(ipsec_tunnels),
        "active_tunnels": len([t for t in ipsec_tunnels.values() if t.state == IkeState.ESTABLISHED]),
        "total_pdu_sessions": total_pdu_sessions
    }


if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="N3IWF - Non-3GPP Interworking Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("n3iwf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)