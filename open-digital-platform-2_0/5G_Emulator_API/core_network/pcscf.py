# File location: 5G_Emulator_API/core_network/pcscf.py
# P-CSCF (Proxy Call Session Control Function) - IMS Entry Point
# Inspired by Kamailio ims_registrar_pcscf module
# Reference: 3GPP TS 24.229 (IMS Call Control Protocol)

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib
import hmac
import asyncio
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PCSCF")

app = FastAPI(
    title="P-CSCF - Proxy Call Session Control Function",
    description="IMS P-CSCF per 3GPP TS 24.229 - Entry point for UE into IMS network",
    version="1.0.0"
)

# ============================================================================
# Data Models - Based on Kamailio ims_registrar_pcscf structures
# ============================================================================

class RegistrationState(str, Enum):
    """Registration states per Kamailio PCONTACT_* flags"""
    NOT_REGISTERED = "not_registered"
    REGISTERED = "registered"
    REG_PENDING = "reg_pending"
    REG_PENDING_AAR = "reg_pending_aar"  # Pending AAR to PCRF
    DEREGISTERED = "deregistered"

class SecurityMechanism(str, Enum):
    """Security mechanisms supported by P-CSCF"""
    IPSEC_3GPP = "ipsec-3gpp"
    TLS = "tls"
    DIGEST = "digest"
    DIGEST_AKAV1_MD5 = "digest-akav1-md5"
    DIGEST_AKAV2_MD5 = "digest-akav2-md5"

class ServiceRouteInfo(BaseModel):
    """Service route for routing requests through IMS"""
    uri: str
    priority: int = 0

class PublicIdentity(BaseModel):
    """Public User Identity (IMPU)"""
    impu: str  # sip:user@domain or tel:+123456
    is_default: bool = False
    barring_indication: bool = False

class SecurityAssociation(BaseModel):
    """IPSec Security Association for P-CSCF"""
    spi_c: int  # SPI client
    spi_s: int  # SPI server
    port_c: int  # Client protected port
    port_s: int  # Server protected port
    ealg: str = "aes-cbc"  # Encryption algorithm
    alg: str = "hmac-sha1-96"  # Integrity algorithm
    ck: str = ""  # Cipher key
    ik: str = ""  # Integrity key

class ContactBinding(BaseModel):
    """P-CSCF Contact binding - based on Kamailio pcontact structure"""
    contact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    aor: str  # Address of Record (Contact URI)
    received_host: str  # IP where request was received from
    received_port: int
    received_proto: str = "UDP"
    via_host: str
    via_port: int
    via_proto: str = "UDP"
    expires: datetime
    state: RegistrationState = RegistrationState.NOT_REGISTERED
    public_ids: List[PublicIdentity] = []
    service_routes: List[ServiceRouteInfo] = []
    security_assoc: Optional[SecurityAssociation] = None
    security_mechanism: SecurityMechanism = SecurityMechanism.DIGEST
    reg_state: RegistrationState = RegistrationState.NOT_REGISTERED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)

class SIPRequest(BaseModel):
    """SIP Request model for API simulation"""
    method: str  # REGISTER, INVITE, etc.
    request_uri: str
    from_uri: str
    to_uri: str
    call_id: str
    cseq: int
    via: List[str] = []
    contact: Optional[str] = None
    expires: Optional[int] = None
    authorization: Optional[str] = None
    security_client: Optional[str] = None
    require: Optional[str] = None
    supported: Optional[str] = None
    p_access_network_info: Optional[str] = None  # 3GPP access info
    content_type: Optional[str] = None
    body: Optional[str] = None

class SIPResponse(BaseModel):
    """SIP Response model"""
    status_code: int
    reason: str
    via: List[str] = []
    from_uri: str
    to_uri: str
    call_id: str
    cseq: int
    contact: Optional[str] = None
    www_authenticate: Optional[str] = None
    service_route: Optional[List[str]] = None
    p_associated_uri: Optional[List[str]] = None
    security_server: Optional[str] = None
    expires: Optional[int] = None
    path: Optional[str] = None

class RxSessionInfo(BaseModel):
    """Rx interface session to PCRF (for QoS)"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    impu: str
    ip_address: str
    media_type: str = "audio"
    bandwidth_ul: int = 64000  # bps
    bandwidth_dl: int = 64000
    qci: int = 1  # QoS Class Identifier for VoLTE
    flow_status: str = "enabled"
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ============================================================================
# In-Memory Storage - Based on Kamailio usrloc_pcscf
# ============================================================================

class PCscfUsrLoc:
    """User Location storage for P-CSCF - similar to Kamailio ims_usrloc_pcscf"""

    def __init__(self):
        self.contacts: Dict[str, ContactBinding] = {}  # keyed by contact_id
        self.aor_index: Dict[str, str] = {}  # aor -> contact_id
        self.rx_sessions: Dict[str, RxSessionInfo] = {}  # Rx sessions to PCRF

    def save_contact(self, contact: ContactBinding) -> ContactBinding:
        """Save or update contact binding"""
        # Check for existing by AOR
        if contact.aor in self.aor_index:
            existing_id = self.aor_index[contact.aor]
            contact.contact_id = existing_id

        contact.last_modified = datetime.utcnow()
        self.contacts[contact.contact_id] = contact
        self.aor_index[contact.aor] = contact.contact_id

        logger.info(f"[USRLOC] Saved contact: {contact.aor} -> {contact.contact_id}")
        return contact

    def lookup_contact(self, aor: str) -> Optional[ContactBinding]:
        """Lookup contact by AOR"""
        if aor in self.aor_index:
            contact_id = self.aor_index[aor]
            return self.contacts.get(contact_id)
        return None

    def lookup_by_received(self, host: str, port: int) -> Optional[ContactBinding]:
        """Lookup by received host/port - used for routing"""
        for contact in self.contacts.values():
            if contact.received_host == host and contact.received_port == port:
                return contact
        return None

    def delete_contact(self, contact_id: str) -> bool:
        """Delete contact binding"""
        if contact_id in self.contacts:
            contact = self.contacts[contact_id]
            if contact.aor in self.aor_index:
                del self.aor_index[contact.aor]
            del self.contacts[contact_id]
            logger.info(f"[USRLOC] Deleted contact: {contact_id}")
            return True
        return False

    def get_all_contacts(self) -> List[ContactBinding]:
        """Get all registered contacts"""
        return list(self.contacts.values())

# Global storage
usrloc = PCscfUsrLoc()

# ============================================================================
# P-CSCF Configuration
# ============================================================================

class PCscfConfig:
    """P-CSCF configuration parameters - from Kamailio pcscf.cfg"""
    pcscf_uri: str = "sip:pcscf.ims.example.com:5060"
    pcscf_ip: str = "127.0.0.1"
    pcscf_port: int = 5060
    pending_reg_expires: int = 30  # Timeout for pending registrations
    default_expires: int = 3600
    min_expires: int = 60
    max_expires: int = 7200
    icscf_uri: str = "sip:icscf.ims.example.com:5060"
    realm: str = "ims.example.com"
    # Security
    ipsec_enabled: bool = True
    tls_enabled: bool = True
    # Rx interface (to PCRF)
    pcrf_enabled: bool = True
    pcrf_uri: str = "http://localhost:9006"  # PCF in our emulator

config = PCscfConfig()

# ============================================================================
# P-CSCF Core Functions - Based on Kamailio save.c, service_routes.c
# ============================================================================

def generate_nonce() -> str:
    """Generate nonce for authentication challenge"""
    return hashlib.sha256(f"{uuid.uuid4()}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:32]

def parse_via_header(via: str) -> Dict[str, Any]:
    """Parse Via header to extract received/rport"""
    result = {"host": "", "port": 5060, "proto": "UDP", "received": None, "rport": None}
    # Simple parsing - production would use full SIP parser
    parts = via.split(";")
    if parts:
        main = parts[0].strip()
        if "UDP" in main.upper():
            result["proto"] = "UDP"
        elif "TCP" in main.upper():
            result["proto"] = "TCP"
        elif "TLS" in main.upper():
            result["proto"] = "TLS"

        # Extract host:port
        if " " in main:
            _, hostport = main.rsplit(" ", 1)
            if ":" in hostport:
                result["host"], port_str = hostport.rsplit(":", 1)
                result["port"] = int(port_str) if port_str.isdigit() else 5060
            else:
                result["host"] = hostport

        # Extract received and rport parameters
        for param in parts[1:]:
            param = param.strip()
            if param.startswith("received="):
                result["received"] = param.split("=", 1)[1]
            elif param.startswith("rport="):
                rport = param.split("=", 1)[1]
                if rport.isdigit():
                    result["rport"] = int(rport)

    return result

def extract_public_ids_from_p_associated_uri(header: str) -> List[PublicIdentity]:
    """Extract public identities from P-Associated-URI header"""
    public_ids = []
    if header:
        uris = header.split(",")
        for i, uri in enumerate(uris):
            uri = uri.strip().strip("<>")
            public_ids.append(PublicIdentity(
                impu=uri,
                is_default=(i == 0)
            ))
    return public_ids

def build_service_route_header(routes: List[ServiceRouteInfo]) -> List[str]:
    """Build Service-Route headers for response"""
    return [f"<{r.uri}>;lr" for r in routes]

async def send_rx_aar(contact: ContactBinding) -> bool:
    """
    Send AA-Request to PCRF via Rx interface
    This is where P-CSCF requests QoS for IMS session
    Based on Kamailio ims_qos module
    """
    if not config.pcrf_enabled:
        return True

    rx_session = RxSessionInfo(
        impu=contact.public_ids[0].impu if contact.public_ids else contact.aor,
        ip_address=contact.received_host,
        media_type="signaling",  # Registration is signaling
        qci=5  # IMS signaling QCI
    )

    usrloc.rx_sessions[rx_session.session_id] = rx_session
    logger.info(f"[Rx] Created AAR session {rx_session.session_id} for {rx_session.impu}")

    # In real implementation, this would make Diameter Rx request to PCRF
    return True

# ============================================================================
# SIP Message Handlers - Based on Kamailio routing logic
# ============================================================================

@app.post("/sip/register", response_model=SIPResponse)
async def handle_register(request: SIPRequest, client_ip: str = "127.0.0.1", client_port: int = 5060):
    """
    Handle SIP REGISTER request at P-CSCF
    Implements: Kamailio pcscf_save() and pcscf_save_pending()

    P-CSCF acts as proxy:
    1. Validates request
    2. Adds Path header (for routing responses back)
    3. Forwards to I-CSCF
    4. Stores contact binding on 200 OK
    """
    logger.info(f"[REGISTER] From: {request.from_uri}, Contact: {request.contact}")

    # Parse Via to get received info
    via_info = parse_via_header(request.via[0]) if request.via else {
        "host": client_ip, "port": client_port, "proto": "UDP"
    }

    # Determine received host/port (NAT handling)
    received_host = via_info.get("received") or via_info.get("host") or client_ip
    received_port = via_info.get("rport") or via_info.get("port") or client_port

    # Check for Security-Client header (IPSec negotiation)
    security_mechanism = SecurityMechanism.DIGEST
    security_assoc = None

    if request.security_client:
        if "ipsec-3gpp" in request.security_client.lower():
            security_mechanism = SecurityMechanism.IPSEC_3GPP
            # In production, parse security parameters and set up IPSec SA
            logger.info(f"[SECURITY] IPSec-3GPP requested by UE")

    # Check if this is a de-registration (Expires: 0)
    expires = request.expires if request.expires is not None else config.default_expires

    if expires == 0:
        # De-registration
        existing = usrloc.lookup_contact(request.contact or request.from_uri)
        if existing:
            usrloc.delete_contact(existing.contact_id)
            logger.info(f"[REGISTER] De-registration successful for {request.from_uri}")
            return SIPResponse(
                status_code=200,
                reason="OK",
                via=request.via,
                from_uri=request.from_uri,
                to_uri=request.to_uri,
                call_id=request.call_id,
                cseq=request.cseq,
                expires=0
            )

    # Enforce expires limits
    expires = max(config.min_expires, min(expires, config.max_expires))

    # Create pending registration
    # In production, P-CSCF forwards to I-CSCF and waits for response
    contact = ContactBinding(
        aor=request.contact or request.from_uri,
        received_host=received_host,
        received_port=received_port,
        received_proto=via_info.get("proto", "UDP"),
        via_host=via_info.get("host", client_ip),
        via_port=via_info.get("port", client_port),
        via_proto=via_info.get("proto", "UDP"),
        expires=datetime.utcnow() + timedelta(seconds=expires),
        state=RegistrationState.REG_PENDING,
        security_mechanism=security_mechanism,
        security_assoc=security_assoc
    )

    # Save pending registration
    contact = usrloc.save_contact(contact)

    # Simulate successful registration (in production, this comes from S-CSCF via I-CSCF)
    # Update with public identities and service routes
    contact.public_ids = [
        PublicIdentity(impu=request.from_uri, is_default=True)
    ]
    contact.service_routes = [
        ServiceRouteInfo(uri=f"sip:orig@scscf.ims.example.com:6060;lr", priority=0)
    ]
    contact.state = RegistrationState.REGISTERED
    contact.reg_state = RegistrationState.REGISTERED

    usrloc.save_contact(contact)

    # Send AAR to PCRF for QoS
    await send_rx_aar(contact)

    logger.info(f"[REGISTER] Registration successful: {contact.aor}")

    # Build response
    response = SIPResponse(
        status_code=200,
        reason="OK",
        via=request.via,
        from_uri=request.from_uri,
        to_uri=request.to_uri,
        call_id=request.call_id,
        cseq=request.cseq,
        contact=contact.aor,
        expires=expires,
        service_route=build_service_route_header(contact.service_routes),
        p_associated_uri=[f"<{p.impu}>" for p in contact.public_ids],
        path=f"<sip:{config.pcscf_ip}:{config.pcscf_port};lr>"
    )

    # Add Security-Server if IPSec was negotiated
    if security_mechanism == SecurityMechanism.IPSEC_3GPP:
        response.security_server = "ipsec-3gpp;alg=hmac-sha-1-96;ealg=aes-cbc;spi-c=12345;spi-s=12346;port-c=5061;port-s=5062"

    return response

@app.post("/sip/invite", response_model=SIPResponse)
async def handle_invite(request: SIPRequest, client_ip: str = "127.0.0.1"):
    """
    Handle SIP INVITE at P-CSCF (Mobile Originating call)
    Implements service route enforcement per 3GPP TS 24.229
    """
    logger.info(f"[INVITE] From: {request.from_uri} To: {request.to_uri}")

    # Check if caller is registered
    contact = usrloc.lookup_contact(request.contact or request.from_uri)

    if not contact or contact.state != RegistrationState.REGISTERED:
        logger.warning(f"[INVITE] Caller not registered: {request.from_uri}")
        return SIPResponse(
            status_code=403,
            reason="Forbidden - Not Registered",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

    # Verify service routes are being followed (pcscf_follows_service_routes)
    # In production, check Route headers match stored service routes

    # P-Asserted-Identity handling
    # P-CSCF asserts the identity based on registration
    asserted_identity = contact.public_ids[0].impu if contact.public_ids else request.from_uri

    logger.info(f"[INVITE] Asserting identity: {asserted_identity}")

    # Create Rx session for media QoS
    if request.body and "m=audio" in request.body:
        rx_session = RxSessionInfo(
            impu=asserted_identity,
            ip_address=contact.received_host,
            media_type="audio",
            qci=1,  # VoLTE QCI
            bandwidth_ul=64000,
            bandwidth_dl=64000
        )
        usrloc.rx_sessions[rx_session.session_id] = rx_session
        logger.info(f"[Rx] Created media session {rx_session.session_id}")

    # Forward to S-CSCF via service route
    return SIPResponse(
        status_code=100,
        reason="Trying",
        via=request.via,
        from_uri=request.from_uri,
        to_uri=request.to_uri,
        call_id=request.call_id,
        cseq=request.cseq
    )

@app.post("/sip/message", response_model=SIPResponse)
async def handle_sip_message(request: SIPRequest, client_ip: str = "127.0.0.1"):
    """Generic SIP message handler - routes based on method"""
    method = request.method.upper()

    if method == "REGISTER":
        return await handle_register(request, client_ip)
    elif method == "INVITE":
        return await handle_invite(request, client_ip)
    elif method == "OPTIONS":
        return SIPResponse(
            status_code=200,
            reason="OK",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )
    else:
        # Check registration for other methods
        contact = usrloc.lookup_contact(request.from_uri)
        if not contact or contact.state != RegistrationState.REGISTERED:
            return SIPResponse(
                status_code=403,
                reason="Forbidden",
                via=request.via,
                from_uri=request.from_uri,
                to_uri=request.to_uri,
                call_id=request.call_id,
                cseq=request.cseq
            )

        return SIPResponse(
            status_code=100,
            reason="Trying",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

# ============================================================================
# P-CSCF Management APIs
# ============================================================================

@app.get("/contacts")
async def list_contacts():
    """List all registered contacts"""
    contacts = usrloc.get_all_contacts()
    return {
        "count": len(contacts),
        "contacts": [
            {
                "contact_id": c.contact_id,
                "aor": c.aor,
                "state": c.state.value,
                "received": f"{c.received_host}:{c.received_port}",
                "expires": c.expires.isoformat(),
                "public_ids": [p.impu for p in c.public_ids],
                "security": c.security_mechanism.value
            }
            for c in contacts
        ]
    }

@app.get("/contacts/{contact_id}")
async def get_contact(contact_id: str):
    """Get specific contact details"""
    if contact_id in usrloc.contacts:
        return usrloc.contacts[contact_id]
    raise HTTPException(status_code=404, detail="Contact not found")

@app.delete("/contacts/{contact_id}")
async def delete_contact(contact_id: str):
    """Administrative contact deletion"""
    if usrloc.delete_contact(contact_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Contact not found")

@app.get("/rx-sessions")
async def list_rx_sessions():
    """List active Rx sessions (QoS sessions to PCRF)"""
    return {
        "count": len(usrloc.rx_sessions),
        "sessions": list(usrloc.rx_sessions.values())
    }

@app.post("/rx-sessions/{session_id}/terminate")
async def terminate_rx_session(session_id: str):
    """Terminate Rx session (send STR to PCRF)"""
    if session_id in usrloc.rx_sessions:
        del usrloc.rx_sessions[session_id]
        logger.info(f"[Rx] Terminated session {session_id}")
        return {"status": "terminated"}
    raise HTTPException(status_code=404, detail="Session not found")

# ============================================================================
# Security Association Management
# ============================================================================

@app.post("/security/ipsec/setup")
async def setup_ipsec_sa(contact_id: str, spi_c: int, spi_s: int, ck: str, ik: str):
    """
    Set up IPSec Security Association for a contact
    Based on Kamailio ims_ipsec_pcscf module
    """
    if contact_id not in usrloc.contacts:
        raise HTTPException(status_code=404, detail="Contact not found")

    contact = usrloc.contacts[contact_id]
    contact.security_assoc = SecurityAssociation(
        spi_c=spi_c,
        spi_s=spi_s,
        port_c=5061,  # Would be dynamically allocated
        port_s=5062,
        ck=ck,
        ik=ik
    )
    contact.security_mechanism = SecurityMechanism.IPSEC_3GPP
    usrloc.save_contact(contact)

    logger.info(f"[IPSec] SA established for {contact.aor}")
    return {"status": "established", "spi_c": spi_c, "spi_s": spi_s}

@app.delete("/security/ipsec/{contact_id}")
async def teardown_ipsec_sa(contact_id: str):
    """Tear down IPSec SA"""
    if contact_id not in usrloc.contacts:
        raise HTTPException(status_code=404, detail="Contact not found")

    contact = usrloc.contacts[contact_id]
    contact.security_assoc = None
    contact.security_mechanism = SecurityMechanism.DIGEST
    usrloc.save_contact(contact)

    return {"status": "removed"}

# ============================================================================
# Health and Status
# ============================================================================

@app.get("/")
async def root():
    """P-CSCF status endpoint"""
    return {
        "nf_type": "P-CSCF",
        "nf_name": "Proxy-CSCF",
        "status": "running",
        "description": "IMS P-CSCF - Entry point for UE into IMS network",
        "version": "1.0.0",
        "pcscf_uri": config.pcscf_uri,
        "realm": config.realm,
        "features": {
            "ipsec": config.ipsec_enabled,
            "tls": config.tls_enabled,
            "rx_interface": config.pcrf_enabled
        },
        "statistics": {
            "registered_contacts": len(usrloc.contacts),
            "active_rx_sessions": len(usrloc.rx_sessions)
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "P-CSCF", "compliance": "3GPP TS 24.229", "version": "1.0.0"}

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="P-CSCF - Proxy Call Session Control Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("pcscf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)