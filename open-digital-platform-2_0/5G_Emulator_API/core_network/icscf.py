# File location: 5G_Emulator_API/core_network/icscf.py
# I-CSCF (Interrogating Call Session Control Function) - IMS Edge/Query Function
# Inspired by Kamailio ims_icscf module
# Reference: 3GPP TS 24.229, TS 29.228 (Cx interface), TS 29.229 (Cx/Dx)

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib
import asyncio
import logging
import httpx
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ICSCF")

app = FastAPI(
    title="I-CSCF - Interrogating Call Session Control Function",
    description="IMS I-CSCF per 3GPP TS 24.229 - Entry point query function",
    version="1.0.0"
)

# ============================================================================
# Data Models - Based on Kamailio ims_icscf structures
# ============================================================================

class DiameterResultCode(int, Enum):
    """Diameter result codes per RFC 6733 and 3GPP TS 29.229"""
    DIAMETER_SUCCESS = 2001
    DIAMETER_FIRST_REGISTRATION = 2001
    DIAMETER_SUBSEQUENT_REGISTRATION = 2002
    DIAMETER_UNREGISTERED_SERVICE = 2003
    DIAMETER_USER_UNKNOWN = 5001
    DIAMETER_IDENTITIES_DONT_MATCH = 5002
    DIAMETER_IDENTITY_NOT_REGISTERED = 5003
    DIAMETER_ROAMING_NOT_ALLOWED = 5004
    DIAMETER_IDENTITY_ALREADY_REGISTERED = 5005
    DIAMETER_AUTH_SCHEME_NOT_SUPPORTED = 5006
    DIAMETER_SERVER_SELECTION_ERROR = 5007

class UserAuthorizationType(int, Enum):
    """User Authorization Type per 3GPP TS 29.229"""
    REGISTRATION = 0
    DE_REGISTRATION = 1
    REGISTRATION_AND_CAPABILITIES = 2

class ServerCapabilities(BaseModel):
    """S-CSCF Capabilities per 3GPP TS 29.229"""
    mandatory_capabilities: List[int] = []
    optional_capabilities: List[int] = []
    server_name: Optional[str] = None

class ScscfEntry(BaseModel):
    """S-CSCF entry in I-CSCF's S-CSCF list"""
    scscf_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scscf_name: str  # SIP URI of S-CSCF
    scscf_uri: str   # Same as name typically
    priority: int = 0
    weight: int = 100
    capabilities: List[int] = []
    current_load: float = 0.0  # 0.0 to 1.0
    max_subscribers: int = 100000
    current_subscribers: int = 0
    is_active: bool = True
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)

class UARRequest(BaseModel):
    """User-Authorization-Request (UAR) - Cx interface"""
    public_identity: str  # IMPU
    private_identity: str  # IMPI
    visited_network_id: str = "ims.example.com"
    authorization_type: UserAuthorizationType = UserAuthorizationType.REGISTRATION
    routing_info_needed: bool = True

class UARResponse(BaseModel):
    """User-Authorization-Answer (UAA) - Cx interface"""
    result_code: int
    experimental_result_code: Optional[int] = None
    server_name: Optional[str] = None  # Assigned S-CSCF
    server_capabilities: Optional[ServerCapabilities] = None

class LIRRequest(BaseModel):
    """Location-Info-Request (LIR) - Cx interface"""
    public_identity: str  # IMPU
    originating_request: bool = False  # True if originating, False if terminating

class LIRResponse(BaseModel):
    """Location-Info-Answer (LIA) - Cx interface"""
    result_code: int
    experimental_result_code: Optional[int] = None
    server_name: Optional[str] = None  # S-CSCF serving the user
    server_capabilities: Optional[ServerCapabilities] = None
    wildcarded_public_identity: Optional[str] = None

class SIPRequest(BaseModel):
    """SIP Request model"""
    method: str
    request_uri: str
    from_uri: str
    to_uri: str
    call_id: str
    cseq: int
    via: List[str] = []
    route: List[str] = []
    contact: Optional[str] = None
    authorization: Optional[str] = None
    p_visited_network_id: Optional[str] = None

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
    route: Optional[List[str]] = None

# ============================================================================
# S-CSCF Pool Management - Based on Kamailio scscf_list.c
# ============================================================================

class ScscfPool:
    """
    S-CSCF Pool/List management
    Based on Kamailio ims_icscf/scscf_list.c

    Maintains list of available S-CSCFs and handles:
    - Capability matching
    - Load balancing
    - Failover
    """

    def __init__(self):
        self.scscf_list: Dict[str, ScscfEntry] = {}
        self.user_scscf_map: Dict[str, str] = {}  # IMPU -> scscf_id
        self._init_default_scscfs()

    def _init_default_scscfs(self):
        """Initialize with default S-CSCF entries"""
        default_scscf = ScscfEntry(
            scscf_name="sip:scscf.ims.example.com:6060",
            scscf_uri="sip:scscf.ims.example.com:6060",
            priority=0,
            weight=100,
            capabilities=[1, 2, 3],  # Example capabilities
            is_active=True
        )
        self.scscf_list[default_scscf.scscf_id] = default_scscf
        logger.info(f"[SCSCF-POOL] Initialized default S-CSCF: {default_scscf.scscf_uri}")

    def add_scscf(self, entry: ScscfEntry) -> ScscfEntry:
        """Add S-CSCF to pool"""
        self.scscf_list[entry.scscf_id] = entry
        logger.info(f"[SCSCF-POOL] Added S-CSCF: {entry.scscf_uri}")
        return entry

    def remove_scscf(self, scscf_id: str) -> bool:
        """Remove S-CSCF from pool"""
        if scscf_id in self.scscf_list:
            del self.scscf_list[scscf_id]
            return True
        return False

    def get_scscf_for_capabilities(
        self,
        mandatory_caps: List[int],
        optional_caps: List[int]
    ) -> Optional[ScscfEntry]:
        """
        Select S-CSCF based on capabilities
        Based on Kamailio I_scscf_select()
        """
        candidates = []

        for scscf in self.scscf_list.values():
            if not scscf.is_active:
                continue

            # Check mandatory capabilities
            if mandatory_caps:
                has_all_mandatory = all(cap in scscf.capabilities for cap in mandatory_caps)
                if not has_all_mandatory:
                    continue

            # Calculate score based on optional capabilities and load
            score = 0
            for cap in optional_caps:
                if cap in scscf.capabilities:
                    score += 10

            # Factor in load (lower is better)
            score -= int(scscf.current_load * 100)

            # Factor in weight
            score += scscf.weight

            candidates.append((scscf, score))

        if not candidates:
            return None

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def get_scscf_by_name(self, name: str) -> Optional[ScscfEntry]:
        """Get S-CSCF by name/URI"""
        for scscf in self.scscf_list.values():
            if scscf.scscf_name == name or scscf.scscf_uri == name:
                return scscf
        return None

    def get_scscf_for_user(self, impu: str) -> Optional[ScscfEntry]:
        """Get assigned S-CSCF for a user"""
        if impu in self.user_scscf_map:
            scscf_id = self.user_scscf_map[impu]
            return self.scscf_list.get(scscf_id)
        return None

    def assign_scscf_to_user(self, impu: str, scscf_id: str):
        """Assign S-CSCF to user"""
        self.user_scscf_map[impu] = scscf_id
        if scscf_id in self.scscf_list:
            self.scscf_list[scscf_id].current_subscribers += 1

    def unassign_scscf_from_user(self, impu: str):
        """Remove S-CSCF assignment for user"""
        if impu in self.user_scscf_map:
            scscf_id = self.user_scscf_map[impu]
            if scscf_id in self.scscf_list:
                self.scscf_list[scscf_id].current_subscribers -= 1
            del self.user_scscf_map[impu]

    def get_all_scscfs(self) -> List[ScscfEntry]:
        """Get all S-CSCFs in pool"""
        return list(self.scscf_list.values())

# Global S-CSCF pool
scscf_pool = ScscfPool()

# ============================================================================
# I-CSCF Configuration
# ============================================================================

class IcscfConfig:
    """I-CSCF configuration - based on Kamailio icscf.cfg"""
    icscf_uri: str = "sip:icscf.ims.example.com:5060"
    icscf_ip: str = "127.0.0.1"
    icscf_port: int = 5060
    realm: str = "ims.example.com"
    # HSS (Diameter Cx) configuration
    hss_uri: str = "http://localhost:9040"  # IMS HSS
    hss_realm: str = "ims.example.com"
    # Routing
    route_on_user_unknown: bool = False  # Route even if user unknown
    default_scscf_uri: str = "sip:scscf.ims.example.com:6060"
    # Cache
    scscf_entry_expiry: int = 300  # seconds

config = IcscfConfig()

# ============================================================================
# Cx Interface - HSS Communication (Based on Kamailio cxdx_uar.c, cxdx_lir.c)
# ============================================================================

async def send_uar_to_hss(uar: UARRequest) -> UARResponse:
    """
    Send User-Authorization-Request to HSS
    Based on Kamailio I_perform_user_authorization_request()

    Returns S-CSCF name or capabilities for selection
    """
    logger.info(f"[Cx-UAR] Public: {uar.public_identity}, Private: {uar.private_identity}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.hss_uri}/cx/uar",
                json=uar.dict(),
                timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                return UARResponse(**data)
    except Exception as e:
        logger.error(f"[Cx-UAR] HSS communication error: {e}")

    # Fallback: local decision
    # Check if user already has an assigned S-CSCF
    existing_scscf = scscf_pool.get_scscf_for_user(uar.public_identity)
    if existing_scscf:
        return UARResponse(
            result_code=DiameterResultCode.DIAMETER_SUBSEQUENT_REGISTRATION,
            server_name=existing_scscf.scscf_uri
        )

    # Select new S-CSCF
    selected = scscf_pool.get_scscf_for_capabilities([], [])
    if selected:
        return UARResponse(
            result_code=DiameterResultCode.DIAMETER_FIRST_REGISTRATION,
            server_name=selected.scscf_uri,
            server_capabilities=ServerCapabilities(
                mandatory_capabilities=[],
                optional_capabilities=selected.capabilities,
                server_name=selected.scscf_uri
            )
        )

    return UARResponse(
        result_code=DiameterResultCode.DIAMETER_SERVER_SELECTION_ERROR
    )

async def send_lir_to_hss(lir: LIRRequest) -> LIRResponse:
    """
    Send Location-Info-Request to HSS
    Based on Kamailio I_perform_location_information_request()

    Used for terminating calls to find which S-CSCF serves the user
    """
    logger.info(f"[Cx-LIR] Public: {lir.public_identity}, Originating: {lir.originating_request}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.hss_uri}/cx/lir",
                json=lir.dict(),
                timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                return LIRResponse(**data)
    except Exception as e:
        logger.error(f"[Cx-LIR] HSS communication error: {e}")

    # Fallback: check local cache
    existing_scscf = scscf_pool.get_scscf_for_user(lir.public_identity)
    if existing_scscf:
        return LIRResponse(
            result_code=DiameterResultCode.DIAMETER_SUCCESS,
            server_name=existing_scscf.scscf_uri
        )

    return LIRResponse(
        result_code=DiameterResultCode.DIAMETER_USER_UNKNOWN
    )

# ============================================================================
# SIP Message Handlers - Based on Kamailio I-CSCF routing
# ============================================================================

@app.post("/sip/register")
async def handle_register(request: SIPRequest):
    """
    Handle REGISTER at I-CSCF
    I-CSCF queries HSS to determine which S-CSCF should handle registration

    Flow:
    1. Receive REGISTER from P-CSCF
    2. Send UAR to HSS (Cx interface)
    3. Get S-CSCF assignment from UAA
    4. Forward REGISTER to assigned S-CSCF
    """
    logger.info(f"[REGISTER] From: {request.from_uri}")

    # Extract IMPI from Authorization header or From
    impi = request.from_uri  # Simplified - real impl parses Authorization
    impu = request.to_uri or request.from_uri

    # Extract visited network from P-Visited-Network-ID
    visited_network = request.p_visited_network_id or config.realm

    # Step 1: Send UAR to HSS
    uar = UARRequest(
        public_identity=impu,
        private_identity=impi,
        visited_network_id=visited_network,
        authorization_type=UserAuthorizationType.REGISTRATION
    )

    uaa = await send_uar_to_hss(uar)

    # Step 2: Process UAA response
    if uaa.result_code in [
        DiameterResultCode.DIAMETER_FIRST_REGISTRATION,
        DiameterResultCode.DIAMETER_SUBSEQUENT_REGISTRATION
    ]:
        # Got S-CSCF assignment
        scscf_uri = uaa.server_name or config.default_scscf_uri

        # Record assignment
        scscf = scscf_pool.get_scscf_by_name(scscf_uri)
        if scscf:
            scscf_pool.assign_scscf_to_user(impu, scscf.scscf_id)

        logger.info(f"[REGISTER] Forwarding to S-CSCF: {scscf_uri}")

        # In production: forward request to S-CSCF
        # For simulation: return 100 Trying
        return SIPResponse(
            status_code=100,
            reason="Trying",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq,
            route=[f"<{scscf_uri};lr>"]
        )

    elif uaa.result_code == DiameterResultCode.DIAMETER_USER_UNKNOWN:
        logger.warning(f"[REGISTER] User unknown: {impu}")
        return SIPResponse(
            status_code=404,
            reason="Not Found",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

    elif uaa.result_code == DiameterResultCode.DIAMETER_ROAMING_NOT_ALLOWED:
        logger.warning(f"[REGISTER] Roaming not allowed: {impu}")
        return SIPResponse(
            status_code=403,
            reason="Forbidden - Roaming Not Allowed",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

    else:
        logger.error(f"[REGISTER] S-CSCF selection failed: {uaa.result_code}")
        return SIPResponse(
            status_code=500,
            reason="Server Selection Error",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

@app.post("/sip/invite")
async def handle_invite(request: SIPRequest):
    """
    Handle INVITE at I-CSCF (Terminating leg)
    I-CSCF queries HSS to find S-CSCF serving the called party

    Flow:
    1. Receive INVITE from external network or S-CSCF
    2. Send LIR to HSS
    3. Get S-CSCF from LIA
    4. Forward to S-CSCF
    """
    logger.info(f"[INVITE] From: {request.from_uri} To: {request.to_uri}")

    # Extract called party IMPU
    called_impu = request.to_uri

    # Step 1: Send LIR to HSS
    lir = LIRRequest(
        public_identity=called_impu,
        originating_request=False  # Terminating
    )

    lia = await send_lir_to_hss(lir)

    # Step 2: Process LIA response
    if lia.result_code == DiameterResultCode.DIAMETER_SUCCESS and lia.server_name:
        scscf_uri = lia.server_name

        logger.info(f"[INVITE] Forwarding to S-CSCF: {scscf_uri}")

        return SIPResponse(
            status_code=100,
            reason="Trying",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq,
            route=[f"<{scscf_uri};lr>"]
        )

    elif lia.result_code == DiameterResultCode.DIAMETER_USER_UNKNOWN:
        logger.warning(f"[INVITE] User not found: {called_impu}")
        return SIPResponse(
            status_code=404,
            reason="User Not Found",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

    elif lia.result_code == DiameterResultCode.DIAMETER_IDENTITY_NOT_REGISTERED:
        logger.warning(f"[INVITE] User not registered: {called_impu}")
        return SIPResponse(
            status_code=480,
            reason="Temporarily Unavailable",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

    else:
        logger.error(f"[INVITE] Location query failed: {lia.result_code}")
        return SIPResponse(
            status_code=500,
            reason="Internal Server Error",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

@app.post("/sip/message")
async def handle_sip_message(request: SIPRequest):
    """Generic SIP message handler"""
    method = request.method.upper()

    if method == "REGISTER":
        return await handle_register(request)
    elif method == "INVITE":
        return await handle_invite(request)
    else:
        # For other methods, do LIR to find S-CSCF
        lir = LIRRequest(
            public_identity=request.to_uri,
            originating_request=False
        )
        lia = await send_lir_to_hss(lir)

        if lia.server_name:
            return SIPResponse(
                status_code=100,
                reason="Trying",
                via=request.via,
                from_uri=request.from_uri,
                to_uri=request.to_uri,
                call_id=request.call_id,
                cseq=request.cseq,
                route=[f"<{lia.server_name};lr>"]
            )

        return SIPResponse(
            status_code=404,
            reason="Not Found",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

# ============================================================================
# Cx Interface Endpoints (for HSS callbacks)
# ============================================================================

@app.post("/cx/uar", response_model=UARResponse)
async def process_uar(uar: UARRequest):
    """Process UAR request (for testing/simulation)"""
    return await send_uar_to_hss(uar)

@app.post("/cx/lir", response_model=LIRResponse)
async def process_lir(lir: LIRRequest):
    """Process LIR request (for testing/simulation)"""
    return await send_lir_to_hss(lir)

# ============================================================================
# S-CSCF Pool Management APIs
# ============================================================================

@app.get("/scscf-pool")
async def list_scscfs():
    """List all S-CSCFs in pool"""
    scscfs = scscf_pool.get_all_scscfs()
    return {
        "count": len(scscfs),
        "scscfs": [
            {
                "scscf_id": s.scscf_id,
                "uri": s.scscf_uri,
                "priority": s.priority,
                "weight": s.weight,
                "capabilities": s.capabilities,
                "is_active": s.is_active,
                "current_subscribers": s.current_subscribers,
                "load": s.current_load
            }
            for s in scscfs
        ]
    }

@app.post("/scscf-pool")
async def add_scscf(
    scscf_uri: str,
    priority: int = 0,
    weight: int = 100,
    capabilities: List[int] = []
):
    """Add S-CSCF to pool"""
    entry = ScscfEntry(
        scscf_name=scscf_uri,
        scscf_uri=scscf_uri,
        priority=priority,
        weight=weight,
        capabilities=capabilities
    )
    scscf_pool.add_scscf(entry)
    return {"status": "added", "scscf_id": entry.scscf_id}

@app.delete("/scscf-pool/{scscf_id}")
async def remove_scscf(scscf_id: str):
    """Remove S-CSCF from pool"""
    if scscf_pool.remove_scscf(scscf_id):
        return {"status": "removed"}
    raise HTTPException(status_code=404, detail="S-CSCF not found")

@app.put("/scscf-pool/{scscf_id}/load")
async def update_scscf_load(scscf_id: str, load: float):
    """Update S-CSCF load (for load balancing)"""
    if scscf_id in scscf_pool.scscf_list:
        scscf_pool.scscf_list[scscf_id].current_load = max(0.0, min(1.0, load))
        return {"status": "updated"}
    raise HTTPException(status_code=404, detail="S-CSCF not found")

@app.get("/user-assignments")
async def list_user_assignments():
    """List user to S-CSCF assignments"""
    return {
        "count": len(scscf_pool.user_scscf_map),
        "assignments": scscf_pool.user_scscf_map
    }

# ============================================================================
# Health and Status
# ============================================================================

@app.get("/")
async def root():
    """I-CSCF status endpoint"""
    return {
        "nf_type": "I-CSCF",
        "nf_name": "Interrogating-CSCF",
        "status": "running",
        "description": "IMS I-CSCF - Query/Entry point for registration and call routing",
        "version": "1.0.0",
        "icscf_uri": config.icscf_uri,
        "realm": config.realm,
        "hss_uri": config.hss_uri,
        "statistics": {
            "scscf_pool_size": len(scscf_pool.scscf_list),
            "user_assignments": len(scscf_pool.user_scscf_map)
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "I-CSCF", "compliance": "3GPP TS 24.229", "version": "1.0.0"}

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="I-CSCF - Interrogating Call Session Control Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("icscf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)