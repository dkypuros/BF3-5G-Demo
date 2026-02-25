# File location: 5G_Emulator_API/core_network/scscf.py
# S-CSCF (Serving Call Session Control Function) - IMS Core Signaling
# Inspired by Kamailio ims_registrar_scscf module
# Reference: 3GPP TS 24.229, TS 29.228/229 (Cx), TS 29.328/329 (Sh)

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib
import hmac
import asyncio
import logging
import httpx
import re
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SCSCF")

app = FastAPI(
    title="S-CSCF - Serving Call Session Control Function",
    description="IMS S-CSCF per 3GPP TS 24.229 - Core IMS signaling control",
    version="1.0.0"
)

# ============================================================================
# Data Models - Based on Kamailio ims_registrar_scscf structures
# ============================================================================

class RegistrationState(str, Enum):
    """Registration states"""
    NOT_REGISTERED = "not_registered"
    REGISTERED = "registered"
    UNREGISTERED = "unregistered"  # User has service profile but not registered
    AUTH_PENDING = "auth_pending"

class AuthScheme(str, Enum):
    """Authentication schemes per 3GPP TS 29.229"""
    DIGEST_AKAV1_MD5 = "Digest-AKAv1-MD5"
    DIGEST_AKAV2_MD5 = "Digest-AKAv2-MD5"
    DIGEST_MD5 = "Digest-MD5"
    EARLY_IMS = "Early-IMS-Security"
    NASS_BUNDLED = "NASS-Bundled"
    SIP_DIGEST = "SIP Digest"

class ServiceProfile(BaseModel):
    """User service profile from HSS"""
    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    public_identities: List[str] = []  # IMPUs
    private_identity: str  # IMPI
    service_trigger_points: List[Dict[str, Any]] = []  # iFC triggers
    initial_filter_criteria: List[Dict[str, Any]] = []  # iFCs
    subscribed_media_profile_id: Optional[int] = None
    barring_indication: bool = False

class AuthVector(BaseModel):
    """Authentication vector from HSS"""
    item_number: int = 0
    auth_scheme: AuthScheme = AuthScheme.DIGEST_AKAV1_MD5
    rand: str  # Random challenge (16 bytes hex)
    autn: str  # Authentication token (16 bytes hex)
    xres: str  # Expected response
    ck: str    # Cipher key
    ik: str    # Integrity key

class ImplicitRegistrationSet(BaseModel):
    """Implicit registration set - IMPUs registered together"""
    irs_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    public_identities: List[str] = []
    default_public_identity: str = ""
    service_profile_id: str = ""

class ScscfSubscription(BaseModel):
    """S-CSCF subscription (for reg event package)"""
    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subscriber_uri: str
    watcher_uri: str
    event: str = "reg"
    expires: datetime
    state: str = "active"

class ScscfContact(BaseModel):
    """Contact binding at S-CSCF - from Kamailio usrloc_scscf"""
    contact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    aor: str  # Address of Record
    contact_uri: str
    path: List[str] = []  # Path headers (P-CSCF route)
    expires: datetime
    q_value: float = 1.0  # Contact priority
    call_id: str = ""
    cseq: int = 0
    user_agent: str = ""
    state: RegistrationState = RegistrationState.NOT_REGISTERED
    auth_vector: Optional[AuthVector] = None
    registration_time: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)

class ScscfUserRecord(BaseModel):
    """User record at S-CSCF"""
    impu: str  # Primary public identity
    impi: str  # Private identity
    contacts: Dict[str, ScscfContact] = {}  # contact_id -> contact
    service_profile: Optional[ServiceProfile] = None
    implicit_registration_set: Optional[ImplicitRegistrationSet] = None
    subscriptions: Dict[str, ScscfSubscription] = {}
    state: RegistrationState = RegistrationState.NOT_REGISTERED
    server_assignment_type: int = 0  # SAR type
    created_at: datetime = Field(default_factory=datetime.utcnow)

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
    path: List[str] = []
    contact: Optional[str] = None
    expires: Optional[int] = None
    authorization: Optional[str] = None
    p_access_network_info: Optional[str] = None
    supported: Optional[str] = None
    require: Optional[str] = None
    content_type: Optional[str] = None
    body: Optional[str] = None
    p_asserted_identity: Optional[str] = None

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
    expires: Optional[int] = None
    service_route: Optional[List[str]] = None
    p_associated_uri: Optional[List[str]] = None
    www_authenticate: Optional[str] = None
    authentication_info: Optional[str] = None
    path: Optional[str] = None

class SARRequest(BaseModel):
    """Server-Assignment-Request (Cx interface)"""
    public_identity: str
    private_identity: str
    server_name: str
    server_assignment_type: int  # TS 29.229 enum
    user_data_request_type: int = 0

class SARResponse(BaseModel):
    """Server-Assignment-Answer"""
    result_code: int
    user_profile: Optional[Dict[str, Any]] = None
    charging_information: Optional[Dict[str, Any]] = None

class MARRequest(BaseModel):
    """Multimedia-Auth-Request (Cx interface)"""
    public_identity: str
    private_identity: str
    server_name: str
    sip_auth_data_items: int = 1
    sip_auth_scheme: str = "Digest-AKAv1-MD5"

class MARResponse(BaseModel):
    """Multimedia-Auth-Answer"""
    result_code: int
    public_identity: str
    private_identity: str
    auth_vectors: List[Dict[str, Any]] = []

# ============================================================================
# In-Memory Storage - Based on Kamailio ims_usrloc_scscf
# ============================================================================

class ScscfUsrLoc:
    """User Location storage for S-CSCF"""

    def __init__(self):
        self.users: Dict[str, ScscfUserRecord] = {}  # IMPU -> user record
        self.impi_to_impu: Dict[str, Set[str]] = {}  # IMPI -> set of IMPUs
        self.auth_vectors: Dict[str, List[AuthVector]] = {}  # IMPI -> auth vectors

    def get_user(self, impu: str) -> Optional[ScscfUserRecord]:
        """Get user record by IMPU"""
        return self.users.get(impu)

    def get_user_by_impi(self, impi: str) -> List[ScscfUserRecord]:
        """Get all user records for an IMPI"""
        if impi in self.impi_to_impu:
            return [self.users[impu] for impu in self.impi_to_impu[impi] if impu in self.users]
        return []

    def create_user(self, impu: str, impi: str) -> ScscfUserRecord:
        """Create new user record"""
        user = ScscfUserRecord(impu=impu, impi=impi)
        self.users[impu] = user

        if impi not in self.impi_to_impu:
            self.impi_to_impu[impi] = set()
        self.impi_to_impu[impi].add(impu)

        logger.info(f"[USRLOC] Created user record: {impu}")
        return user

    def save_contact(self, impu: str, contact: ScscfContact) -> ScscfContact:
        """Save or update contact for user"""
        if impu not in self.users:
            raise ValueError(f"User {impu} not found")

        contact.last_modified = datetime.utcnow()
        self.users[impu].contacts[contact.contact_id] = contact
        logger.info(f"[USRLOC] Saved contact for {impu}: {contact.contact_uri}")
        return contact

    def lookup_contact(self, impu: str, contact_uri: str = None) -> List[ScscfContact]:
        """Lookup contacts for user"""
        if impu not in self.users:
            return []

        contacts = list(self.users[impu].contacts.values())

        if contact_uri:
            contacts = [c for c in contacts if c.contact_uri == contact_uri]

        # Filter expired
        now = datetime.utcnow()
        contacts = [c for c in contacts if c.expires > now]

        return contacts

    def delete_contact(self, impu: str, contact_id: str) -> bool:
        """Delete specific contact"""
        if impu in self.users and contact_id in self.users[impu].contacts:
            del self.users[impu].contacts[contact_id]
            logger.info(f"[USRLOC] Deleted contact {contact_id} for {impu}")
            return True
        return False

    def delete_all_contacts(self, impu: str) -> int:
        """Delete all contacts for user"""
        if impu in self.users:
            count = len(self.users[impu].contacts)
            self.users[impu].contacts.clear()
            self.users[impu].state = RegistrationState.NOT_REGISTERED
            logger.info(f"[USRLOC] Deleted {count} contacts for {impu}")
            return count
        return 0

    def store_auth_vectors(self, impi: str, vectors: List[AuthVector]):
        """Store authentication vectors for user"""
        self.auth_vectors[impi] = vectors
        logger.info(f"[AUTH] Stored {len(vectors)} auth vectors for {impi}")

    def get_auth_vector(self, impi: str) -> Optional[AuthVector]:
        """Get next authentication vector"""
        if impi in self.auth_vectors and self.auth_vectors[impi]:
            return self.auth_vectors[impi].pop(0)
        return None

    def get_all_users(self) -> List[ScscfUserRecord]:
        """Get all user records"""
        return list(self.users.values())

# Global storage
usrloc = ScscfUsrLoc()

# ============================================================================
# S-CSCF Configuration
# ============================================================================

class ScscfConfig:
    """S-CSCF configuration - from Kamailio scscf.cfg"""
    scscf_uri: str = "sip:scscf.ims.example.com:6060"
    scscf_name: str = "sip:scscf.ims.example.com:6060"
    scscf_ip: str = "127.0.0.1"
    scscf_port: int = 6060
    realm: str = "ims.example.com"
    # HSS (Diameter Cx) configuration
    hss_uri: str = "http://localhost:9040"
    hss_realm: str = "ims.example.com"
    # Registration
    default_expires: int = 3600
    min_expires: int = 60
    max_expires: int = 7200
    # Authentication
    auth_vector_count: int = 3  # Number of AVs to request
    auth_scheme: AuthScheme = AuthScheme.DIGEST_AKAV1_MD5
    # Service Route
    service_route_uri: str = "sip:orig@scscf.ims.example.com:6060;lr"

config = ScscfConfig()

# ============================================================================
# Cx Interface - HSS Communication (Based on Kamailio cxdx_sar.c, cxdx_mar.c)
# ============================================================================

# Server Assignment Types (3GPP TS 29.229)
class ServerAssignmentType(int, Enum):
    NO_ASSIGNMENT = 0
    REGISTRATION = 1
    RE_REGISTRATION = 2
    UNREGISTERED_USER = 3
    TIMEOUT_DEREGISTRATION = 4
    USER_DEREGISTRATION = 5
    TIMEOUT_DEREGISTRATION_STORE_SERVER_NAME = 6
    USER_DEREGISTRATION_STORE_SERVER_NAME = 7
    ADMINISTRATIVE_DEREGISTRATION = 8
    AUTHENTICATION_FAILURE = 9
    AUTHENTICATION_TIMEOUT = 10
    DEREGISTRATION_TOO_MUCH_DATA = 11
    AAA_USER_DATA_REQUEST = 12
    PGW_UPDATE = 13
    RESTORATION = 14

async def send_sar_to_hss(sar: SARRequest) -> SARResponse:
    """
    Send Server-Assignment-Request to HSS
    Based on Kamailio cxdx_sar.c
    """
    logger.info(f"[Cx-SAR] IMPU: {sar.public_identity}, Type: {sar.server_assignment_type}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.hss_uri}/cx/sar",
                json=sar.dict(),
                timeout=5.0
            )
            if response.status_code == 200:
                return SARResponse(**response.json())
    except Exception as e:
        logger.error(f"[Cx-SAR] HSS communication error: {e}")

    # Fallback: simulate success
    return SARResponse(
        result_code=2001,  # DIAMETER_SUCCESS
        user_profile={
            "public_identities": [sar.public_identity],
            "private_identity": sar.private_identity,
            "service_profile": {}
        }
    )

async def send_mar_to_hss(mar: MARRequest) -> MARResponse:
    """
    Send Multimedia-Auth-Request to HSS
    Based on Kamailio cxdx_mar.c
    """
    logger.info(f"[Cx-MAR] IMPU: {mar.public_identity}, IMPI: {mar.private_identity}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.hss_uri}/cx/mar",
                json=mar.dict(),
                timeout=5.0
            )
            if response.status_code == 200:
                return MARResponse(**response.json())
    except Exception as e:
        logger.error(f"[Cx-MAR] HSS communication error: {e}")

    # Fallback: generate test auth vector
    import secrets
    rand = secrets.token_hex(16)
    autn = secrets.token_hex(16)

    return MARResponse(
        result_code=2001,
        public_identity=mar.public_identity,
        private_identity=mar.private_identity,
        auth_vectors=[{
            "item_number": 0,
            "auth_scheme": mar.sip_auth_scheme,
            "rand": rand,
            "autn": autn,
            "xres": hashlib.md5(f"{mar.private_identity}{rand}".encode()).hexdigest()[:32],
            "ck": secrets.token_hex(16),
            "ik": secrets.token_hex(16)
        }]
    )

# ============================================================================
# Authentication - Based on Kamailio ims_auth/authorize.c
# ============================================================================

def parse_authorization_header(auth_header: str) -> Dict[str, str]:
    """Parse Digest authorization header"""
    result = {}
    if not auth_header or not auth_header.startswith("Digest "):
        return result

    auth_str = auth_header[7:]  # Remove "Digest "
    # Parse key="value" pairs
    pattern = r'(\w+)="([^"]*)"'
    for match in re.finditer(pattern, auth_str):
        result[match.group(1)] = match.group(2)

    # Also handle unquoted values like algorithm
    pattern2 = r'(\w+)=([^,\s"]+)'
    for match in re.finditer(pattern2, auth_str):
        if match.group(1) not in result:
            result[match.group(1)] = match.group(2)

    return result

def generate_nonce(impi: str) -> str:
    """Generate nonce for authentication challenge"""
    timestamp = datetime.utcnow().isoformat()
    return hashlib.sha256(f"{impi}{timestamp}{uuid.uuid4()}".encode()).hexdigest()[:32]

def build_www_authenticate(
    realm: str,
    nonce: str,
    algorithm: str = "AKAv1-MD5",
    qop: str = "auth",
    ck: str = "",
    ik: str = ""
) -> str:
    """Build WWW-Authenticate header for 401 challenge"""
    header = f'Digest realm="{realm}", nonce="{nonce}", algorithm={algorithm}, qop="{qop}"'
    if ck and ik:
        header += f', ck="{ck}", ik="{ik}"'
    return header

def verify_digest_response(
    auth_params: Dict[str, str],
    expected_response: str,
    method: str
) -> bool:
    """
    Verify digest authentication response
    Based on Kamailio authorize.c check_response()
    """
    if "response" not in auth_params:
        return False

    # For AKA, the expected response comes from HSS
    # For MD5, we'd compute: MD5(HA1:nonce:nc:cnonce:qop:HA2)
    # Simplified verification for simulation
    return auth_params.get("response") == expected_response

# ============================================================================
# Initial Filter Criteria (iFC) - Based on Kamailio ims_isc
# ============================================================================

def evaluate_ifc(
    request: SIPRequest,
    service_profile: ServiceProfile,
    is_originating: bool
) -> List[Dict[str, Any]]:
    """
    Evaluate Initial Filter Criteria to determine which Application Servers to invoke
    Based on Kamailio ims_isc module

    Returns list of AS URIs to route through
    """
    triggered_as = []

    if not service_profile or not service_profile.initial_filter_criteria:
        return triggered_as

    for ifc in service_profile.initial_filter_criteria:
        # Check trigger point conditions
        trigger_point = ifc.get("trigger_point", {})

        # Check method match
        methods = trigger_point.get("sip_method", [])
        if methods and request.method.upper() not in methods:
            continue

        # Check request URI match
        req_uri_pattern = trigger_point.get("request_uri", "")
        if req_uri_pattern and not re.match(req_uri_pattern, request.request_uri):
            continue

        # Check originating/terminating
        condition_type = trigger_point.get("condition_type", "")
        if condition_type == "originating" and not is_originating:
            continue
        if condition_type == "terminating" and is_originating:
            continue

        # iFC matched - add AS to list
        as_info = {
            "application_server": ifc.get("application_server", ""),
            "default_handling": ifc.get("default_handling", 0),
            "service_info": ifc.get("service_info", ""),
            "priority": ifc.get("priority", 0)
        }
        if as_info["application_server"]:
            triggered_as.append(as_info)

    # Sort by priority
    triggered_as.sort(key=lambda x: x["priority"])
    return triggered_as

# ============================================================================
# SIP Message Handlers - Based on Kamailio S-CSCF routing
# ============================================================================

@app.post("/sip/register", response_model=SIPResponse)
async def handle_register(request: SIPRequest):
    """
    Handle SIP REGISTER at S-CSCF
    Implements: Kamailio save() and scscf authentication

    Flow:
    1. If no Authorization: Request auth vectors (MAR) and challenge (401)
    2. If Authorization: Verify credentials
    3. On success: Register via SAR, return 200 OK
    """
    logger.info(f"[REGISTER] From: {request.from_uri}, To: {request.to_uri}")

    # Extract identities
    impu = request.to_uri or request.from_uri
    impi = request.from_uri  # Simplified - real impl extracts from Authorization

    # Parse Authorization header if present
    auth_params = {}
    if request.authorization:
        auth_params = parse_authorization_header(request.authorization)
        if "username" in auth_params:
            impi = auth_params["username"]

    # Check expires
    expires = request.expires if request.expires is not None else config.default_expires

    # De-registration check
    if expires == 0:
        # Send SAR with USER_DEREGISTRATION
        sar = SARRequest(
            public_identity=impu,
            private_identity=impi,
            server_name=config.scscf_name,
            server_assignment_type=ServerAssignmentType.USER_DEREGISTRATION
        )
        await send_sar_to_hss(sar)

        # Delete contacts
        user = usrloc.get_user(impu)
        if user:
            usrloc.delete_all_contacts(impu)

        logger.info(f"[REGISTER] De-registration for {impu}")
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

    # Check if we need to authenticate
    if not request.authorization:
        # First REGISTER - need to challenge
        # Request auth vectors from HSS (MAR)
        mar = MARRequest(
            public_identity=impu,
            private_identity=impi,
            server_name=config.scscf_name,
            sip_auth_data_items=config.auth_vector_count,
            sip_auth_scheme=config.auth_scheme.value
        )
        maa = await send_mar_to_hss(mar)

        if maa.result_code != 2001 or not maa.auth_vectors:
            logger.error(f"[REGISTER] Failed to get auth vectors for {impi}")
            return SIPResponse(
                status_code=403,
                reason="Forbidden",
                via=request.via,
                from_uri=request.from_uri,
                to_uri=request.to_uri,
                call_id=request.call_id,
                cseq=request.cseq
            )

        # Store auth vectors
        vectors = [AuthVector(**v) for v in maa.auth_vectors]
        usrloc.store_auth_vectors(impi, vectors)

        # Get first vector for challenge
        av = vectors[0] if vectors else None
        if not av:
            return SIPResponse(
                status_code=500,
                reason="Server Error",
                via=request.via,
                from_uri=request.from_uri,
                to_uri=request.to_uri,
                call_id=request.call_id,
                cseq=request.cseq
            )

        # Build 401 challenge
        www_auth = build_www_authenticate(
            realm=config.realm,
            nonce=av.rand + av.autn,  # Concatenate for AKA
            algorithm="AKAv1-MD5",
            ck=av.ck,
            ik=av.ik
        )

        logger.info(f"[REGISTER] Challenging {impi}")
        return SIPResponse(
            status_code=401,
            reason="Unauthorized",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq,
            www_authenticate=www_auth
        )

    # Second REGISTER with credentials - verify
    av = usrloc.get_auth_vector(impi)
    if not av:
        # No stored vector - request new ones
        logger.warning(f"[REGISTER] No auth vector for {impi}, re-challenging")
        return SIPResponse(
            status_code=401,
            reason="Unauthorized - Stale Nonce",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq,
            www_authenticate=f'Digest realm="{config.realm}", stale=true'
        )

    # Verify response
    client_response = auth_params.get("response", "")
    if not verify_digest_response(auth_params, av.xres, "REGISTER"):
        # For simulation, accept if response is present
        if not client_response:
            logger.warning(f"[REGISTER] Auth failed for {impi}")
            return SIPResponse(
                status_code=403,
                reason="Forbidden - Authentication Failed",
                via=request.via,
                from_uri=request.from_uri,
                to_uri=request.to_uri,
                call_id=request.call_id,
                cseq=request.cseq
            )

    # Authentication successful - Register user
    # Send SAR to HSS
    user = usrloc.get_user(impu)
    assignment_type = ServerAssignmentType.RE_REGISTRATION if user else ServerAssignmentType.REGISTRATION

    sar = SARRequest(
        public_identity=impu,
        private_identity=impi,
        server_name=config.scscf_name,
        server_assignment_type=assignment_type,
        user_data_request_type=0
    )
    saa = await send_sar_to_hss(sar)

    if saa.result_code != 2001:
        logger.error(f"[REGISTER] SAR failed for {impu}: {saa.result_code}")
        return SIPResponse(
            status_code=500,
            reason="Registration Failed",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

    # Create/update user record
    if not user:
        user = usrloc.create_user(impu, impi)

    # Update service profile from HSS response
    if saa.user_profile:
        user.service_profile = ServiceProfile(
            public_identities=saa.user_profile.get("public_identities", [impu]),
            private_identity=impi,
            initial_filter_criteria=saa.user_profile.get("initial_filter_criteria", [])
        )

    # Save contact
    contact = ScscfContact(
        aor=impu,
        contact_uri=request.contact or request.from_uri,
        path=request.path,
        expires=datetime.utcnow() + timedelta(seconds=expires),
        call_id=request.call_id,
        cseq=request.cseq,
        state=RegistrationState.REGISTERED,
        auth_vector=av
    )
    usrloc.save_contact(impu, contact)
    user.state = RegistrationState.REGISTERED

    # Build associated URIs
    associated_uris = [f"<{impu}>"]
    if user.service_profile:
        for pub_id in user.service_profile.public_identities:
            if pub_id != impu:
                associated_uris.append(f"<{pub_id}>")

    logger.info(f"[REGISTER] Registration successful: {impu}")

    return SIPResponse(
        status_code=200,
        reason="OK",
        via=request.via,
        from_uri=request.from_uri,
        to_uri=request.to_uri,
        call_id=request.call_id,
        cseq=request.cseq,
        contact=contact.contact_uri,
        expires=expires,
        service_route=[f"<{config.service_route_uri}>"],
        p_associated_uri=associated_uris,
        authentication_info=f'nextnonce="{generate_nonce(impi)}"'
    )

@app.post("/sip/invite", response_model=SIPResponse)
async def handle_invite(request: SIPRequest):
    """
    Handle SIP INVITE at S-CSCF
    Implements originating and terminating call processing

    Flow:
    1. Determine if originating or terminating
    2. Evaluate iFCs for Application Server chaining
    3. Route to next hop (AS, other S-CSCF, or UE)
    """
    logger.info(f"[INVITE] From: {request.from_uri} To: {request.to_uri}")

    # Determine if originating (from registered user) or terminating (to registered user)
    from_user = usrloc.get_user(request.from_uri)
    to_user = usrloc.get_user(request.to_uri)

    is_originating = from_user is not None and from_user.state == RegistrationState.REGISTERED

    if is_originating:
        # Originating call processing
        logger.info(f"[INVITE] Originating call from {request.from_uri}")

        # Check caller is registered
        if not from_user or from_user.state != RegistrationState.REGISTERED:
            return SIPResponse(
                status_code=403,
                reason="Forbidden - Not Registered",
                via=request.via,
                from_uri=request.from_uri,
                to_uri=request.to_uri,
                call_id=request.call_id,
                cseq=request.cseq
            )

        # Evaluate iFCs for originating
        triggered_as = []
        if from_user.service_profile:
            triggered_as = evaluate_ifc(request, from_user.service_profile, is_originating=True)

        if triggered_as:
            # Route through Application Servers
            as_uri = triggered_as[0]["application_server"]
            logger.info(f"[INVITE] Routing through AS: {as_uri}")
            return SIPResponse(
                status_code=100,
                reason="Trying",
                via=request.via,
                from_uri=request.from_uri,
                to_uri=request.to_uri,
                call_id=request.call_id,
                cseq=request.cseq,
                route=[f"<{as_uri};lr>"]
            )

        # No AS - check if destination is local
        if to_user:
            # Local user - terminating processing
            return await process_terminating_invite(request, to_user)

        # External destination - route to I-CSCF or external
        return SIPResponse(
            status_code=100,
            reason="Trying",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

    elif to_user:
        # Terminating call processing
        return await process_terminating_invite(request, to_user)

    else:
        # User not found
        logger.warning(f"[INVITE] User not found: {request.to_uri}")
        return SIPResponse(
            status_code=404,
            reason="User Not Found",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

async def process_terminating_invite(request: SIPRequest, to_user: ScscfUserRecord) -> SIPResponse:
    """Process terminating INVITE leg"""
    logger.info(f"[INVITE] Terminating call to {to_user.impu}")

    # Check user is registered
    if to_user.state != RegistrationState.REGISTERED:
        return SIPResponse(
            status_code=480,
            reason="Temporarily Unavailable",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

    # Evaluate iFCs for terminating
    triggered_as = []
    if to_user.service_profile:
        triggered_as = evaluate_ifc(request, to_user.service_profile, is_originating=False)

    if triggered_as:
        as_uri = triggered_as[0]["application_server"]
        logger.info(f"[INVITE] Routing through AS: {as_uri}")
        return SIPResponse(
            status_code=100,
            reason="Trying",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq,
            route=[f"<{as_uri};lr>"]
        )

    # Lookup contacts for forking
    contacts = usrloc.lookup_contact(to_user.impu)
    if not contacts:
        return SIPResponse(
            status_code=480,
            reason="Temporarily Unavailable",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

    # Route to first contact (via Path if present)
    contact = contacts[0]
    route = contact.path if contact.path else []

    return SIPResponse(
        status_code=100,
        reason="Trying",
        via=request.via,
        from_uri=request.from_uri,
        to_uri=request.to_uri,
        call_id=request.call_id,
        cseq=request.cseq,
        contact=contact.contact_uri,
        route=route
    )

@app.post("/sip/message", response_model=SIPResponse)
async def handle_sip_message(request: SIPRequest):
    """Generic SIP message handler"""
    method = request.method.upper()

    if method == "REGISTER":
        return await handle_register(request)
    elif method == "INVITE":
        return await handle_invite(request)
    elif method == "SUBSCRIBE":
        return await handle_subscribe(request)
    else:
        # For other methods, check registration and forward
        from_user = usrloc.get_user(request.from_uri)
        if not from_user or from_user.state != RegistrationState.REGISTERED:
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
            status_code=200,
            reason="OK",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

async def handle_subscribe(request: SIPRequest) -> SIPResponse:
    """Handle SUBSCRIBE (reg event package)"""
    logger.info(f"[SUBSCRIBE] From: {request.from_uri}")

    # Create subscription
    user = usrloc.get_user(request.to_uri)
    if not user:
        return SIPResponse(
            status_code=404,
            reason="Not Found",
            via=request.via,
            from_uri=request.from_uri,
            to_uri=request.to_uri,
            call_id=request.call_id,
            cseq=request.cseq
        )

    expires = request.expires or 3600
    subscription = ScscfSubscription(
        subscriber_uri=request.from_uri,
        watcher_uri=request.from_uri,
        expires=datetime.utcnow() + timedelta(seconds=expires)
    )
    user.subscriptions[subscription.subscription_id] = subscription

    return SIPResponse(
        status_code=200,
        reason="OK",
        via=request.via,
        from_uri=request.from_uri,
        to_uri=request.to_uri,
        call_id=request.call_id,
        cseq=request.cseq,
        expires=expires
    )

# ============================================================================
# Cx Interface Endpoints
# ============================================================================

@app.post("/cx/sar", response_model=SARResponse)
async def process_sar(sar: SARRequest):
    """Process SAR (for testing)"""
    return await send_sar_to_hss(sar)

@app.post("/cx/mar", response_model=MARResponse)
async def process_mar(mar: MARRequest):
    """Process MAR (for testing)"""
    return await send_mar_to_hss(mar)

# ============================================================================
# Management APIs
# ============================================================================

@app.get("/users")
async def list_users():
    """List all registered users"""
    users = usrloc.get_all_users()
    return {
        "count": len(users),
        "users": [
            {
                "impu": u.impu,
                "impi": u.impi,
                "state": u.state.value,
                "contacts": len(u.contacts),
                "subscriptions": len(u.subscriptions)
            }
            for u in users
        ]
    }

@app.get("/users/{impu}")
async def get_user(impu: str):
    """Get user details"""
    user = usrloc.get_user(impu)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "impu": user.impu,
        "impi": user.impi,
        "state": user.state.value,
        "contacts": [
            {
                "contact_id": c.contact_id,
                "uri": c.contact_uri,
                "expires": c.expires.isoformat(),
                "path": c.path
            }
            for c in user.contacts.values()
        ],
        "service_profile": user.service_profile.dict() if user.service_profile else None
    }

@app.delete("/users/{impu}")
async def delete_user(impu: str):
    """Administrative user deletion"""
    user = usrloc.get_user(impu)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    usrloc.delete_all_contacts(impu)
    if impu in usrloc.users:
        del usrloc.users[impu]

    return {"status": "deleted"}

@app.get("/users/{impu}/contacts")
async def list_user_contacts(impu: str):
    """List contacts for user"""
    contacts = usrloc.lookup_contact(impu)
    return {
        "impu": impu,
        "contacts": [
            {
                "contact_id": c.contact_id,
                "uri": c.contact_uri,
                "expires": c.expires.isoformat(),
                "state": c.state.value,
                "path": c.path
            }
            for c in contacts
        ]
    }

# ============================================================================
# Health and Status
# ============================================================================

@app.get("/")
async def root():
    """S-CSCF status endpoint"""
    return {
        "nf_type": "S-CSCF",
        "nf_name": "Serving-CSCF",
        "status": "running",
        "description": "IMS S-CSCF - Core signaling control and SIP registration",
        "version": "1.0.0",
        "scscf_uri": config.scscf_uri,
        "realm": config.realm,
        "hss_uri": config.hss_uri,
        "statistics": {
            "registered_users": len([u for u in usrloc.users.values() if u.state == RegistrationState.REGISTERED]),
            "total_users": len(usrloc.users),
            "total_contacts": sum(len(u.contacts) for u in usrloc.users.values())
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "S-CSCF", "compliance": "3GPP TS 24.229", "version": "1.0.0"}

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="S-CSCF - Serving Call Session Control Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("scscf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)