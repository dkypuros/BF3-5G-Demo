# File location: 5G_Emulator_API/core_network/ims_hss.py
# IMS HSS (Home Subscriber Server) - IMS User Database
# Implements Cx/Dx interfaces per 3GPP TS 29.228/29.229
# Inspired by Kamailio CDP (Cx Diameter Protocol) and Open5GS HSS

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib
import hmac
import secrets
import logging
import struct
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IMS-HSS")

app = FastAPI(
    title="IMS HSS - Home Subscriber Server",
    description="IMS HSS per 3GPP TS 29.228/29.229 - Cx/Dx Interface",
    version="1.0.0"
)

# ============================================================================
# Data Models - Based on 3GPP TS 29.228/29.229
# ============================================================================

class DiameterResultCode(int, Enum):
    """Diameter result codes"""
    DIAMETER_SUCCESS = 2001
    DIAMETER_FIRST_REGISTRATION = 2001
    DIAMETER_SUBSEQUENT_REGISTRATION = 2002
    DIAMETER_UNREGISTERED_SERVICE = 2003
    DIAMETER_SUCCESS_SERVER_NAME_NOT_STORED = 2004
    DIAMETER_ERROR_USER_UNKNOWN = 5001
    DIAMETER_ERROR_IDENTITIES_DONT_MATCH = 5002
    DIAMETER_ERROR_IDENTITY_NOT_REGISTERED = 5003
    DIAMETER_ERROR_ROAMING_NOT_ALLOWED = 5004
    DIAMETER_ERROR_IDENTITY_ALREADY_REGISTERED = 5005
    DIAMETER_ERROR_AUTH_SCHEME_NOT_SUPPORTED = 5006
    DIAMETER_ERROR_IN_ASSIGNMENT_TYPE = 5007
    DIAMETER_ERROR_TOO_MUCH_DATA = 5008
    DIAMETER_ERROR_NOT_SUPPORTED_USER_DATA = 5009

class UserAuthorizationType(int, Enum):
    """User Authorization Type per 3GPP TS 29.229"""
    REGISTRATION = 0
    DE_REGISTRATION = 1
    REGISTRATION_AND_CAPABILITIES = 2

class ServerAssignmentType(int, Enum):
    """Server Assignment Type per 3GPP TS 29.229"""
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

class AuthScheme(str, Enum):
    """SIP Authentication Scheme"""
    DIGEST_AKAV1_MD5 = "Digest-AKAv1-MD5"
    DIGEST_AKAV2_MD5 = "Digest-AKAv2-MD5"
    DIGEST_MD5 = "Digest-MD5"
    EARLY_IMS = "Early-IMS-Security"
    NASS_BUNDLED = "NASS-Bundled"
    SIP_DIGEST = "SIP Digest"

class RegistrationState(str, Enum):
    """IMS Registration State"""
    NOT_REGISTERED = "not_registered"
    REGISTERED = "registered"
    UNREGISTERED = "unregistered"
    PENDING = "pending"

class ServiceProfileData(BaseModel):
    """Service Profile for IMS user"""
    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_name: str = "default"
    # Initial Filter Criteria (iFCs)
    initial_filter_criteria: List[Dict[str, Any]] = []
    # Subscribed media profile
    subscribed_media_profile_id: Optional[int] = None
    # Shared iFCs
    shared_ifc_set_ids: List[int] = []

class PublicIdentity(BaseModel):
    """Public User Identity (IMPU)"""
    impu: str  # sip:user@domain or tel:+123456
    is_default: bool = False
    barring_indication: bool = False
    display_name: str = ""
    service_profile_id: str = ""
    wildcarded_psi: Optional[str] = None  # For wildcarded Public Service Identities

class ImplicitRegistrationSet(BaseModel):
    """Implicit Registration Set - IMPUs registered together"""
    irs_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    public_identities: List[str] = []  # List of IMPUs
    default_impu: str = ""

class ChargingInfo(BaseModel):
    """Charging information"""
    primary_charging_collection_function: str = ""
    secondary_charging_collection_function: str = ""
    primary_event_charging_function: str = ""
    secondary_event_charging_function: str = ""

class ImsSubscription(BaseModel):
    """IMS Subscription data - main subscriber record"""
    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # Private Identity (IMPI)
    impi: str
    # Authentication data
    k: str  # Secret key (128 bits hex)
    op: Optional[str] = None  # Operator key
    opc: Optional[str] = None  # Computed OPc
    amf: str = "8000"  # Authentication Management Field
    sqn: int = 0  # Sequence number
    # Public Identities
    public_identities: Dict[str, PublicIdentity] = {}
    # Service Profiles
    service_profiles: Dict[str, ServiceProfileData] = {}
    # Implicit Registration Sets
    implicit_registration_sets: Dict[str, ImplicitRegistrationSet] = {}
    # Assigned S-CSCF
    scscf_name: Optional[str] = None
    scscf_capabilities: List[int] = []
    # Registration state
    registration_state: RegistrationState = RegistrationState.NOT_REGISTERED
    registration_timestamp: Optional[datetime] = None
    # Charging
    charging_info: Optional[ChargingInfo] = None
    # Roaming
    visited_network_ids: List[str] = []
    roaming_allowed: bool = True
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)

# ============================================================================
# Cx Interface Request/Response Models
# ============================================================================

class UARRequest(BaseModel):
    """User-Authorization-Request (UAR)"""
    public_identity: str
    private_identity: str
    visited_network_id: str = ""
    authorization_type: int = 0
    ue_srvcc_capability: Optional[bool] = None

class UARResponse(BaseModel):
    """User-Authorization-Answer (UAA)"""
    result_code: int
    experimental_result_code: Optional[int] = None
    server_name: Optional[str] = None
    server_capabilities: Optional[Dict[str, Any]] = None
    ue_srvcc_capability: Optional[bool] = None

class LIRRequest(BaseModel):
    """Location-Info-Request (LIR)"""
    public_identity: str
    originating_request: bool = False
    ue_srvcc_capability: Optional[bool] = None

class LIRResponse(BaseModel):
    """Location-Info-Answer (LIA)"""
    result_code: int
    experimental_result_code: Optional[int] = None
    server_name: Optional[str] = None
    server_capabilities: Optional[Dict[str, Any]] = None
    wildcarded_public_identity: Optional[str] = None
    lia_flags: int = 0

class MARRequest(BaseModel):
    """Multimedia-Auth-Request (MAR)"""
    public_identity: str
    private_identity: str
    server_name: str
    sip_auth_data_items: int = 1
    sip_auth_scheme: str = "Digest-AKAv1-MD5"

class MARResponse(BaseModel):
    """Multimedia-Auth-Answer (MAA)"""
    result_code: int
    public_identity: str
    private_identity: str
    auth_vectors: List[Dict[str, Any]] = []

class SARRequest(BaseModel):
    """Server-Assignment-Request (SAR)"""
    public_identity: str
    private_identity: str
    server_name: str
    server_assignment_type: int
    user_data_request_type: int = 0
    user_data_already_available: int = 0

class SARResponse(BaseModel):
    """Server-Assignment-Answer (SAA)"""
    result_code: int
    user_profile: Optional[Dict[str, Any]] = None
    charging_information: Optional[Dict[str, Any]] = None
    associated_identities: Optional[List[str]] = None

class RTRRequest(BaseModel):
    """Registration-Termination-Request (RTR)"""
    public_identities: List[str]
    private_identity: str
    deregistration_reason: int

class RTRResponse(BaseModel):
    """Registration-Termination-Answer (RTA)"""
    result_code: int
    associated_identities: Optional[List[str]] = None

class PPRRequest(BaseModel):
    """Push-Profile-Request (PPR)"""
    user_identity: str
    user_data: Dict[str, Any]
    charging_information: Optional[Dict[str, Any]] = None

class PPRResponse(BaseModel):
    """Push-Profile-Answer (PPA)"""
    result_code: int

# ============================================================================
# Milenage Algorithm Implementation - Same as 4G HSS
# ============================================================================

def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings"""
    return bytes(x ^ y for x, y in zip(a, b))

def aes_encrypt(key: bytes, data: bytes) -> bytes:
    """AES-128 encryption (simplified - uses hashlib for simulation)"""
    # In production, use proper AES
    # For simulation, use a deterministic transform
    combined = key + data
    return hashlib.md5(combined).digest()

def milenage_f1(k: bytes, rand: bytes, sqn: bytes, amf: bytes, op: bytes) -> bytes:
    """Milenage f1 function - Network authentication (MAC-A)"""
    opc = aes_encrypt(k, op)
    temp = aes_encrypt(k, xor_bytes(rand, opc))

    in1 = sqn + amf + sqn + amf
    rijndael_input = xor_bytes(xor_bytes(temp, opc), in1)

    # Rotate and XOR for f1
    c1 = bytes([0] * 16)
    r1 = 8
    rotated = rijndael_input[r1:] + rijndael_input[:r1]
    out1 = aes_encrypt(k, xor_bytes(rotated, c1))
    out1 = xor_bytes(out1, opc)

    return out1[:8]  # MAC-A

def milenage_f2345(k: bytes, rand: bytes, op: bytes) -> tuple:
    """Milenage f2, f3, f4, f5 functions"""
    opc = aes_encrypt(k, op)
    temp = aes_encrypt(k, xor_bytes(rand, opc))

    # f2 - RES
    c2 = bytes([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    r2 = 0
    rotated = temp[r2:] + temp[:r2]
    out2 = aes_encrypt(k, xor_bytes(xor_bytes(rotated, c2), opc))
    out2 = xor_bytes(out2, opc)
    res = out2[8:16]  # f2 output

    # f3 - CK
    c3 = bytes([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2])
    r3 = 4
    rotated = temp[r3:] + temp[:r3]
    out3 = aes_encrypt(k, xor_bytes(xor_bytes(rotated, c3), opc))
    ck = xor_bytes(out3, opc)

    # f4 - IK
    c4 = bytes([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4])
    r4 = 8
    rotated = temp[r4:] + temp[:r4]
    out4 = aes_encrypt(k, xor_bytes(xor_bytes(rotated, c4), opc))
    ik = xor_bytes(out4, opc)

    # f5 - AK
    c5 = bytes([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8])
    r5 = 12
    rotated = temp[r5:] + temp[:r5]
    out5 = aes_encrypt(k, xor_bytes(xor_bytes(rotated, c5), opc))
    out5 = xor_bytes(out5, opc)
    ak = out5[:6]  # f5 output

    return res, ck, ik, ak

def generate_auth_vector(subscription: ImsSubscription, auth_scheme: str = "Digest-AKAv1-MD5") -> Dict[str, Any]:
    """Generate authentication vector for subscriber"""
    # Get keys
    k = bytes.fromhex(subscription.k)
    op = bytes.fromhex(subscription.op or "00000000000000000000000000000000")

    # Generate RAND
    rand = secrets.token_bytes(16)

    # Increment SQN
    subscription.sqn += 1
    sqn_bytes = subscription.sqn.to_bytes(6, 'big')

    # AMF
    amf = bytes.fromhex(subscription.amf)

    # Generate authentication values
    mac_a = milenage_f1(k, rand, sqn_bytes, amf, op)
    xres, ck, ik, ak = milenage_f2345(k, rand, op)

    # AUTN = SQN XOR AK || AMF || MAC-A
    sqn_ak = xor_bytes(sqn_bytes, ak)
    autn = sqn_ak + amf + mac_a

    return {
        "item_number": 0,
        "auth_scheme": auth_scheme,
        "rand": rand.hex(),
        "autn": autn.hex(),
        "xres": xres.hex(),
        "ck": ck.hex(),
        "ik": ik.hex()
    }

# ============================================================================
# In-Memory Database
# ============================================================================

class HssDatabase:
    """IMS HSS Database"""

    def __init__(self):
        self.subscriptions: Dict[str, ImsSubscription] = {}  # IMPI -> subscription
        self.impu_to_impi: Dict[str, str] = {}  # IMPU -> IMPI mapping
        self._init_test_subscribers()

    def _init_test_subscribers(self):
        """Initialize test subscribers"""
        test_users = [
            {
                "impi": "user1@ims.example.com",
                "impu": "sip:user1@ims.example.com",
                "tel": "tel:+15551001",
                "k": "465b5ce8b199b49faa5f0a2ee238a6bc",
                "op": "cdc202d5123e20f62b6d676ac72cb318"
            },
            {
                "impi": "user2@ims.example.com",
                "impu": "sip:user2@ims.example.com",
                "tel": "tel:+15551002",
                "k": "0396eb317b6d1c36f19c1c84cd6ffd16",
                "op": "cdc202d5123e20f62b6d676ac72cb318"
            },
            {
                "impi": "user3@ims.example.com",
                "impu": "sip:user3@ims.example.com",
                "tel": "tel:+15551003",
                "k": "fec86ba6eb707ed08905757b1bb44b8f",
                "op": "cdc202d5123e20f62b6d676ac72cb318"
            }
        ]

        for user in test_users:
            # Create service profile
            profile = ServiceProfileData(
                profile_name=f"profile_{user['impi']}",
                initial_filter_criteria=[
                    {
                        "priority": 0,
                        "trigger_point": {
                            "condition_type": "originating",
                            "sip_method": ["INVITE"]
                        },
                        "application_server": "sip:as.ims.example.com:5065",
                        "default_handling": 0
                    }
                ]
            )

            # Create subscription
            subscription = ImsSubscription(
                impi=user["impi"],
                k=user["k"],
                op=user["op"],
                public_identities={
                    user["impu"]: PublicIdentity(
                        impu=user["impu"],
                        is_default=True,
                        service_profile_id=profile.profile_id
                    ),
                    user["tel"]: PublicIdentity(
                        impu=user["tel"],
                        is_default=False,
                        service_profile_id=profile.profile_id
                    )
                },
                service_profiles={profile.profile_id: profile},
                implicit_registration_sets={
                    "irs1": ImplicitRegistrationSet(
                        public_identities=[user["impu"], user["tel"]],
                        default_impu=user["impu"]
                    )
                }
            )

            self.subscriptions[user["impi"]] = subscription
            self.impu_to_impi[user["impu"]] = user["impi"]
            self.impu_to_impi[user["tel"]] = user["impi"]

            logger.info(f"[DB] Initialized subscriber: {user['impi']}")

    def get_subscription_by_impi(self, impi: str) -> Optional[ImsSubscription]:
        """Get subscription by IMPI"""
        return self.subscriptions.get(impi)

    def get_subscription_by_impu(self, impu: str) -> Optional[ImsSubscription]:
        """Get subscription by IMPU"""
        impi = self.impu_to_impi.get(impu)
        if impi:
            return self.subscriptions.get(impi)
        return None

    def create_subscription(self, subscription: ImsSubscription) -> ImsSubscription:
        """Create new subscription"""
        self.subscriptions[subscription.impi] = subscription
        for impu in subscription.public_identities:
            self.impu_to_impi[impu] = subscription.impi
        subscription.last_modified = datetime.utcnow()
        logger.info(f"[DB] Created subscription: {subscription.impi}")
        return subscription

    def update_subscription(self, subscription: ImsSubscription) -> ImsSubscription:
        """Update subscription"""
        subscription.last_modified = datetime.utcnow()
        self.subscriptions[subscription.impi] = subscription
        return subscription

    def delete_subscription(self, impi: str) -> bool:
        """Delete subscription"""
        if impi in self.subscriptions:
            sub = self.subscriptions[impi]
            for impu in sub.public_identities:
                if impu in self.impu_to_impi:
                    del self.impu_to_impi[impu]
            del self.subscriptions[impi]
            return True
        return False

    def get_all_subscriptions(self) -> List[ImsSubscription]:
        """Get all subscriptions"""
        return list(self.subscriptions.values())

# Global database
db = HssDatabase()

# ============================================================================
# Cx Interface Handlers
# ============================================================================

@app.post("/cx/uar", response_model=UARResponse)
async def user_authorization_request(uar: UARRequest):
    """
    User-Authorization-Request (UAR) - I-CSCF -> HSS
    Triggered on REGISTER to determine S-CSCF assignment

    Returns:
    - Assigned S-CSCF name if user already registered
    - Server capabilities if new registration (I-CSCF selects S-CSCF)
    """
    logger.info(f"[Cx-UAR] IMPU: {uar.public_identity}, IMPI: {uar.private_identity}")

    # Look up subscription
    subscription = db.get_subscription_by_impu(uar.public_identity)

    if not subscription:
        logger.warning(f"[Cx-UAR] User unknown: {uar.public_identity}")
        return UARResponse(
            result_code=DiameterResultCode.DIAMETER_ERROR_USER_UNKNOWN
        )

    # Verify IMPI matches
    if subscription.impi != uar.private_identity:
        # Check if IMPI is in a valid format for this subscription
        if uar.private_identity and "@" in uar.private_identity:
            # Allow if domain matches
            domain = subscription.impi.split("@")[1] if "@" in subscription.impi else ""
            req_domain = uar.private_identity.split("@")[1] if "@" in uar.private_identity else ""
            if domain != req_domain:
                logger.warning(f"[Cx-UAR] IMPI mismatch: {uar.private_identity} vs {subscription.impi}")
                return UARResponse(
                    result_code=DiameterResultCode.DIAMETER_ERROR_IDENTITIES_DONT_MATCH
                )

    # Check roaming
    if uar.visited_network_id and not subscription.roaming_allowed:
        if uar.visited_network_id not in subscription.visited_network_ids:
            logger.warning(f"[Cx-UAR] Roaming not allowed: {uar.visited_network_id}")
            return UARResponse(
                result_code=DiameterResultCode.DIAMETER_ERROR_ROAMING_NOT_ALLOWED
            )

    # Check if already registered with S-CSCF
    if subscription.scscf_name and subscription.registration_state == RegistrationState.REGISTERED:
        logger.info(f"[Cx-UAR] Already registered at {subscription.scscf_name}")
        return UARResponse(
            result_code=DiameterResultCode.DIAMETER_SUBSEQUENT_REGISTRATION,
            server_name=subscription.scscf_name
        )

    # First registration - return capabilities for I-CSCF to select S-CSCF
    logger.info(f"[Cx-UAR] First registration for {uar.public_identity}")
    return UARResponse(
        result_code=DiameterResultCode.DIAMETER_FIRST_REGISTRATION,
        server_capabilities={
            "mandatory_capabilities": [],
            "optional_capabilities": [1, 2, 3],  # Example capabilities
            "server_name": None  # I-CSCF will select
        }
    )

@app.post("/cx/lir", response_model=LIRResponse)
async def location_info_request(lir: LIRRequest):
    """
    Location-Info-Request (LIR) - I-CSCF -> HSS
    Triggered on incoming session to find serving S-CSCF

    Returns the S-CSCF currently serving the user
    """
    logger.info(f"[Cx-LIR] IMPU: {lir.public_identity}")

    subscription = db.get_subscription_by_impu(lir.public_identity)

    if not subscription:
        logger.warning(f"[Cx-LIR] User unknown: {lir.public_identity}")
        return LIRResponse(
            result_code=DiameterResultCode.DIAMETER_ERROR_USER_UNKNOWN
        )

    if not subscription.scscf_name:
        logger.warning(f"[Cx-LIR] User not registered: {lir.public_identity}")
        return LIRResponse(
            result_code=DiameterResultCode.DIAMETER_ERROR_IDENTITY_NOT_REGISTERED
        )

    logger.info(f"[Cx-LIR] User {lir.public_identity} served by {subscription.scscf_name}")
    return LIRResponse(
        result_code=DiameterResultCode.DIAMETER_SUCCESS,
        server_name=subscription.scscf_name
    )

@app.post("/cx/mar", response_model=MARResponse)
async def multimedia_auth_request(mar: MARRequest):
    """
    Multimedia-Auth-Request (MAR) - S-CSCF -> HSS
    Request authentication vectors for user registration

    Returns authentication vectors (RAND, AUTN, XRES, CK, IK)
    """
    logger.info(f"[Cx-MAR] IMPU: {mar.public_identity}, IMPI: {mar.private_identity}")

    subscription = db.get_subscription_by_impi(mar.private_identity)

    if not subscription:
        subscription = db.get_subscription_by_impu(mar.public_identity)

    if not subscription:
        logger.warning(f"[Cx-MAR] User unknown: {mar.private_identity}")
        return MARResponse(
            result_code=DiameterResultCode.DIAMETER_ERROR_USER_UNKNOWN,
            public_identity=mar.public_identity,
            private_identity=mar.private_identity
        )

    # Check auth scheme support
    supported_schemes = [AuthScheme.DIGEST_AKAV1_MD5.value, AuthScheme.DIGEST_MD5.value, AuthScheme.SIP_DIGEST.value]
    if mar.sip_auth_scheme not in supported_schemes:
        logger.warning(f"[Cx-MAR] Unsupported auth scheme: {mar.sip_auth_scheme}")
        return MARResponse(
            result_code=DiameterResultCode.DIAMETER_ERROR_AUTH_SCHEME_NOT_SUPPORTED,
            public_identity=mar.public_identity,
            private_identity=mar.private_identity
        )

    # Generate authentication vectors
    auth_vectors = []
    for i in range(mar.sip_auth_data_items):
        av = generate_auth_vector(subscription, mar.sip_auth_scheme)
        av["item_number"] = i
        auth_vectors.append(av)

    # Save updated SQN
    db.update_subscription(subscription)

    logger.info(f"[Cx-MAR] Generated {len(auth_vectors)} auth vectors for {mar.private_identity}")
    return MARResponse(
        result_code=DiameterResultCode.DIAMETER_SUCCESS,
        public_identity=mar.public_identity,
        private_identity=mar.private_identity,
        auth_vectors=auth_vectors
    )

@app.post("/cx/sar", response_model=SARResponse)
async def server_assignment_request(sar: SARRequest):
    """
    Server-Assignment-Request (SAR) - S-CSCF -> HSS
    Register/deregister S-CSCF assignment for user

    Updates HSS with current S-CSCF serving the user
    """
    logger.info(f"[Cx-SAR] IMPU: {sar.public_identity}, Type: {sar.server_assignment_type}")

    subscription = db.get_subscription_by_impu(sar.public_identity)

    if not subscription:
        logger.warning(f"[Cx-SAR] User unknown: {sar.public_identity}")
        return SARResponse(result_code=DiameterResultCode.DIAMETER_ERROR_USER_UNKNOWN)

    assignment_type = ServerAssignmentType(sar.server_assignment_type)

    # Handle different assignment types
    if assignment_type in [ServerAssignmentType.REGISTRATION, ServerAssignmentType.RE_REGISTRATION]:
        # Register user at S-CSCF
        subscription.scscf_name = sar.server_name
        subscription.registration_state = RegistrationState.REGISTERED
        subscription.registration_timestamp = datetime.utcnow()
        logger.info(f"[Cx-SAR] Registered {sar.public_identity} at {sar.server_name}")

    elif assignment_type == ServerAssignmentType.USER_DEREGISTRATION:
        # User initiated de-registration
        subscription.registration_state = RegistrationState.NOT_REGISTERED
        subscription.scscf_name = None
        logger.info(f"[Cx-SAR] De-registered {sar.public_identity}")

    elif assignment_type == ServerAssignmentType.TIMEOUT_DEREGISTRATION:
        # Registration expired
        subscription.registration_state = RegistrationState.NOT_REGISTERED
        subscription.scscf_name = None
        logger.info(f"[Cx-SAR] Timeout de-registration for {sar.public_identity}")

    elif assignment_type == ServerAssignmentType.UNREGISTERED_USER:
        # S-CSCF serving unregistered user (for terminating calls)
        subscription.scscf_name = sar.server_name
        subscription.registration_state = RegistrationState.UNREGISTERED
        logger.info(f"[Cx-SAR] Unregistered user service for {sar.public_identity}")

    db.update_subscription(subscription)

    # Build user profile response
    user_profile = None
    if sar.user_data_request_type == 0 or sar.user_data_already_available == 0:
        # Return user data
        default_profile_id = None
        for pub_id in subscription.public_identities.values():
            if pub_id.is_default:
                default_profile_id = pub_id.service_profile_id
                break

        user_profile = {
            "public_identities": list(subscription.public_identities.keys()),
            "private_identity": subscription.impi,
            "service_profile": subscription.service_profiles.get(default_profile_id, {}) if default_profile_id else {},
            "initial_filter_criteria": []
        }

        # Add iFCs
        if default_profile_id and default_profile_id in subscription.service_profiles:
            profile = subscription.service_profiles[default_profile_id]
            user_profile["initial_filter_criteria"] = profile.initial_filter_criteria

    # Get associated identities (from implicit registration sets)
    associated_identities = []
    for irs in subscription.implicit_registration_sets.values():
        associated_identities.extend(irs.public_identities)

    return SARResponse(
        result_code=DiameterResultCode.DIAMETER_SUCCESS,
        user_profile=user_profile,
        associated_identities=associated_identities,
        charging_information=subscription.charging_info.dict() if subscription.charging_info else None
    )

@app.post("/cx/rtr", response_model=RTRResponse)
async def registration_termination_request(rtr: RTRRequest):
    """
    Registration-Termination-Request (RTR) - HSS -> S-CSCF
    Administrative de-registration (HSS initiated)
    """
    logger.info(f"[Cx-RTR] IMPI: {rtr.private_identity}, Reason: {rtr.deregistration_reason}")

    subscription = db.get_subscription_by_impi(rtr.private_identity)

    if not subscription:
        return RTRResponse(result_code=DiameterResultCode.DIAMETER_ERROR_USER_UNKNOWN)

    # Clear registration
    subscription.registration_state = RegistrationState.NOT_REGISTERED
    subscription.scscf_name = None
    db.update_subscription(subscription)

    return RTRResponse(
        result_code=DiameterResultCode.DIAMETER_SUCCESS,
        associated_identities=list(subscription.public_identities.keys())
    )

@app.post("/cx/ppr", response_model=PPRResponse)
async def push_profile_request(ppr: PPRRequest):
    """
    Push-Profile-Request (PPR) - HSS -> S-CSCF
    Push updated user profile to S-CSCF
    """
    logger.info(f"[Cx-PPR] User: {ppr.user_identity}")

    subscription = db.get_subscription_by_impu(ppr.user_identity)
    if not subscription:
        subscription = db.get_subscription_by_impi(ppr.user_identity)

    if not subscription:
        return PPRResponse(result_code=DiameterResultCode.DIAMETER_ERROR_USER_UNKNOWN)

    # Update profile (simplified - would parse XML user data in production)
    if "charging_info" in ppr.user_data:
        subscription.charging_info = ChargingInfo(**ppr.user_data["charging_info"])

    db.update_subscription(subscription)

    return PPRResponse(result_code=DiameterResultCode.DIAMETER_SUCCESS)

# ============================================================================
# Subscription Management APIs
# ============================================================================

@app.get("/subscriptions")
async def list_subscriptions():
    """List all subscriptions"""
    subs = db.get_all_subscriptions()
    return {
        "count": len(subs),
        "subscriptions": [
            {
                "impi": s.impi,
                "public_identities": list(s.public_identities.keys()),
                "registration_state": s.registration_state.value,
                "scscf_name": s.scscf_name,
                "created_at": s.created_at.isoformat()
            }
            for s in subs
        ]
    }

@app.get("/subscriptions/{impi}")
async def get_subscription(impi: str):
    """Get subscription details"""
    sub = db.get_subscription_by_impi(impi)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")

    return {
        "impi": sub.impi,
        "public_identities": [
            {"impu": impu, "is_default": pub.is_default, "barring": pub.barring_indication}
            for impu, pub in sub.public_identities.items()
        ],
        "service_profiles": list(sub.service_profiles.keys()),
        "registration_state": sub.registration_state.value,
        "scscf_name": sub.scscf_name,
        "roaming_allowed": sub.roaming_allowed
    }

@app.post("/subscriptions")
async def create_subscription(
    impi: str,
    impu: str,
    k: str,
    op: Optional[str] = None,
    tel: Optional[str] = None
):
    """Create new subscription"""
    if db.get_subscription_by_impi(impi):
        raise HTTPException(status_code=409, detail="Subscription already exists")

    profile = ServiceProfileData(profile_name=f"profile_{impi}")

    public_ids = {
        impu: PublicIdentity(
            impu=impu,
            is_default=True,
            service_profile_id=profile.profile_id
        )
    }

    if tel:
        public_ids[tel] = PublicIdentity(
            impu=tel,
            is_default=False,
            service_profile_id=profile.profile_id
        )

    subscription = ImsSubscription(
        impi=impi,
        k=k,
        op=op or "cdc202d5123e20f62b6d676ac72cb318",
        public_identities=public_ids,
        service_profiles={profile.profile_id: profile}
    )

    db.create_subscription(subscription)

    return {"status": "created", "impi": impi}

@app.delete("/subscriptions/{impi}")
async def delete_subscription(impi: str):
    """Delete subscription"""
    if db.delete_subscription(impi):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Subscription not found")

@app.put("/subscriptions/{impi}/ifc")
async def update_ifc(impi: str, ifcs: List[Dict[str, Any]]):
    """Update Initial Filter Criteria for subscription"""
    sub = db.get_subscription_by_impi(impi)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")

    # Update first service profile
    if sub.service_profiles:
        profile_id = list(sub.service_profiles.keys())[0]
        sub.service_profiles[profile_id].initial_filter_criteria = ifcs
        db.update_subscription(sub)

    return {"status": "updated"}

# ============================================================================
# Health and Status
# ============================================================================

@app.get("/")
async def root():
    """IMS HSS status endpoint"""
    registered = len([s for s in db.subscriptions.values() if s.registration_state == RegistrationState.REGISTERED])
    return {
        "nf_type": "IMS-HSS",
        "nf_name": "IMS Home Subscriber Server",
        "status": "running",
        "description": "IMS HSS - Cx/Dx Interface for IMS Authentication and User Data",
        "version": "1.0.0",
        "interfaces": ["Cx (I-CSCF/S-CSCF)", "Dx (AS)"],
        "statistics": {
            "total_subscriptions": len(db.subscriptions),
            "registered_users": registered,
            "public_identities": len(db.impu_to_impi)
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "IMS-HSS", "compliance": "3GPP TS 29.228", "version": "1.0.0"}

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="IMS-HSS - IMS Home Subscriber Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("ims_hss"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)