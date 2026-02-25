# File: core_network/mme.py
# MME - Mobility Management Entity (4G/LTE EPC)
# Inspired by Open5GS src/mme implementation
# 3GPP TS 23.401 - GPRS enhancements for E-UTRAN access

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid
import hashlib
import secrets

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(jaeger_exporter))
tracer = trace.get_tracer(__name__)

app = FastAPI(
    title="MME - Mobility Management Entity",
    description="4G/LTE EPC MME - Manages UE attach, authentication, and mobility",
    version="1.0.0"
)
FastAPIInstrumentor.instrument_app(app)

# ============================================================================
# Data Models - Inspired by Open5GS mme-context.h
# ============================================================================

class EmmState(str, Enum):
    """EMM (EPS Mobility Management) States"""
    DEREGISTERED = "EMM-DEREGISTERED"
    REGISTERED_INITIATED = "EMM-REGISTERED-INITIATED"
    REGISTERED = "EMM-REGISTERED"
    DEREGISTERED_INITIATED = "EMM-DEREGISTERED-INITIATED"
    SERVICE_REQUEST_INITIATED = "EMM-SERVICE-REQUEST-INITIATED"
    TAU_INITIATED = "EMM-TAU-INITIATED"

class EsmState(str, Enum):
    """ESM (EPS Session Management) States"""
    INACTIVE = "BEARER-INACTIVE"
    ACTIVE_PENDING = "BEARER-ACTIVE-PENDING"
    ACTIVE = "BEARER-ACTIVE"
    INACTIVE_PENDING = "BEARER-INACTIVE-PENDING"
    MODIFY_PENDING = "BEARER-MODIFY-PENDING"

class AttachType(str, Enum):
    """EPS Attach Types - TS 24.301"""
    EPS_ATTACH = "EPS_ATTACH"
    COMBINED_EPS_IMSI = "COMBINED_EPS_IMSI_ATTACH"
    EPS_EMERGENCY = "EPS_EMERGENCY_ATTACH"

class TAI(BaseModel):
    """Tracking Area Identity"""
    plmn_id: str = Field(..., description="PLMN ID (MCC+MNC)")
    tac: int = Field(..., description="Tracking Area Code")

class ECGI(BaseModel):
    """E-UTRAN Cell Global Identifier"""
    plmn_id: str = Field(..., description="PLMN ID (MCC+MNC)")
    cell_id: int = Field(..., description="Cell ID (28 bits)")

class GUTI(BaseModel):
    """Globally Unique Temporary Identifier"""
    plmn_id: str
    mme_gid: int = Field(..., description="MME Group ID")
    mme_code: int = Field(..., description="MME Code")
    m_tmsi: str = Field(..., description="M-TMSI")

class SecurityContext(BaseModel):
    """NAS Security Context"""
    ksi: int = Field(default=0, description="Key Set Identifier")
    kasme: Optional[str] = Field(None, description="Key ASME (hex)")
    k_nas_enc: Optional[str] = Field(None, description="NAS encryption key")
    k_nas_int: Optional[str] = Field(None, description="NAS integrity key")
    ul_count: int = Field(default=0, description="Uplink NAS COUNT")
    dl_count: int = Field(default=0, description="Downlink NAS COUNT")

class EpsBearer(BaseModel):
    """EPS Bearer Context"""
    ebi: int = Field(..., description="EPS Bearer ID (5-15)")
    qci: int = Field(default=9, description="QoS Class Identifier")
    arp_priority: int = Field(default=15, description="ARP Priority Level")
    gbr_ul: Optional[int] = Field(None, description="Guaranteed Bit Rate UL (kbps)")
    gbr_dl: Optional[int] = Field(None, description="Guaranteed Bit Rate DL (kbps)")
    mbr_ul: Optional[int] = Field(None, description="Maximum Bit Rate UL (kbps)")
    mbr_dl: Optional[int] = Field(None, description="Maximum Bit Rate DL (kbps)")
    state: EsmState = Field(default=EsmState.INACTIVE)
    sgw_s1u_teid: Optional[str] = None
    enb_s1u_teid: Optional[str] = None

class PdnConnection(BaseModel):
    """PDN Connection"""
    pdn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    apn: str = Field(..., description="Access Point Name")
    pdn_type: str = Field(default="IPv4", description="IPv4, IPv6, or IPv4v6")
    ip_address: Optional[str] = None
    default_bearer: Optional[EpsBearer] = None
    dedicated_bearers: List[EpsBearer] = Field(default_factory=list)

class EnbUeContext(BaseModel):
    """eNB UE Context - S1AP association"""
    enb_ue_s1ap_id: int
    mme_ue_s1ap_id: int
    enb_id: str
    tai: TAI
    ecgi: ECGI

class MmeUeContext(BaseModel):
    """MME UE Context - Full UE state"""
    ue_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    imsi: str
    msisdn: Optional[str] = None
    imei: Optional[str] = None
    guti: Optional[GUTI] = None
    emm_state: EmmState = Field(default=EmmState.DEREGISTERED)
    security_context: Optional[SecurityContext] = None
    enb_ue: Optional[EnbUeContext] = None
    pdn_connections: List[PdnConnection] = Field(default_factory=list)
    attach_type: Optional[AttachType] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)

# Request/Response Models
class AttachRequest(BaseModel):
    """Initial Attach Request from UE via eNB"""
    imsi: Optional[str] = None
    guti: Optional[GUTI] = None
    attach_type: AttachType = AttachType.EPS_ATTACH
    tai: TAI
    ecgi: ECGI
    enb_ue_s1ap_id: int
    enb_id: str
    ue_network_capability: Optional[Dict[str, Any]] = None
    esm_message: Optional[Dict[str, Any]] = None

class AttachAccept(BaseModel):
    """Attach Accept sent to UE"""
    guti: GUTI
    tai_list: List[TAI]
    eps_bearer_context: EpsBearer
    pdn_address: str
    apn: str

class AttachReject(BaseModel):
    """Attach Reject sent to UE"""
    emm_cause: int
    esm_cause: Optional[int] = None

class ServiceRequest(BaseModel):
    """Service Request from UE"""
    m_tmsi: str
    tai: TAI
    ecgi: ECGI
    enb_ue_s1ap_id: int
    enb_id: str
    ksi: int

class TauRequest(BaseModel):
    """Tracking Area Update Request"""
    guti: GUTI
    tai: TAI
    ecgi: ECGI
    enb_ue_s1ap_id: int
    enb_id: str
    eps_update_type: str = "TA_UPDATING"

class HandoverRequest(BaseModel):
    """S1 Handover Request"""
    mme_ue_s1ap_id: int
    target_enb_id: str
    target_tai: TAI
    target_ecgi: ECGI
    cause: str = "handover-desirable-for-radio-reasons"

# ============================================================================
# MME Context Storage
# ============================================================================

class MmeContext:
    """MME Context Manager - Inspired by Open5GS mme_context_t"""

    def __init__(self):
        self.mme_id = str(uuid.uuid4())[:8]
        self.mme_name = "MME-01"
        self.mme_gid = 1  # MME Group ID
        self.mme_code = 1  # MME Code
        self.plmn_id = "00101"  # MCC=001, MNC=01

        # Served TAI list
        self.served_tai_list: List[TAI] = [
            TAI(plmn_id="00101", tac=1),
            TAI(plmn_id="00101", tac=2),
            TAI(plmn_id="00101", tac=3)
        ]

        # UE contexts indexed by various keys
        self.ue_by_id: Dict[str, MmeUeContext] = {}
        self.ue_by_imsi: Dict[str, MmeUeContext] = {}
        self.ue_by_guti: Dict[str, MmeUeContext] = {}
        self.ue_by_s1ap_id: Dict[int, MmeUeContext] = {}

        # eNB connections
        self.enb_connections: Dict[str, Dict[str, Any]] = {}

        # M-TMSI pool
        self.m_tmsi_counter = 0

        # Statistics
        self.stats = {
            "attach_requests": 0,
            "attach_accepts": 0,
            "attach_rejects": 0,
            "service_requests": 0,
            "tau_requests": 0,
            "handovers": 0,
            "detaches": 0
        }

    def allocate_m_tmsi(self) -> str:
        """Allocate a new M-TMSI"""
        self.m_tmsi_counter += 1
        return f"{self.m_tmsi_counter:08x}"

    def allocate_mme_ue_s1ap_id(self) -> int:
        """Allocate MME UE S1AP ID"""
        return len(self.ue_by_s1ap_id) + 1

    def allocate_guti(self) -> GUTI:
        """Allocate a new GUTI for UE"""
        return GUTI(
            plmn_id=self.plmn_id,
            mme_gid=self.mme_gid,
            mme_code=self.mme_code,
            m_tmsi=self.allocate_m_tmsi()
        )

    def add_ue(self, ue: MmeUeContext) -> None:
        """Add UE to context"""
        self.ue_by_id[ue.ue_id] = ue
        if ue.imsi:
            self.ue_by_imsi[ue.imsi] = ue
        if ue.guti:
            guti_key = f"{ue.guti.plmn_id}-{ue.guti.mme_gid}-{ue.guti.mme_code}-{ue.guti.m_tmsi}"
            self.ue_by_guti[guti_key] = ue
        if ue.enb_ue:
            self.ue_by_s1ap_id[ue.enb_ue.mme_ue_s1ap_id] = ue

    def find_ue_by_imsi(self, imsi: str) -> Optional[MmeUeContext]:
        return self.ue_by_imsi.get(imsi)

    def find_ue_by_guti(self, guti: GUTI) -> Optional[MmeUeContext]:
        guti_key = f"{guti.plmn_id}-{guti.mme_gid}-{guti.mme_code}-{guti.m_tmsi}"
        return self.ue_by_guti.get(guti_key)

    def find_ue_by_s1ap_id(self, mme_ue_s1ap_id: int) -> Optional[MmeUeContext]:
        return self.ue_by_s1ap_id.get(mme_ue_s1ap_id)

    def remove_ue(self, ue_id: str) -> None:
        """Remove UE from context"""
        ue = self.ue_by_id.get(ue_id)
        if ue:
            del self.ue_by_id[ue_id]
            if ue.imsi and ue.imsi in self.ue_by_imsi:
                del self.ue_by_imsi[ue.imsi]
            if ue.guti:
                guti_key = f"{ue.guti.plmn_id}-{ue.guti.mme_gid}-{ue.guti.mme_code}-{ue.guti.m_tmsi}"
                if guti_key in self.ue_by_guti:
                    del self.ue_by_guti[guti_key]
            if ue.enb_ue and ue.enb_ue.mme_ue_s1ap_id in self.ue_by_s1ap_id:
                del self.ue_by_s1ap_id[ue.enb_ue.mme_ue_s1ap_id]

# Global MME context
mme_ctx = MmeContext()

# ============================================================================
# Security Functions - Inspired by Open5GS nas-security.c
# ============================================================================

def derive_kasme(ck: bytes, ik: bytes, plmn_id: str, sqn_xor_ak: bytes) -> bytes:
    """Derive KASME from CK, IK per TS 33.401"""
    # Simplified key derivation for emulation
    key_material = ck + ik + plmn_id.encode() + sqn_xor_ak
    return hashlib.sha256(key_material).digest()

def derive_nas_keys(kasme: bytes) -> tuple:
    """Derive NAS encryption and integrity keys from KASME"""
    # K_NAS_enc = KDF(KASME, 0x15, alg_type=1, alg_id)
    k_nas_enc = hashlib.sha256(kasme + b'\x15\x01\x02').digest()[:16]
    # K_NAS_int = KDF(KASME, 0x15, alg_type=2, alg_id)
    k_nas_int = hashlib.sha256(kasme + b'\x15\x02\x02').digest()[:16]
    return k_nas_enc, k_nas_int

def derive_kenb(kasme: bytes, ul_count: int) -> bytes:
    """Derive KeNB from KASME and UL NAS COUNT"""
    ul_count_bytes = ul_count.to_bytes(4, 'big')
    return hashlib.sha256(kasme + b'\x11' + ul_count_bytes).digest()

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "MME",
        "compliance": "3GPP TS 23.401",
        "version": "1.0.0",
        "mme_id": mme_ctx.mme_id,
        "mme_name": mme_ctx.mme_name,
        "connected_ues": len(mme_ctx.ue_by_id),
        "connected_enbs": len(mme_ctx.enb_connections)
    }

@app.get("/mme/v1/configuration")
async def get_configuration():
    """Get MME configuration"""
    return {
        "mme_id": mme_ctx.mme_id,
        "mme_name": mme_ctx.mme_name,
        "mme_gid": mme_ctx.mme_gid,
        "mme_code": mme_ctx.mme_code,
        "plmn_id": mme_ctx.plmn_id,
        "served_tai_list": [tai.dict() for tai in mme_ctx.served_tai_list]
    }

# ----------------------------------------------------------------------------
# S1AP Interface (eNB <-> MME)
# ----------------------------------------------------------------------------

@app.post("/s1ap/v1/enb/setup")
async def s1ap_enb_setup(
    enb_id: str,
    enb_name: str,
    supported_ta_list: List[TAI],
    global_enb_id: Dict[str, Any]
):
    """S1 Setup Request from eNB - TS 36.413"""
    with tracer.start_as_current_span("s1ap_enb_setup"):
        # Validate supported TAs against served TAs
        served_tacs = {tai.tac for tai in mme_ctx.served_tai_list}
        matching_tas = [ta for ta in supported_ta_list if ta.tac in served_tacs]

        if not matching_tas:
            return {
                "status": "failure",
                "cause": "unknown-PLMN",
                "message": "No matching TAI"
            }

        # Register eNB
        mme_ctx.enb_connections[enb_id] = {
            "enb_id": enb_id,
            "enb_name": enb_name,
            "global_enb_id": global_enb_id,
            "supported_ta_list": [ta.dict() for ta in supported_ta_list],
            "connected_at": datetime.utcnow().isoformat(),
            "ue_count": 0
        }

        return {
            "status": "success",
            "mme_name": mme_ctx.mme_name,
            "served_gummei_list": [{
                "plmn_id": mme_ctx.plmn_id,
                "mme_gid": mme_ctx.mme_gid,
                "mme_code": mme_ctx.mme_code
            }],
            "relative_mme_capacity": 100
        }

# ----------------------------------------------------------------------------
# EMM Procedures - TS 24.301
# ----------------------------------------------------------------------------

@app.post("/emm/v1/attach", response_model=Dict[str, Any])
async def emm_attach_request(request: AttachRequest):
    """
    Handle Initial Attach Request - TS 24.301 Section 5.5.1
    Inspired by Open5GS emm_handle_attach_request()
    """
    with tracer.start_as_current_span("emm_attach_request") as span:
        mme_ctx.stats["attach_requests"] += 1

        # Find or create UE context
        ue = None
        if request.imsi:
            ue = mme_ctx.find_ue_by_imsi(request.imsi)
            span.set_attribute("ue.imsi", request.imsi)
        elif request.guti:
            ue = mme_ctx.find_ue_by_guti(request.guti)
            span.set_attribute("ue.guti", request.guti.m_tmsi)

        if not ue:
            if not request.imsi:
                # Need identity request
                return {
                    "status": "identity_required",
                    "identity_type": "IMSI",
                    "message": "Send Identity Request to UE"
                }

            # Create new UE context
            ue = MmeUeContext(
                imsi=request.imsi,
                attach_type=request.attach_type,
                emm_state=EmmState.REGISTERED_INITIATED
            )

        # Set up eNB UE context
        mme_ue_s1ap_id = mme_ctx.allocate_mme_ue_s1ap_id()
        ue.enb_ue = EnbUeContext(
            enb_ue_s1ap_id=request.enb_ue_s1ap_id,
            mme_ue_s1ap_id=mme_ue_s1ap_id,
            enb_id=request.enb_id,
            tai=request.tai,
            ecgi=request.ecgi
        )

        # Simulate authentication (would go to HSS in production)
        # Generate security context
        rand = secrets.token_bytes(16)
        autn = secrets.token_bytes(16)
        kasme = derive_kasme(rand, autn, mme_ctx.plmn_id, b'\x00' * 6)
        k_nas_enc, k_nas_int = derive_nas_keys(kasme)

        ue.security_context = SecurityContext(
            ksi=0,
            kasme=kasme.hex(),
            k_nas_enc=k_nas_enc.hex(),
            k_nas_int=k_nas_int.hex()
        )

        # Allocate GUTI
        ue.guti = mme_ctx.allocate_guti()

        # Create default bearer and PDN connection
        default_bearer = EpsBearer(
            ebi=5,  # First EBI is 5
            qci=9,  # Default QCI for internet
            arp_priority=15,
            state=EsmState.ACTIVE
        )

        pdn = PdnConnection(
            apn="internet",
            pdn_type="IPv4",
            ip_address=f"10.45.0.{len(mme_ctx.ue_by_id) + 2}",
            default_bearer=default_bearer
        )
        ue.pdn_connections = [pdn]

        # Update state
        ue.emm_state = EmmState.REGISTERED
        ue.last_activity = datetime.utcnow()

        # Add to context
        mme_ctx.add_ue(ue)
        mme_ctx.stats["attach_accepts"] += 1

        span.set_attribute("ue.state", ue.emm_state.value)

        return {
            "status": "attach_accept",
            "ue_id": ue.ue_id,
            "mme_ue_s1ap_id": mme_ue_s1ap_id,
            "guti": ue.guti.dict(),
            "tai_list": [tai.dict() for tai in mme_ctx.served_tai_list],
            "eps_bearer": {
                "ebi": default_bearer.ebi,
                "qci": default_bearer.qci,
                "apn": pdn.apn,
                "pdn_address": pdn.ip_address,
                "pdn_type": pdn.pdn_type
            },
            "nas_security": {
                "ksi": ue.security_context.ksi,
                "algorithm": "EEA2/EIA2"
            }
        }

@app.post("/emm/v1/detach")
async def emm_detach_request(
    imsi: Optional[str] = None,
    guti: Optional[GUTI] = None,
    detach_type: str = "UE_ORIGINATED",
    switch_off: bool = False
):
    """Handle Detach Request - TS 24.301 Section 5.5.2"""
    with tracer.start_as_current_span("emm_detach_request"):
        mme_ctx.stats["detaches"] += 1

        ue = None
        if imsi:
            ue = mme_ctx.find_ue_by_imsi(imsi)
        elif guti:
            ue = mme_ctx.find_ue_by_guti(guti)

        if not ue:
            raise HTTPException(status_code=404, detail="UE not found")

        # Update state
        ue.emm_state = EmmState.DEREGISTERED

        # Remove from context
        mme_ctx.remove_ue(ue.ue_id)

        return {
            "status": "detach_accept",
            "ue_id": ue.ue_id,
            "switch_off": switch_off
        }

@app.post("/emm/v1/service-request")
async def emm_service_request(request: ServiceRequest):
    """Handle Service Request - TS 24.301 Section 5.6.1"""
    with tracer.start_as_current_span("emm_service_request"):
        mme_ctx.stats["service_requests"] += 1

        # Find UE by M-TMSI
        ue = None
        for u in mme_ctx.ue_by_id.values():
            if u.guti and u.guti.m_tmsi == request.m_tmsi:
                ue = u
                break

        if not ue:
            return {
                "status": "service_reject",
                "emm_cause": 10,  # Implicitly detached
                "message": "UE not found"
            }

        # Verify KSI
        if ue.security_context and ue.security_context.ksi != request.ksi:
            return {
                "status": "service_reject",
                "emm_cause": 9,  # UE identity cannot be derived
                "message": "KSI mismatch"
            }

        # Update eNB context for the new cell
        ue.enb_ue = EnbUeContext(
            enb_ue_s1ap_id=request.enb_ue_s1ap_id,
            mme_ue_s1ap_id=mme_ctx.allocate_mme_ue_s1ap_id(),
            enb_id=request.enb_id,
            tai=request.tai,
            ecgi=request.ecgi
        )
        ue.last_activity = datetime.utcnow()

        return {
            "status": "service_accept",
            "ue_id": ue.ue_id,
            "mme_ue_s1ap_id": ue.enb_ue.mme_ue_s1ap_id
        }

@app.post("/emm/v1/tau")
async def emm_tau_request(request: TauRequest):
    """Handle Tracking Area Update - TS 24.301 Section 5.5.3"""
    with tracer.start_as_current_span("emm_tau_request"):
        mme_ctx.stats["tau_requests"] += 1

        ue = mme_ctx.find_ue_by_guti(request.guti)
        if not ue:
            return {
                "status": "tau_reject",
                "emm_cause": 10,  # Implicitly detached
                "message": "UE not found"
            }

        # Update TAI
        ue.enb_ue = EnbUeContext(
            enb_ue_s1ap_id=request.enb_ue_s1ap_id,
            mme_ue_s1ap_id=mme_ctx.allocate_mme_ue_s1ap_id(),
            enb_id=request.enb_id,
            tai=request.tai,
            ecgi=request.ecgi
        )

        # Optionally allocate new GUTI
        if request.eps_update_type in ["TA_UPDATING", "COMBINED_TA_LA_UPDATING"]:
            old_guti_key = f"{ue.guti.plmn_id}-{ue.guti.mme_gid}-{ue.guti.mme_code}-{ue.guti.m_tmsi}"
            if old_guti_key in mme_ctx.ue_by_guti:
                del mme_ctx.ue_by_guti[old_guti_key]

            ue.guti = mme_ctx.allocate_guti()
            new_guti_key = f"{ue.guti.plmn_id}-{ue.guti.mme_gid}-{ue.guti.mme_code}-{ue.guti.m_tmsi}"
            mme_ctx.ue_by_guti[new_guti_key] = ue

        ue.last_activity = datetime.utcnow()

        return {
            "status": "tau_accept",
            "ue_id": ue.ue_id,
            "guti": ue.guti.dict() if ue.guti else None,
            "tai_list": [tai.dict() for tai in mme_ctx.served_tai_list]
        }

# ----------------------------------------------------------------------------
# ESM Procedures - TS 24.301
# ----------------------------------------------------------------------------

@app.post("/esm/v1/pdn-connectivity")
async def esm_pdn_connectivity_request(
    ue_id: str,
    apn: str,
    pdn_type: str = "IPv4",
    request_type: str = "initial"
):
    """Handle PDN Connectivity Request - TS 24.301 Section 6.5.1"""
    with tracer.start_as_current_span("esm_pdn_connectivity"):
        ue = mme_ctx.ue_by_id.get(ue_id)
        if not ue:
            raise HTTPException(status_code=404, detail="UE not found")

        # Create new PDN connection
        ebi = 5 + len(ue.pdn_connections)
        if ebi > 15:
            return {
                "status": "reject",
                "esm_cause": 65,  # Maximum number of EPS bearers reached
                "message": "No EBI available"
            }

        default_bearer = EpsBearer(
            ebi=ebi,
            qci=9,
            arp_priority=15,
            state=EsmState.ACTIVE
        )

        pdn = PdnConnection(
            apn=apn,
            pdn_type=pdn_type,
            ip_address=f"10.45.{len(mme_ctx.ue_by_id)}.{len(ue.pdn_connections) + 2}",
            default_bearer=default_bearer
        )
        ue.pdn_connections.append(pdn)

        return {
            "status": "accept",
            "pdn_id": pdn.pdn_id,
            "apn": pdn.apn,
            "pdn_type": pdn.pdn_type,
            "pdn_address": pdn.ip_address,
            "eps_bearer_id": default_bearer.ebi,
            "qci": default_bearer.qci
        }

@app.post("/esm/v1/pdn-disconnect")
async def esm_pdn_disconnect_request(ue_id: str, pdn_id: str):
    """Handle PDN Disconnect Request - TS 24.301 Section 6.5.2"""
    with tracer.start_as_current_span("esm_pdn_disconnect"):
        ue = mme_ctx.ue_by_id.get(ue_id)
        if not ue:
            raise HTTPException(status_code=404, detail="UE not found")

        # Find and remove PDN connection
        pdn_to_remove = None
        for pdn in ue.pdn_connections:
            if pdn.pdn_id == pdn_id:
                pdn_to_remove = pdn
                break

        if not pdn_to_remove:
            raise HTTPException(status_code=404, detail="PDN connection not found")

        # Cannot remove last PDN if UE is attached
        if len(ue.pdn_connections) == 1 and ue.emm_state == EmmState.REGISTERED:
            return {
                "status": "reject",
                "esm_cause": 49,  # Last PDN disconnection not allowed
                "message": "Cannot disconnect last PDN"
            }

        ue.pdn_connections.remove(pdn_to_remove)

        return {
            "status": "accept",
            "pdn_id": pdn_id
        }

# ----------------------------------------------------------------------------
# S1 Handover - TS 36.413
# ----------------------------------------------------------------------------

@app.post("/s1ap/v1/handover/required")
async def s1ap_handover_required(request: HandoverRequest):
    """Handle S1 Handover Required - Intra-MME handover"""
    with tracer.start_as_current_span("s1ap_handover"):
        mme_ctx.stats["handovers"] += 1

        ue = mme_ctx.find_ue_by_s1ap_id(request.mme_ue_s1ap_id)
        if not ue:
            raise HTTPException(status_code=404, detail="UE not found")

        # Verify target eNB is connected
        if request.target_enb_id not in mme_ctx.enb_connections:
            return {
                "status": "handover_preparation_failure",
                "cause": "unknown-targetID",
                "message": "Target eNB not connected"
            }

        # Prepare handover context
        handover_id = str(uuid.uuid4())[:8]

        return {
            "status": "handover_request",
            "handover_id": handover_id,
            "mme_ue_s1ap_id": request.mme_ue_s1ap_id,
            "target_enb_id": request.target_enb_id,
            "ue_security_capabilities": {
                "eea": ["EEA0", "EEA1", "EEA2"],
                "eia": ["EIA1", "EIA2"]
            },
            "nas_security_parameters": {
                "dl_count": ue.security_context.dl_count if ue.security_context else 0
            }
        }

@app.post("/s1ap/v1/handover/notify")
async def s1ap_handover_notify(
    handover_id: str,
    mme_ue_s1ap_id: int,
    target_ecgi: ECGI,
    target_tai: TAI
):
    """Handle Handover Notify - Update UE location after handover"""
    with tracer.start_as_current_span("s1ap_handover_notify"):
        ue = mme_ctx.find_ue_by_s1ap_id(mme_ue_s1ap_id)
        if not ue:
            raise HTTPException(status_code=404, detail="UE not found")

        # Update UE location
        if ue.enb_ue:
            ue.enb_ue.ecgi = target_ecgi
            ue.enb_ue.tai = target_tai

        ue.last_activity = datetime.utcnow()

        return {
            "status": "success",
            "ue_id": ue.ue_id,
            "new_ecgi": target_ecgi.dict(),
            "new_tai": target_tai.dict()
        }

# ----------------------------------------------------------------------------
# UE Context Management
# ----------------------------------------------------------------------------

@app.get("/mme/v1/ue")
async def list_ue_contexts(
    imsi: Optional[str] = None,
    state: Optional[EmmState] = None
):
    """List UE contexts"""
    ues = list(mme_ctx.ue_by_id.values())

    if imsi:
        ues = [u for u in ues if u.imsi == imsi]
    if state:
        ues = [u for u in ues if u.emm_state == state]

    return {
        "ue_count": len(ues),
        "ues": [
            {
                "ue_id": ue.ue_id,
                "imsi": ue.imsi,
                "guti": ue.guti.dict() if ue.guti else None,
                "emm_state": ue.emm_state.value,
                "pdn_count": len(ue.pdn_connections),
                "last_activity": ue.last_activity.isoformat()
            }
            for ue in ues
        ]
    }

@app.get("/mme/v1/ue/{ue_id}")
async def get_ue_context(ue_id: str):
    """Get detailed UE context"""
    ue = mme_ctx.ue_by_id.get(ue_id)
    if not ue:
        raise HTTPException(status_code=404, detail="UE not found")

    return ue.dict()

@app.get("/mme/v1/statistics")
async def get_statistics():
    """Get MME statistics"""
    return {
        "mme_id": mme_ctx.mme_id,
        "mme_name": mme_ctx.mme_name,
        "connected_ues": len(mme_ctx.ue_by_id),
        "registered_ues": len([u for u in mme_ctx.ue_by_id.values()
                               if u.emm_state == EmmState.REGISTERED]),
        "connected_enbs": len(mme_ctx.enb_connections),
        "procedures": mme_ctx.stats
    }

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="MME - Mobility Management Entity")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("mme"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)