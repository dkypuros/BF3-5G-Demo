# File: core_network/hss.py
# HSS - Home Subscriber Server (4G/LTE EPC)
# Inspired by Open5GS src/hss implementation
# 3GPP TS 29.272 - MME and SGSN related interfaces based on Diameter

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid
import hashlib
import hmac
import secrets
import struct

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(agent_host_name="localhost", agent_port=6831)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(jaeger_exporter))
tracer = trace.get_tracer(__name__)

app = FastAPI(
    title="HSS - Home Subscriber Server",
    description="4G/LTE EPC HSS - Subscriber database and authentication",
    version="1.0.0"
)
FastAPIInstrumentor.instrument_app(app)

# ============================================================================
# Data Models - Inspired by Open5GS hss-context.h
# ============================================================================

class SubscriberStatus(str, Enum):
    """Subscriber Status"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"

class NetworkAccessMode(str, Enum):
    """Network Access Mode"""
    PACKET_ONLY = "PACKET_ONLY"
    CIRCUIT_ONLY = "CIRCUIT_ONLY"
    PACKET_AND_CIRCUIT = "PACKET_AND_CIRCUIT"

class ApnConfiguration(BaseModel):
    """APN Configuration for subscriber"""
    apn: str
    pdn_type: str = Field(default="IPv4")
    qci: int = Field(default=9)
    arp_priority: int = Field(default=15)
    ambr_ul: int = Field(default=100000, description="kbps")
    ambr_dl: int = Field(default=100000, description="kbps")
    default_apn: bool = Field(default=False)

class AuthenticationKey(BaseModel):
    """USIM Authentication Key (K)"""
    k: str = Field(..., description="128-bit key in hex")
    opc: Optional[str] = Field(None, description="OPc in hex (derived from K and OP)")
    op: Optional[str] = Field(None, description="OP in hex")
    amf: str = Field(default="8000", description="Authentication Management Field")
    sqn: int = Field(default=0, description="Sequence Number")

class SubscriptionData(BaseModel):
    """Subscriber Subscription Data"""
    imsi: str
    msisdn: Optional[str] = None
    imei: Optional[str] = None
    status: SubscriberStatus = Field(default=SubscriberStatus.ACTIVE)
    network_access_mode: NetworkAccessMode = Field(default=NetworkAccessMode.PACKET_ONLY)

    # Authentication
    auth_key: AuthenticationKey

    # APN configurations
    apn_configurations: List[ApnConfiguration] = Field(default_factory=list)

    # Aggregate Maximum Bit Rate
    subscribed_ambr_ul: int = Field(default=100000, description="kbps")
    subscribed_ambr_dl: int = Field(default=100000, description="kbps")

    # RAT restrictions
    rat_restrictions: List[str] = Field(default_factory=list)

    # Subscriber locator
    mme_id: Optional[str] = Field(None, description="Serving MME")
    mme_realm: Optional[str] = None
    mme_host: Optional[str] = None

    # IMS
    ims_private_identity: Optional[str] = None
    ims_public_identities: List[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Diameter S6a Messages
class AuthInfoRequest(BaseModel):
    """Authentication Information Request (AIR) - TS 29.272"""
    imsi: str
    visited_plmn_id: str
    num_vectors: int = Field(default=1, description="Number of auth vectors requested")
    immediate_response_preferred: bool = Field(default=True)
    re_synchronization_info: Optional[Dict[str, str]] = Field(None, description="AUTS, RAND for resync")

class AuthVector(BaseModel):
    """EPS Authentication Vector"""
    rand: str = Field(..., description="128-bit RAND in hex")
    xres: str = Field(..., description="XRES in hex")
    autn: str = Field(..., description="128-bit AUTN in hex")
    kasme: str = Field(..., description="256-bit KASME in hex")

class AuthInfoAnswer(BaseModel):
    """Authentication Information Answer (AIA)"""
    result_code: int
    imsi: str
    auth_vectors: List[AuthVector] = Field(default_factory=list)

class UpdateLocationRequest(BaseModel):
    """Update Location Request (ULR) - TS 29.272"""
    imsi: str
    mme_id: str
    mme_realm: str
    mme_host: str
    visited_plmn_id: str
    rat_type: str = Field(default="EUTRAN")
    ulr_flags: int = Field(default=0)

class UpdateLocationAnswer(BaseModel):
    """Update Location Answer (ULA)"""
    result_code: int
    imsi: str
    subscription_data: Optional[Dict[str, Any]] = None

class PurgeUeRequest(BaseModel):
    """Purge UE Request (PUR) - TS 29.272"""
    imsi: str
    mme_id: str

# ============================================================================
# Milenage Algorithm Implementation
# ============================================================================

def aes_encrypt(key: bytes, data: bytes) -> bytes:
    """AES-128 encryption (simplified for emulation)"""
    # In production, use cryptography library
    # For emulation, use a hash-based approximation
    return hashlib.sha256(key + data).digest()[:16]

def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings"""
    return bytes(x ^ y for x, y in zip(a, b))

def rotate_bytes(data: bytes, bits: int) -> bytes:
    """Rotate bytes left by bits"""
    byte_shift = bits // 8
    return data[byte_shift:] + data[:byte_shift]

def milenage_f1(k: bytes, rand: bytes, sqn: bytes, amf: bytes, opc: bytes) -> tuple:
    """
    Milenage f1 and f1* functions
    Returns (MAC-A, MAC-S)
    """
    # temp = AES_K(RAND XOR OPc)
    temp = aes_encrypt(k, xor_bytes(rand, opc))

    # IN1 = SQN || AMF || SQN || AMF
    in1 = sqn + amf + sqn + amf

    # OUT1 = AES_K(rot(temp XOR OPc, r1) XOR c1) XOR OPc
    r1 = 64  # rotation constant
    c1 = bytes([0] * 16)
    out1 = aes_encrypt(k, xor_bytes(rotate_bytes(xor_bytes(xor_bytes(temp, opc), in1), r1), c1))
    out1 = xor_bytes(out1, opc)

    mac_a = out1[:8]
    mac_s = out1[8:16]

    return mac_a, mac_s

def milenage_f2345(k: bytes, rand: bytes, opc: bytes) -> tuple:
    """
    Milenage f2, f3, f4, f5 functions
    Returns (RES, CK, IK, AK)
    """
    # temp = AES_K(RAND XOR OPc)
    temp = aes_encrypt(k, xor_bytes(rand, opc))

    # f2: OUT2 for RES
    r2 = 0
    c2 = bytes([0] * 15 + [1])
    out2 = aes_encrypt(k, xor_bytes(rotate_bytes(xor_bytes(temp, opc), r2), c2))
    out2 = xor_bytes(out2, opc)
    res = out2[8:16]

    # f3: OUT3 for CK
    r3 = 32
    c3 = bytes([0] * 15 + [2])
    out3 = aes_encrypt(k, xor_bytes(rotate_bytes(xor_bytes(temp, opc), r3), c3))
    ck = xor_bytes(out3, opc)

    # f4: OUT4 for IK
    r4 = 64
    c4 = bytes([0] * 15 + [4])
    out4 = aes_encrypt(k, xor_bytes(rotate_bytes(xor_bytes(temp, opc), r4), c4))
    ik = xor_bytes(out4, opc)

    # f5: OUT5 for AK
    r5 = 96
    c5 = bytes([0] * 15 + [8])
    out5 = aes_encrypt(k, xor_bytes(rotate_bytes(xor_bytes(temp, opc), r5), c5))
    ak = xor_bytes(out5, opc)[:6]

    return res, ck, ik, ak

def derive_kasme(ck: bytes, ik: bytes, plmn_id: bytes, sqn_xor_ak: bytes) -> bytes:
    """Derive KASME per TS 33.401"""
    # Key = CK || IK
    key = ck + ik

    # S = FC || P0 || L0 || P1 || L1
    fc = b'\x10'
    p0 = plmn_id
    l0 = struct.pack('>H', len(p0))
    p1 = sqn_xor_ak
    l1 = struct.pack('>H', len(p1))
    s = fc + p0 + l0 + p1 + l1

    # KASME = HMAC-SHA-256(Key, S)
    return hmac.new(key, s, hashlib.sha256).digest()

def generate_auth_vector(k: bytes, opc: bytes, sqn: int, plmn_id: str, amf: bytes) -> AuthVector:
    """Generate EPS Authentication Vector"""
    # Generate RAND
    rand = secrets.token_bytes(16)

    # Convert SQN to 6 bytes
    sqn_bytes = sqn.to_bytes(6, 'big')

    # Run Milenage
    res, ck, ik, ak = milenage_f2345(k, rand, opc)
    mac_a, _ = milenage_f1(k, rand, sqn_bytes, amf, opc)

    # SQN XOR AK
    sqn_xor_ak = xor_bytes(sqn_bytes, ak)

    # AUTN = SQN XOR AK || AMF || MAC-A
    autn = sqn_xor_ak + amf + mac_a

    # Derive KASME
    plmn_bytes = bytes.fromhex(plmn_id.ljust(6, '0')[:6])
    kasme = derive_kasme(ck, ik, plmn_bytes, sqn_xor_ak)

    return AuthVector(
        rand=rand.hex(),
        xres=res.hex(),
        autn=autn.hex(),
        kasme=kasme.hex()
    )

# ============================================================================
# HSS Context Storage
# ============================================================================

class HssContext:
    """HSS Context Manager"""

    def __init__(self):
        self.hss_id = str(uuid.uuid4())[:8]
        self.hss_name = "HSS-01"
        self.realm = "epc.mnc001.mcc001.3gppnetwork.org"

        # Subscriber database
        self.subscribers: Dict[str, SubscriptionData] = {}
        self.subscribers_by_msisdn: Dict[str, SubscriptionData] = {}

        # Default authentication key for testing
        self.default_k = "465B5CE8B199B49FAA5F0A2EE238A6BC"
        self.default_opc = "E8ED289DEBA952E4283B54E88E6183CA"

        # Pre-populate some test subscribers
        self._init_test_subscribers()

        # Statistics
        self.stats = {
            "air_requests": 0,
            "ulr_requests": 0,
            "pur_requests": 0,
            "auth_failures": 0
        }

    def _init_test_subscribers(self):
        """Initialize test subscribers"""
        for i in range(1, 11):
            imsi = f"001010000000{i:03d}"
            msisdn = f"1234567{i:04d}"

            sub = SubscriptionData(
                imsi=imsi,
                msisdn=msisdn,
                auth_key=AuthenticationKey(
                    k=self.default_k,
                    opc=self.default_opc
                ),
                apn_configurations=[
                    ApnConfiguration(
                        apn="internet",
                        pdn_type="IPv4",
                        qci=9,
                        ambr_ul=100000,
                        ambr_dl=100000,
                        default_apn=True
                    ),
                    ApnConfiguration(
                        apn="ims",
                        pdn_type="IPv4v6",
                        qci=5,
                        ambr_ul=50000,
                        ambr_dl=50000
                    )
                ]
            )
            self.add_subscriber(sub)

    def add_subscriber(self, sub: SubscriptionData) -> None:
        """Add subscriber to database"""
        self.subscribers[sub.imsi] = sub
        if sub.msisdn:
            self.subscribers_by_msisdn[sub.msisdn] = sub

    def get_subscriber(self, imsi: str) -> Optional[SubscriptionData]:
        """Get subscriber by IMSI"""
        return self.subscribers.get(imsi)

    def get_subscriber_by_msisdn(self, msisdn: str) -> Optional[SubscriptionData]:
        """Get subscriber by MSISDN"""
        return self.subscribers_by_msisdn.get(msisdn)

    def update_subscriber_location(self, imsi: str, mme_id: str, mme_realm: str, mme_host: str) -> bool:
        """Update subscriber serving MME"""
        sub = self.subscribers.get(imsi)
        if sub:
            sub.mme_id = mme_id
            sub.mme_realm = mme_realm
            sub.mme_host = mme_host
            sub.updated_at = datetime.utcnow()
            return True
        return False

    def increment_sqn(self, imsi: str) -> int:
        """Increment and return SQN for subscriber"""
        sub = self.subscribers.get(imsi)
        if sub:
            sub.auth_key.sqn += 1
            return sub.auth_key.sqn
        return 0


# Global HSS context
hss_ctx = HssContext()

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "HSS",
        "compliance": "3GPP TS 29.272",
        "version": "1.0.0",
        "hss_id": hss_ctx.hss_id,
        "hss_name": hss_ctx.hss_name,
        "realm": hss_ctx.realm,
        "subscriber_count": len(hss_ctx.subscribers)
    }

@app.get("/hss/v1/configuration")
async def get_configuration():
    """Get HSS configuration"""
    return {
        "hss_id": hss_ctx.hss_id,
        "hss_name": hss_ctx.hss_name,
        "realm": hss_ctx.realm,
        "subscriber_count": len(hss_ctx.subscribers)
    }

# ----------------------------------------------------------------------------
# S6a Interface (MME <-> HSS) - Diameter
# ----------------------------------------------------------------------------

@app.post("/s6a/v1/air", response_model=AuthInfoAnswer)
async def authentication_information_request(request: AuthInfoRequest):
    """
    Authentication Information Request (AIR)
    MME requests authentication vectors from HSS
    """
    with tracer.start_as_current_span("s6a_air") as span:
        hss_ctx.stats["air_requests"] += 1
        span.set_attribute("imsi", request.imsi)

        sub = hss_ctx.get_subscriber(request.imsi)
        if not sub:
            hss_ctx.stats["auth_failures"] += 1
            return AuthInfoAnswer(
                result_code=5001,  # DIAMETER_ERROR_USER_UNKNOWN
                imsi=request.imsi
            )

        if sub.status != SubscriberStatus.ACTIVE:
            hss_ctx.stats["auth_failures"] += 1
            return AuthInfoAnswer(
                result_code=5004,  # DIAMETER_ERROR_ROAMING_NOT_ALLOWED
                imsi=request.imsi
            )

        # Handle re-synchronization
        if request.re_synchronization_info:
            # In production: verify AUTS and resync SQN
            # For emulation: just continue with new vectors
            pass

        # Get authentication key
        k = bytes.fromhex(sub.auth_key.k)
        opc = bytes.fromhex(sub.auth_key.opc) if sub.auth_key.opc else None

        if not opc and sub.auth_key.op:
            # Derive OPc from K and OP
            op = bytes.fromhex(sub.auth_key.op)
            opc = xor_bytes(aes_encrypt(k, op), op)
        elif not opc:
            opc = bytes.fromhex(hss_ctx.default_opc)

        amf = bytes.fromhex(sub.auth_key.amf)

        # Generate authentication vectors
        vectors = []
        for _ in range(request.num_vectors):
            sqn = hss_ctx.increment_sqn(request.imsi)
            vector = generate_auth_vector(k, opc, sqn, request.visited_plmn_id, amf)
            vectors.append(vector)

        return AuthInfoAnswer(
            result_code=2001,  # DIAMETER_SUCCESS
            imsi=request.imsi,
            auth_vectors=vectors
        )

@app.post("/s6a/v1/ulr", response_model=UpdateLocationAnswer)
async def update_location_request(request: UpdateLocationRequest):
    """
    Update Location Request (ULR)
    MME registers with HSS during attach
    """
    with tracer.start_as_current_span("s6a_ulr") as span:
        hss_ctx.stats["ulr_requests"] += 1
        span.set_attribute("imsi", request.imsi)
        span.set_attribute("mme_id", request.mme_id)

        sub = hss_ctx.get_subscriber(request.imsi)
        if not sub:
            return UpdateLocationAnswer(
                result_code=5001,  # DIAMETER_ERROR_USER_UNKNOWN
                imsi=request.imsi
            )

        if sub.status != SubscriberStatus.ACTIVE:
            return UpdateLocationAnswer(
                result_code=5004,  # DIAMETER_ERROR_ROAMING_NOT_ALLOWED
                imsi=request.imsi
            )

        # Update subscriber location
        hss_ctx.update_subscriber_location(
            request.imsi,
            request.mme_id,
            request.mme_realm,
            request.mme_host
        )

        # Build subscription data for response
        subscription_data = {
            "msisdn": sub.msisdn,
            "network_access_mode": sub.network_access_mode.value,
            "subscribed_rau_tau_timer": 1800,  # 30 minutes
            "ambr": {
                "ul": sub.subscribed_ambr_ul,
                "dl": sub.subscribed_ambr_dl
            },
            "apn_configurations": [
                {
                    "context_id": idx + 1,
                    "apn": apn.apn,
                    "pdn_type": apn.pdn_type,
                    "qci": apn.qci,
                    "arp_priority": apn.arp_priority,
                    "ambr_ul": apn.ambr_ul,
                    "ambr_dl": apn.ambr_dl
                }
                for idx, apn in enumerate(sub.apn_configurations)
            ],
            "rat_restrictions": sub.rat_restrictions
        }

        return UpdateLocationAnswer(
            result_code=2001,  # DIAMETER_SUCCESS
            imsi=request.imsi,
            subscription_data=subscription_data
        )

@app.post("/s6a/v1/pur")
async def purge_ue_request(request: PurgeUeRequest):
    """
    Purge UE Request (PUR)
    MME notifies HSS that UE is detached
    """
    with tracer.start_as_current_span("s6a_pur"):
        hss_ctx.stats["pur_requests"] += 1

        sub = hss_ctx.get_subscriber(request.imsi)
        if not sub:
            return {
                "result_code": 5001,
                "imsi": request.imsi
            }

        # Clear MME registration
        if sub.mme_id == request.mme_id:
            sub.mme_id = None
            sub.mme_realm = None
            sub.mme_host = None
            sub.updated_at = datetime.utcnow()

        return {
            "result_code": 2001,
            "imsi": request.imsi
        }

@app.post("/s6a/v1/clr")
async def cancel_location_request(imsi: str, cancellation_type: str = "MME_UPDATE_PROCEDURE"):
    """
    Cancel Location Request (CLR)
    HSS initiated - cancel UE registration at old MME
    """
    with tracer.start_as_current_span("s6a_clr"):
        sub = hss_ctx.get_subscriber(imsi)
        if not sub:
            return {
                "result_code": 5001,
                "imsi": imsi
            }

        old_mme = {
            "mme_id": sub.mme_id,
            "mme_host": sub.mme_host
        }

        return {
            "result_code": 2001,
            "imsi": imsi,
            "cancellation_type": cancellation_type,
            "old_mme": old_mme
        }

@app.post("/s6a/v1/idr")
async def insert_subscriber_data_request(imsi: str, subscription_data: Dict[str, Any]):
    """
    Insert Subscriber Data Request (IDR)
    HSS pushes updated subscription data to MME
    """
    sub = hss_ctx.get_subscriber(imsi)
    if not sub:
        return {
            "result_code": 5001,
            "imsi": imsi
        }

    return {
        "result_code": 2001,
        "imsi": imsi,
        "mme_id": sub.mme_id,
        "data_pushed": subscription_data
    }

# ----------------------------------------------------------------------------
# Subscriber Management
# ----------------------------------------------------------------------------

@app.get("/hss/v1/subscribers")
async def list_subscribers(
    status: Optional[SubscriberStatus] = None,
    limit: int = 100,
    offset: int = 0
):
    """List all subscribers"""
    subs = list(hss_ctx.subscribers.values())

    if status:
        subs = [s for s in subs if s.status == status]

    total = len(subs)
    subs = subs[offset:offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "subscribers": [
            {
                "imsi": s.imsi,
                "msisdn": s.msisdn,
                "status": s.status.value,
                "mme_id": s.mme_id,
                "apn_count": len(s.apn_configurations)
            }
            for s in subs
        ]
    }

@app.get("/hss/v1/subscribers/{imsi}")
async def get_subscriber(imsi: str):
    """Get subscriber details"""
    sub = hss_ctx.get_subscriber(imsi)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscriber not found")

    return {
        "imsi": sub.imsi,
        "msisdn": sub.msisdn,
        "status": sub.status.value,
        "network_access_mode": sub.network_access_mode.value,
        "subscribed_ambr": {
            "ul": sub.subscribed_ambr_ul,
            "dl": sub.subscribed_ambr_dl
        },
        "apn_configurations": [apn.dict() for apn in sub.apn_configurations],
        "serving_mme": {
            "mme_id": sub.mme_id,
            "mme_realm": sub.mme_realm,
            "mme_host": sub.mme_host
        },
        "auth_sqn": sub.auth_key.sqn,
        "created_at": sub.created_at.isoformat(),
        "updated_at": sub.updated_at.isoformat()
    }

@app.post("/hss/v1/subscribers")
async def create_subscriber(
    imsi: str,
    msisdn: Optional[str] = None,
    k: Optional[str] = None,
    opc: Optional[str] = None,
    apn_configs: Optional[List[ApnConfiguration]] = None
):
    """Create new subscriber"""
    if hss_ctx.get_subscriber(imsi):
        raise HTTPException(status_code=409, detail="Subscriber already exists")

    sub = SubscriptionData(
        imsi=imsi,
        msisdn=msisdn,
        auth_key=AuthenticationKey(
            k=k or hss_ctx.default_k,
            opc=opc or hss_ctx.default_opc
        ),
        apn_configurations=apn_configs or [
            ApnConfiguration(apn="internet", default_apn=True)
        ]
    )

    hss_ctx.add_subscriber(sub)

    return {
        "status": "created",
        "imsi": sub.imsi
    }

@app.put("/hss/v1/subscribers/{imsi}")
async def update_subscriber(
    imsi: str,
    msisdn: Optional[str] = None,
    status: Optional[SubscriberStatus] = None,
    subscribed_ambr_ul: Optional[int] = None,
    subscribed_ambr_dl: Optional[int] = None
):
    """Update subscriber"""
    sub = hss_ctx.get_subscriber(imsi)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscriber not found")

    if msisdn is not None:
        # Update MSISDN index
        if sub.msisdn and sub.msisdn in hss_ctx.subscribers_by_msisdn:
            del hss_ctx.subscribers_by_msisdn[sub.msisdn]
        sub.msisdn = msisdn
        hss_ctx.subscribers_by_msisdn[msisdn] = sub

    if status is not None:
        sub.status = status
    if subscribed_ambr_ul is not None:
        sub.subscribed_ambr_ul = subscribed_ambr_ul
    if subscribed_ambr_dl is not None:
        sub.subscribed_ambr_dl = subscribed_ambr_dl

    sub.updated_at = datetime.utcnow()

    return {
        "status": "updated",
        "imsi": sub.imsi
    }

@app.delete("/hss/v1/subscribers/{imsi}")
async def delete_subscriber(imsi: str):
    """Delete subscriber"""
    sub = hss_ctx.get_subscriber(imsi)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscriber not found")

    del hss_ctx.subscribers[imsi]
    if sub.msisdn and sub.msisdn in hss_ctx.subscribers_by_msisdn:
        del hss_ctx.subscribers_by_msisdn[sub.msisdn]

    return {
        "status": "deleted",
        "imsi": imsi
    }

@app.post("/hss/v1/subscribers/{imsi}/apn")
async def add_apn_configuration(imsi: str, apn_config: ApnConfiguration):
    """Add APN configuration to subscriber"""
    sub = hss_ctx.get_subscriber(imsi)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscriber not found")

    # Check if APN already exists
    for existing in sub.apn_configurations:
        if existing.apn == apn_config.apn:
            raise HTTPException(status_code=409, detail="APN already configured")

    sub.apn_configurations.append(apn_config)
    sub.updated_at = datetime.utcnow()

    return {
        "status": "added",
        "imsi": imsi,
        "apn": apn_config.apn
    }

@app.get("/hss/v1/statistics")
async def get_statistics():
    """Get HSS statistics"""
    return {
        "hss_id": hss_ctx.hss_id,
        "hss_name": hss_ctx.hss_name,
        "subscriber_count": len(hss_ctx.subscribers),
        "active_subscribers": len([s for s in hss_ctx.subscribers.values()
                                   if s.status == SubscriberStatus.ACTIVE]),
        "registered_subscribers": len([s for s in hss_ctx.subscribers.values()
                                       if s.mme_id is not None]),
        "procedures": hss_ctx.stats
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

    parser = argparse.ArgumentParser(description="HSS - Home Subscriber Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("hss"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)