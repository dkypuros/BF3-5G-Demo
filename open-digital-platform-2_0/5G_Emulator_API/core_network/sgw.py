# File: core_network/sgw.py
# SGW - Serving Gateway (4G/LTE EPC)
# Inspired by Open5GS src/sgwc and src/sgwu implementation
# 3GPP TS 23.401 - GPRS enhancements for E-UTRAN access

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid
import random

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
    title="SGW - Serving Gateway",
    description="4G/LTE EPC SGW - Routes user data between eNB and PGW",
    version="1.0.0"
)
FastAPIInstrumentor.instrument_app(app)

# ============================================================================
# Data Models - Inspired by Open5GS sgwc/context.h
# ============================================================================

class BearerState(str, Enum):
    """Bearer States"""
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    MODIFY_PENDING = "MODIFY_PENDING"

class GtpTunnel(BaseModel):
    """GTP-U Tunnel Endpoint"""
    teid: str = Field(..., description="Tunnel Endpoint ID")
    ip_address: str = Field(..., description="Transport IP address")
    port: int = Field(default=2152, description="GTP-U port")

class SgwBearer(BaseModel):
    """SGW Bearer Context"""
    bearer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ebi: int = Field(..., description="EPS Bearer ID")
    state: BearerState = Field(default=BearerState.INACTIVE)

    # S1-U: eNB <-> SGW
    s1u_sgw: Optional[GtpTunnel] = None
    s1u_enb: Optional[GtpTunnel] = None

    # S5/S8: SGW <-> PGW
    s5_sgw: Optional[GtpTunnel] = None
    s5_pgw: Optional[GtpTunnel] = None

    # QoS parameters
    qci: int = Field(default=9)
    arp_priority: int = Field(default=15)
    mbr_ul: Optional[int] = None
    mbr_dl: Optional[int] = None
    gbr_ul: Optional[int] = None
    gbr_dl: Optional[int] = None

    # Traffic counters
    ul_bytes: int = Field(default=0)
    dl_bytes: int = Field(default=0)
    ul_packets: int = Field(default=0)
    dl_packets: int = Field(default=0)

class SgwSession(BaseModel):
    """SGW Session (PDN Connection)"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    apn: str
    pdn_type: str = Field(default="IPv4")

    # S11: MME <-> SGW
    s11_sgw_teid: str = Field(default_factory=lambda: f"{random.randint(1, 0xFFFFFFFF):08x}")
    s11_mme_teid: Optional[str] = None

    # S5/S8: SGW <-> PGW  (control plane)
    s5c_sgw_teid: str = Field(default_factory=lambda: f"{random.randint(1, 0xFFFFFFFF):08x}")
    s5c_pgw_teid: Optional[str] = None

    bearers: List[SgwBearer] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SgwUeContext(BaseModel):
    """SGW UE Context"""
    ue_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    imsi: str
    msisdn: Optional[str] = None

    # MME info
    mme_id: Optional[str] = None
    mme_s11_teid: Optional[str] = None
    mme_ip: Optional[str] = None

    # Sessions
    sessions: List[SgwSession] = Field(default_factory=list)

    # Location
    serving_enb_id: Optional[str] = None
    tai: Optional[Dict[str, Any]] = None
    ecgi: Optional[Dict[str, Any]] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)

# Request/Response Models
class CreateSessionRequest(BaseModel):
    """GTPv2-C Create Session Request"""
    imsi: str
    msisdn: Optional[str] = None
    mme_id: str
    mme_s11_teid: str
    mme_ip: str
    apn: str
    pdn_type: str = "IPv4"
    rat_type: str = "EUTRAN"
    serving_network: Dict[str, Any]
    bearer_contexts: List[Dict[str, Any]]
    ue_timezone: Optional[str] = None
    charging_characteristics: Optional[str] = None

class ModifyBearerRequest(BaseModel):
    """GTPv2-C Modify Bearer Request"""
    s11_sgw_teid: str
    enb_s1u_teid: str
    enb_s1u_ip: str
    ebi: int
    indication_flags: Optional[Dict[str, bool]] = None

class DeleteSessionRequest(BaseModel):
    """GTPv2-C Delete Session Request"""
    s11_sgw_teid: str
    ebi: Optional[int] = None
    cause: Optional[str] = None

class CreateBearerRequest(BaseModel):
    """GTPv2-C Create Bearer Request (SGW -> MME)"""
    s11_mme_teid: str
    linked_ebi: int
    bearer_context: Dict[str, Any]
    pti: Optional[int] = None

# ============================================================================
# SGW Context Storage
# ============================================================================

class SgwContext:
    """SGW Context Manager"""

    def __init__(self):
        self.sgw_id = str(uuid.uuid4())[:8]
        self.sgw_name = "SGW-01"
        self.sgw_ip = "10.11.0.1"

        # UE contexts
        self.ue_by_id: Dict[str, SgwUeContext] = {}
        self.ue_by_imsi: Dict[str, SgwUeContext] = {}
        self.session_by_s11_teid: Dict[str, SgwSession] = {}

        # Connected PGW
        self.pgw_pool: List[Dict[str, Any]] = [
            {"pgw_id": "PGW-01", "ip": "10.12.0.1", "apn_list": ["internet", "ims"]}
        ]

        # TEID allocation
        self.teid_counter = 0

        # Statistics
        self.stats = {
            "create_session_requests": 0,
            "modify_bearer_requests": 0,
            "delete_session_requests": 0,
            "create_bearer_requests": 0,
            "total_bearers": 0,
            "ul_bytes": 0,
            "dl_bytes": 0
        }

    def allocate_teid(self) -> str:
        """Allocate a unique TEID"""
        self.teid_counter += 1
        return f"{self.teid_counter:08x}"

    def find_ue_by_imsi(self, imsi: str) -> Optional[SgwUeContext]:
        return self.ue_by_imsi.get(imsi)

    def find_session_by_s11_teid(self, teid: str) -> Optional[tuple]:
        """Find session and parent UE by S11 TEID"""
        for ue in self.ue_by_id.values():
            for session in ue.sessions:
                if session.s11_sgw_teid == teid:
                    return ue, session
        return None, None

    def select_pgw(self, apn: str) -> Optional[Dict[str, Any]]:
        """Select PGW for APN"""
        for pgw in self.pgw_pool:
            if apn in pgw.get("apn_list", []):
                return pgw
        # Return first PGW as default
        return self.pgw_pool[0] if self.pgw_pool else None


# Global SGW context
sgw_ctx = SgwContext()

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SGW",
        "compliance": "3GPP TS 23.401",
        "version": "1.0.0",
        "sgw_id": sgw_ctx.sgw_id,
        "sgw_name": sgw_ctx.sgw_name,
        "active_sessions": sum(len(ue.sessions) for ue in sgw_ctx.ue_by_id.values()),
        "active_bearers": sgw_ctx.stats["total_bearers"]
    }

@app.get("/sgw/v1/configuration")
async def get_configuration():
    """Get SGW configuration"""
    return {
        "sgw_id": sgw_ctx.sgw_id,
        "sgw_name": sgw_ctx.sgw_name,
        "sgw_ip": sgw_ctx.sgw_ip,
        "pgw_pool": sgw_ctx.pgw_pool
    }

# ----------------------------------------------------------------------------
# S11 Interface (MME <-> SGW) - GTPv2-C
# ----------------------------------------------------------------------------

@app.post("/s11/v1/create-session")
async def create_session(request: CreateSessionRequest):
    """
    Handle Create Session Request from MME
    Inspired by Open5GS sgwc_s11_handle_create_session_request()
    """
    with tracer.start_as_current_span("s11_create_session") as span:
        sgw_ctx.stats["create_session_requests"] += 1
        span.set_attribute("imsi", request.imsi)

        # Find or create UE context
        ue = sgw_ctx.find_ue_by_imsi(request.imsi)
        if not ue:
            ue = SgwUeContext(
                imsi=request.imsi,
                msisdn=request.msisdn,
                mme_id=request.mme_id,
                mme_s11_teid=request.mme_s11_teid,
                mme_ip=request.mme_ip
            )
            sgw_ctx.ue_by_id[ue.ue_id] = ue
            sgw_ctx.ue_by_imsi[ue.imsi] = ue

        # Select PGW
        pgw = sgw_ctx.select_pgw(request.apn)
        if not pgw:
            return {
                "cause": "NO_RESOURCES_AVAILABLE",
                "message": "No PGW available for APN"
            }

        # Create session
        session = SgwSession(
            apn=request.apn,
            pdn_type=request.pdn_type,
            s11_mme_teid=request.mme_s11_teid
        )

        # Create bearers from request
        bearer_contexts_created = []
        for bc in request.bearer_contexts:
            ebi = bc.get("ebi", 5)

            bearer = SgwBearer(
                ebi=ebi,
                qci=bc.get("qci", 9),
                arp_priority=bc.get("arp_priority", 15),
                state=BearerState.ACTIVE
            )

            # Allocate S1-U tunnel (SGW side)
            bearer.s1u_sgw = GtpTunnel(
                teid=sgw_ctx.allocate_teid(),
                ip_address=sgw_ctx.sgw_ip
            )

            # Allocate S5 tunnel (SGW side)
            bearer.s5_sgw = GtpTunnel(
                teid=sgw_ctx.allocate_teid(),
                ip_address=sgw_ctx.sgw_ip
            )

            # Simulate PGW response (would be S5/S8 in production)
            bearer.s5_pgw = GtpTunnel(
                teid=f"{random.randint(1, 0xFFFFFFFF):08x}",
                ip_address=pgw["ip"]
            )

            session.bearers.append(bearer)
            sgw_ctx.stats["total_bearers"] += 1

            bearer_contexts_created.append({
                "ebi": ebi,
                "cause": "REQUEST_ACCEPTED",
                "s1u_sgw_teid": bearer.s1u_sgw.teid,
                "s1u_sgw_ip": bearer.s1u_sgw.ip_address,
                "s5_pgw_teid": bearer.s5_pgw.teid,
                "s5_pgw_ip": bearer.s5_pgw.ip_address
            })

        # Store session
        ue.sessions.append(session)
        sgw_ctx.session_by_s11_teid[session.s11_sgw_teid] = session

        return {
            "cause": "REQUEST_ACCEPTED",
            "s11_sgw_teid": session.s11_sgw_teid,
            "s5c_pgw_teid": f"{random.randint(1, 0xFFFFFFFF):08x}",
            "pgw_ip": pgw["ip"],
            "pdn_address": f"10.45.0.{len(sgw_ctx.ue_by_id) + 1}",
            "bearer_contexts": bearer_contexts_created
        }

@app.post("/s11/v1/modify-bearer")
async def modify_bearer(request: ModifyBearerRequest):
    """
    Handle Modify Bearer Request from MME
    Updates S1-U tunnel endpoint after Initial Context Setup
    """
    with tracer.start_as_current_span("s11_modify_bearer"):
        sgw_ctx.stats["modify_bearer_requests"] += 1

        ue, session = sgw_ctx.find_session_by_s11_teid(request.s11_sgw_teid)
        if not session:
            return {
                "cause": "CONTEXT_NOT_FOUND",
                "message": "Session not found"
            }

        # Find bearer by EBI
        bearer = None
        for b in session.bearers:
            if b.ebi == request.ebi:
                bearer = b
                break

        if not bearer:
            return {
                "cause": "CONTEXT_NOT_FOUND",
                "message": "Bearer not found"
            }

        # Update S1-U eNB tunnel info
        bearer.s1u_enb = GtpTunnel(
            teid=request.enb_s1u_teid,
            ip_address=request.enb_s1u_ip
        )
        bearer.state = BearerState.ACTIVE

        if ue:
            ue.last_activity = datetime.utcnow()

        return {
            "cause": "REQUEST_ACCEPTED",
            "bearer_contexts": [{
                "ebi": bearer.ebi,
                "cause": "REQUEST_ACCEPTED",
                "s1u_sgw_teid": bearer.s1u_sgw.teid if bearer.s1u_sgw else None,
                "s1u_sgw_ip": bearer.s1u_sgw.ip_address if bearer.s1u_sgw else None
            }]
        }

@app.post("/s11/v1/delete-session")
async def delete_session(request: DeleteSessionRequest):
    """Handle Delete Session Request from MME"""
    with tracer.start_as_current_span("s11_delete_session"):
        sgw_ctx.stats["delete_session_requests"] += 1

        ue, session = sgw_ctx.find_session_by_s11_teid(request.s11_sgw_teid)
        if not session:
            return {
                "cause": "CONTEXT_NOT_FOUND",
                "message": "Session not found"
            }

        # Remove bearers
        sgw_ctx.stats["total_bearers"] -= len(session.bearers)

        # Remove session
        if ue:
            ue.sessions = [s for s in ue.sessions if s.session_id != session.session_id]
            # Remove UE if no sessions left
            if not ue.sessions:
                del sgw_ctx.ue_by_id[ue.ue_id]
                if ue.imsi in sgw_ctx.ue_by_imsi:
                    del sgw_ctx.ue_by_imsi[ue.imsi]

        if request.s11_sgw_teid in sgw_ctx.session_by_s11_teid:
            del sgw_ctx.session_by_s11_teid[request.s11_sgw_teid]

        return {
            "cause": "REQUEST_ACCEPTED"
        }

@app.post("/s11/v1/release-access-bearers")
async def release_access_bearers(s11_sgw_teid: str):
    """
    Release Access Bearers Request - UE moves to IDLE
    Releases S1-U but maintains S5/S8
    """
    with tracer.start_as_current_span("s11_release_access_bearers"):
        ue, session = sgw_ctx.find_session_by_s11_teid(s11_sgw_teid)
        if not session:
            return {
                "cause": "CONTEXT_NOT_FOUND",
                "message": "Session not found"
            }

        # Release S1-U tunnels
        for bearer in session.bearers:
            bearer.s1u_enb = None
            bearer.state = BearerState.INACTIVE

        return {
            "cause": "REQUEST_ACCEPTED"
        }

# ----------------------------------------------------------------------------
# S5/S8 Interface (SGW <-> PGW) - GTPv2-C
# ----------------------------------------------------------------------------

@app.post("/s5/v1/create-bearer")
async def create_dedicated_bearer(request: CreateBearerRequest):
    """
    Create Bearer Request - PGW initiated dedicated bearer
    SGW forwards to MME
    """
    with tracer.start_as_current_span("s5_create_bearer"):
        sgw_ctx.stats["create_bearer_requests"] += 1

        # Find session by linked EBI
        for ue in sgw_ctx.ue_by_id.values():
            for session in ue.sessions:
                for bearer in session.bearers:
                    if bearer.ebi == request.linked_ebi:
                        # Allocate new EBI
                        new_ebi = max(b.ebi for b in session.bearers) + 1
                        if new_ebi > 15:
                            return {
                                "cause": "NO_RESOURCES_AVAILABLE",
                                "message": "No EBI available"
                            }

                        # Create new bearer
                        new_bearer = SgwBearer(
                            ebi=new_ebi,
                            qci=request.bearer_context.get("qci", 5),
                            arp_priority=request.bearer_context.get("arp_priority", 10),
                            state=BearerState.ACTIVE
                        )

                        new_bearer.s1u_sgw = GtpTunnel(
                            teid=sgw_ctx.allocate_teid(),
                            ip_address=sgw_ctx.sgw_ip
                        )
                        new_bearer.s5_sgw = GtpTunnel(
                            teid=sgw_ctx.allocate_teid(),
                            ip_address=sgw_ctx.sgw_ip
                        )

                        session.bearers.append(new_bearer)
                        sgw_ctx.stats["total_bearers"] += 1

                        return {
                            "cause": "REQUEST_ACCEPTED",
                            "ebi": new_ebi,
                            "s1u_sgw_teid": new_bearer.s1u_sgw.teid,
                            "s1u_sgw_ip": new_bearer.s1u_sgw.ip_address
                        }

        return {
            "cause": "CONTEXT_NOT_FOUND",
            "message": "Linked bearer not found"
        }

# ----------------------------------------------------------------------------
# User Plane Management
# ----------------------------------------------------------------------------

@app.post("/sgw/v1/data-notification")
async def downlink_data_notification(s11_sgw_teid: str):
    """
    Downlink Data Notification - Trigger paging
    When downlink data arrives for IDLE UE
    """
    with tracer.start_as_current_span("downlink_data_notification"):
        ue, session = sgw_ctx.find_session_by_s11_teid(s11_sgw_teid)
        if not session:
            return {
                "cause": "CONTEXT_NOT_FOUND"
            }

        # Check if any bearer has no S1-U (UE is idle)
        idle_bearers = [b for b in session.bearers if b.s1u_enb is None]
        if not idle_bearers:
            return {
                "cause": "UE_ALREADY_CONNECTED",
                "message": "S1-U already established"
            }

        return {
            "cause": "REQUEST_ACCEPTED",
            "action": "TRIGGER_PAGING",
            "imsi": ue.imsi if ue else None,
            "s11_mme_teid": session.s11_mme_teid
        }

@app.post("/sgw/v1/traffic")
async def report_traffic(
    s11_sgw_teid: str,
    ebi: int,
    ul_bytes: int = 0,
    dl_bytes: int = 0,
    ul_packets: int = 0,
    dl_packets: int = 0
):
    """Report traffic statistics for a bearer"""
    ue, session = sgw_ctx.find_session_by_s11_teid(s11_sgw_teid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    for bearer in session.bearers:
        if bearer.ebi == ebi:
            bearer.ul_bytes += ul_bytes
            bearer.dl_bytes += dl_bytes
            bearer.ul_packets += ul_packets
            bearer.dl_packets += dl_packets
            sgw_ctx.stats["ul_bytes"] += ul_bytes
            sgw_ctx.stats["dl_bytes"] += dl_bytes
            return {"status": "recorded"}

    raise HTTPException(status_code=404, detail="Bearer not found")

# ----------------------------------------------------------------------------
# Context Management
# ----------------------------------------------------------------------------

@app.get("/sgw/v1/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for ue in sgw_ctx.ue_by_id.values():
        for session in ue.sessions:
            sessions.append({
                "session_id": session.session_id,
                "imsi": ue.imsi,
                "apn": session.apn,
                "s11_sgw_teid": session.s11_sgw_teid,
                "bearer_count": len(session.bearers),
                "created_at": session.created_at.isoformat()
            })

    return {
        "session_count": len(sessions),
        "sessions": sessions
    }

@app.get("/sgw/v1/sessions/{s11_sgw_teid}")
async def get_session(s11_sgw_teid: str):
    """Get detailed session info"""
    ue, session = sgw_ctx.find_session_by_s11_teid(s11_sgw_teid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "imsi": ue.imsi if ue else None,
        "apn": session.apn,
        "pdn_type": session.pdn_type,
        "s11_sgw_teid": session.s11_sgw_teid,
        "s11_mme_teid": session.s11_mme_teid,
        "s5c_sgw_teid": session.s5c_sgw_teid,
        "s5c_pgw_teid": session.s5c_pgw_teid,
        "bearers": [
            {
                "ebi": b.ebi,
                "state": b.state.value,
                "qci": b.qci,
                "s1u_sgw": b.s1u_sgw.dict() if b.s1u_sgw else None,
                "s1u_enb": b.s1u_enb.dict() if b.s1u_enb else None,
                "s5_sgw": b.s5_sgw.dict() if b.s5_sgw else None,
                "s5_pgw": b.s5_pgw.dict() if b.s5_pgw else None,
                "ul_bytes": b.ul_bytes,
                "dl_bytes": b.dl_bytes
            }
            for b in session.bearers
        ]
    }

@app.get("/sgw/v1/statistics")
async def get_statistics():
    """Get SGW statistics"""
    return {
        "sgw_id": sgw_ctx.sgw_id,
        "sgw_name": sgw_ctx.sgw_name,
        "active_ues": len(sgw_ctx.ue_by_id),
        "active_sessions": sum(len(ue.sessions) for ue in sgw_ctx.ue_by_id.values()),
        "procedures": sgw_ctx.stats
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

    parser = argparse.ArgumentParser(description="SGW - Serving Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("sgw"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)