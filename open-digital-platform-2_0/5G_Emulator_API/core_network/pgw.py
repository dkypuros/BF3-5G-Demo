# File: core_network/pgw.py
# PGW - PDN Gateway (4G/LTE EPC)
# Inspired by Open5GS SMF (combined PGW-C/PGW-U for 5G)
# 3GPP TS 23.401 - GPRS enhancements for E-UTRAN access

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid
import random
import ipaddress

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
    title="PGW - PDN Gateway",
    description="4G/LTE EPC PGW - Connects UE to external PDN, IP allocation, policy enforcement",
    version="1.0.0"
)
FastAPIInstrumentor.instrument_app(app)

# ============================================================================
# Data Models
# ============================================================================

class PdnType(str, Enum):
    """PDN Types"""
    IPV4 = "IPv4"
    IPV6 = "IPv6"
    IPV4V6 = "IPv4v6"

class BearerState(str, Enum):
    """Bearer States"""
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"

class IpPool(BaseModel):
    """IP Address Pool"""
    pool_id: str
    name: str
    network: str
    prefix_len: int
    allocated: List[str] = Field(default_factory=list)
    pdn_type: PdnType = PdnType.IPV4

class ApnConfiguration(BaseModel):
    """APN Configuration"""
    apn: str
    pdn_type: PdnType = PdnType.IPV4
    ip_pool_id: str
    qci: int = Field(default=9, description="Default QCI")
    arp_priority: int = Field(default=15)
    ambr_ul: int = Field(default=100000, description="Aggregate MBR UL (kbps)")
    ambr_dl: int = Field(default=100000, description="Aggregate MBR DL (kbps)")
    dns_primary: str = Field(default="8.8.8.8")
    dns_secondary: str = Field(default="8.8.4.4")

class GtpTunnel(BaseModel):
    """GTP-U Tunnel Endpoint"""
    teid: str
    ip_address: str
    port: int = Field(default=2152)

class PccRule(BaseModel):
    """Policy and Charging Control Rule"""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    rule_name: str
    precedence: int = Field(default=100)
    flow_direction: str = Field(default="BIDIRECTIONAL")
    flow_description: Optional[str] = None
    qci: int = Field(default=9)
    gbr_ul: Optional[int] = None
    gbr_dl: Optional[int] = None
    mbr_ul: Optional[int] = None
    mbr_dl: Optional[int] = None
    charging_key: Optional[int] = None

class PgwBearer(BaseModel):
    """PGW Bearer Context"""
    bearer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ebi: int
    state: BearerState = Field(default=BearerState.INACTIVE)

    # S5/S8 GTP-U tunnel
    s5u_pgw: Optional[GtpTunnel] = None
    s5u_sgw: Optional[GtpTunnel] = None

    # QoS
    qci: int = Field(default=9)
    arp_priority: int = Field(default=15)
    gbr_ul: Optional[int] = None
    gbr_dl: Optional[int] = None
    mbr_ul: Optional[int] = None
    mbr_dl: Optional[int] = None

    # PCC rules applied to this bearer
    pcc_rules: List[PccRule] = Field(default_factory=list)

    # Usage
    ul_bytes: int = Field(default=0)
    dl_bytes: int = Field(default=0)

class PgwSession(BaseModel):
    """PGW Session (PDN Connection)"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    imsi: str
    msisdn: Optional[str] = None
    apn: str
    pdn_type: PdnType

    # Allocated IP
    ip_address: Optional[str] = None
    ipv6_prefix: Optional[str] = None

    # S5/S8-C tunnel
    s5c_pgw_teid: str = Field(default_factory=lambda: f"{random.randint(1, 0xFFFFFFFF):08x}")
    s5c_sgw_teid: Optional[str] = None
    sgw_ip: Optional[str] = None

    # APN-AMBR
    ambr_ul: int = Field(default=100000)
    ambr_dl: int = Field(default=100000)

    # Bearers
    bearers: List[PgwBearer] = Field(default_factory=list)

    # Charging
    charging_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])

    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)

# Request/Response Models
class CreateSessionRequest(BaseModel):
    """S5/S8 Create Session Request"""
    imsi: str
    msisdn: Optional[str] = None
    apn: str
    pdn_type: str = "IPv4"
    s5c_sgw_teid: str
    sgw_ip: str
    rat_type: str = "EUTRAN"
    bearer_contexts: List[Dict[str, Any]]
    pco: Optional[Dict[str, Any]] = None  # Protocol Configuration Options

class CreateBearerResponse(BaseModel):
    """Create Bearer Response to SGW"""
    cause: str
    ebi: int
    s5u_pgw_teid: str
    s5u_pgw_ip: str
    qci: int
    arp_priority: int

# ============================================================================
# PGW Context Storage
# ============================================================================

class PgwContext:
    """PGW Context Manager"""

    def __init__(self):
        self.pgw_id = str(uuid.uuid4())[:8]
        self.pgw_name = "PGW-01"
        self.pgw_ip = "10.12.0.1"

        # IP Pools
        self.ip_pools: Dict[str, IpPool] = {
            "pool-ipv4-internet": IpPool(
                pool_id="pool-ipv4-internet",
                name="Internet IPv4 Pool",
                network="10.45.0.0",
                prefix_len=16,
                pdn_type=PdnType.IPV4
            ),
            "pool-ipv4-ims": IpPool(
                pool_id="pool-ipv4-ims",
                name="IMS IPv4 Pool",
                network="10.46.0.0",
                prefix_len=16,
                pdn_type=PdnType.IPV4
            ),
            "pool-ipv6": IpPool(
                pool_id="pool-ipv6",
                name="IPv6 Pool",
                network="2001:db8::",
                prefix_len=32,
                pdn_type=PdnType.IPV6
            )
        }

        # APN Configurations
        self.apn_configs: Dict[str, ApnConfiguration] = {
            "internet": ApnConfiguration(
                apn="internet",
                pdn_type=PdnType.IPV4,
                ip_pool_id="pool-ipv4-internet",
                qci=9,
                ambr_ul=100000,
                ambr_dl=100000
            ),
            "ims": ApnConfiguration(
                apn="ims",
                pdn_type=PdnType.IPV4V6,
                ip_pool_id="pool-ipv4-ims",
                qci=5,
                ambr_ul=50000,
                ambr_dl=50000
            )
        }

        # Sessions
        self.sessions_by_id: Dict[str, PgwSession] = {}
        self.sessions_by_imsi: Dict[str, List[PgwSession]] = {}
        self.sessions_by_teid: Dict[str, PgwSession] = {}

        # IP allocation tracking
        self.ip_counter: Dict[str, int] = {}

        # TEID allocation
        self.teid_counter = 0

        # Statistics
        self.stats = {
            "create_session_requests": 0,
            "delete_session_requests": 0,
            "create_bearer_requests": 0,
            "total_sessions": 0,
            "total_bearers": 0,
            "ul_bytes": 0,
            "dl_bytes": 0
        }

    def allocate_teid(self) -> str:
        """Allocate unique TEID"""
        self.teid_counter += 1
        return f"{self.teid_counter:08x}"

    def allocate_ip(self, pool_id: str) -> Optional[str]:
        """Allocate IP from pool"""
        pool = self.ip_pools.get(pool_id)
        if not pool:
            return None

        if pool_id not in self.ip_counter:
            self.ip_counter[pool_id] = 2  # Start from .2

        counter = self.ip_counter[pool_id]

        if pool.pdn_type == PdnType.IPV4:
            network = ipaddress.IPv4Network(f"{pool.network}/{pool.prefix_len}", strict=False)
            if counter >= 2 ** (32 - pool.prefix_len) - 1:
                return None  # Pool exhausted

            ip = str(network.network_address + counter)
            self.ip_counter[pool_id] = counter + 1
            pool.allocated.append(ip)
            return ip

        elif pool.pdn_type == PdnType.IPV6:
            # Allocate /64 prefix
            prefix = f"2001:db8:{counter:x}::/64"
            self.ip_counter[pool_id] = counter + 1
            return prefix

        return None

    def release_ip(self, pool_id: str, ip: str) -> None:
        """Release IP back to pool"""
        pool = self.ip_pools.get(pool_id)
        if pool and ip in pool.allocated:
            pool.allocated.remove(ip)

    def get_apn_config(self, apn: str) -> Optional[ApnConfiguration]:
        """Get APN configuration"""
        return self.apn_configs.get(apn)

    def add_session(self, session: PgwSession) -> None:
        """Add session to context"""
        self.sessions_by_id[session.session_id] = session
        self.sessions_by_teid[session.s5c_pgw_teid] = session

        if session.imsi not in self.sessions_by_imsi:
            self.sessions_by_imsi[session.imsi] = []
        self.sessions_by_imsi[session.imsi].append(session)

        self.stats["total_sessions"] += 1

    def remove_session(self, session_id: str) -> Optional[PgwSession]:
        """Remove session from context"""
        session = self.sessions_by_id.get(session_id)
        if not session:
            return None

        del self.sessions_by_id[session_id]
        if session.s5c_pgw_teid in self.sessions_by_teid:
            del self.sessions_by_teid[session.s5c_pgw_teid]

        if session.imsi in self.sessions_by_imsi:
            self.sessions_by_imsi[session.imsi] = [
                s for s in self.sessions_by_imsi[session.imsi]
                if s.session_id != session_id
            ]

        self.stats["total_sessions"] -= 1
        self.stats["total_bearers"] -= len(session.bearers)

        return session


# Global PGW context
pgw_ctx = PgwContext()

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "PGW",
        "compliance": "3GPP TS 23.401",
        "version": "1.0.0",
        "pgw_id": pgw_ctx.pgw_id,
        "pgw_name": pgw_ctx.pgw_name,
        "active_sessions": pgw_ctx.stats["total_sessions"],
        "active_bearers": pgw_ctx.stats["total_bearers"]
    }

@app.get("/pgw/v1/configuration")
async def get_configuration():
    """Get PGW configuration"""
    return {
        "pgw_id": pgw_ctx.pgw_id,
        "pgw_name": pgw_ctx.pgw_name,
        "pgw_ip": pgw_ctx.pgw_ip,
        "apn_configs": {k: v.dict() for k, v in pgw_ctx.apn_configs.items()},
        "ip_pools": {k: v.dict() for k, v in pgw_ctx.ip_pools.items()}
    }

# ----------------------------------------------------------------------------
# S5/S8 Interface (SGW <-> PGW) - GTPv2-C
# ----------------------------------------------------------------------------

@app.post("/s5/v1/create-session")
async def create_session(request: CreateSessionRequest):
    """
    Handle Create Session Request from SGW
    """
    with tracer.start_as_current_span("s5_create_session") as span:
        pgw_ctx.stats["create_session_requests"] += 1
        span.set_attribute("imsi", request.imsi)
        span.set_attribute("apn", request.apn)

        # Get APN configuration
        apn_config = pgw_ctx.get_apn_config(request.apn)
        if not apn_config:
            # Use default config
            apn_config = ApnConfiguration(
                apn=request.apn,
                ip_pool_id="pool-ipv4-internet"
            )

        # Determine PDN type
        pdn_type = PdnType(request.pdn_type) if request.pdn_type else apn_config.pdn_type

        # Allocate IP address
        ip_address = None
        ipv6_prefix = None

        if pdn_type in [PdnType.IPV4, PdnType.IPV4V6]:
            ip_address = pgw_ctx.allocate_ip(apn_config.ip_pool_id)
            if not ip_address:
                return {
                    "cause": "ALL_DYNAMIC_ADDRESSES_ARE_OCCUPIED",
                    "message": "No IP address available"
                }

        if pdn_type in [PdnType.IPV6, PdnType.IPV4V6]:
            ipv6_prefix = pgw_ctx.allocate_ip("pool-ipv6")

        # Create session
        session = PgwSession(
            imsi=request.imsi,
            msisdn=request.msisdn,
            apn=request.apn,
            pdn_type=pdn_type,
            ip_address=ip_address,
            ipv6_prefix=ipv6_prefix,
            s5c_sgw_teid=request.s5c_sgw_teid,
            sgw_ip=request.sgw_ip,
            ambr_ul=apn_config.ambr_ul,
            ambr_dl=apn_config.ambr_dl
        )

        # Create bearers
        bearer_contexts_created = []
        for bc in request.bearer_contexts:
            ebi = bc.get("ebi", 5)

            bearer = PgwBearer(
                ebi=ebi,
                qci=bc.get("qci", apn_config.qci),
                arp_priority=bc.get("arp_priority", apn_config.arp_priority),
                state=BearerState.ACTIVE
            )

            # Allocate S5-U tunnel
            bearer.s5u_pgw = GtpTunnel(
                teid=pgw_ctx.allocate_teid(),
                ip_address=pgw_ctx.pgw_ip
            )

            # Store SGW tunnel info if provided
            if bc.get("s5u_sgw_teid"):
                bearer.s5u_sgw = GtpTunnel(
                    teid=bc["s5u_sgw_teid"],
                    ip_address=bc.get("s5u_sgw_ip", request.sgw_ip)
                )

            # Create default PCC rule
            default_rule = PccRule(
                rule_name=f"default-{ebi}",
                precedence=255,
                qci=bearer.qci,
                mbr_ul=apn_config.ambr_ul,
                mbr_dl=apn_config.ambr_dl
            )
            bearer.pcc_rules.append(default_rule)

            session.bearers.append(bearer)
            pgw_ctx.stats["total_bearers"] += 1

            bearer_contexts_created.append({
                "ebi": ebi,
                "cause": "REQUEST_ACCEPTED",
                "s5u_pgw_teid": bearer.s5u_pgw.teid,
                "s5u_pgw_ip": bearer.s5u_pgw.ip_address,
                "qci": bearer.qci,
                "arp_priority": bearer.arp_priority
            })

        # Store session
        pgw_ctx.add_session(session)

        # Build PCO response
        pco_response = {
            "dns_primary": apn_config.dns_primary,
            "dns_secondary": apn_config.dns_secondary
        }

        return {
            "cause": "REQUEST_ACCEPTED",
            "s5c_pgw_teid": session.s5c_pgw_teid,
            "pdn_address": {
                "pdn_type": pdn_type.value,
                "ipv4": ip_address,
                "ipv6_prefix": ipv6_prefix
            },
            "apn_ambr": {
                "ul": session.ambr_ul,
                "dl": session.ambr_dl
            },
            "pco": pco_response,
            "charging_id": session.charging_id,
            "bearer_contexts": bearer_contexts_created
        }

@app.post("/s5/v1/delete-session")
async def delete_session(s5c_pgw_teid: str, cause: Optional[str] = None):
    """Handle Delete Session Request from SGW"""
    with tracer.start_as_current_span("s5_delete_session"):
        pgw_ctx.stats["delete_session_requests"] += 1

        session = pgw_ctx.sessions_by_teid.get(s5c_pgw_teid)
        if not session:
            return {
                "cause": "CONTEXT_NOT_FOUND",
                "message": "Session not found"
            }

        # Release IP address
        apn_config = pgw_ctx.get_apn_config(session.apn)
        if apn_config and session.ip_address:
            pgw_ctx.release_ip(apn_config.ip_pool_id, session.ip_address)

        # Remove session
        pgw_ctx.remove_session(session.session_id)

        return {
            "cause": "REQUEST_ACCEPTED"
        }

@app.post("/s5/v1/modify-bearer")
async def modify_bearer(
    s5c_pgw_teid: str,
    ebi: int,
    s5u_sgw_teid: Optional[str] = None,
    s5u_sgw_ip: Optional[str] = None
):
    """Handle Modify Bearer Request from SGW"""
    with tracer.start_as_current_span("s5_modify_bearer"):
        session = pgw_ctx.sessions_by_teid.get(s5c_pgw_teid)
        if not session:
            return {
                "cause": "CONTEXT_NOT_FOUND",
                "message": "Session not found"
            }

        for bearer in session.bearers:
            if bearer.ebi == ebi:
                if s5u_sgw_teid and s5u_sgw_ip:
                    bearer.s5u_sgw = GtpTunnel(
                        teid=s5u_sgw_teid,
                        ip_address=s5u_sgw_ip
                    )
                bearer.state = BearerState.ACTIVE
                session.last_activity = datetime.utcnow()

                return {
                    "cause": "REQUEST_ACCEPTED",
                    "bearer_context": {
                        "ebi": bearer.ebi,
                        "s5u_pgw_teid": bearer.s5u_pgw.teid if bearer.s5u_pgw else None,
                        "s5u_pgw_ip": bearer.s5u_pgw.ip_address if bearer.s5u_pgw else None
                    }
                }

        return {
            "cause": "CONTEXT_NOT_FOUND",
            "message": "Bearer not found"
        }

# ----------------------------------------------------------------------------
# Dedicated Bearer Management
# ----------------------------------------------------------------------------

@app.post("/pgw/v1/create-bearer")
async def create_dedicated_bearer(
    session_id: str,
    linked_ebi: int,
    qci: int,
    arp_priority: int = 10,
    gbr_ul: Optional[int] = None,
    gbr_dl: Optional[int] = None,
    tft: Optional[Dict[str, Any]] = None
):
    """
    Create Dedicated Bearer - PGW initiated
    For QoS flows requiring different treatment
    """
    with tracer.start_as_current_span("pgw_create_bearer"):
        pgw_ctx.stats["create_bearer_requests"] += 1

        session = pgw_ctx.sessions_by_id.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Verify linked bearer exists
        linked_bearer = None
        for b in session.bearers:
            if b.ebi == linked_ebi:
                linked_bearer = b
                break

        if not linked_bearer:
            raise HTTPException(status_code=404, detail="Linked bearer not found")

        # Allocate new EBI
        new_ebi = max(b.ebi for b in session.bearers) + 1
        if new_ebi > 15:
            return {
                "cause": "NO_RESOURCES_AVAILABLE",
                "message": "No EBI available"
            }

        # Create new bearer
        new_bearer = PgwBearer(
            ebi=new_ebi,
            qci=qci,
            arp_priority=arp_priority,
            gbr_ul=gbr_ul,
            gbr_dl=gbr_dl,
            state=BearerState.ACTIVE
        )

        new_bearer.s5u_pgw = GtpTunnel(
            teid=pgw_ctx.allocate_teid(),
            ip_address=pgw_ctx.pgw_ip
        )

        # Create PCC rule for this bearer
        rule = PccRule(
            rule_name=f"dedicated-{new_ebi}",
            precedence=50,
            qci=qci,
            gbr_ul=gbr_ul,
            gbr_dl=gbr_dl
        )
        new_bearer.pcc_rules.append(rule)

        session.bearers.append(new_bearer)
        pgw_ctx.stats["total_bearers"] += 1

        return {
            "cause": "REQUEST_ACCEPTED",
            "ebi": new_ebi,
            "linked_ebi": linked_ebi,
            "s5u_pgw_teid": new_bearer.s5u_pgw.teid,
            "s5u_pgw_ip": new_bearer.s5u_pgw.ip_address,
            "qci": qci,
            "arp_priority": arp_priority,
            "gbr_ul": gbr_ul,
            "gbr_dl": gbr_dl
        }

@app.delete("/pgw/v1/delete-bearer")
async def delete_dedicated_bearer(session_id: str, ebi: int):
    """Delete Dedicated Bearer"""
    with tracer.start_as_current_span("pgw_delete_bearer"):
        session = pgw_ctx.sessions_by_id.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Find bearer
        bearer_to_remove = None
        for b in session.bearers:
            if b.ebi == ebi:
                bearer_to_remove = b
                break

        if not bearer_to_remove:
            raise HTTPException(status_code=404, detail="Bearer not found")

        # Cannot delete default bearer (EBI 5)
        if ebi == 5:
            return {
                "cause": "SEMANTIC_ERROR",
                "message": "Cannot delete default bearer"
            }

        session.bearers.remove(bearer_to_remove)
        pgw_ctx.stats["total_bearers"] -= 1

        return {
            "cause": "REQUEST_ACCEPTED",
            "ebi": ebi
        }

# ----------------------------------------------------------------------------
# Policy Control (Gx Interface simulation)
# ----------------------------------------------------------------------------

@app.post("/gx/v1/install-rule")
async def install_pcc_rule(
    session_id: str,
    ebi: int,
    rule: PccRule
):
    """Install PCC Rule on bearer"""
    session = pgw_ctx.sessions_by_id.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    for bearer in session.bearers:
        if bearer.ebi == ebi:
            bearer.pcc_rules.append(rule)
            return {
                "status": "installed",
                "rule_id": rule.rule_id
            }

    raise HTTPException(status_code=404, detail="Bearer not found")

@app.delete("/gx/v1/remove-rule")
async def remove_pcc_rule(session_id: str, ebi: int, rule_id: str):
    """Remove PCC Rule from bearer"""
    session = pgw_ctx.sessions_by_id.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    for bearer in session.bearers:
        if bearer.ebi == ebi:
            bearer.pcc_rules = [r for r in bearer.pcc_rules if r.rule_id != rule_id]
            return {"status": "removed"}

    raise HTTPException(status_code=404, detail="Bearer not found")

# ----------------------------------------------------------------------------
# Charging (Gy Interface simulation)
# ----------------------------------------------------------------------------

@app.post("/gy/v1/report-usage")
async def report_usage(
    session_id: str,
    ebi: int,
    ul_bytes: int = 0,
    dl_bytes: int = 0
):
    """Report usage for charging"""
    session = pgw_ctx.sessions_by_id.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    for bearer in session.bearers:
        if bearer.ebi == ebi:
            bearer.ul_bytes += ul_bytes
            bearer.dl_bytes += dl_bytes
            pgw_ctx.stats["ul_bytes"] += ul_bytes
            pgw_ctx.stats["dl_bytes"] += dl_bytes
            return {
                "status": "recorded",
                "charging_id": session.charging_id
            }

    raise HTTPException(status_code=404, detail="Bearer not found")

# ----------------------------------------------------------------------------
# Session Management
# ----------------------------------------------------------------------------

@app.get("/pgw/v1/sessions")
async def list_sessions(imsi: Optional[str] = None, apn: Optional[str] = None):
    """List all sessions"""
    sessions = list(pgw_ctx.sessions_by_id.values())

    if imsi:
        sessions = [s for s in sessions if s.imsi == imsi]
    if apn:
        sessions = [s for s in sessions if s.apn == apn]

    return {
        "session_count": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "imsi": s.imsi,
                "apn": s.apn,
                "pdn_type": s.pdn_type.value,
                "ip_address": s.ip_address,
                "bearer_count": len(s.bearers),
                "created_at": s.created_at.isoformat()
            }
            for s in sessions
        ]
    }

@app.get("/pgw/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get detailed session info"""
    session = pgw_ctx.sessions_by_id.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "imsi": session.imsi,
        "msisdn": session.msisdn,
        "apn": session.apn,
        "pdn_type": session.pdn_type.value,
        "ip_address": session.ip_address,
        "ipv6_prefix": session.ipv6_prefix,
        "s5c_pgw_teid": session.s5c_pgw_teid,
        "s5c_sgw_teid": session.s5c_sgw_teid,
        "ambr": {"ul": session.ambr_ul, "dl": session.ambr_dl},
        "charging_id": session.charging_id,
        "bearers": [
            {
                "ebi": b.ebi,
                "state": b.state.value,
                "qci": b.qci,
                "s5u_pgw": b.s5u_pgw.dict() if b.s5u_pgw else None,
                "s5u_sgw": b.s5u_sgw.dict() if b.s5u_sgw else None,
                "pcc_rules": [r.dict() for r in b.pcc_rules],
                "usage": {"ul_bytes": b.ul_bytes, "dl_bytes": b.dl_bytes}
            }
            for b in session.bearers
        ],
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat()
    }

@app.get("/pgw/v1/statistics")
async def get_statistics():
    """Get PGW statistics"""
    return {
        "pgw_id": pgw_ctx.pgw_id,
        "pgw_name": pgw_ctx.pgw_name,
        "ip_pool_usage": {
            pool_id: {
                "total": 2 ** (32 - pool.prefix_len) - 2,
                "allocated": len(pool.allocated)
            }
            for pool_id, pool in pgw_ctx.ip_pools.items()
            if pool.pdn_type == PdnType.IPV4
        },
        "procedures": pgw_ctx.stats
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

    parser = argparse.ArgumentParser(description="PGW - PDN Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("pgw"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)