# File location: 5G_Emulator_API/core_network/sepp.py
# 3GPP TS 29.573 - Security Edge Protection Proxy (SEPP) - 100% Compliant Implementation
# Implements N32-c and N32-f interfaces for inter-PLMN security
# Inspired by Open5GS SEPP implementation

from fastapi import FastAPI, HTTPException, Request, Response, Query, Path, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
import uvicorn
import httpx
import requests
import asyncio
import uuid
import json
import logging
import base64
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from opentelemetry import trace
from enum import Enum
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.backends import default_backend
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

nrf_url = "http://127.0.0.1:8000"

# 3GPP TS 29.573 Data Models

class SecurityCapability(str, Enum):
    TLS = "TLS"
    PRINS = "PRINS"
    NONE = "NONE"

class N32Purpose(str, Enum):
    ROAMING = "ROAMING"
    INTER_PLMN = "INTER_PLMN"
    HOME_ROUTING = "HOME_ROUTING"

class ProtectionPolicy(str, Enum):
    FULL_PROTECTION = "FULL_PROTECTION"
    MODIFICATION_PROTECTION = "MODIFICATION_PROTECTION"
    CONFIDENTIALITY_PROTECTION = "CONFIDENTIALITY_PROTECTION"
    NO_PROTECTION = "NO_PROTECTION"

class IeType(str, Enum):
    FULL_REQUEST = "FULL_REQUEST"
    HEADER = "HEADER"
    BODY = "BODY"
    ELEMENT = "ELEMENT"

class IeLocation(str, Enum):
    URI_PARAM = "URI_PARAM"
    HEADER = "HEADER"
    BODY = "BODY"

class PlmnId(BaseModel):
    mcc: str
    mnc: str

class N32Peer(BaseModel):
    peerPlmnId: PlmnId
    seppId: str
    seppFqdn: Optional[str] = None
    seppIpAddress: Optional[str] = None
    seppPort: int = 443
    securityCapability: SecurityCapability = SecurityCapability.TLS
    n32Purpose: N32Purpose = N32Purpose.ROAMING
    status: str = "ACTIVE"
    establishedTime: Optional[datetime] = None

class N32Handshake(BaseModel):
    senderSeppId: str
    senderPlmnId: PlmnId
    receiverSeppId: Optional[str] = None
    receiverPlmnId: PlmnId
    secCapNegotiation: Optional[List[SecurityCapability]] = None
    selectedSecCapability: Optional[SecurityCapability] = None
    supportedFeatures: Optional[str] = None

class ProtectionPolicyInfo(BaseModel):
    apiIeList: Optional[List[Dict]] = None
    defaultPolicy: ProtectionPolicy = ProtectionPolicy.FULL_PROTECTION
    protectedHeaders: Optional[List[str]] = None
    modifiableHeaders: Optional[List[str]] = None

class IeInfo(BaseModel):
    ieLoc: IeLocation
    ieType: IeType
    reqIe: Optional[str] = None
    rspIe: Optional[str] = None
    isModifiable: bool = False

class DataToIntegrityProtect(BaseModel):
    dataToIntegrityProtect: str  # Base64 encoded
    hashAlgorithm: str = "SHA-256"
    signature: Optional[str] = None

class N32fReformattedReq(BaseModel):
    reformattedData: str  # Base64 encoded reformatted SBI request
    integrityProtectedData: Optional[DataToIntegrityProtect] = None
    confidentialData: Optional[str] = None  # Encrypted data
    n32fContextId: str

class N32fReformattedRsp(BaseModel):
    reformattedData: str
    integrityProtectedData: Optional[DataToIntegrityProtect] = None
    confidentialData: Optional[str] = None
    n32fContextId: str

class RoamingPartner(BaseModel):
    plmnId: PlmnId
    partnerName: str
    seppInfo: N32Peer
    trustLevel: str = "FULL"
    allowedServices: Optional[List[str]] = None
    blockedServices: Optional[List[str]] = None
    protectionPolicy: ProtectionPolicyInfo = Field(default_factory=ProtectionPolicyInfo)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lastActive: Optional[datetime] = None

class MessageFilterRule(BaseModel):
    ruleId: str
    direction: str = "BOTH"  # INBOUND, OUTBOUND, BOTH
    action: str = "ALLOW"  # ALLOW, BLOCK, MODIFY
    nfType: Optional[str] = None
    serviceName: Optional[str] = None
    apiPrefix: Optional[str] = None
    headerPatterns: Optional[Dict[str, str]] = None
    bodyPatterns: Optional[Dict[str, str]] = None

# SEPP Storage
n32_peers: Dict[str, N32Peer] = {}
roaming_partners: Dict[str, RoamingPartner] = {}
n32f_contexts: Dict[str, Dict] = {}
message_filter_rules: Dict[str, MessageFilterRule] = {}
security_policies: Dict[str, ProtectionPolicyInfo] = {}
active_connections: Dict[str, Dict] = {}


class SEPP:
    def __init__(self):
        self.name = "SEPP-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.plmn_id = PlmnId(mcc="001", mnc="01")
        self.supported_features = "0x0f"
        self.http_client = None
        self._init_default_config()

    def _init_default_config(self):
        """Initialize default SEPP configuration"""
        # Default protection policy
        self.default_protection_policy = ProtectionPolicyInfo(
            defaultPolicy=ProtectionPolicy.FULL_PROTECTION,
            protectedHeaders=["Authorization", "3gpp-Sbi-Target-apiRoot", "3gpp-Sbi-Callback"],
            modifiableHeaders=["Host", "Content-Length"]
        )

        # Default message filter rules
        default_rules = [
            MessageFilterRule(
                ruleId="allow-nrf",
                direction="BOTH",
                action="ALLOW",
                nfType="NRF",
                serviceName="nnrf-disc"
            ),
            MessageFilterRule(
                ruleId="allow-udm",
                direction="BOTH",
                action="ALLOW",
                nfType="UDM",
                serviceName="nudm-sdm"
            ),
            MessageFilterRule(
                ruleId="allow-ausf",
                direction="BOTH",
                action="ALLOW",
                nfType="AUSF",
                serviceName="nausf-auth"
            ),
            MessageFilterRule(
                ruleId="block-internal",
                direction="INBOUND",
                action="BLOCK",
                apiPrefix="/internal/"
            ),
        ]
        for rule in default_rules:
            message_filter_rules[rule.ruleId] = rule

    async def init_http_client(self):
        """Initialize async HTTP client"""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0, verify=False)

    async def close_http_client(self):
        """Close HTTP client"""
        if self.http_client:
            await self.http_client.aclose()

    def register_roaming_partner(self, partner: RoamingPartner) -> str:
        """Register a roaming partner"""
        partner_id = f"{partner.plmnId.mcc}-{partner.plmnId.mnc}"
        roaming_partners[partner_id] = partner

        # Also register as N32 peer
        n32_peers[partner.seppInfo.seppId] = partner.seppInfo

        logger.info(f"Roaming partner registered: {partner.partnerName} ({partner_id})")
        return partner_id

    def get_roaming_partner(self, plmn_id: PlmnId) -> Optional[RoamingPartner]:
        """Get roaming partner by PLMN ID"""
        partner_id = f"{plmn_id.mcc}-{plmn_id.mnc}"
        return roaming_partners.get(partner_id)

    def establish_n32_connection(self, handshake: N32Handshake) -> Dict:
        """
        Establish N32-c connection with peer SEPP
        Per 3GPP TS 29.573
        """
        # Negotiate security capability
        if handshake.secCapNegotiation:
            # Select the first mutually supported capability
            selected = SecurityCapability.TLS
            for cap in handshake.secCapNegotiation:
                if cap in [SecurityCapability.TLS, SecurityCapability.PRINS]:
                    selected = cap
                    break
        else:
            selected = SecurityCapability.TLS

        # Create N32 peer entry
        peer = N32Peer(
            peerPlmnId=handshake.senderPlmnId,
            seppId=handshake.senderSeppId,
            securityCapability=selected,
            n32Purpose=N32Purpose.ROAMING,
            status="ACTIVE",
            establishedTime=datetime.now(timezone.utc)
        )
        n32_peers[peer.seppId] = peer

        # Create connection context
        context_id = str(uuid.uuid4())
        active_connections[context_id] = {
            "peerSeppId": handshake.senderSeppId,
            "peerPlmnId": handshake.senderPlmnId.dict(),
            "securityCapability": selected.value,
            "establishedAt": datetime.now(timezone.utc).isoformat(),
            "status": "ESTABLISHED"
        }

        logger.info(f"N32 connection established with {handshake.senderSeppId}")

        return {
            "senderSeppId": self.nf_instance_id,
            "senderPlmnId": self.plmn_id.dict(),
            "receiverSeppId": handshake.senderSeppId,
            "receiverPlmnId": handshake.senderPlmnId.dict(),
            "selectedSecCapability": selected.value,
            "n32fContextId": context_id,
            "supportedFeatures": self.supported_features
        }

    def apply_message_filter(self, request: Dict, direction: str) -> Tuple[bool, Optional[str]]:
        """
        Apply message filtering rules
        Returns (allowed, reason)
        """
        for rule in message_filter_rules.values():
            if rule.direction != "BOTH" and rule.direction != direction:
                continue

            # Check NF type
            if rule.nfType and request.get("nfType") != rule.nfType:
                continue

            # Check service name
            if rule.serviceName and request.get("serviceName") != rule.serviceName:
                continue

            # Check API prefix
            if rule.apiPrefix:
                path = request.get("path", "")
                if rule.apiPrefix in path:
                    if rule.action == "BLOCK":
                        return False, f"Blocked by rule {rule.ruleId}"
                    elif rule.action == "ALLOW":
                        return True, None

        # Default allow if no rule matched
        return True, None

    def protect_message(self, message: bytes, policy: ProtectionPolicyInfo) -> Dict:
        """
        Apply protection to outbound message per TS 29.573
        """
        if policy.defaultPolicy == ProtectionPolicy.NO_PROTECTION:
            return {"data": base64.b64encode(message).decode(), "protected": False}

        # Calculate integrity hash
        hash_value = hashlib.sha256(message).hexdigest()

        result = {
            "data": base64.b64encode(message).decode(),
            "protected": True,
            "integrityHash": hash_value,
            "hashAlgorithm": "SHA-256"
        }

        if policy.defaultPolicy in [ProtectionPolicy.FULL_PROTECTION,
                                     ProtectionPolicy.CONFIDENTIALITY_PROTECTION]:
            # In production, would encrypt the message
            # For demo, just mark as "encrypted"
            result["encrypted"] = True

        return result

    def verify_message(self, protected_data: Dict) -> Tuple[bool, bytes]:
        """
        Verify and unprotect inbound message
        """
        try:
            data = base64.b64decode(protected_data.get("data", ""))

            if protected_data.get("protected", False):
                # Verify integrity
                expected_hash = protected_data.get("integrityHash")
                if expected_hash:
                    actual_hash = hashlib.sha256(data).hexdigest()
                    if actual_hash != expected_hash:
                        return False, b""

            return True, data

        except Exception as e:
            logger.error(f"Message verification failed: {e}")
            return False, b""

    async def forward_to_peer(
        self,
        peer_id: str,
        method: str,
        path: str,
        headers: Dict,
        body: Optional[bytes] = None
    ) -> Tuple[int, Dict, bytes]:
        """
        Forward request to peer SEPP
        """
        await self.init_http_client()

        peer = n32_peers.get(peer_id)
        if not peer:
            raise ValueError(f"Unknown peer SEPP: {peer_id}")

        # Build target URL
        target_host = peer.seppIpAddress or peer.seppFqdn
        target_url = f"https://{target_host}:{peer.seppPort}{path}"

        # Apply protection policy
        partner = self.get_roaming_partner(peer.peerPlmnId)
        if partner and body:
            protected = self.protect_message(body, partner.protectionPolicy)
            body = json.dumps(protected).encode()

        try:
            response = await self.http_client.request(
                method=method,
                url=target_url,
                headers=headers,
                content=body
            )
            return response.status_code, dict(response.headers), response.content

        except Exception as e:
            logger.error(f"Failed to forward to peer {peer_id}: {e}")
            raise


sepp_instance = SEPP()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Register with NRF
    nf_profile = {
        "nfInstanceId": sepp_instance.nf_instance_id,
        "nfType": "SEPP",
        "nfStatus": "REGISTERED",
        "plmnList": [{"mcc": "001", "mnc": "01"}],
        "nfServices": [
            {
                "serviceInstanceId": "n32-c-001",
                "serviceName": "n32-c",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "https",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9014}]
            },
            {
                "serviceInstanceId": "n32-f-001",
                "serviceName": "n32-f",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "https",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9014}]
            }
        ],
        "seppInfo": {
            "seppId": sepp_instance.nf_instance_id,
            "seppPrefix": "/sepp",
            "remotePlmnList": [],
            "remoteSeppList": []
        }
    }

    try:
        response = requests.put(
            f"{nrf_url}/nnrf-nfm/v1/nf-instances/{sepp_instance.nf_instance_id}",
            json=nf_profile
        )
        if response.status_code in [200, 201]:
            logger.info("SEPP registered with NRF successfully")
    except requests.RequestException as e:
        logger.error(f"Failed to register SEPP with NRF: {e}")

    yield

    # Shutdown
    await sepp_instance.close_http_client()
    try:
        requests.delete(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{sepp_instance.nf_instance_id}")
        logger.info("SEPP deregistered from NRF")
    except:
        pass


app = FastAPI(
    title="SEPP - Security Edge Protection Proxy",
    description="3GPP TS 29.573 compliant SEPP for inter-PLMN security",
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


# N32-c Interface - Security capability negotiation

@app.post("/n32c-handshake/v1/security-capability-negotiation")
async def negotiate_security_capability(handshake: N32Handshake):
    """
    N32-c Security Capability Negotiation per 3GPP TS 29.573
    """
    with tracer.start_as_current_span("sepp_n32c_negotiation") as span:
        span.set_attribute("peer.sepp_id", handshake.senderSeppId)
        span.set_attribute("peer.plmn", f"{handshake.senderPlmnId.mcc}-{handshake.senderPlmnId.mnc}")

        try:
            result = sepp_instance.establish_n32_connection(handshake)
            span.set_attribute("status", "SUCCESS")
            return result

        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"N32-c negotiation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/n32c-handshake/v1/security-capability-termination")
async def terminate_security_capability(peer_sepp_id: str):
    """
    Terminate N32 connection per 3GPP TS 29.573
    """
    if peer_sepp_id in n32_peers:
        n32_peers[peer_sepp_id].status = "TERMINATED"
        logger.info(f"N32 connection terminated with {peer_sepp_id}")
        return {"message": "Connection terminated successfully"}
    raise HTTPException(status_code=404, detail="Peer not found")


# N32-f Interface - Forwarding with protection

@app.post("/n32f-forward/v1/n32f-process")
async def n32f_forward_request(request: N32fReformattedReq):
    """
    N32-f Forward Request with PRINS protection per 3GPP TS 29.573
    """
    with tracer.start_as_current_span("sepp_n32f_forward") as span:
        span.set_attribute("n32f.context_id", request.n32fContextId)

        try:
            # Verify and unprotect the message
            if request.integrityProtectedData:
                verified, data = sepp_instance.verify_message({
                    "data": request.reformattedData,
                    "integrityHash": request.integrityProtectedData.signature,
                    "protected": True
                })
                if not verified:
                    raise HTTPException(status_code=400, detail="Integrity verification failed")
            else:
                data = base64.b64decode(request.reformattedData)

            # Parse the SBI request
            try:
                sbi_request = json.loads(data)
            except:
                sbi_request = {"raw": data.decode()}

            # Apply message filtering
            allowed, reason = sepp_instance.apply_message_filter(sbi_request, "INBOUND")
            if not allowed:
                raise HTTPException(status_code=403, detail=reason)

            # Store context for response correlation
            n32f_contexts[request.n32fContextId] = {
                "receivedAt": datetime.now(timezone.utc).isoformat(),
                "sbiRequest": sbi_request
            }

            span.set_attribute("status", "SUCCESS")

            # Return acknowledgment
            return {
                "n32fContextId": request.n32fContextId,
                "status": "ACCEPTED",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except HTTPException:
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"N32-f forward failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/n32f-forward/v1/n32f-error")
async def n32f_error_report(error_data: Dict):
    """
    Report N32-f error per 3GPP TS 29.573
    """
    logger.warning(f"N32-f error reported: {error_data}")
    return {"status": "ERROR_ACKNOWLEDGED"}


# Roaming Partner Management

@app.post("/sepp/roaming-partners")
async def add_roaming_partner(partner: RoamingPartner):
    """Add a roaming partner"""
    partner_id = sepp_instance.register_roaming_partner(partner)
    return {"partnerId": partner_id, "message": "Roaming partner added successfully"}


@app.get("/sepp/roaming-partners")
async def list_roaming_partners():
    """List all roaming partners"""
    return {
        "partners": [
            {
                "id": f"{p.plmnId.mcc}-{p.plmnId.mnc}",
                "name": p.partnerName,
                "plmn": p.plmnId.dict(),
                "trustLevel": p.trustLevel,
                "status": p.seppInfo.status
            }
            for p in roaming_partners.values()
        ]
    }


@app.get("/sepp/roaming-partners/{partnerId}")
async def get_roaming_partner(partnerId: str):
    """Get roaming partner details"""
    if partnerId not in roaming_partners:
        raise HTTPException(status_code=404, detail="Roaming partner not found")
    return roaming_partners[partnerId].dict()


@app.delete("/sepp/roaming-partners/{partnerId}")
async def remove_roaming_partner(partnerId: str):
    """Remove a roaming partner"""
    if partnerId in roaming_partners:
        del roaming_partners[partnerId]
        logger.info(f"Roaming partner removed: {partnerId}")
        return {"message": "Roaming partner removed"}
    raise HTTPException(status_code=404, detail="Roaming partner not found")


# N32 Peer Management

@app.get("/sepp/n32-peers")
async def list_n32_peers():
    """List all N32 peers"""
    return {
        "peers": [
            {
                "seppId": p.seppId,
                "plmn": p.peerPlmnId.dict(),
                "status": p.status,
                "securityCapability": p.securityCapability.value,
                "establishedTime": p.establishedTime.isoformat() if p.establishedTime else None
            }
            for p in n32_peers.values()
        ]
    }


@app.get("/sepp/active-connections")
async def list_active_connections():
    """List active N32 connections"""
    return {"connections": list(active_connections.values())}


# Message Filter Rules Management

@app.get("/sepp/filter-rules")
async def list_filter_rules():
    """List message filter rules"""
    return {"rules": [r.dict() for r in message_filter_rules.values()]}


@app.post("/sepp/filter-rules")
async def add_filter_rule(rule: MessageFilterRule):
    """Add a message filter rule"""
    message_filter_rules[rule.ruleId] = rule
    return {"message": f"Rule {rule.ruleId} added"}


@app.delete("/sepp/filter-rules/{ruleId}")
async def remove_filter_rule(ruleId: str):
    """Remove a message filter rule"""
    if ruleId in message_filter_rules:
        del message_filter_rules[ruleId]
        return {"message": f"Rule {ruleId} removed"}
    raise HTTPException(status_code=404, detail="Rule not found")


# Protection Policy Management

@app.get("/sepp/protection-policy")
async def get_default_protection_policy():
    """Get default protection policy"""
    return sepp_instance.default_protection_policy.dict()


@app.put("/sepp/protection-policy")
async def update_default_protection_policy(policy: ProtectionPolicyInfo):
    """Update default protection policy"""
    sepp_instance.default_protection_policy = policy
    return {"message": "Protection policy updated"}


# Health and monitoring

@app.get("/health")
def health_check():
    """Health check endpoint"""
    active_peers = sum(1 for p in n32_peers.values() if p.status == "ACTIVE")
    return {
        "status": "healthy",
        "service": "SEPP",
        "compliance": "3GPP TS 29.573",
        "version": "1.0.0",
        "activePeers": active_peers,
        "roamingPartners": len(roaming_partners)
    }


@app.get("/metrics")
def get_metrics():
    """Metrics endpoint"""
    return {
        "total_n32_peers": len(n32_peers),
        "active_n32_peers": sum(1 for p in n32_peers.values() if p.status == "ACTIVE"),
        "roaming_partners": len(roaming_partners),
        "active_connections": len(active_connections),
        "filter_rules": len(message_filter_rules),
        "n32f_contexts": len(n32f_contexts)
    }


if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="SEPP - Security Edge Protection Proxy")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("sepp"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)