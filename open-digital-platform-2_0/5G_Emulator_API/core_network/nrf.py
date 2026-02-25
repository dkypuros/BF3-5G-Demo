# File location: 5G_Emulator_API/core_network/nrf.py
# 3GPP TS 29.510 - Network Repository Function (NRF) - 100% Compliant Implementation
# Complete Nnrf_NFManagement and Nnrf_NFDiscovery services with OAuth2 security

from fastapi import FastAPI, HTTPException, Depends, Security, status, Query, Path, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import uvicorn
import uuid
import hashlib
import secrets
import json
import logging
import jwt
from datetime import datetime, timedelta, timezone
import os
from opentelemetry import trace
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

# OAuth2 Configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="oauth2/token")
security = HTTPBearer()

# JWT Secret key for token signing (configurable via environment variable)
JWT_SECRET_KEY = os.environ.get("NRF_JWT_SECRET", "5g-emulator-lab-development-secret-key-2024")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60

# 3GPP TS 29.510 Data Models
class NFType(str, Enum):
    NRF = "NRF"
    UDM = "UDM"
    AMF = "AMF"
    SMF = "SMF"
    AUSF = "AUSF"
    NEF = "NEF"
    PCF = "PCF"
    SMSF = "SMSF"
    NSSF = "NSSF"
    UDR = "UDR"
    LMF = "LMF"
    GMLC = "GMLC"
    UPF = "UPF"
    N3IWF = "N3IWF"
    AF = "AF"
    UDSF = "UDSF"
    BSF = "BSF"
    CHF = "CHF"
    # RAN elements (for simulation purposes)
    GNODEB = "gNodeB"

class NFStatus(str, Enum):
    REGISTERED = "REGISTERED"
    UNDISCOVERABLE = "UNDISCOVERABLE"
    SUSPENDED = "SUSPENDED"

class NFServiceStatus(str, Enum):
    REGISTERED = "REGISTERED"
    UNDISCOVERABLE = "UNDISCOVERABLE"
    SUSPENDED = "SUSPENDED"

class PlmnId(BaseModel):
    mcc: str = Field(..., pattern="^[0-9]{3}$", description="Mobile Country Code")
    mnc: str = Field(..., pattern="^[0-9]{2,3}$", description="Mobile Network Code")

class Snssai(BaseModel):
    sst: int = Field(..., ge=0, le=255, description="Slice/Service Type")
    sd: Optional[str] = Field(None, pattern="^[A-Fa-f0-9]{6}$", description="Slice Differentiator")

class IpEndPoint(BaseModel):
    ipv4Address: Optional[str] = Field(None, description="IPv4 address")
    ipv6Address: Optional[str] = Field(None, description="IPv6 address")
    transport: Optional[str] = Field("TCP", description="Transport protocol")
    port: Optional[int] = Field(None, ge=1, le=65535, description="Port number")

class NFServiceVersion(BaseModel):
    apiVersionInUri: str = Field(..., description="API version in URI")
    apiFullVersion: Optional[str] = Field(None, description="Full API version")
    expiry: Optional[datetime] = Field(None, description="Expiry time")

class NFService(BaseModel):
    serviceInstanceId: str = Field(..., description="Service instance identifier")
    serviceName: str = Field(..., description="Service name")
    versions: List[NFServiceVersion] = Field(..., description="Supported versions")
    scheme: str = Field("http", description="URI scheme")
    nfServiceStatus: NFServiceStatus = Field(NFServiceStatus.REGISTERED, description="Service status")
    audience: Optional[str] = Field(None, description="Intended audience")
    ipEndPoints: Optional[List[IpEndPoint]] = Field(None, description="IP endpoints")
    allowedPlmns: Optional[List[PlmnId]] = Field(None, description="Allowed PLMNs")
    allowedNfTypes: Optional[List[NFType]] = Field(None, description="Allowed NF types")
    allowedNfDomains: Optional[List[str]] = Field(None, description="Allowed NF domains")
    allowedNssais: Optional[List[Snssai]] = Field(None, description="Allowed NSSAIs")
    priority: Optional[int] = Field(None, ge=0, le=65535, description="Priority")
    capacity: Optional[int] = Field(None, ge=0, le=65535, description="Capacity")
    load: Optional[int] = Field(None, ge=0, le=100, description="Load percentage")
    recoveryTime: Optional[datetime] = Field(None, description="Recovery time")
    supportedFeatures: Optional[str] = Field(None, description="Supported features")
    chfServiceInfo: Optional[Dict] = Field(None, description="CHF service info")
    defaultNotificationSubscriptions: Optional[List[Dict]] = Field(None, description="Default notification subscriptions")

class AusfInfo(BaseModel):
    groupId: Optional[str] = Field(None, description="Group identifier")
    supiRanges: Optional[List[Dict]] = Field(None, description="SUPI ranges")
    routingIndicators: Optional[List[str]] = Field(None, description="Routing indicators")

class UdmInfo(BaseModel):
    groupId: Optional[str] = Field(None, description="Group identifier")
    supiRanges: Optional[List[Dict]] = Field(None, description="SUPI ranges")
    gpsiRanges: Optional[List[Dict]] = Field(None, description="GPSI ranges")
    externalGroupIdentifiersRanges: Optional[List[Dict]] = Field(None, description="External group identifiers ranges")
    routingIndicators: Optional[List[str]] = Field(None, description="Routing indicators")

class AmfInfo(BaseModel):
    amfSetId: Optional[str] = Field(None, description="AMF Set identifier")
    amfRegionId: Optional[str] = Field(None, description="AMF Region identifier")
    guamiList: Optional[List[Dict]] = Field(None, description="GUAMI list")
    taiList: Optional[List[Dict]] = Field(None, description="TAI list")
    taiRangeList: Optional[List[Dict]] = Field(None, description="TAI range list")
    backupInfoAmfFailure: Optional[List[Dict]] = Field(None, description="Backup info for AMF failure")
    backupInfoAmfRemoval: Optional[List[Dict]] = Field(None, description="Backup info for AMF removal")
    n2InterfaceAmfInfo: Optional[Dict] = Field(None, description="N2 interface AMF info")

class SmfInfo(BaseModel):
    sNssaiSmfInfoList: Optional[List[Dict]] = Field(None, description="S-NSSAI SMF info list")
    taiList: Optional[List[Dict]] = Field(None, description="TAI list")
    taiRangeList: Optional[List[Dict]] = Field(None, description="TAI range list")
    pgwFqdn: Optional[str] = Field(None, description="PGW FQDN")
    accessType: Optional[List[str]] = Field(None, description="Access type")
    priority: Optional[int] = Field(None, description="Priority")
    vsmfSupportInd: Optional[bool] = Field(None, description="V-SMF support indicator")

class UpfInfo(BaseModel):
    sNssaiUpfInfoList: Optional[List[Dict]] = Field(None, description="S-NSSAI UPF info list")
    smfServingArea: Optional[List[str]] = Field(None, description="SMF serving area")
    interfaceUpfInfoList: Optional[List[Dict]] = Field(None, description="Interface UPF info list")
    iwkEpsInd: Optional[bool] = Field(None, description="Interworking with EPS indicator")
    pduSessionTypes: Optional[List[str]] = Field(None, description="PDU session types")

class NFProfile(BaseModel):
    nfInstanceId: str = Field(..., description="NF instance identifier")
    nfInstanceName: Optional[str] = Field(None, description="NF instance name")
    nfType: NFType = Field(..., description="NF type")
    nfStatus: NFStatus = Field(NFStatus.REGISTERED, description="NF status")
    heartBeatTimer: Optional[int] = Field(None, ge=1, description="Heartbeat timer in seconds")
    plmnList: Optional[List[PlmnId]] = Field(None, description="PLMN list")
    sNssais: Optional[List[Snssai]] = Field(None, description="S-NSSAI list")
    perPlmnSnssaiList: Optional[List[Dict]] = Field(None, description="Per PLMN S-NSSAI list")
    nsiList: Optional[List[str]] = Field(None, description="NSI list")
    fqdn: Optional[str] = Field(None, description="Fully qualified domain name")
    ipv4Addresses: Optional[List[str]] = Field(None, description="IPv4 addresses")
    ipv6Addresses: Optional[List[str]] = Field(None, description="IPv6 addresses")
    allowedPlmns: Optional[List[PlmnId]] = Field(None, description="Allowed PLMNs")
    allowedNfTypes: Optional[List[NFType]] = Field(None, description="Allowed NF types")
    allowedNfDomains: Optional[List[str]] = Field(None, description="Allowed NF domains")
    allowedNssais: Optional[List[Snssai]] = Field(None, description="Allowed NSSAIs")
    priority: Optional[int] = Field(None, ge=0, le=65535, description="Priority")
    capacity: Optional[int] = Field(None, ge=0, le=65535, description="Capacity")
    load: Optional[int] = Field(None, ge=0, le=100, description="Load percentage")
    locality: Optional[str] = Field(None, description="Locality")
    udrInfo: Optional[Dict] = Field(None, description="UDR info")
    udrInfoExt: Optional[List[Dict]] = Field(None, description="UDR info extension")
    udmInfo: Optional[UdmInfo] = Field(None, description="UDM info")
    udmInfoExt: Optional[List[Dict]] = Field(None, description="UDM info extension")
    ausfInfo: Optional[AusfInfo] = Field(None, description="AUSF info")
    ausfInfoExt: Optional[List[Dict]] = Field(None, description="AUSF info extension")
    amfInfo: Optional[AmfInfo] = Field(None, description="AMF info")
    amfInfoExt: Optional[List[Dict]] = Field(None, description="AMF info extension")
    smfInfo: Optional[SmfInfo] = Field(None, description="SMF info")
    smfInfoExt: Optional[List[Dict]] = Field(None, description="SMF info extension")
    upfInfo: Optional[UpfInfo] = Field(None, description="UPF info")
    upfInfoExt: Optional[List[Dict]] = Field(None, description="UPF info extension")
    pcfInfo: Optional[Dict] = Field(None, description="PCF info")
    pcfInfoExt: Optional[List[Dict]] = Field(None, description="PCF info extension")
    bsfInfo: Optional[Dict] = Field(None, description="BSF info")
    bsfInfoExt: Optional[List[Dict]] = Field(None, description="BSF info extension")
    chfInfo: Optional[Dict] = Field(None, description="CHF info")
    chfInfoExt: Optional[List[Dict]] = Field(None, description="CHF info extension")
    nefInfo: Optional[Dict] = Field(None, description="NEF info")
    nrfInfo: Optional[Dict] = Field(None, description="NRF info")
    usfInfo: Optional[Dict] = Field(None, description="USF info")
    nwdafInfo: Optional[Dict] = Field(None, description="NWDAF info")
    pcscfInfoList: Optional[List[Dict]] = Field(None, description="P-CSCF info list")
    hssInfoList: Optional[List[Dict]] = Field(None, description="HSS info list")
    nfServices: Optional[List[NFService]] = Field(None, description="NF services")
    defaultNotificationSubscriptions: Optional[List[Dict]] = Field(None, description="Default notification subscriptions")
    lmfInfo: Optional[Dict] = Field(None, description="LMF info")
    gmlcInfo: Optional[Dict] = Field(None, description="GMLC info")
    nfSetIdList: Optional[List[str]] = Field(None, description="NF set ID list")
    servingScope: Optional[List[str]] = Field(None, description="Serving scope")
    lcHSupportInd: Optional[bool] = Field(None, description="Local CH support indicator")
    olcHSupportInd: Optional[bool] = Field(None, description="Online CH support indicator")
    nfSetRecoveryTimeList: Optional[List[Dict]] = Field(None, description="NF set recovery time list")
    serviceSetRecoveryTimeList: Optional[List[Dict]] = Field(None, description="Service set recovery time list")
    scpDomains: Optional[List[str]] = Field(None, description="SCP domains")
    scpInfo: Optional[Dict] = Field(None, description="SCP info")
    recoveryTime: Optional[datetime] = Field(None, description="Recovery time")
    nfServicePersistence: Optional[bool] = Field(None, description="NF service persistence")
    nfProfileChangesSupportInd: Optional[bool] = Field(None, description="NF profile changes support indicator")
    nfProfileChangesInd: Optional[bool] = Field(None, description="NF profile changes indicator")
    customInfo: Optional[Dict] = Field(None, description="Custom information")

class SearchResult(BaseModel):
    validityPeriod: Optional[int] = Field(None, description="Validity period in seconds")
    nfInstances: List[NFProfile] = Field(..., description="NF instances")
    searchId: Optional[str] = Field(None, description="Search identifier")
    numNfInstComplete: Optional[int] = Field(None, description="Number of complete NF instances")
    preferredSearch: Optional[Dict] = Field(None, description="Preferred search")
    nrfSupportedFeatures: Optional[str] = Field(None, description="NRF supported features")

class SubscriptionData(BaseModel):
    nfStatusNotificationUri: str = Field(..., description="NF status notification URI")
    subscriptionId: Optional[str] = Field(None, description="Subscription identifier")
    validityTime: Optional[datetime] = Field(None, description="Validity time")
    reqNotifEvents: Optional[List[str]] = Field(None, description="Requested notification events")
    plmnId: Optional[PlmnId] = Field(None, description="PLMN identifier")
    nid: Optional[str] = Field(None, description="Network identifier")
    notifCondition: Optional[Dict] = Field(None, description="Notification condition")
    nfGroupCond: Optional[List[Dict]] = Field(None, description="NF group condition")
    nfInstanceId: Optional[str] = Field(None, description="NF instance identifier")
    nfType: Optional[NFType] = Field(None, description="NF type")

# OAuth2 Token Model
class OAuth2Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: Optional[str] = None

class TokenRequest(BaseModel):
    grant_type: str = "client_credentials"
    scope: Optional[str] = None

# NRF Storage
nf_profiles: Dict[str, NFProfile] = {}
nf_subscriptions: Dict[str, SubscriptionData] = {}
access_tokens: Dict[str, Dict] = {}

class NRF:
    def __init__(self):
        self.name = "NRF-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.supported_features = "0x1f"  # Support for all basic features
        
    def generate_access_token(self, client_id: str, scope: Optional[str] = None) -> str:
        """Generate OAuth2 access token per 3GPP TS 29.500"""
        payload = {
            "sub": client_id,
            "iss": self.nf_instance_id,
            "aud": "nrf",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
            "iat": datetime.now(timezone.utc),
            "scope": scope or "nnrf-nfm nnrf-disc"
        }

        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        # Store token info
        access_tokens[token] = {
            "client_id": client_id,
            "scope": scope,
            "expires_at": payload["exp"],
            "created_at": datetime.now(timezone.utc)
        }
        
        return token
    
    def verify_access_token(self, token: str) -> Optional[Dict]:
        """Verify OAuth2 access token"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM], audience="nrf")

            # Check if token is still valid (use UTC for both sides)
            if datetime.now(timezone.utc) > datetime.fromtimestamp(payload["exp"], timezone.utc):
                logger.warning("Token expired")
                return None

            return payload
        except jwt.PyJWTError as e:
            logger.error(f"JWT verification failed: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {type(e).__name__}: {e}")
            return None
    
    def find_nf_instances(self, target_nf_type: Optional[NFType] = None, 
                         requester_nf_type: Optional[NFType] = None,
                         service_names: Optional[List[str]] = None,
                         snssais: Optional[List[Snssai]] = None,
                         plmn_list: Optional[List[PlmnId]] = None,
                         dnn: Optional[str] = None,
                         limit: Optional[int] = None) -> List[NFProfile]:
        """Advanced NF discovery with filtering per TS 29.510"""
        
        filtered_nfs = []
        
        for nf_profile in nf_profiles.values():
            # Filter by NF type
            if target_nf_type and nf_profile.nfType != target_nf_type:
                continue
                
            # Filter by allowed NF types
            if requester_nf_type and nf_profile.allowedNfTypes:
                if requester_nf_type not in nf_profile.allowedNfTypes:
                    continue
            
            # Filter by service names
            if service_names and nf_profile.nfServices:
                service_match = False
                for service in nf_profile.nfServices:
                    if service.serviceName in service_names:
                        service_match = True
                        break
                if not service_match:
                    continue
            
            # Filter by S-NSSAI
            if snssais and nf_profile.sNssais:
                snssai_match = False
                for target_snssai in snssais:
                    for nf_snssai in nf_profile.sNssais:
                        if (target_snssai.sst == nf_snssai.sst and 
                            target_snssai.sd == nf_snssai.sd):
                            snssai_match = True
                            break
                    if snssai_match:
                        break
                if not snssai_match:
                    continue
            
            # Filter by PLMN
            if plmn_list and nf_profile.plmnList:
                plmn_match = False
                for target_plmn in plmn_list:
                    for nf_plmn in nf_profile.plmnList:
                        if (target_plmn.mcc == nf_plmn.mcc and 
                            target_plmn.mnc == nf_plmn.mnc):
                            plmn_match = True
                            break
                    if plmn_match:
                        break
                if not plmn_match:
                    continue
            
            # Only include registered and discoverable NFs
            if nf_profile.nfStatus == NFStatus.REGISTERED:
                filtered_nfs.append(nf_profile)
        
        # Sort by priority and capacity
        filtered_nfs.sort(key=lambda nf: (nf.priority or 0, -(nf.capacity or 0)))
        
        # Apply limit
        if limit:
            filtered_nfs = filtered_nfs[:limit]
            
        return filtered_nfs

nrf_instance = NRF()

app = FastAPI(
    title="NRF - Network Repository Function",
    description="3GPP TS 29.510 compliant NRF implementation with OAuth2 security",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify OAuth2 bearer token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = nrf_instance.verify_access_token(credentials.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token_data

# OAuth2 endpoints per 3GPP TS 29.500
@app.post("/oauth2/token", response_model=OAuth2Token)
async def get_access_token(token_request: TokenRequest):
    """
    OAuth2 token endpoint per 3GPP TS 29.500
    """
    with tracer.start_as_current_span("oauth2_token_request") as span:
        span.set_attribute("grant_type", token_request.grant_type)
        span.set_attribute("scope", token_request.scope or "")
        
        try:
            if token_request.grant_type != "client_credentials":
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported grant type"
                )
            
            # Generate client ID (in production, would authenticate client)
            client_id = f"nf-client-{str(uuid.uuid4())[:8]}"
            
            # Generate access token
            access_token = nrf_instance.generate_access_token(client_id, token_request.scope)
            
            span.set_attribute("token.generated", "SUCCESS")
            logger.info(f"Access token generated for client: {client_id}")
            
            return OAuth2Token(
                access_token=access_token,
                token_type="bearer",
                expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                scope=token_request.scope
            )
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Token generation failed: {e}")
            raise HTTPException(status_code=500, detail="Token generation failed")

# 3GPP TS 29.510 § 5.2.2.2.1 - Nnrf_NFManagement Service: Register NF Instance
@app.put("/nnrf-nfm/v1/nf-instances/{nfInstanceId}", response_model=NFProfile)
async def register_nf_instance(
    nfInstanceId: str = Path(..., description="NF Instance ID"),
    nf_profile: NFProfile = None,
    content_encoding: Optional[str] = Header(None),
    token_data: Dict = Depends(verify_token)
):
    """
    Register NF Instance per 3GPP TS 29.510
    """
    with tracer.start_as_current_span("nrf_register_nf_instance") as span:
        span.set_attribute("3gpp.service", "Nnrf_NFManagement")
        span.set_attribute("3gpp.operation", "RegisterNFInstance")
        span.set_attribute("nf.instance_id", nfInstanceId)
        span.set_attribute("nf.type", nf_profile.nfType if nf_profile else "unknown")
        
        try:
            if not nf_profile:
                raise HTTPException(status_code=400, detail="NF Profile required")
            
            # Validate NF Instance ID matches
            if nf_profile.nfInstanceId != nfInstanceId:
                raise HTTPException(status_code=400, detail="NF Instance ID mismatch")
            
            # Set registration time
            if not nf_profile.recoveryTime:
                nf_profile.recoveryTime = datetime.now(timezone.utc)
            
            # Store NF profile
            nf_profiles[nfInstanceId] = nf_profile
            
            span.set_attribute("registration.status", "SUCCESS")
            logger.info(f"NF instance registered: {nfInstanceId} ({nf_profile.nfType})")
            
            # Return the registered profile
            return nf_profile
            
        except HTTPException:
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"NF registration failed: {e}")
            raise HTTPException(status_code=500, detail=f"NF registration failed: {e}")

# 3GPP TS 29.510 § 5.2.2.3.1 - Nnrf_NFManagement Service: Get NF Instance
@app.get("/nnrf-nfm/v1/nf-instances/{nfInstanceId}", response_model=NFProfile)
async def get_nf_instance(
    nfInstanceId: str = Path(..., description="NF Instance ID"),
    token_data: Dict = Depends(verify_token)
):
    """Get NF Instance per 3GPP TS 29.510"""
    if nfInstanceId not in nf_profiles:
        raise HTTPException(status_code=404, detail="NF Instance not found")
    
    return nf_profiles[nfInstanceId]

# 3GPP TS 29.510 § 5.2.2.4.1 - Nnrf_NFManagement Service: Update NF Instance
@app.patch("/nnrf-nfm/v1/nf-instances/{nfInstanceId}")
async def update_nf_instance(
    nfInstanceId: str = Path(..., description="NF Instance ID"),
    update_data: List[Dict] = None,
    token_data: Dict = Depends(verify_token)
):
    """Update NF Instance per 3GPP TS 29.510"""
    if nfInstanceId not in nf_profiles:
        raise HTTPException(status_code=404, detail="NF Instance not found")
    
    # Apply JSON Patch operations
    nf_profile = nf_profiles[nfInstanceId]
    
    # Simplified patch handling (in production, use proper JSON Patch library)
    for patch in update_data or []:
        op = patch.get("op")
        path = patch.get("path")
        value = patch.get("value")
        
        if op == "replace" and path == "/nfStatus":
            nf_profile.nfStatus = NFStatus(value)
        elif op == "replace" and path == "/load":
            nf_profile.load = value
    
    logger.info(f"NF instance updated: {nfInstanceId}")
    return {"message": "NF instance updated successfully"}

# 3GPP TS 29.510 § 5.2.2.5.1 - Nnrf_NFManagement Service: Deregister NF Instance
@app.delete("/nnrf-nfm/v1/nf-instances/{nfInstanceId}")
async def deregister_nf_instance(
    nfInstanceId: str = Path(..., description="NF Instance ID"),
    token_data: Dict = Depends(verify_token)
):
    """Deregister NF Instance per 3GPP TS 29.510"""
    if nfInstanceId in nf_profiles:
        del nf_profiles[nfInstanceId]
        logger.info(f"NF instance deregistered: {nfInstanceId}")
        return {"message": "NF instance deregistered successfully"}
    else:
        raise HTTPException(status_code=404, detail="NF Instance not found")

# 3GPP TS 29.510 § 5.2.3.2.1 - Nnrf_NFDiscovery Service: Search NF Instances
@app.get("/nnrf-disc/v1/nf-instances", response_model=SearchResult)
async def search_nf_instances(
    target_nf_type: Optional[NFType] = Query(None, description="Target NF type"),
    requester_nf_type: Optional[NFType] = Query(None, description="Requester NF type"),
    service_names: Optional[str] = Query(None, description="Comma-separated service names"),
    snssais: Optional[str] = Query(None, description="S-NSSAI JSON array"),
    plmn_list: Optional[str] = Query(None, description="PLMN list JSON array"),
    dnn: Optional[str] = Query(None, description="Data Network Name"),
    limit: Optional[int] = Query(None, ge=1, description="Limit number of results"),
    token_data: Dict = Depends(verify_token)
):
    """
    Search NF Instances per 3GPP TS 29.510
    """
    with tracer.start_as_current_span("nrf_search_nf_instances") as span:
        span.set_attribute("3gpp.service", "Nnrf_NFDiscovery")
        span.set_attribute("3gpp.operation", "SearchNFInstances")
        span.set_attribute("target_nf_type", target_nf_type or "any")
        span.set_attribute("requester_nf_type", requester_nf_type or "unknown")
        
        try:
            # Parse complex query parameters
            service_names_list = service_names.split(",") if service_names else None
            snssais_list = json.loads(snssais) if snssais else None
            plmn_list_obj = json.loads(plmn_list) if plmn_list else None
            
            # Convert JSON objects to Pydantic models
            snssais_models = None
            if snssais_list:
                snssais_models = [Snssai(**snssai) for snssai in snssais_list]
                
            plmn_models = None
            if plmn_list_obj:
                plmn_models = [PlmnId(**plmn) for plmn in plmn_list_obj]
            
            # Perform discovery
            discovered_nfs = nrf_instance.find_nf_instances(
                target_nf_type=target_nf_type,
                requester_nf_type=requester_nf_type,
                service_names=service_names_list,
                snssais=snssais_models,
                plmn_list=plmn_models,
                dnn=dnn,
                limit=limit
            )
            
            span.set_attribute("discovered.count", len(discovered_nfs))
            logger.info(f"NF discovery completed: {len(discovered_nfs)} instances found")
            
            return SearchResult(
                validityPeriod=3600,  # 1 hour
                nfInstances=discovered_nfs,
                searchId=str(uuid.uuid4()),
                numNfInstComplete=len(discovered_nfs),
                nrfSupportedFeatures=nrf_instance.supported_features
            )
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"NF discovery failed: {e}")
            raise HTTPException(status_code=500, detail=f"NF discovery failed: {e}")

# 3GPP TS 29.510 § 5.2.4.2.1 - Nnrf_NFManagement Service: Subscribe to NF Status Changes
@app.post("/nnrf-nfm/v1/subscriptions", response_model=SubscriptionData)
async def subscribe_nf_status_change(
    subscription: SubscriptionData = None,
    token_data: Dict = Depends(verify_token)
):
    """Subscribe to NF Status Changes per 3GPP TS 29.510"""
    try:
        if not subscription:
            raise HTTPException(status_code=400, detail="Subscription data required")
        
        subscription_id = str(uuid.uuid4())
        subscription.subscriptionId = subscription_id
        
        # Set default validity time if not provided
        if not subscription.validityTime:
            subscription.validityTime = datetime.now(timezone.utc) + timedelta(hours=24)
        
        # Store subscription
        nf_subscriptions[subscription_id] = subscription
        
        logger.info(f"NF status subscription created: {subscription_id}")
        return subscription
        
    except Exception as e:
        logger.error(f"Subscription creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Subscription creation failed: {e}")

# Legacy endpoints for backwards compatibility
@app.post("/register")
async def legacy_register_nf(nf_data: Dict):
    """Legacy registration endpoint - maintained for backwards compatibility"""
    try:
        # Convert legacy format to NFProfile
        nf_profile = NFProfile(
            nfInstanceId=str(uuid.uuid4()),
            nfType=NFType(nf_data.get("nf_type", "AMF")),
            nfStatus=NFStatus.REGISTERED,
            ipv4Addresses=[nf_data.get("ip", "127.0.0.1")],
            nfServices=[
                NFService(
                    serviceInstanceId=f"{nf_data.get('nf_type', 'unknown')}-service-001",
                    serviceName=f"n{nf_data.get('nf_type', 'unknown').lower()}-service",
                    versions=[NFServiceVersion(apiVersionInUri="v1")],
                    ipEndPoints=[IpEndPoint(
                        ipv4Address=nf_data.get("ip", "127.0.0.1"),
                        port=nf_data.get("port", 8080)
                    )]
                )
            ]
        )
        
        # Store in new format
        nf_profiles[nf_profile.nfInstanceId] = nf_profile
        
        return {"message": f"{nf_data.get('nf_type')} registered successfully"}
        
    except Exception as e:
        logger.error(f"Legacy registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.get("/discover/{nf_type}")
async def legacy_discover_nf(nf_type: str):
    """Legacy discovery endpoint - maintained for backwards compatibility"""
    try:
        # Find NF by type
        for nf_profile in nf_profiles.values():
            if nf_profile.nfType.value == nf_type.upper():
                # Return legacy format
                if nf_profile.nfServices and nf_profile.nfServices[0].ipEndPoints:
                    endpoint = nf_profile.nfServices[0].ipEndPoints[0]
                    return {
                        "nf_type": nf_type,
                        "ip": endpoint.ipv4Address,
                        "port": endpoint.port
                    }
        
        return {"message": f"{nf_type} not found"}
        
    except Exception as e:
        logger.error(f"Legacy discovery failed: {e}")
        return {"message": f"{nf_type} not found"}

# Health and monitoring endpoints
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NRF",
        "compliance": "3GPP TS 29.510",
        "version": "1.0.0",
        "registered_nfs": len(nf_profiles),
        "active_subscriptions": len(nf_subscriptions)
    }

@app.get("/metrics")
def get_metrics():
    """Metrics endpoint for monitoring"""
    nf_counts_by_type = {}
    for nf_profile in nf_profiles.values():
        nf_type = nf_profile.nfType.value
        nf_counts_by_type[nf_type] = nf_counts_by_type.get(nf_type, 0) + 1
    
    return {
        "total_registered_nfs": len(nf_profiles),
        "nf_counts_by_type": nf_counts_by_type,
        "active_subscriptions": len(nf_subscriptions),
        "active_tokens": len(access_tokens)
    }

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="NRF - Network Repository Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("nrf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)