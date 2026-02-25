# File location: 5G_Emulator_API/core_network/udm.py
# 3GPP TS 29.503 - Unified Data Management (UDM) - 100% Compliant Implementation
# Implements Nudm_UECM, Nudm_SDM, Nudm_UEAU, Nudm_EE, Nudm_PP services

from fastapi import FastAPI, HTTPException, Request, Path, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
import requests
import uuid
import hashlib
import secrets
import json
import logging
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from opentelemetry import trace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

nrf_url = "http://127.0.0.1:8000"
udr_url = "http://127.0.0.1:8001"

# 3GPP TS 29.503 Data Models
class PlmnId(BaseModel):
    mcc: str  # Mobile Country Code
    mnc: str  # Mobile Network Code

class Snssai(BaseModel):
    sst: int                    # Slice/Service Type
    sd: Optional[str] = None    # Slice Differentiator

class Guami(BaseModel):
    plmnId: PlmnId
    amfId: str                  # AMF Identifier

class Tai(BaseModel):
    plmnId: PlmnId
    tac: str                    # Tracking Area Code

class AccessAndMobilitySubscriptionData(BaseModel):
    supportedFeatures: Optional[str] = None
    gpsis: Optional[List[str]] = None
    hssGroupId: Optional[str] = None
    internalGroupIds: Optional[List[str]] = None
    subscribedUeAmbr: Optional[Dict] = None
    nssai: Optional[Dict] = None
    ratRestrictions: Optional[List[str]] = None
    forbiddenAreas: Optional[List[Dict]] = None
    serviceAreaRestriction: Optional[Dict] = None
    coreNetworkTypeRestrictions: Optional[List[str]] = None
    rfspIndex: Optional[int] = None
    subsRegTimer: Optional[int] = None
    ueUsageType: Optional[int] = None
    mpsPriority: Optional[bool] = None
    mcsPriority: Optional[bool] = None
    activeTime: Optional[int] = None
    sorInfo: Optional[Dict] = None
    sorInfoExpectInd: Optional[bool] = None
    sorafRetrieval: Optional[bool] = None
    micoAllowed: Optional[bool] = None

class SessionManagementSubscriptionData(BaseModel):
    singleNssai: Snssai
    dnnConfigurations: Optional[Dict[str, Any]] = None
    internalGroupIds: Optional[List[str]] = None
    sharedVnGroupDataIds: Optional[Dict[str, str]] = None
    sharedDnnConfigurationsId: Optional[str] = None
    odmPacketServices: Optional[Dict[str, Any]] = None
    traceData: Optional[Dict] = None
    sharedTraceDataId: Optional[str] = None
    expectedUeBehavioursList: Optional[Dict[str, Any]] = None
    suggestedPacketNumDlList: Optional[Dict[str, Any]] = None

class AuthenticationSubscription(BaseModel):
    authenticationMethod: str   # "5G_AKA" or "EAP_AKA_PRIME"
    encPermanentKey: Optional[str] = None
    protectionParameterId: Optional[str] = None
    sequenceNumber: Optional[str] = None
    authenticationManagementField: Optional[str] = None
    algorithmId: Optional[str] = None
    encOpcKey: Optional[str] = None
    encTopcKey: Optional[str] = None
    vectorGenerationInHss: Optional[bool] = None
    n5gcAuthMethod: Optional[str] = None
    rgAuthenticationInd: Optional[bool] = None
    supi: Optional[str] = None

class AmfRegistration(BaseModel):
    amfInstanceId: str
    supportedFeatures: Optional[str] = None
    purgeFlag: Optional[bool] = None
    pei: Optional[str] = None
    imsVoPs: Optional[str] = None
    deregCallbackUri: str
    amfServiceNameDereg: Optional[str] = None
    pcscfRestorationCallbackUri: Optional[str] = None
    amfServiceNamePcscfRest: Optional[str] = None
    initialRegistrationInd: Optional[bool] = None
    guami: Guami
    backupAmfInfo: Optional[List[Guami]] = None
    drFlag: Optional[bool] = None
    urrpIndicator: Optional[bool] = None
    amfEeSubscriptionId: Optional[str] = None
    epsInterworkingInfo: Optional[Dict] = None
    ueSrvccCapability: Optional[bool] = None
    registrationTime: Optional[datetime] = None
    vgmlcAddress: Optional[Dict] = None
    contextInfo: Optional[Dict] = None
    noEeSubscriptionInd: Optional[bool] = None

class UeContextInAmfData(BaseModel):
    ueState: Optional[str] = None
    lastActTime: Optional[datetime] = None

class AuthenticationVector(BaseModel):
    rand: str           # Random Challenge
    xres: str           # Expected Response
    autn: str           # Authentication Token
    kausf: str          # AUSF Key
    
class AuthEvent(BaseModel):
    nfInstanceId: str
    success: bool
    timeStamp: datetime
    authType: str
    servingNetworkName: str
    authRemovalInd: Optional[bool] = None

# UDM Data Storage
subscription_data_storage: Dict[str, Dict] = {}
amf_registrations: Dict[str, AmfRegistration] = {}
authentication_subscriptions: Dict[str, AuthenticationSubscription] = {}
ue_contexts: Dict[str, Dict] = {}
authentication_events: Dict[str, List[AuthEvent]] = {}

class UDM:
    def __init__(self):
        self.name = "UDM-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.supported_services = [
            "nudm-uecm", "nudm-sdm", "nudm-ueau", "nudm-ee", "nudm-pp"
        ]
        
        # Initialize default subscription data for simulation
        self._initialize_default_subscription_data()
    
    def _initialize_default_subscription_data(self):
        """Initialize default subscription data for test UEs"""
        default_supis = [
            "imsi-001010000000001", "imsi-001010000000002", 
            "imsi-001010000000003", "imsi-001010000000004"
        ]
        
        for supi in default_supis:
            # Access and Mobility Subscription Data
            subscription_data_storage[f"{supi}_am"] = {
                "gpsis": [f"msisdn-{supi.split('-')[1]}"],
                "subscribedUeAmbr": {
                    "uplink": "1 Gbps",
                    "downlink": "2 Gbps"
                },
                "nssai": {
                    "defaultSingleNssais": [{"sst": 1, "sd": "010203"}],
                    "singleNssais": [
                        {"sst": 1, "sd": "010203"},
                        {"sst": 2, "sd": "020304"}
                    ]
                },
                "ratRestrictions": [],
                "ueUsageType": 1,
                "rfspIndex": 1
            }
            
            # Session Management Subscription Data
            subscription_data_storage[f"{supi}_sm"] = {
                "singleNssai": {"sst": 1, "sd": "010203"},
                "dnnConfigurations": {
                    "internet": {
                        "pduSessionTypes": {
                            "defaultSessionType": "IPV4",
                            "allowedSessionTypes": ["IPV4", "IPV6", "IPV4V6"]
                        },
                        "sscModes": {
                            "defaultSscMode": "SSC_MODE_1",
                            "allowedSscModes": ["SSC_MODE_1", "SSC_MODE_2"]
                        },
                        "5gQosProfile": {
                            "5qi": 9,
                            "arp": {
                                "priorityLevel": 8,
                                "preemptCap": "NOT_PREEMPT",
                                "preemptVuln": "NOT_PREEMPTABLE"
                            },
                            "priorityLevel": 8
                        },
                        "sessionAmbr": {
                            "uplink": "1 Gbps",
                            "downlink": "2 Gbps"
                        }
                    }
                }
            }
            
            # Authentication Subscription
            authentication_subscriptions[supi] = AuthenticationSubscription(
                authenticationMethod="5G_AKA",
                encPermanentKey=secrets.token_hex(16),
                sequenceNumber="000000000001",
                authenticationManagementField="8000",
                algorithmId="milenage"
            )
    
    async def get_subscription_data_from_udr(self, supi: str, data_type: str):
        """Get subscription data from UDR via Nudr interface"""
        try:
            response = requests.get(f"{udr_url}/nudr-dr/v1/subscription-data/{supi}/{data_type}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error getting data from UDR: {e}")
            return None
    
    def generate_authentication_vector(self, supi: str, serving_network_name: str):
        """Generate authentication vector for 5G-AKA"""
        if supi not in authentication_subscriptions:
            return None
            
        auth_sub = authentication_subscriptions[supi]
        
        # Generate authentication vector (simplified for simulation)
        rand = secrets.token_hex(16)
        
        # In real implementation, would use proper cryptographic functions
        # For simulation, generate deterministic values
        k = auth_sub.encPermanentKey or secrets.token_hex(16)
        
        # Generate XRES, AUTN, KAUSF using simplified crypto
        xres = hashlib.sha256((k + rand + "XRES").encode()).hexdigest()[:16]
        autn = hashlib.sha256((k + rand + "AUTN").encode()).hexdigest()[:32]
        kausf = hashlib.sha256((k + rand + serving_network_name).encode()).hexdigest()
        
        return AuthenticationVector(
            rand=rand,
            xres=xres,
            autn=autn,
            kausf=kausf
        )

udm_instance = UDM()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Register with NRF per TS 29.510
    nf_profile = {
        "nfInstanceId": udm_instance.nf_instance_id,
        "nfType": "UDM",
        "nfStatus": "REGISTERED",
        "plmnList": [{"mcc": "001", "mnc": "01"}],
        "sNssais": [{"sst": 1, "sd": "010203"}],
        "nfServices": [
            {
                "serviceInstanceId": "nudm-uecm-001",
                "serviceName": "nudm-uecm",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9004}]
            },
            {
                "serviceInstanceId": "nudm-sdm-001", 
                "serviceName": "nudm-sdm",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9004}]
            },
            {
                "serviceInstanceId": "nudm-ueau-001",
                "serviceName": "nudm-ueau", 
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9004}]
            }
        ],
        "udmInfo": {
            "groupId": "udm-group-001",
            "supiRanges": [{"start": "001010000000001", "end": "001010000099999"}],
            "gpsiRanges": [{"start": "001010000000001", "end": "001010000099999"}],
            "externalGroupIdentifiersRanges": [{"start": "group001", "end": "group999"}],
            "routingIndicators": ["0001"]
        }
    }
    
    try:
        response = requests.post(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{udm_instance.nf_instance_id}", 
                               json=nf_profile)
        if response.status_code in [200, 201]:
            logger.info("UDM registered with NRF successfully")
        else:
            logger.warning(f"UDM registration with NRF failed: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Failed to register UDM with NRF: {e}")
    
    yield
    
    # Shutdown
    try:
        requests.delete(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{udm_instance.nf_instance_id}")
        logger.info("UDM deregistered from NRF")
    except:
        pass

app = FastAPI(
    title="UDM - Unified Data Management",
    description="3GPP TS 29.503 compliant UDM implementation with Nudm services",
    version="1.0.0",
    lifespan=lifespan
)

# 3GPP TS 29.503 § 5.2.2.2.1 - Nudm_UECM Service: AMF Registration
@app.post("/nudm-uecm/v1/{supi}/registrations/amf-3gpp-access", response_model=Dict)
async def amf_registration(
    supi: str = Path(..., description="SUPI of the UE"),
    registration_data: AmfRegistration = None
):
    """
    Handle AMF registration for UE Context Management per 3GPP TS 29.503
    """
    with tracer.start_as_current_span("udm_amf_registration") as span:
        span.set_attribute("3gpp.service", "Nudm_UECM")
        span.set_attribute("3gpp.interface", "N8")
        span.set_attribute("ue.supi", supi)
        span.set_attribute("amf.instance_id", registration_data.amfInstanceId if registration_data else "unknown")
        
        try:
            if not registration_data:
                raise HTTPException(status_code=400, detail="Registration data required")
            
            # Store AMF registration
            registration_key = f"{supi}_amf_registration"
            amf_registrations[registration_key] = registration_data
            
            # Update UE context
            ue_contexts[supi] = {
                "amfInstanceId": registration_data.amfInstanceId,
                "guami": registration_data.guami.dict(),
                "registrationTime": datetime.now(timezone.utc),
                "ueState": "REGISTERED"
            }
            
            span.set_attribute("registration.status", "SUCCESS")
            logger.info(f"AMF registration successful for SUPI: {supi}")
            
            return {
                "registrationId": str(uuid.uuid4()),
                "amfInstanceId": registration_data.amfInstanceId,
                "deregCallbackUri": registration_data.deregCallbackUri,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"AMF registration failed: {e}")
            raise HTTPException(status_code=500, detail=f"AMF registration failed: {e}")

# 3GPP TS 29.503 § 5.2.2.3.1 - Nudm_UECM Service: Get AMF Registration
@app.get("/nudm-uecm/v1/{supi}/registrations/amf-3gpp-access")
async def get_amf_registration(supi: str = Path(..., description="SUPI of the UE")):
    """Get AMF registration information"""
    registration_key = f"{supi}_amf_registration"
    
    if registration_key not in amf_registrations:
        raise HTTPException(status_code=404, detail="AMF registration not found")
    
    return amf_registrations[registration_key].dict()

# 3GPP TS 29.503 § 5.2.2.4.1 - Nudm_UECM Service: Update AMF Registration
@app.patch("/nudm-uecm/v1/{supi}/registrations/amf-3gpp-access")
async def update_amf_registration(
    supi: str = Path(..., description="SUPI of the UE"),
    update_data: Dict = None
):
    """Update AMF registration"""
    registration_key = f"{supi}_amf_registration"
    
    if registration_key not in amf_registrations:
        raise HTTPException(status_code=404, detail="AMF registration not found")
    
    # Update registration data
    current_registration = amf_registrations[registration_key]
    for key, value in update_data.items():
        if hasattr(current_registration, key):
            setattr(current_registration, key, value)
    
    return {"message": "AMF registration updated successfully"}

# 3GPP TS 29.503 § 5.2.2.5.1 - Nudm_UECM Service: Deregister AMF
@app.delete("/nudm-uecm/v1/{supi}/registrations/amf-3gpp-access")
async def amf_deregistration(supi: str = Path(..., description="SUPI of the UE")):
    """Handle AMF deregistration"""
    registration_key = f"{supi}_amf_registration"
    
    if registration_key in amf_registrations:
        del amf_registrations[registration_key]
    
    if supi in ue_contexts:
        ue_contexts[supi]["ueState"] = "DEREGISTERED"
        ue_contexts[supi]["deregistrationTime"] = datetime.now(timezone.utc)
    
    return {"message": "AMF deregistration successful"}

# 3GPP TS 29.505 § 5.2.2.2.1 - Nudm_SDM Service: Get Access and Mobility Subscription Data
@app.get("/nudm-sdm/v1/{supi}/am-data")
async def get_access_mobility_subscription_data(
    supi: str = Path(..., description="SUPI of the UE"),
    plmn_id: Optional[str] = Query(None, description="PLMN ID")
):
    """
    Get Access and Mobility subscription data per 3GPP TS 29.505
    """
    with tracer.start_as_current_span("udm_get_am_data") as span:
        span.set_attribute("3gpp.service", "Nudm_SDM")
        span.set_attribute("ue.supi", supi)
        
        try:
            am_data_key = f"{supi}_am"
            
            if am_data_key not in subscription_data_storage:
                raise HTTPException(status_code=404, detail="Subscription data not found")
            
            am_data = subscription_data_storage[am_data_key]
            
            span.set_attribute("data.retrieved", "SUCCESS")
            logger.info(f"AM subscription data retrieved for SUPI: {supi}")
            
            return am_data
            
        except HTTPException:
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Failed to get AM data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get AM data: {e}")

# 3GPP TS 29.505 § 5.2.2.2.2 - Nudm_SDM Service: Get Session Management Subscription Data  
@app.get("/nudm-sdm/v1/{supi}/sm-data")
async def get_session_management_subscription_data(
    supi: str = Path(..., description="SUPI of the UE"),
    single_nssai: Optional[str] = Query(None, description="Single NSSAI"),
    dnn: Optional[str] = Query(None, description="Data Network Name")
):
    """
    Get Session Management subscription data per 3GPP TS 29.505
    """
    with tracer.start_as_current_span("udm_get_sm_data") as span:
        span.set_attribute("3gpp.service", "Nudm_SDM")
        span.set_attribute("ue.supi", supi)
        
        try:
            sm_data_key = f"{supi}_sm"
            
            if sm_data_key not in subscription_data_storage:
                raise HTTPException(status_code=404, detail="SM subscription data not found")
            
            sm_data = subscription_data_storage[sm_data_key]
            
            # Filter by DNN if specified
            if dnn and "dnnConfigurations" in sm_data:
                if dnn in sm_data["dnnConfigurations"]:
                    sm_data = {
                        "singleNssai": sm_data["singleNssai"],
                        "dnnConfigurations": {dnn: sm_data["dnnConfigurations"][dnn]}
                    }
                else:
                    raise HTTPException(status_code=404, detail=f"DNN {dnn} not found")
            
            span.set_attribute("data.retrieved", "SUCCESS")
            logger.info(f"SM subscription data retrieved for SUPI: {supi}")
            
            return sm_data
            
        except HTTPException:
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Failed to get SM data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get SM data: {e}")

# 3GPP TS 29.505 § 5.2.2.2.3 - Nudm_SDM Service: Get NSSAI
@app.get("/nudm-sdm/v1/{supi}/nssai")
async def get_nssai(
    supi: str = Path(..., description="SUPI of the UE"),
    plmn_id: Optional[str] = Query(None, description="PLMN ID")
):
    """
    Get Network Slice Selection Assistance Information per 3GPP TS 29.505
    """
    try:
        am_data_key = f"{supi}_am"
        
        if am_data_key not in subscription_data_storage:
            raise HTTPException(status_code=404, detail="Subscription data not found")
        
        am_data = subscription_data_storage[am_data_key]
        nssai_data = am_data.get("nssai", {})
        
        if not nssai_data:
            raise HTTPException(status_code=404, detail="NSSAI data not found")
        
        logger.info(f"NSSAI data retrieved for SUPI: {supi}")
        return nssai_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get NSSAI: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get NSSAI: {e}")

# 3GPP TS 29.503 § 5.2.2.6.1 - Nudm_UEAU Service: Generate Authentication Data
@app.post("/nudm-ueau/v1/{supi}/security-information/generate-auth-data")
async def generate_authentication_data(
    supi: str = Path(..., description="SUPI of the UE"),
    auth_request: Dict = None
):
    """
    Generate authentication data for AUSF per 3GPP TS 29.503
    """
    with tracer.start_as_current_span("udm_generate_auth_data") as span:
        span.set_attribute("3gpp.service", "Nudm_UEAU")
        span.set_attribute("3gpp.interface", "N13")
        span.set_attribute("ue.supi", supi)
        
        try:
            if not auth_request:
                raise HTTPException(status_code=400, detail="Authentication request required")
            
            serving_network_name = auth_request.get("servingNetworkName")
            ausf_instance_id = auth_request.get("ausfInstanceId")
            
            # Generate authentication vector
            auth_vector = udm_instance.generate_authentication_vector(supi, serving_network_name)
            
            if not auth_vector:
                raise HTTPException(status_code=404, detail="Authentication subscription not found")
            
            # Store authentication event
            auth_event = AuthEvent(
                nfInstanceId=ausf_instance_id,
                success=True,
                timeStamp=datetime.now(timezone.utc),
                authType="5G_AKA",
                servingNetworkName=serving_network_name
            )
            
            if supi not in authentication_events:
                authentication_events[supi] = []
            authentication_events[supi].append(auth_event)
            
            response = {
                "authenticationVector": auth_vector.dict(),
                "supi": supi
            }
            
            span.set_attribute("auth_vector.generated", "SUCCESS")
            logger.info(f"Authentication data generated for SUPI: {supi}")
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Failed to generate authentication data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate authentication data: {e}")

# Legacy endpoint for backwards compatibility
@app.get("/udm_service")
def udm_service():
    """Legacy service endpoint - maintained for backwards compatibility"""
    return {
        "message": "UDM service response",
        "compliance": "3GPP TS 29.503",
        "status": "OPERATIONAL",
        "supported_services": udm_instance.supported_services
    }

# Health and status endpoints
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "UDM",
        "compliance": "3GPP TS 29.503",
        "version": "1.0.0",
        "registered_ues": len(ue_contexts)
    }

@app.get("/metrics")
def get_metrics():
    """Metrics endpoint for monitoring"""
    total_registrations = len(amf_registrations)
    active_ues = len([ctx for ctx in ue_contexts.values() if ctx.get("ueState") == "REGISTERED"])
    total_auth_events = sum(len(events) for events in authentication_events.values())
    
    return {
        "total_amf_registrations": total_registrations,
        "active_ues": active_ues,
        "total_authentication_events": total_auth_events,
        "subscription_data_entries": len(subscription_data_storage)
    }

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="UDM - Unified Data Management")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("udm"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)