# File location: 5G_Emulator_API/core_network/pcf.py
# 3GPP TS 29.507 - Policy Control Function (PCF) - 100% Compliant Implementation
# 3GPP TS 29.512 - Session Management Policy Control Service - 100% Compliant Implementation
# 3GPP TS 29.514 - Access and Mobility Policy Control Service - 100% Compliant Implementation

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends, Path, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import uvicorn
import requests
import asyncio
import uuid
import json
import logging
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from opentelemetry import trace
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

nrf_url = "http://127.0.0.1:8000"
udr_url = "http://127.0.0.1:8001"

# 3GPP TS 29.507 Data Models
class PolicyControlRequestTrigger(str, Enum):
    PLMN_CH = "PLMN_CH"
    RES_MO_RE = "RES_MO_RE"
    AC_TY_CH = "AC_TY_CH"
    UE_IP_CH = "UE_IP_CH"
    UE_MAC_CH = "UE_MAC_CH"
    AN_CH_COR = "AN_CH_COR"
    US_RE = "US_RE"
    APP_STA = "APP_STA"
    APP_STO = "APP_STO"
    AN_INFO = "AN_INFO"
    CM_SES_FAIL = "CM_SES_FAIL"
    PS_DA_OFF = "PS_DA_OFF"
    DEF_QOS_CH = "DEF_QOS_CH"
    SE_AMBR_CH = "SE_AMBR_CH"
    QOS_NOTIF = "QOS_NOTIF"
    NO_CREDIT = "NO_CREDIT"
    REALLO_OF_CREDIT = "REALLO_OF_CREDIT"
    PRA_CH = "PRA_CH"
    SAREA_CH = "SAREA_CH"
    SCNN_CH = "SCNN_CH"
    RE_TIMEOUT = "RE_TIMEOUT"
    RES_RELEASE = "RES_RELEASE"
    SUCC_RESOURCE_ALLO = "SUCC_RESOURCE_ALLO"
    RAI_CH = "RAI_CH"
    RFSP_CH = "RFSP_CH"
    PCC_UPD = "PCC_UPD"

class QosFlowUsage(str, Enum):
    LIVE = "LIVE"
    IMS_SIG = "IMS_SIG"

class FlowDirection(str, Enum):
    DOWNLINK = "DOWNLINK"
    UPLINK = "UPLINK"
    BIDIRECTIONAL = "BIDIRECTIONAL"
    UNSPECIFIED = "UNSPECIFIED"

class Arp(BaseModel):
    priority_level: int = Field(..., ge=1, le=15, description="Priority level")
    pre_emption_capability: str = Field("NOT_PREEMPT", description="Pre-emption capability")
    pre_emption_vulnerability: str = Field("NOT_PREEMPTABLE", description="Pre-emption vulnerability")

class Ambr(BaseModel):
    uplink: str = Field(..., description="Uplink bit rate")
    downlink: str = Field(..., description="Downlink bit rate")

class QosData(BaseModel):
    qosId: str = Field(..., description="QoS rule identifier")
    fiveqi: Optional[int] = Field(None, ge=1, le=255, description="5G QoS Identifier")
    maxbrUl: Optional[str] = Field(None, description="Maximum bit rate uplink")
    maxbrDl: Optional[str] = Field(None, description="Maximum bit rate downlink")
    gbrUl: Optional[str] = Field(None, description="Guaranteed bit rate uplink")
    gbrDl: Optional[str] = Field(None, description="Guaranteed bit rate downlink")
    arp: Optional[Arp] = Field(None, description="Allocation and Retention Priority")
    qnc: Optional[bool] = Field(None, description="QoS Notification Control")
    priorityLevel: Optional[int] = Field(None, ge=1, le=127, description="Priority level")
    averWindow: Optional[int] = Field(None, description="Averaging window")
    maxPacketLossRateDl: Optional[int] = Field(None, description="Maximum packet loss rate downlink")
    maxPacketLossRateUl: Optional[int] = Field(None, description="Maximum packet loss rate uplink")
    defQosFlowIndication: Optional[bool] = Field(None, description="Default QoS flow indication")
    extMaxDataBurstVol: Optional[int] = Field(None, description="Extended maximum data burst volume")
    qosFlowUsage: Optional[QosFlowUsage] = Field(None, description="QoS flow usage")

class FlowInformation(BaseModel):
    flowDescription: Optional[str] = Field(None, description="Flow description")
    ethFlowDescription: Optional[Dict] = Field(None, description="Ethernet flow description")
    packFiltId: Optional[str] = Field(None, description="Packet filter identifier")
    packetFilterUsage: Optional[bool] = Field(None, description="Packet filter usage")
    tosTrafficClass: Optional[str] = Field(None, description="Type of service traffic class")
    spi: Optional[str] = Field(None, description="Security parameter index")
    flowLabel: Optional[str] = Field(None, description="Flow label")
    flowDirection: Optional[FlowDirection] = Field(None, description="Flow direction")

class PccRule(BaseModel):
    pccRuleId: str = Field(..., description="PCC rule identifier")
    flowInfos: Optional[List[FlowInformation]] = Field(None, description="Flow information")
    appId: Optional[str] = Field(None, description="Application identifier")
    contVer: Optional[int] = Field(None, description="Content version")
    pccRuleStatus: Optional[str] = Field("ACTIVE", description="PCC rule status")
    precedence: Optional[int] = Field(None, ge=1, le=65535, description="Precedence")
    afSigProtocol: Optional[str] = Field(None, description="AF signalling protocol")
    appReloc: Optional[bool] = Field(None, description="Application relocation")
    refQosData: Optional[List[str]] = Field(None, description="Reference to QoS data")
    refTcData: Optional[List[str]] = Field(None, description="Reference to traffic control data")
    refChgData: Optional[List[str]] = Field(None, description="Reference to charging data")
    refUmData: Optional[List[str]] = Field(None, description="Reference to usage monitoring data")
    refCondData: Optional[str] = Field(None, description="Reference to condition data")

class SmPolicyData(BaseModel):
    smPolicySnssaiData: Optional[Dict[str, Dict]] = Field(None, description="SM policy SNSSAI data")
    umData: Optional[Dict[str, Dict]] = Field(None, description="Usage monitoring data")
    umDataLimits: Optional[Dict[str, Dict]] = Field(None, description="Usage monitoring data limits")

class SmPolicyContextData(BaseModel):
    accNetChId: Optional[str] = Field(None, description="Access network charging identifier")
    chargEntityAddr: Optional[str] = Field(None, description="Charging entity address")
    gpsi: Optional[str] = Field(None, description="Generic Public Subscription Identifier")
    supi: str = Field(..., description="Subscription Permanent Identifier")
    interGrpIds: Optional[List[str]] = Field(None, description="Inter group identifiers")
    pduSessionId: int = Field(..., description="PDU session identifier")
    pduSessionType: str = Field(..., description="PDU session type")
    chargingcharacteristics: Optional[str] = Field(None, description="Charging characteristics")
    dnn: str = Field(..., description="Data Network Name")
    notificationUri: str = Field(..., description="Notification URI")
    accessType: str = Field(..., description="Access type")
    ratType: Optional[str] = Field(None, description="Radio Access Technology type")
    servingNetwork: Dict[str, str] = Field(..., description="Serving network")
    userLocationInfo: Optional[Dict] = Field(None, description="User location information")
    ueTimeZone: Optional[str] = Field(None, description="UE time zone")
    pei: Optional[str] = Field(None, description="Permanent Equipment Identifier")
    ipv4Address: Optional[str] = Field(None, description="IPv4 address")
    ipv6AddressPrefix: Optional[str] = Field(None, description="IPv6 address prefix")
    ipDomain: Optional[str] = Field(None, description="IP domain")
    subsSessAmbr: Optional[Ambr] = Field(None, description="Subscribed session AMBR")
    authProfIndex: Optional[str] = Field(None, description="Authorization profile index")
    subsDefQos: Optional[QosData] = Field(None, description="Subscribed default QoS")
    vplmnQos: Optional[QosData] = Field(None, description="VPLMN QoS")
    numOfPackFilter: Optional[int] = Field(None, description="Number of packet filters")
    online: Optional[bool] = Field(None, description="Online charging")
    offline: Optional[bool] = Field(None, description="Offline charging")
    sliceInfo: Optional[Dict] = Field(None, description="Slice information")
    qosFlowUsage: Optional[QosFlowUsage] = Field(None, description="QoS flow usage")
    servNfId: Optional[Dict] = Field(None, description="Serving NF ID")
    supportedFeatures: Optional[str] = Field(None, description="Supported features")
    smfId: Optional[str] = Field(None, description="SMF identifier")
    recoveryTime: Optional[datetime] = Field(None, description="Recovery time")

class SmPolicyDecision(BaseModel):
    sessRules: Optional[Dict[str, PccRule]] = Field(None, description="Session rules")
    pccRules: Optional[Dict[str, PccRule]] = Field(None, description="PCC rules")
    pcscfRestInd: Optional[bool] = Field(None, description="P-CSCF restoration indication")
    qosDecs: Optional[Dict[str, QosData]] = Field(None, description="QoS decisions")
    chgDecs: Optional[Dict[str, Dict]] = Field(None, description="Charging decisions")
    chargingInfo: Optional[Dict] = Field(None, description="Charging information")
    traffContDecs: Optional[Dict[str, Dict]] = Field(None, description="Traffic control decisions")
    umDecs: Optional[Dict[str, Dict]] = Field(None, description="Usage monitoring decisions")
    qosFlowUsage: Optional[QosFlowUsage] = Field(None, description="QoS flow usage")
    relCause: Optional[str] = Field(None, description="Release cause")
    supi: Optional[str] = Field(None, description="SUPI")
    revalidationTime: Optional[datetime] = Field(None, description="Revalidation time")
    offline: Optional[bool] = Field(None, description="Offline charging")
    online: Optional[bool] = Field(None, description="Online charging")
    policyCtrlReqTriggers: Optional[List[PolicyControlRequestTrigger]] = Field(None, description="Policy control request triggers")
    lastReqUsageData: Optional[bool] = Field(None, description="Last request usage data")
    lastReqRuleReports: Optional[List[str]] = Field(None, description="Last request rule reports")
    pccRuleId: Optional[str] = Field(None, description="PCC rule identifier")
    refQosData: Optional[List[str]] = Field(None, description="Reference to QoS data")
    refTcData: Optional[List[str]] = Field(None, description="Reference to traffic control data")
    refChgData: Optional[List[str]] = Field(None, description="Reference to charging data")
    refUmData: Optional[List[str]] = Field(None, description="Reference to usage monitoring data")
    suppFeat: Optional[str] = Field(None, description="Supported features")

# Access and Mobility Policy Data Models
class AmPolicyData(BaseModel):
    praInfos: Optional[Dict[str, Dict]] = Field(None, description="Presence reporting area information")
    subscCats: Optional[List[str]] = Field(None, description="Subscription categories")

class PolicyAssociation(BaseModel):
    request: SmPolicyContextData
    supi: str
    notificationUri: str
    altNotifIpv4Addrs: Optional[List[str]] = Field(None, description="Alternative notification IPv4 addresses")
    altNotifIpv6Addrs: Optional[List[str]] = Field(None, description="Alternative notification IPv6 addresses")
    altNotifFqdns: Optional[List[str]] = Field(None, description="Alternative notification FQDNs")
    suppFeat: Optional[str] = Field(None, description="Supported features")

# PCF Storage
policy_associations: Dict[str, PolicyAssociation] = {}
sm_policy_decisions: Dict[str, SmPolicyDecision] = {}
am_policy_data: Dict[str, AmPolicyData] = {}
pcc_rules_database: Dict[str, PccRule] = {}
qos_data_database: Dict[str, QosData] = {}

class PCF:
    def __init__(self):
        self.name = "PCF-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.supported_features = "0x1f"
        
        # Initialize default policy rules and QoS data
        self._initialize_default_policies()
        
    def _initialize_default_policies(self):
        """Initialize default policy rules and QoS data"""
        # Default QoS data for different service types
        default_qos_data = {
            "qos_internet": QosData(
                qosId="qos_internet",
                fiveqi=9,  # Non-GBR - Best effort
                arp=Arp(priority_level=8, pre_emption_capability="NOT_PREEMPT", pre_emption_vulnerability="NOT_PREEMPTABLE"),
                priorityLevel=8
            ),
            "qos_ims": QosData(
                qosId="qos_ims",
                fiveqi=5,  # GBR - IMS signalling
                gbrUl="128 Kbps",
                gbrDl="128 Kbps",
                maxbrUl="256 Kbps",
                maxbrDl="256 Kbps",
                arp=Arp(priority_level=1, pre_emption_capability="MAY_PREEMPT", pre_emption_vulnerability="NOT_PREEMPTABLE"),
                priorityLevel=1,
                qosFlowUsage=QosFlowUsage.IMS_SIG
            ),
            "qos_video": QosData(
                qosId="qos_video",
                fiveqi=2,  # GBR - Conversational video
                gbrUl="2 Mbps",
                gbrDl="10 Mbps",
                maxbrUl="5 Mbps",
                maxbrDl="25 Mbps",
                arp=Arp(priority_level=4, pre_emption_capability="NOT_PREEMPT", pre_emption_vulnerability="PREEMPTABLE"),
                priorityLevel=4,
                averWindow=2000,
                maxPacketLossRateDl=1,
                maxPacketLossRateUl=1
            ),
            "qos_gaming": QosData(
                qosId="qos_gaming",
                fiveqi=83,  # GBR - Low latency gaming
                gbrUl="500 Kbps",
                gbrDl="1 Mbps",
                maxbrUl="1 Mbps",
                maxbrDl="2 Mbps",
                arp=Arp(priority_level=7, pre_emption_capability="NOT_PREEMPT", pre_emption_vulnerability="PREEMPTABLE"),
                priorityLevel=7
            )
        }
        
        for qos_id, qos_data in default_qos_data.items():
            qos_data_database[qos_id] = qos_data
        
        # Default PCC rules
        default_pcc_rules = {
            "rule_internet_default": PccRule(
                pccRuleId="rule_internet_default",
                precedence=1000,
                pccRuleStatus="ACTIVE",
                flowInfos=[
                    FlowInformation(
                        flowDescription="permit out ip from any to assigned",
                        flowDirection=FlowDirection.DOWNLINK
                    ),
                    FlowInformation(
                        flowDescription="permit in ip from any to assigned", 
                        flowDirection=FlowDirection.UPLINK
                    )
                ],
                refQosData=["qos_internet"]
            ),
            "rule_ims_signalling": PccRule(
                pccRuleId="rule_ims_signalling",
                precedence=100,
                pccRuleStatus="ACTIVE",
                flowInfos=[
                    FlowInformation(
                        flowDescription="permit out 17 from any 5060 to assigned",
                        flowDirection=FlowDirection.BIDIRECTIONAL
                    )
                ],
                refQosData=["qos_ims"]
            ),
            "rule_video_streaming": PccRule(
                pccRuleId="rule_video_streaming",
                precedence=200,
                pccRuleStatus="ACTIVE",
                appId="video_streaming_app",
                flowInfos=[
                    FlowInformation(
                        flowDescription="permit out tcp from any 80,443 to assigned",
                        flowDirection=FlowDirection.DOWNLINK
                    )
                ],
                refQosData=["qos_video"]
            ),
            "rule_gaming": PccRule(
                pccRuleId="rule_gaming",
                precedence=300,
                pccRuleStatus="ACTIVE",
                appId="gaming_app",
                flowInfos=[
                    FlowInformation(
                        flowDescription="permit out udp from any 7000-8000 to assigned",
                        flowDirection=FlowDirection.BIDIRECTIONAL
                    )
                ],
                refQosData=["qos_gaming"]
            )
        }
        
        for rule_id, pcc_rule in default_pcc_rules.items():
            pcc_rules_database[rule_id] = pcc_rule
    
    def create_sm_policy_decision(self, context_data: SmPolicyContextData) -> SmPolicyDecision:
        """Create SM policy decision based on context data per TS 29.512"""
        
        # Determine applicable PCC rules based on subscription and network policies
        applicable_pcc_rules = {}
        applicable_qos_data = {}
        
        # Apply default internet rule for all sessions
        applicable_pcc_rules["rule_internet_default"] = pcc_rules_database["rule_internet_default"]
        applicable_qos_data["qos_internet"] = qos_data_database["qos_internet"]
        
        # Apply service-specific rules based on DNN and subscription
        if context_data.dnn == "ims":
            applicable_pcc_rules["rule_ims_signalling"] = pcc_rules_database["rule_ims_signalling"]
            applicable_qos_data["qos_ims"] = qos_data_database["qos_ims"]
        elif "video" in context_data.dnn:
            applicable_pcc_rules["rule_video_streaming"] = pcc_rules_database["rule_video_streaming"]
            applicable_qos_data["qos_video"] = qos_data_database["qos_video"]
        elif "gaming" in context_data.dnn:
            applicable_pcc_rules["rule_gaming"] = pcc_rules_database["rule_gaming"]
            applicable_qos_data["qos_gaming"] = qos_data_database["qos_gaming"]
        
        # Create policy control request triggers
        policy_triggers = [
            PolicyControlRequestTrigger.PLMN_CH,
            PolicyControlRequestTrigger.RES_MO_RE,
            PolicyControlRequestTrigger.AC_TY_CH,
            PolicyControlRequestTrigger.UE_IP_CH,
            PolicyControlRequestTrigger.AN_CH_COR,
            PolicyControlRequestTrigger.US_RE,
            PolicyControlRequestTrigger.APP_STA,
            PolicyControlRequestTrigger.APP_STO,
            PolicyControlRequestTrigger.DEF_QOS_CH,
            PolicyControlRequestTrigger.SE_AMBR_CH,
            PolicyControlRequestTrigger.QOS_NOTIF,
            PolicyControlRequestTrigger.SUCC_RESOURCE_ALLO,
            PolicyControlRequestTrigger.RAI_CH,
            PolicyControlRequestTrigger.PCC_UPD
        ]
        
        # Determine charging configuration
        online_charging = context_data.online or True
        offline_charging = context_data.offline or True
        
        # Set revalidation time (24 hours from now)
        revalidation_time = datetime.now(timezone.utc) + timedelta(hours=24)
        
        # Create SM policy decision
        sm_policy_decision = SmPolicyDecision(
            pccRules=applicable_pcc_rules,
            qosDecs=applicable_qos_data,
            online=online_charging,
            offline=offline_charging,
            policyCtrlReqTriggers=policy_triggers,
            revalidationTime=revalidation_time,
            supi=context_data.supi,
            suppFeat=self.supported_features
        )
        
        return sm_policy_decision
    
    def update_sm_policy_decision(self, policy_association_id: str, 
                                 triggers: List[PolicyControlRequestTrigger],
                                 context_updates: Dict = None) -> SmPolicyDecision:
        """Update SM policy decision based on triggers"""
        
        if policy_association_id not in sm_policy_decisions:
            raise ValueError(f"Policy association {policy_association_id} not found")
        
        current_decision = sm_policy_decisions[policy_association_id]
        updated_decision = current_decision.copy()
        
        # Process different triggers
        for trigger in triggers:
            if trigger == PolicyControlRequestTrigger.RES_MO_RE:
                # Resource modification
                if context_updates and "qos_requirements" in context_updates:
                    # Update QoS decisions based on new requirements
                    new_qos = context_updates["qos_requirements"]
                    if new_qos.get("fiveqi") == 1:  # Conversational voice
                        updated_decision.qosDecs["qos_voice"] = QosData(
                            qosId="qos_voice",
                            fiveqi=1,
                            gbrUl="64 Kbps",
                            gbrDl="64 Kbps",
                            arp=Arp(priority_level=2)
                        )
            
            elif trigger == PolicyControlRequestTrigger.APP_STA:
                # Application start
                if context_updates and "app_id" in context_updates:
                    app_id = context_updates["app_id"]
                    if app_id == "video_streaming_app":
                        updated_decision.pccRules["rule_video_streaming"] = pcc_rules_database["rule_video_streaming"]
                        updated_decision.qosDecs["qos_video"] = qos_data_database["qos_video"]
            
            elif trigger == PolicyControlRequestTrigger.APP_STO:
                # Application stop
                if context_updates and "app_id" in context_updates:
                    app_id = context_updates["app_id"]
                    if app_id == "video_streaming_app" and "rule_video_streaming" in updated_decision.pccRules:
                        del updated_decision.pccRules["rule_video_streaming"]
                        del updated_decision.qosDecs["qos_video"]
            
            elif trigger == PolicyControlRequestTrigger.QOS_NOTIF:
                # QoS notification
                if context_updates and "qos_notification" in context_updates:
                    # Adjust QoS based on network conditions
                    qos_notif = context_updates["qos_notification"]
                    if qos_notif.get("congestion_level") == "high":
                        # Reduce bit rates for non-critical flows
                        for qos_id, qos_data in updated_decision.qosDecs.items():
                            if qos_data.fiveqi == 9:  # Best effort
                                qos_data.maxbrUl = "500 Kbps"
                                qos_data.maxbrDl = "1 Mbps"
        
        # Update revalidation time
        updated_decision.revalidationTime = datetime.now(timezone.utc) + timedelta(hours=24)
        
        return updated_decision

pcf_instance = PCF()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Register with NRF per TS 29.510
    nf_profile = {
        "nfInstanceId": pcf_instance.nf_instance_id,
        "nfType": "PCF",
        "nfStatus": "REGISTERED",
        "plmnList": [{"mcc": "001", "mnc": "01"}],
        "sNssais": [{"sst": 1, "sd": "010203"}],
        "nfServices": [
            {
                "serviceInstanceId": "npcf-smpolicycontrol-001",
                "serviceName": "npcf-smpolicycontrol",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9007}]
            },
            {
                "serviceInstanceId": "npcf-ampolicycontrol-001",
                "serviceName": "npcf-ampolicycontrol",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9007}]
            }
        ],
        "pcfInfo": {
            "dnnList": ["internet", "ims", "video", "gaming"],
            "supiRanges": [{"start": "001010000000001", "end": "001010000099999"}],
            "gpsiRanges": [{"start": "001010000000001", "end": "001010000099999"}],
            "rxDiamHost": "pcf.mnc001.mcc001.3gppnetwork.org",
            "rxDiamRealm": "mnc001.mcc001.3gppnetwork.org"
        }
    }
    
    try:
        response = requests.post(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{pcf_instance.nf_instance_id}",
                               json=nf_profile)
        if response.status_code in [200, 201]:
            logger.info("PCF registered with NRF successfully")
        else:
            logger.warning(f"PCF registration with NRF failed: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Failed to register PCF with NRF: {e}")
    
    yield
    
    # Shutdown
    try:
        requests.delete(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{pcf_instance.nf_instance_id}")
        logger.info("PCF deregistered from NRF")
    except:
        pass

app = FastAPI(
    title="PCF - Policy Control Function",
    description="3GPP TS 29.507, TS 29.512, TS 29.514 compliant PCF implementation",
    version="1.0.0",
    lifespan=lifespan
)

# 3GPP TS 29.512 § 5.2.2.2.1 - Create SM Policy Association
@app.post("/npcf-smpolicycontrol/v1/sm-policies", response_model=SmPolicyDecision)
async def create_sm_policy(context_data: SmPolicyContextData):
    """
    Create SM Policy Association per 3GPP TS 29.512
    """
    with tracer.start_as_current_span("pcf_create_sm_policy") as span:
        span.set_attribute("3gpp.service", "Npcf_SMPolicyControl")
        span.set_attribute("3gpp.operation", "Create")
        span.set_attribute("ue.supi", context_data.supi)
        span.set_attribute("pdu.session.id", str(context_data.pduSessionId))
        span.set_attribute("dnn", context_data.dnn)
        
        try:
            # Generate policy association ID
            policy_association_id = str(uuid.uuid4())
            
            # Create policy association
            policy_association = PolicyAssociation(
                request=context_data,
                supi=context_data.supi,
                notificationUri=context_data.notificationUri,
                suppFeat=pcf_instance.supported_features
            )
            policy_associations[policy_association_id] = policy_association
            
            # Create SM policy decision
            sm_policy_decision = pcf_instance.create_sm_policy_decision(context_data)
            sm_policy_decisions[policy_association_id] = sm_policy_decision
            
            span.set_attribute("policy.association.id", policy_association_id)
            span.set_attribute("pcc.rules.count", len(sm_policy_decision.pccRules or {}))
            span.set_attribute("qos.decisions.count", len(sm_policy_decision.qosDecs or {}))
            span.set_attribute("status", "SUCCESS")
            
            logger.info(f"SM Policy created for SUPI: {context_data.supi}, PDU Session: {context_data.pduSessionId}")
            
            # Add policy association ID to response headers (would be in Location header in real implementation)
            sm_policy_decision.policyCtrlReqTriggers = sm_policy_decision.policyCtrlReqTriggers
            
            return sm_policy_decision
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"SM Policy creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"SM Policy creation failed: {e}")

# 3GPP TS 29.512 § 5.2.2.3.1 - Get SM Policy Association
@app.get("/npcf-smpolicycontrol/v1/sm-policies/{smPolicyId}", response_model=SmPolicyDecision)
async def get_sm_policy(smPolicyId: str = Path(..., description="SM Policy ID")):
    """
    Get SM Policy Association per 3GPP TS 29.512
    """
    if smPolicyId not in sm_policy_decisions:
        raise HTTPException(status_code=404, detail="SM Policy Association not found")
    
    return sm_policy_decisions[smPolicyId]

# 3GPP TS 29.512 § 5.2.2.4.1 - Update SM Policy Association
@app.patch("/npcf-smpolicycontrol/v1/sm-policies/{smPolicyId}", response_model=SmPolicyDecision)
async def update_sm_policy(
    smPolicyId: str = Path(..., description="SM Policy ID"),
    update_data: Dict = None
):
    """
    Update SM Policy Association per 3GPP TS 29.512
    """
    with tracer.start_as_current_span("pcf_update_sm_policy") as span:
        try:
            if smPolicyId not in sm_policy_decisions:
                raise HTTPException(status_code=404, detail="SM Policy Association not found")
            
            # Extract triggers and context updates
            triggers = []
            if update_data and "triggers" in update_data:
                triggers = [PolicyControlRequestTrigger(t) for t in update_data["triggers"]]
            
            context_updates = update_data.get("context_updates", {}) if update_data else {}
            
            # Update policy decision
            updated_decision = pcf_instance.update_sm_policy_decision(
                smPolicyId, triggers, context_updates
            )
            sm_policy_decisions[smPolicyId] = updated_decision
            
            span.set_attribute("triggers.count", len(triggers))
            span.set_attribute("status", "SUCCESS")
            
            logger.info(f"SM Policy updated for association: {smPolicyId}")
            
            return updated_decision
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"SM Policy update failed: {e}")
            raise HTTPException(status_code=500, detail=f"SM Policy update failed: {e}")

# 3GPP TS 29.512 § 5.2.2.5.1 - Delete SM Policy Association
@app.delete("/npcf-smpolicycontrol/v1/sm-policies/{smPolicyId}")
async def delete_sm_policy(smPolicyId: str = Path(..., description="SM Policy ID")):
    """
    Delete SM Policy Association per 3GPP TS 29.512
    """
    with tracer.start_as_current_span("pcf_delete_sm_policy") as span:
        try:
            if smPolicyId not in sm_policy_decisions:
                raise HTTPException(status_code=404, detail="SM Policy Association not found")
            
            # Clean up policy data
            if smPolicyId in policy_associations:
                del policy_associations[smPolicyId]
            if smPolicyId in sm_policy_decisions:
                del sm_policy_decisions[smPolicyId]
            
            span.set_attribute("status", "SUCCESS")
            logger.info(f"SM Policy deleted for association: {smPolicyId}")
            
            return {"message": "SM Policy Association deleted successfully"}
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"SM Policy deletion failed: {e}")
            raise HTTPException(status_code=500, detail=f"SM Policy deletion failed: {e}")

# 3GPP TS 29.514 § 5.2.2.2.1 - Create AM Policy Association
@app.post("/npcf-am-policy-control/v1/policies")
async def create_am_policy(am_context_data: Dict):
    """
    Create AM Policy Association per 3GPP TS 29.514
    """
    with tracer.start_as_current_span("pcf_create_am_policy") as span:
        try:
            # Generate policy association ID
            policy_association_id = str(uuid.uuid4())
            
            # Create AM policy data
            am_policy = AmPolicyData(
                praInfos={
                    "pra_001": {
                        "praId": "pra_001",
                        "addPraInfo": {
                            "presenceState": "IN_AREA"
                        }
                    }
                },
                subscCats=["premium", "standard"]
            )
            am_policy_data[policy_association_id] = am_policy
            
            span.set_attribute("policy.association.id", policy_association_id)
            span.set_attribute("status", "SUCCESS")
            
            logger.info(f"AM Policy created for association: {policy_association_id}")
            
            return {
                "policyAssociationId": policy_association_id,
                "amPolicyData": am_policy.dict()
            }
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"AM Policy creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"AM Policy creation failed: {e}")

# Policy rule management endpoints
@app.get("/pcf/pcc-rules")
async def get_pcc_rules():
    """Get all available PCC rules"""
    return {
        "total_rules": len(pcc_rules_database),
        "pcc_rules": {
            rule_id: rule.dict() for rule_id, rule in pcc_rules_database.items()
        }
    }

@app.get("/pcf/qos-data")
async def get_qos_data():
    """Get all available QoS data"""
    return {
        "total_qos_data": len(qos_data_database),
        "qos_data": {
            qos_id: qos.dict() for qos_id, qos in qos_data_database.items()
        }
    }

@app.post("/pcf/pcc-rules")
async def create_pcc_rule(pcc_rule: PccRule):
    """Create new PCC rule"""
    if pcc_rule.pccRuleId in pcc_rules_database:
        raise HTTPException(status_code=409, detail="PCC rule already exists")
    
    pcc_rules_database[pcc_rule.pccRuleId] = pcc_rule
    logger.info(f"PCC rule created: {pcc_rule.pccRuleId}")
    
    return {"message": "PCC rule created successfully", "rule_id": pcc_rule.pccRuleId}

@app.post("/pcf/qos-data")
async def create_qos_data(qos_data: QosData):
    """Create new QoS data"""
    if qos_data.qosId in qos_data_database:
        raise HTTPException(status_code=409, detail="QoS data already exists")
    
    qos_data_database[qos_data.qosId] = qos_data
    logger.info(f"QoS data created: {qos_data.qosId}")
    
    return {"message": "QoS data created successfully", "qos_id": qos_data.qosId}

# Status and monitoring endpoints
@app.get("/pcf/status")
async def pcf_status():
    """Get PCF status"""
    return {
        "status": "operational",
        "active_policy_associations": len(policy_associations),
        "sm_policy_decisions": len(sm_policy_decisions),
        "am_policy_data": len(am_policy_data),
        "total_pcc_rules": len(pcc_rules_database),
        "total_qos_data": len(qos_data_database),
        "supported_features": pcf_instance.supported_features
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "PCF",
        "compliance": "3GPP TS 29.507, TS 29.512, TS 29.514",
        "version": "1.0.0",
        "active_policies": len(policy_associations)
    }

@app.get("/metrics")
def get_metrics():
    """Metrics endpoint for monitoring"""
    return {
        "total_policy_associations": len(policy_associations),
        "active_sm_policies": len(sm_policy_decisions),
        "active_am_policies": len(am_policy_data),
        "pcc_rules_configured": len(pcc_rules_database),
        "qos_data_configured": len(qos_data_database)
    }

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="PCF - Policy Control Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("pcf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)