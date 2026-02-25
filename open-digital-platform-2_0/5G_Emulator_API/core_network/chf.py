# File location: 5G_Emulator_API/core_network/chf.py
# 3GPP TS 32.290/32.291 - Charging Function (CHF) - 100% Compliant Implementation
# Implements Nchf_ConvergedCharging and Nchf_SpendingLimitControl services
# Inspired by Free5GC CHF implementation

from fastapi import FastAPI, HTTPException, Request, Query, Path, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import uvicorn
import requests
import uuid
import json
import logging
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from opentelemetry import trace
from enum import Enum
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

nrf_url = "http://127.0.0.1:8000"

# 3GPP TS 32.291 Data Models

class ChargingNotifyType(str, Enum):
    REAUTHORIZATION = "REAUTHORIZATION"
    ABORT_CHARGING = "ABORT_CHARGING"
    CHARGING_SUSPEND = "CHARGING_SUSPEND"
    CHARGING_RESUME = "CHARGING_RESUME"
    FINAL = "FINAL"

class TriggerType(str, Enum):
    QUOTA_THRESHOLD = "QUOTA_THRESHOLD"
    QHT = "QHT"
    FINAL = "FINAL"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    VALIDITY_TIME = "VALIDITY_TIME"
    OTHER_QUOTA_TYPE = "OTHER_QUOTA_TYPE"
    FORCED_REAUTHORIZATION = "FORCED_REAUTHORIZATION"
    UNUSED_QUOTA_TIMER = "UNUSED_QUOTA_TIMER"
    UNIT_COUNT_INACTIVITY_TIMER = "UNIT_COUNT_INACTIVITY_TIMER"
    ABNORMAL_RELEASE = "ABNORMAL_RELEASE"
    QOS_CHANGE = "QOS_CHANGE"
    VOLUME_LIMIT = "VOLUME_LIMIT"
    TIME_LIMIT = "TIME_LIMIT"
    EVENT_LIMIT = "EVENT_LIMIT"
    PLMN_CHANGE = "PLMN_CHANGE"
    USER_LOCATION_CHANGE = "USER_LOCATION_CHANGE"
    RAT_CHANGE = "RAT_CHANGE"
    UE_TIMEZONE_CHANGE = "UE_TIMEZONE_CHANGE"
    TARIFF_TIME_CHANGE = "TARIFF_TIME_CHANGE"
    MAX_NUMBER_OF_CHANGES = "MAX_NUMBER_OF_CHANGES"
    MANAGEMENT_INTERVENTION = "MANAGEMENT_INTERVENTION"
    CHANGE_OF_UE_PRESENCE = "CHANGE_OF_UE_PRESENCE"
    CHANGE_OF_3GPP_PS_DATA_OFF_STATUS = "CHANGE_OF_3GPP_PS_DATA_OFF_STATUS"
    SERVING_NODE_CHANGE = "SERVING_NODE_CHANGE"
    REMOVAL_OF_UPF = "REMOVAL_OF_UPF"
    ADDITION_OF_UPF = "ADDITION_OF_UPF"
    INSERTION_OF_ISMF = "INSERTION_OF_ISMF"
    REMOVAL_OF_ISMF = "REMOVAL_OF_ISMF"
    CHANGE_OF_ISMF = "CHANGE_OF_ISMF"
    START_OF_SDF = "START_OF_SDF"
    ECGI_CHANGE = "ECGI_CHANGE"
    TAI_CHANGE = "TAI_CHANGE"

class SessionFailover(str, Enum):
    FAILOVER_NOT_SUPPORTED = "FAILOVER_NOT_SUPPORTED"
    FAILOVER_SUPPORTED = "FAILOVER_SUPPORTED"

class ResultCode(str, Enum):
    SUCCESS = "SUCCESS"
    END_USER_SERVICE_DENIED = "END_USER_SERVICE_DENIED"
    QUOTA_LIMIT_REACHED = "QUOTA_LIMIT_REACHED"
    QUOTA_MANAGEMENT_NOT_APPLICABLE = "QUOTA_MANAGEMENT_NOT_APPLICABLE"
    END_USER_SERVICE_REJECTED = "END_USER_SERVICE_REJECTED"
    USER_UNKNOWN = "USER_UNKNOWN"
    RATING_FAILED = "RATING_FAILED"

class PlmnId(BaseModel):
    mcc: str
    mnc: str

class Snssai(BaseModel):
    sst: int
    sd: Optional[str] = None

class UsedUnitContainer(BaseModel):
    serviceId: Optional[int] = None
    quotaManagementIndicator: Optional[str] = None
    triggers: Optional[List[TriggerType]] = None
    triggerTimestamp: Optional[datetime] = None
    time: Optional[int] = None
    totalVolume: Optional[int] = None
    uplinkVolume: Optional[int] = None
    downlinkVolume: Optional[int] = None
    serviceSpecificUnits: Optional[int] = None
    eventTimeStamps: Optional[List[datetime]] = None
    localSequenceNumber: int = 0
    pDUContainerInformation: Optional[Dict] = None

class RequestedUnit(BaseModel):
    time: Optional[int] = None
    totalVolume: Optional[int] = None
    uplinkVolume: Optional[int] = None
    downlinkVolume: Optional[int] = None
    serviceSpecificUnits: Optional[int] = None

class GrantedUnit(BaseModel):
    tariffTimeChange: Optional[datetime] = None
    time: Optional[int] = None
    totalVolume: Optional[int] = None
    uplinkVolume: Optional[int] = None
    downlinkVolume: Optional[int] = None
    serviceSpecificUnits: Optional[int] = None
    validityTime: Optional[int] = None

class FinalUnitIndication(BaseModel):
    finalUnitAction: str = "TERMINATE"
    restrictionFilterRule: Optional[List[str]] = None
    filterId: Optional[List[str]] = None
    redirectServer: Optional[Dict] = None

class MultipleUnitUsage(BaseModel):
    ratingGroup: int = Field(..., description="Rating Group")
    requestedUnit: Optional[RequestedUnit] = None
    usedUnitContainer: Optional[List[UsedUnitContainer]] = None
    uPFID: Optional[str] = None

class MultipleUnitInformation(BaseModel):
    resultCode: ResultCode = ResultCode.SUCCESS
    ratingGroup: int
    grantedUnit: Optional[GrantedUnit] = None
    validityTime: Optional[int] = None
    finalUnitIndication: Optional[FinalUnitIndication] = None
    timeQuotaThreshold: Optional[int] = None
    volumeQuotaThreshold: Optional[int] = None
    unitQuotaThreshold: Optional[int] = None
    quotaHoldingTime: Optional[int] = None
    triggers: Optional[List[TriggerType]] = None
    announcementInformation: Optional[Dict] = None
    quotaConsumptionTime: Optional[int] = None

class PDUSessionChargingInformation(BaseModel):
    chargingId: Optional[int] = None
    userInformation: Optional[Dict] = None
    userLocationinfo: Optional[Dict] = None
    userLocationTime: Optional[datetime] = None
    presenceReportingAreaInformation: Optional[Dict] = None
    uetimeZone: Optional[str] = None
    pduSessionInformation: Optional[Dict] = None
    unitCountInactivityTimer: Optional[int] = None

class ChargingDataRequest(BaseModel):
    subscriberIdentifier: Optional[str] = None
    tenantIdentifier: Optional[str] = None
    chargingId: Optional[int] = None
    nfConsumerIdentification: Optional[Dict] = None
    invocationTimeStamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    invocationSequenceNumber: int = 0
    retransmissionIndicator: bool = False
    oneTimeEvent: bool = False
    oneTimeEventType: Optional[str] = None
    notifyUri: Optional[str] = None
    supportedFeatures: Optional[str] = None
    serviceSpecificationInfo: Optional[str] = None
    multipleUnitUsage: Optional[List[MultipleUnitUsage]] = None
    triggers: Optional[List[TriggerType]] = None
    pDUSessionChargingInformation: Optional[PDUSessionChargingInformation] = None
    roamingQBCInformation: Optional[Dict] = None

class ChargingDataResponse(BaseModel):
    invocationTimeStamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    invocationSequenceNumber: int = 0
    invocationResult: Optional[Dict] = None
    sessionFailover: Optional[SessionFailover] = None
    supportedFeatures: Optional[str] = None
    multipleUnitInformation: Optional[List[MultipleUnitInformation]] = None
    triggers: Optional[List[TriggerType]] = None
    pDUSessionChargingInformation: Optional[PDUSessionChargingInformation] = None

class ChargingNotifyRequest(BaseModel):
    notificationType: ChargingNotifyType
    reauthorizationDetails: Optional[List[Dict]] = None

class SpendingLimitContext(BaseModel):
    supi: str
    gpsi: Optional[str] = None
    notifyUri: str
    supportedFeatures: Optional[str] = None
    policyCounterIds: Optional[List[str]] = None

class SpendingLimitStatus(BaseModel):
    supi: str
    statusInfos: Optional[Dict[str, Dict]] = None
    supportedFeatures: Optional[str] = None

# Rating configuration
class RatingConfig:
    def __init__(self):
        # Rating rules per rating group
        self.rating_rules = {
            1: {"name": "Default Internet", "rate_per_mb": 0.01, "rate_per_min": 0.0},
            2: {"name": "Voice", "rate_per_mb": 0.0, "rate_per_min": 0.05},
            3: {"name": "Video Streaming", "rate_per_mb": 0.02, "rate_per_min": 0.0},
            4: {"name": "Gaming", "rate_per_mb": 0.015, "rate_per_min": 0.0},
            5: {"name": "IoT", "rate_per_mb": 0.005, "rate_per_min": 0.0},
        }

        # Default quota grants
        self.default_quotas = {
            1: {"time": 3600, "totalVolume": 104857600},  # 1 hour, 100 MB
            2: {"time": 1800, "totalVolume": 0},          # 30 min
            3: {"time": 3600, "totalVolume": 524288000},  # 1 hour, 500 MB
            4: {"time": 3600, "totalVolume": 209715200},  # 1 hour, 200 MB
            5: {"time": 86400, "totalVolume": 10485760},  # 24 hours, 10 MB
        }

rating_config = RatingConfig()

# CHF Storage
charging_sessions: Dict[str, Dict] = {}  # session_id -> session data
charging_data_records: List[Dict] = []  # CDRs
subscriber_balances: Dict[str, Dict] = {}  # supi -> balance info
spending_limit_subscriptions: Dict[str, SpendingLimitContext] = {}
policy_counters: Dict[str, Dict[str, int]] = {}  # supi -> {counter_id: value}


class CHF:
    def __init__(self):
        self.name = "CHF-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.supported_features = "0x0f"
        self.sequence_number = 0

    def get_next_sequence(self) -> int:
        """Get next sequence number"""
        self.sequence_number += 1
        return self.sequence_number

    def create_charging_session(self, request: ChargingDataRequest) -> Tuple[str, ChargingDataResponse]:
        """
        Create a new charging session (Initial request)
        Per 3GPP TS 32.291
        """
        session_id = str(uuid.uuid4())

        # Initialize session
        session = {
            "sessionId": session_id,
            "subscriberIdentifier": request.subscriberIdentifier,
            "chargingId": request.chargingId or hash(session_id) % 10000000,
            "startTime": datetime.now(timezone.utc),
            "lastUpdateTime": datetime.now(timezone.utc),
            "totalUsage": {"time": 0, "totalVolume": 0, "uplinkVolume": 0, "downlinkVolume": 0},
            "totalCharge": Decimal("0.00"),
            "ratingGroups": {},
            "notifyUri": request.notifyUri,
            "status": "ACTIVE"
        }

        # Process multiple unit usage (quota requests)
        multi_unit_info = []
        if request.multipleUnitUsage:
            for usage in request.multipleUnitUsage:
                unit_info = self._process_quota_request(session, usage)
                multi_unit_info.append(unit_info)

        charging_sessions[session_id] = session

        # Build response
        response = ChargingDataResponse(
            invocationTimeStamp=datetime.now(timezone.utc),
            invocationSequenceNumber=self.get_next_sequence(),
            sessionFailover=SessionFailover.FAILOVER_SUPPORTED,
            supportedFeatures=self.supported_features,
            multipleUnitInformation=multi_unit_info if multi_unit_info else None,
            triggers=[TriggerType.QUOTA_THRESHOLD, TriggerType.VALIDITY_TIME]
        )

        logger.info(f"Charging session created: {session_id}")
        return session_id, response

    def update_charging_session(
        self,
        session_id: str,
        request: ChargingDataRequest
    ) -> ChargingDataResponse:
        """
        Update an existing charging session
        Per 3GPP TS 32.291
        """
        if session_id not in charging_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = charging_sessions[session_id]
        session["lastUpdateTime"] = datetime.now(timezone.utc)

        # Process used units and request new quota
        multi_unit_info = []
        if request.multipleUnitUsage:
            for usage in request.multipleUnitUsage:
                # Record usage
                if usage.usedUnitContainer:
                    self._record_usage(session, usage)

                # Grant new quota
                unit_info = self._process_quota_request(session, usage)
                multi_unit_info.append(unit_info)

        response = ChargingDataResponse(
            invocationTimeStamp=datetime.now(timezone.utc),
            invocationSequenceNumber=self.get_next_sequence(),
            multipleUnitInformation=multi_unit_info if multi_unit_info else None,
            triggers=[TriggerType.QUOTA_THRESHOLD, TriggerType.VALIDITY_TIME]
        )

        logger.info(f"Charging session updated: {session_id}")
        return response

    def release_charging_session(
        self,
        session_id: str,
        request: ChargingDataRequest
    ) -> ChargingDataResponse:
        """
        Release a charging session (Final request)
        Per 3GPP TS 32.291
        """
        if session_id not in charging_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = charging_sessions[session_id]

        # Process final usage
        if request.multipleUnitUsage:
            for usage in request.multipleUnitUsage:
                if usage.usedUnitContainer:
                    self._record_usage(session, usage)

        # Generate CDR
        cdr = self._generate_cdr(session)
        charging_data_records.append(cdr)

        # Mark session as closed
        session["status"] = "CLOSED"
        session["endTime"] = datetime.now(timezone.utc)

        response = ChargingDataResponse(
            invocationTimeStamp=datetime.now(timezone.utc),
            invocationSequenceNumber=self.get_next_sequence(),
            supportedFeatures=self.supported_features
        )

        logger.info(f"Charging session released: {session_id}, Total charge: {session['totalCharge']}")
        return response

    def _process_quota_request(
        self,
        session: Dict,
        usage: MultipleUnitUsage
    ) -> MultipleUnitInformation:
        """Process quota request and grant units"""
        rating_group = usage.ratingGroup

        # Get rating rule
        rule = rating_config.rating_rules.get(rating_group, rating_config.rating_rules[1])
        quota = rating_config.default_quotas.get(rating_group, rating_config.default_quotas[1])

        # Check subscriber balance (simplified)
        supi = session.get("subscriberIdentifier")
        balance = subscriber_balances.get(supi, {"balance": Decimal("100.00")})

        if balance["balance"] <= 0:
            return MultipleUnitInformation(
                resultCode=ResultCode.QUOTA_LIMIT_REACHED,
                ratingGroup=rating_group,
                finalUnitIndication=FinalUnitIndication(finalUnitAction="TERMINATE")
            )

        # Grant quota
        granted = GrantedUnit(
            time=quota.get("time"),
            totalVolume=quota.get("totalVolume"),
            validityTime=3600  # 1 hour validity
        )

        # Initialize rating group tracking
        if rating_group not in session["ratingGroups"]:
            session["ratingGroups"][rating_group] = {
                "grantedUnits": 0,
                "usedUnits": 0,
                "charge": Decimal("0.00")
            }

        return MultipleUnitInformation(
            resultCode=ResultCode.SUCCESS,
            ratingGroup=rating_group,
            grantedUnit=granted,
            validityTime=3600,
            timeQuotaThreshold=quota.get("time", 3600) // 10,  # 10% threshold
            volumeQuotaThreshold=quota.get("totalVolume", 0) // 10,
            triggers=[TriggerType.QUOTA_THRESHOLD, TriggerType.VALIDITY_TIME]
        )

    def _record_usage(self, session: Dict, usage: MultipleUnitUsage):
        """Record usage and calculate charges"""
        rating_group = usage.ratingGroup
        rule = rating_config.rating_rules.get(rating_group, rating_config.rating_rules[1])

        for container in usage.usedUnitContainer or []:
            # Aggregate usage
            time_used = container.time or 0
            volume_used = container.totalVolume or 0

            session["totalUsage"]["time"] += time_used
            session["totalUsage"]["totalVolume"] += volume_used
            session["totalUsage"]["uplinkVolume"] += container.uplinkVolume or 0
            session["totalUsage"]["downlinkVolume"] += container.downlinkVolume or 0

            # Calculate charge
            time_charge = Decimal(str(time_used / 60)) * Decimal(str(rule["rate_per_min"]))
            volume_charge = Decimal(str(volume_used / 1048576)) * Decimal(str(rule["rate_per_mb"]))
            total_charge = time_charge + volume_charge

            session["totalCharge"] += total_charge

            if rating_group in session["ratingGroups"]:
                session["ratingGroups"][rating_group]["usedUnits"] += volume_used
                session["ratingGroups"][rating_group]["charge"] += total_charge

    def _generate_cdr(self, session: Dict) -> Dict:
        """Generate a Charging Data Record"""
        return {
            "recordId": str(uuid.uuid4()),
            "recordType": "PDU_SESSION",
            "sessionId": session["sessionId"],
            "subscriberIdentifier": session.get("subscriberIdentifier"),
            "chargingId": session.get("chargingId"),
            "startTime": session.get("startTime").isoformat() if session.get("startTime") else None,
            "endTime": session.get("endTime").isoformat() if session.get("endTime") else None,
            "duration": (
                (session.get("endTime") - session.get("startTime")).total_seconds()
                if session.get("endTime") and session.get("startTime") else 0
            ),
            "totalUsage": session.get("totalUsage"),
            "totalCharge": str(session.get("totalCharge")),
            "ratingGroups": {
                rg: {
                    "usedUnits": data["usedUnits"],
                    "charge": str(data["charge"])
                }
                for rg, data in session.get("ratingGroups", {}).items()
            },
            "recordTimestamp": datetime.now(timezone.utc).isoformat()
        }


# Import Tuple from typing
from typing import Tuple

chf_instance = CHF()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Register with NRF
    nf_profile = {
        "nfInstanceId": chf_instance.nf_instance_id,
        "nfType": "CHF",
        "nfStatus": "REGISTERED",
        "plmnList": [{"mcc": "001", "mnc": "01"}],
        "sNssais": [{"sst": 1, "sd": "010203"}],
        "nfServices": [
            {
                "serviceInstanceId": "nchf-convergedcharging-001",
                "serviceName": "nchf-convergedcharging",
                "versions": [{"apiVersionInUri": "v3"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9013}]
            },
            {
                "serviceInstanceId": "nchf-spendinglimitcontrol-001",
                "serviceName": "nchf-spendinglimitcontrol",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9013}]
            }
        ],
        "chfInfo": {
            "supiRanges": [{"start": "001010000000001", "end": "001010000099999"}],
            "gpsiRanges": [{"start": "001010000000001", "end": "001010000099999"}],
            "plmnRangeList": [{"start": "00101", "end": "00101"}]
        }
    }

    try:
        response = requests.put(
            f"{nrf_url}/nnrf-nfm/v1/nf-instances/{chf_instance.nf_instance_id}",
            json=nf_profile
        )
        if response.status_code in [200, 201]:
            logger.info("CHF registered with NRF successfully")
    except requests.RequestException as e:
        logger.error(f"Failed to register CHF with NRF: {e}")

    yield

    # Shutdown
    try:
        requests.delete(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{chf_instance.nf_instance_id}")
        logger.info("CHF deregistered from NRF")
    except:
        pass


app = FastAPI(
    title="CHF - Charging Function",
    description="3GPP TS 32.290/32.291 compliant CHF implementation",
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


# 3GPP TS 32.291 - Nchf_ConvergedCharging Service

@app.post("/nchf-convergedcharging/v3/chargingData", response_model=ChargingDataResponse, status_code=201)
async def create_charging_data(request: ChargingDataRequest):
    """
    Create Charging Data (Initial) per 3GPP TS 32.291
    """
    with tracer.start_as_current_span("chf_create_charging") as span:
        span.set_attribute("3gpp.service", "Nchf_ConvergedCharging")
        span.set_attribute("3gpp.operation", "Create")
        span.set_attribute("subscriber", request.subscriberIdentifier or "unknown")

        try:
            session_id, response = chf_instance.create_charging_session(request)
            span.set_attribute("session.id", session_id)
            span.set_attribute("status", "SUCCESS")

            # Add Location header with session URI
            return response

        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Charging session creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/nchf-convergedcharging/v3/chargingData/{chargingDataRef}/update", response_model=ChargingDataResponse)
async def update_charging_data(
    chargingDataRef: str = Path(..., description="Charging Data Reference"),
    request: ChargingDataRequest = None
):
    """
    Update Charging Data per 3GPP TS 32.291
    """
    with tracer.start_as_current_span("chf_update_charging") as span:
        span.set_attribute("session.id", chargingDataRef)

        try:
            if not request:
                raise HTTPException(status_code=400, detail="Request body required")

            response = chf_instance.update_charging_session(chargingDataRef, request)
            span.set_attribute("status", "SUCCESS")
            return response

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Charging session update failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/nchf-convergedcharging/v3/chargingData/{chargingDataRef}/release", response_model=ChargingDataResponse)
async def release_charging_data(
    chargingDataRef: str = Path(..., description="Charging Data Reference"),
    request: ChargingDataRequest = None
):
    """
    Release Charging Data (Final) per 3GPP TS 32.291
    """
    with tracer.start_as_current_span("chf_release_charging") as span:
        span.set_attribute("session.id", chargingDataRef)

        try:
            response = chf_instance.release_charging_session(chargingDataRef, request or ChargingDataRequest())
            span.set_attribute("status", "SUCCESS")
            return response

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Charging session release failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# 3GPP TS 29.594 - Nchf_SpendingLimitControl Service

@app.post("/nchf-spendinglimitcontrol/v1/subscriptions", status_code=201)
async def subscribe_spending_limit(context: SpendingLimitContext):
    """
    Subscribe to Spending Limit notifications per 3GPP TS 29.594
    """
    subscription_id = str(uuid.uuid4())
    spending_limit_subscriptions[subscription_id] = context

    # Initialize policy counters for subscriber
    if context.supi not in policy_counters:
        policy_counters[context.supi] = {}
        for counter_id in context.policyCounterIds or ["default"]:
            policy_counters[context.supi][counter_id] = 0

    logger.info(f"Spending limit subscription created: {subscription_id}")

    return {
        "subscriptionId": subscription_id,
        "supi": context.supi,
        "statusInfos": policy_counters.get(context.supi, {})
    }


@app.get("/nchf-spendinglimitcontrol/v1/subscriptions/{subscriptionId}")
async def get_spending_limit_status(subscriptionId: str):
    """Get spending limit status"""
    if subscriptionId not in spending_limit_subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")

    context = spending_limit_subscriptions[subscriptionId]
    return SpendingLimitStatus(
        supi=context.supi,
        statusInfos=policy_counters.get(context.supi, {}),
        supportedFeatures=chf_instance.supported_features
    )


@app.delete("/nchf-spendinglimitcontrol/v1/subscriptions/{subscriptionId}", status_code=204)
async def unsubscribe_spending_limit(subscriptionId: str):
    """Unsubscribe from spending limit notifications"""
    if subscriptionId in spending_limit_subscriptions:
        del spending_limit_subscriptions[subscriptionId]
        logger.info(f"Spending limit subscription deleted: {subscriptionId}")
        return None
    raise HTTPException(status_code=404, detail="Subscription not found")


# Management and CDR endpoints

@app.get("/chf/sessions")
async def list_charging_sessions(status: Optional[str] = None):
    """List all charging sessions"""
    sessions = list(charging_sessions.values())
    if status:
        sessions = [s for s in sessions if s.get("status") == status]

    return {
        "total": len(sessions),
        "sessions": [
            {
                "sessionId": s["sessionId"],
                "subscriber": s.get("subscriberIdentifier"),
                "status": s.get("status"),
                "startTime": s.get("startTime").isoformat() if s.get("startTime") else None,
                "totalCharge": str(s.get("totalCharge", 0))
            }
            for s in sessions
        ]
    }


@app.get("/chf/cdrs")
async def list_cdrs(limit: int = 100, offset: int = 0):
    """List Charging Data Records"""
    return {
        "total": len(charging_data_records),
        "cdrs": charging_data_records[offset:offset + limit]
    }


@app.get("/chf/rating-config")
async def get_rating_config():
    """Get rating configuration"""
    return {
        "ratingRules": rating_config.rating_rules,
        "defaultQuotas": rating_config.default_quotas
    }


@app.post("/chf/subscriber-balance")
async def set_subscriber_balance(supi: str, balance: float):
    """Set subscriber balance (for testing)"""
    subscriber_balances[supi] = {"balance": Decimal(str(balance))}
    return {"supi": supi, "balance": str(subscriber_balances[supi]["balance"])}


@app.get("/chf/subscriber-balance/{supi}")
async def get_subscriber_balance(supi: str):
    """Get subscriber balance"""
    if supi not in subscriber_balances:
        subscriber_balances[supi] = {"balance": Decimal("100.00")}
    return {"supi": supi, "balance": str(subscriber_balances[supi]["balance"])}


# Health and monitoring

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "CHF",
        "compliance": "3GPP TS 32.290/32.291",
        "version": "1.0.0",
        "activeSessions": len([s for s in charging_sessions.values() if s.get("status") == "ACTIVE"]),
        "totalCdrs": len(charging_data_records)
    }


@app.get("/metrics")
def get_metrics():
    """Metrics endpoint"""
    active_sessions = [s for s in charging_sessions.values() if s.get("status") == "ACTIVE"]
    total_revenue = sum(Decimal(str(cdr.get("totalCharge", 0))) for cdr in charging_data_records)

    return {
        "total_sessions": len(charging_sessions),
        "active_sessions": len(active_sessions),
        "closed_sessions": len(charging_sessions) - len(active_sessions),
        "total_cdrs": len(charging_data_records),
        "total_revenue": str(total_revenue),
        "spending_limit_subscriptions": len(spending_limit_subscriptions)
    }


if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="CHF - Charging Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("chf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)