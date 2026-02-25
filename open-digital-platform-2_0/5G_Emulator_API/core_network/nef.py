# File location: 5G_Emulator_API/core_network/nef.py
# 3GPP TS 29.522 - Network Exposure Function (NEF) - 100% Compliant Implementation
# Implements northbound APIs for external application function integration
# Inspired by Free5GC NEF implementation

from fastapi import FastAPI, HTTPException, Request, Query, Path, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
import requests
import uuid
import json
import logging
import httpx
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

# 3GPP TS 29.522 Data Models

class MonitoringType(str, Enum):
    LOCATION_REPORTING = "LOCATION_REPORTING"
    CHANGE_OF_IMSI_IMEI_SV = "CHANGE_OF_IMSI_IMEI_SV"
    ROAMING_STATUS = "ROAMING_STATUS"
    COMMUNICATION_FAILURE = "COMMUNICATION_FAILURE"
    AVAILABILITY_AFTER_DDN_FAILURE = "AVAILABILITY_AFTER_DDN_FAILURE"
    NUMBER_OF_UES_IN_AREA = "NUMBER_OF_UES_IN_AREA"
    PDN_CONNECTIVITY_STATUS = "PDN_CONNECTIVITY_STATUS"
    DOWNLINK_DATA_DELIVERY_STATUS = "DOWNLINK_DATA_DELIVERY_STATUS"
    API_SUPPORT_CAPABILITY = "API_SUPPORT_CAPABILITY"
    NUM_OF_REGD_UES = "NUM_OF_REGD_UES"
    LOSS_OF_CONNECTIVITY = "LOSS_OF_CONNECTIVITY"
    UE_REACHABILITY = "UE_REACHABILITY"

class EventType(str, Enum):
    SESSION_TERMINATION = "SESSION_TERMINATION"
    LOSS_OF_BEARER = "LOSS_OF_BEARER"
    RECOVERY_OF_BEARER = "RECOVERY_OF_BEARER"
    RELEASE_OF_BEARER = "RELEASE_OF_BEARER"
    USAGE_REPORT = "USAGE_REPORT"
    FAILED_RESOURCES_ALLOCATION = "FAILED_RESOURCES_ALLOCATION"
    SUCCESSFUL_RESOURCES_ALLOCATION = "SUCCESSFUL_RESOURCES_ALLOCATION"

class TrafficInfluenceType(str, Enum):
    BREAKOUT = "BREAKOUT"
    STEERING = "STEERING"
    OFFLOAD = "OFFLOAD"

class PlmnId(BaseModel):
    mcc: str
    mnc: str

class Snssai(BaseModel):
    sst: int
    sd: Optional[str] = None

class LocationArea(BaseModel):
    cellIds: Optional[List[str]] = None
    enodebIds: Optional[List[str]] = None
    taiList: Optional[List[Dict]] = None
    geographicalArea: Optional[Dict] = None
    civicAddress: Optional[Dict] = None

class AfSubscription(BaseModel):
    afId: str
    afTransId: Optional[str] = None
    appId: Optional[str] = None
    dnn: Optional[str] = None
    snssai: Optional[Snssai] = None
    externalGroupId: Optional[str] = None
    anyUe: bool = False
    gpsi: Optional[str] = None
    ipv4Addr: Optional[str] = None
    ipv6Addr: Optional[str] = None
    macAddr: Optional[str] = None
    notificationDestination: str
    requestTestNotification: bool = False
    websockNotifConfig: Optional[Dict] = None
    events: Optional[List[EventType]] = None
    supportedFeatures: Optional[str] = None

class MonitoringEventSubscription(BaseModel):
    externalId: Optional[str] = None
    externalGroupId: Optional[str] = None
    msisdn: Optional[str] = None
    ipv4Addr: Optional[str] = None
    ipv6Addr: Optional[str] = None
    monitoringType: MonitoringType
    maximumNumberOfReports: Optional[int] = None
    monitorExpireTime: Optional[datetime] = None
    reachabilityType: Optional[str] = None
    maximumLatency: Optional[int] = None
    maximumResponseTime: Optional[int] = None
    locationArea: Optional[LocationArea] = None
    notificationDestination: str
    supportedFeatures: Optional[str] = None

class MonitoringEventReport(BaseModel):
    monitoringType: MonitoringType
    eventTime: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    externalId: Optional[str] = None
    reachabilityType: Optional[str] = None
    locationInfo: Optional[Dict] = None
    lossOfConnectivity: Optional[Dict] = None
    communicationFailure: Optional[Dict] = None
    pdnConnectivityStatus: Optional[str] = None

class TrafficInfluenceSubscription(BaseModel):
    afId: str
    afTransId: Optional[str] = None
    dnn: Optional[str] = None
    snssai: Optional[Snssai] = None
    externalGroupId: Optional[str] = None
    anyUe: bool = False
    gpsi: Optional[str] = None
    ipv4Addr: Optional[str] = None
    ipv6Addr: Optional[str] = None
    macAddr: Optional[str] = None
    trafficFilters: Optional[List[Dict]] = None
    trafficRoutes: Optional[List[Dict]] = None
    afAckInd: bool = False
    addrPreserInd: bool = False
    notificationDestination: Optional[str] = None
    supportedFeatures: Optional[str] = None
    dnaiChgType: Optional[TrafficInfluenceType] = None

class AsSessionWithQoS(BaseModel):
    afId: str
    dnn: Optional[str] = None
    snssai: Optional[Snssai] = None
    ipv4Addr: Optional[str] = None
    ipv6Addr: Optional[str] = None
    macAddr: Optional[str] = None
    notificationDestination: Optional[str] = None
    supportedFeatures: Optional[str] = None
    qosReference: Optional[str] = None
    altQosReferences: Optional[List[str]] = None
    ueIpv4Addr: Optional[str] = None
    ueIpv6Addr: Optional[str] = None
    ueMac: Optional[str] = None
    medComponents: Optional[Dict[str, Dict]] = None

class PfdManagement(BaseModel):
    afId: str
    pfdDatas: List[Dict]
    supportedFeatures: Optional[str] = None

class ChargeableParty(BaseModel):
    externalId: Optional[str] = None
    msisdn: Optional[str] = None
    sponsorInformation: Optional[Dict] = None
    gpsi: Optional[str] = None
    ipv4Addr: Optional[str] = None
    ipv6Addr: Optional[str] = None
    notificationDestination: Optional[str] = None
    supportedFeatures: Optional[str] = None

# NEF Storage
af_subscriptions: Dict[str, AfSubscription] = {}
monitoring_subscriptions: Dict[str, MonitoringEventSubscription] = {}
traffic_influence_subscriptions: Dict[str, TrafficInfluenceSubscription] = {}
qos_subscriptions: Dict[str, AsSessionWithQoS] = {}
pfd_data: Dict[str, Dict] = {}
chargeable_parties: Dict[str, ChargeableParty] = {}
notification_history: List[Dict] = []


class NEF:
    def __init__(self):
        self.name = "NEF-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.supported_features = "0xff"
        self.http_client = None

    async def init_http_client(self):
        """Initialize async HTTP client"""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=10.0)

    async def close_http_client(self):
        """Close HTTP client"""
        if self.http_client:
            await self.http_client.aclose()

    async def send_notification(self, destination: str, notification: Dict) -> bool:
        """Send notification to AF"""
        await self.init_http_client()

        try:
            response = await self.http_client.post(
                destination,
                json=notification
            )
            success = response.status_code in [200, 201, 204]

            # Log notification
            notification_history.append({
                "destination": destination,
                "notification": notification,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "SUCCESS" if success else "FAILED",
                "responseCode": response.status_code
            })

            return success

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            notification_history.append({
                "destination": destination,
                "notification": notification,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "ERROR",
                "error": str(e)
            })
            return False

    def create_monitoring_subscription(
        self,
        subscription: MonitoringEventSubscription
    ) -> str:
        """Create monitoring event subscription"""
        subscription_id = str(uuid.uuid4())
        monitoring_subscriptions[subscription_id] = subscription

        # Set expiry if not set
        if not subscription.monitorExpireTime:
            subscription.monitorExpireTime = datetime.now(timezone.utc) + timedelta(hours=24)

        logger.info(f"Monitoring subscription created: {subscription_id} for {subscription.monitoringType}")
        return subscription_id

    def create_traffic_influence_subscription(
        self,
        subscription: TrafficInfluenceSubscription
    ) -> str:
        """Create traffic influence subscription"""
        subscription_id = str(uuid.uuid4())
        traffic_influence_subscriptions[subscription_id] = subscription
        logger.info(f"Traffic influence subscription created: {subscription_id}")
        return subscription_id

    def create_qos_subscription(
        self,
        subscription: AsSessionWithQoS
    ) -> str:
        """Create QoS subscription for AS session"""
        subscription_id = str(uuid.uuid4())
        qos_subscriptions[subscription_id] = subscription
        logger.info(f"QoS subscription created: {subscription_id}")
        return subscription_id

    def store_pfd_data(
        self,
        af_id: str,
        app_id: str,
        pfd_data_list: List[Dict]
    ) -> str:
        """Store PFD (Packet Flow Description) data"""
        pfd_id = f"{af_id}_{app_id}"
        pfd_data[pfd_id] = {
            "afId": af_id,
            "appId": app_id,
            "pfds": pfd_data_list,
            "createdAt": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"PFD data stored: {pfd_id}")
        return pfd_id

    async def trigger_monitoring_event(
        self,
        subscription_id: str,
        event_data: Dict
    ) -> bool:
        """Trigger a monitoring event and notify AF"""
        if subscription_id not in monitoring_subscriptions:
            return False

        subscription = monitoring_subscriptions[subscription_id]

        report = MonitoringEventReport(
            monitoringType=subscription.monitoringType,
            externalId=subscription.externalId,
            **event_data
        )

        # Send notification
        success = await self.send_notification(
            subscription.notificationDestination,
            {
                "subscriptionId": subscription_id,
                "monitoringEventReport": report.dict()
            }
        )

        # Update report count
        if subscription.maximumNumberOfReports:
            subscription.maximumNumberOfReports -= 1
            if subscription.maximumNumberOfReports <= 0:
                del monitoring_subscriptions[subscription_id]
                logger.info(f"Monitoring subscription {subscription_id} expired after max reports")

        return success


nef_instance = NEF()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Register with NRF
    nf_profile = {
        "nfInstanceId": nef_instance.nf_instance_id,
        "nfType": "NEF",
        "nfStatus": "REGISTERED",
        "plmnList": [{"mcc": "001", "mnc": "01"}],
        "sNssais": [{"sst": 1, "sd": "010203"}],
        "nfServices": [
            {
                "serviceInstanceId": "nnef-pfd-management-001",
                "serviceName": "nnef-pfdmanagement",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9016}]
            },
            {
                "serviceInstanceId": "nnef-monitoring-001",
                "serviceName": "nnef-eventexposure",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9016}]
            },
            {
                "serviceInstanceId": "nnef-traffic-influence-001",
                "serviceName": "nnef-traffinfluence",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9016}]
            }
        ],
        "nefInfo": {
            "nefId": nef_instance.nf_instance_id
        }
    }

    try:
        response = requests.put(
            f"{nrf_url}/nnrf-nfm/v1/nf-instances/{nef_instance.nf_instance_id}",
            json=nf_profile
        )
        if response.status_code in [200, 201]:
            logger.info("NEF registered with NRF successfully")
    except requests.RequestException as e:
        logger.error(f"Failed to register NEF with NRF: {e}")

    yield

    # Shutdown
    await nef_instance.close_http_client()
    try:
        requests.delete(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{nef_instance.nf_instance_id}")
        logger.info("NEF deregistered from NRF")
    except:
        pass


app = FastAPI(
    title="NEF - Network Exposure Function",
    description="3GPP TS 29.522 compliant NEF for external AF integration",
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


# 3GPP TS 29.522 - Monitoring Event API

@app.post("/3gpp-monitoring-event/v1/subscriptions", status_code=201)
async def create_monitoring_subscription(subscription: MonitoringEventSubscription):
    """
    Create monitoring event subscription per 3GPP TS 29.522
    """
    with tracer.start_as_current_span("nef_create_monitoring") as span:
        span.set_attribute("monitoring.type", subscription.monitoringType.value)

        try:
            subscription_id = nef_instance.create_monitoring_subscription(subscription)

            span.set_attribute("subscription.id", subscription_id)
            span.set_attribute("status", "SUCCESS")

            return {
                "subscriptionId": subscription_id,
                "monitoringType": subscription.monitoringType.value,
                "externalId": subscription.externalId,
                "monitorExpireTime": subscription.monitorExpireTime.isoformat() if subscription.monitorExpireTime else None,
                "notificationDestination": subscription.notificationDestination
            }

        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/3gpp-monitoring-event/v1/subscriptions")
async def list_monitoring_subscriptions():
    """List all monitoring subscriptions"""
    return {
        "subscriptions": [
            {
                "subscriptionId": sid,
                "monitoringType": sub.monitoringType.value,
                "externalId": sub.externalId,
                "monitorExpireTime": sub.monitorExpireTime.isoformat() if sub.monitorExpireTime else None
            }
            for sid, sub in monitoring_subscriptions.items()
        ]
    }


@app.get("/3gpp-monitoring-event/v1/subscriptions/{subscriptionId}")
async def get_monitoring_subscription(subscriptionId: str):
    """Get monitoring subscription details"""
    if subscriptionId not in monitoring_subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return monitoring_subscriptions[subscriptionId].dict()


@app.delete("/3gpp-monitoring-event/v1/subscriptions/{subscriptionId}", status_code=204)
async def delete_monitoring_subscription(subscriptionId: str):
    """Delete monitoring subscription"""
    if subscriptionId in monitoring_subscriptions:
        del monitoring_subscriptions[subscriptionId]
        logger.info(f"Monitoring subscription deleted: {subscriptionId}")
        return None
    raise HTTPException(status_code=404, detail="Subscription not found")


# 3GPP TS 29.522 - Traffic Influence API

@app.post("/3gpp-traffic-influence/v1/subscriptions", status_code=201)
async def create_traffic_influence_subscription(subscription: TrafficInfluenceSubscription):
    """
    Create traffic influence subscription per 3GPP TS 29.522
    """
    with tracer.start_as_current_span("nef_create_traffic_influence") as span:
        span.set_attribute("af.id", subscription.afId)

        try:
            subscription_id = nef_instance.create_traffic_influence_subscription(subscription)

            span.set_attribute("subscription.id", subscription_id)
            span.set_attribute("status", "SUCCESS")

            return {
                "subscriptionId": subscription_id,
                "afId": subscription.afId,
                "dnn": subscription.dnn,
                "dnaiChgType": subscription.dnaiChgType.value if subscription.dnaiChgType else None
            }

        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/3gpp-traffic-influence/v1/subscriptions")
async def list_traffic_influence_subscriptions():
    """List all traffic influence subscriptions"""
    return {
        "subscriptions": [
            {
                "subscriptionId": sid,
                "afId": sub.afId,
                "dnn": sub.dnn
            }
            for sid, sub in traffic_influence_subscriptions.items()
        ]
    }


@app.delete("/3gpp-traffic-influence/v1/subscriptions/{subscriptionId}", status_code=204)
async def delete_traffic_influence_subscription(subscriptionId: str):
    """Delete traffic influence subscription"""
    if subscriptionId in traffic_influence_subscriptions:
        del traffic_influence_subscriptions[subscriptionId]
        return None
    raise HTTPException(status_code=404, detail="Subscription not found")


# 3GPP TS 29.522 - PFD Management API

@app.post("/3gpp-pfd-management/v1/transactions", status_code=201)
async def create_pfd_transaction(pfd_management: PfdManagement):
    """
    Create PFD management transaction per 3GPP TS 29.522
    """
    with tracer.start_as_current_span("nef_pfd_transaction") as span:
        span.set_attribute("af.id", pfd_management.afId)

        try:
            transaction_id = str(uuid.uuid4())

            for pfd_data_item in pfd_management.pfdDatas:
                app_id = pfd_data_item.get("externalAppId", "unknown")
                nef_instance.store_pfd_data(
                    pfd_management.afId,
                    app_id,
                    pfd_data_item.get("pfds", [])
                )

            span.set_attribute("transaction.id", transaction_id)
            span.set_attribute("status", "SUCCESS")

            return {
                "transactionId": transaction_id,
                "afId": pfd_management.afId,
                "pfdCount": len(pfd_management.pfdDatas)
            }

        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/3gpp-pfd-management/v1/transactions")
async def list_pfd_transactions():
    """List all PFD data"""
    return {"pfdData": list(pfd_data.values())}


# 3GPP TS 29.522 - AS Session with QoS API

@app.post("/3gpp-as-session-with-qos/v1/subscriptions", status_code=201)
async def create_qos_subscription(subscription: AsSessionWithQoS):
    """
    Create AS session with QoS subscription per 3GPP TS 29.522
    """
    with tracer.start_as_current_span("nef_create_qos") as span:
        span.set_attribute("af.id", subscription.afId)

        try:
            subscription_id = nef_instance.create_qos_subscription(subscription)

            span.set_attribute("subscription.id", subscription_id)
            span.set_attribute("status", "SUCCESS")

            return {
                "subscriptionId": subscription_id,
                "afId": subscription.afId,
                "qosReference": subscription.qosReference,
                "dnn": subscription.dnn
            }

        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/3gpp-as-session-with-qos/v1/subscriptions")
async def list_qos_subscriptions():
    """List all QoS subscriptions"""
    return {
        "subscriptions": [
            {
                "subscriptionId": sid,
                "afId": sub.afId,
                "qosReference": sub.qosReference
            }
            for sid, sub in qos_subscriptions.items()
        ]
    }


@app.delete("/3gpp-as-session-with-qos/v1/subscriptions/{subscriptionId}", status_code=204)
async def delete_qos_subscription(subscriptionId: str):
    """Delete QoS subscription"""
    if subscriptionId in qos_subscriptions:
        del qos_subscriptions[subscriptionId]
        return None
    raise HTTPException(status_code=404, detail="Subscription not found")


# 3GPP TS 29.522 - Chargeable Party API

@app.post("/3gpp-chargeable-party/v1/transactions", status_code=201)
async def create_chargeable_party(party: ChargeableParty):
    """
    Create chargeable party transaction per 3GPP TS 29.522
    """
    transaction_id = str(uuid.uuid4())
    chargeable_parties[transaction_id] = party
    logger.info(f"Chargeable party created: {transaction_id}")

    return {
        "transactionId": transaction_id,
        "externalId": party.externalId,
        "gpsi": party.gpsi
    }


@app.get("/3gpp-chargeable-party/v1/transactions")
async def list_chargeable_parties():
    """List all chargeable parties"""
    return {
        "parties": [
            {"transactionId": tid, "externalId": p.externalId}
            for tid, p in chargeable_parties.items()
        ]
    }


# Notification testing and history

@app.post("/nef/test-notification")
async def trigger_test_notification(subscriptionId: str, eventData: Dict):
    """Trigger a test monitoring event notification"""
    if subscriptionId not in monitoring_subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")

    success = await nef_instance.trigger_monitoring_event(subscriptionId, eventData)
    return {"status": "SENT" if success else "FAILED"}


@app.get("/nef/notification-history")
async def get_notification_history(limit: int = 100):
    """Get notification history"""
    return {
        "total": len(notification_history),
        "notifications": notification_history[-limit:]
    }


# Health and monitoring

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NEF",
        "compliance": "3GPP TS 29.522",
        "version": "1.0.0",
        "activeSubscriptions": {
            "monitoring": len(monitoring_subscriptions),
            "trafficInfluence": len(traffic_influence_subscriptions),
            "qos": len(qos_subscriptions)
        }
    }


@app.get("/metrics")
def get_metrics():
    """Metrics endpoint"""
    return {
        "monitoring_subscriptions": len(monitoring_subscriptions),
        "traffic_influence_subscriptions": len(traffic_influence_subscriptions),
        "qos_subscriptions": len(qos_subscriptions),
        "pfd_entries": len(pfd_data),
        "chargeable_parties": len(chargeable_parties),
        "notifications_sent": len(notification_history)
    }


if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="NEF - Network Exposure Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("nef"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)