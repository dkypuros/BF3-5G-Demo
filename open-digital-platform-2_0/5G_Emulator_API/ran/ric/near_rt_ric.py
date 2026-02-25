#!/usr/bin/env python3
"""
Near-RT RIC (Near-Real-Time RAN Intelligent Controller)
ETSI TS 104038/104039/104040 Compliant Implementation

This module implements the Near-RT RIC which provides:
- E2 interface to E2 Nodes (CU, DU, gNB)
- Near-real-time control loop (10ms - 1s)
- xApp hosting and lifecycle management
- RIC services: REPORT, INSERT, CONTROL, POLICY, QUERY

Port: 8095
"""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)


# =============================================================================
# Enumerations (per ETSI TS 104038/104039)
# =============================================================================

class E2NodeType(str, Enum):
    """E2 Node types per O-RAN architecture"""
    GNB = "gNB"
    GNB_CU = "gNB-CU"
    GNB_DU = "gNB-DU"
    GNB_CU_CP = "gNB-CU-CP"
    GNB_CU_UP = "gNB-CU-UP"
    EN_GNB = "en-gNB"
    NG_ENB = "ng-eNB"


class RicServiceType(str, Enum):
    """RIC Service types per ETSI TS 104038 Section 7"""
    REPORT = "REPORT"
    INSERT = "INSERT"
    CONTROL = "CONTROL"
    POLICY = "POLICY"
    QUERY = "QUERY"


class SubscriptionState(str, Enum):
    """Subscription lifecycle states"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    DELETED = "DELETED"


class XAppState(str, Enum):
    """xApp lifecycle states"""
    REGISTERED = "REGISTERED"
    DEPLOYING = "DEPLOYING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"


class IndicationType(str, Enum):
    """RIC Indication types per ETSI TS 104039"""
    REPORT = "REPORT"
    INSERT = "INSERT"


# =============================================================================
# Data Models (Pydantic)
# =============================================================================

class RanFunction(BaseModel):
    """RAN Function definition exposed by E2 Node"""
    ranFunctionId: int = Field(..., description="RAN Function ID")
    ranFunctionDefinition: str = Field(..., description="Function definition")
    ranFunctionRevision: int = Field(default=1, description="Revision number")
    ranFunctionOid: Optional[str] = Field(default=None, description="OID for service model")


class E2NodeConfig(BaseModel):
    """E2 Node configuration from E2 Setup"""
    globalE2NodeId: str = Field(..., description="Global E2 Node ID")
    nodeType: E2NodeType = Field(..., description="E2 Node type")
    ranFunctions: List[RanFunction] = Field(default=[], description="Supported RAN functions")
    e2NodeComponentConfig: Optional[Dict] = Field(default=None, description="Component config")


class E2Node(BaseModel):
    """E2 Node connection state"""
    e2NodeId: str = Field(..., description="E2 Node identifier")
    nodeType: E2NodeType = Field(..., description="Node type")
    globalE2NodeId: str = Field(..., description="Global E2 Node ID")
    ranFunctions: List[RanFunction] = Field(default=[], description="RAN functions")
    connectionState: str = Field(default="CONNECTED", description="Connection state")
    endpoint: str = Field(..., description="E2 Node endpoint URL")
    connectedAt: datetime = Field(default_factory=datetime.utcnow)
    lastHeartbeat: datetime = Field(default_factory=datetime.utcnow)


class EventTrigger(BaseModel):
    """Event trigger definition for subscription"""
    triggerType: str = Field(..., description="Trigger type (periodic, event-based)")
    reportingPeriodMs: Optional[int] = Field(default=None, description="Reporting period in ms")
    eventConditions: Optional[Dict] = Field(default=None, description="Event conditions")


class RicAction(BaseModel):
    """RIC Action in subscription"""
    actionId: int = Field(..., description="Action ID")
    actionType: RicServiceType = Field(..., description="Action type")
    actionDefinition: Optional[Dict] = Field(default=None, description="Action parameters")
    subsequentAction: Optional[str] = Field(default=None, description="Subsequent action")


class RicSubscriptionRequest(BaseModel):
    """RIC Subscription Request per ETSI TS 104039 Section 8.2.1"""
    e2NodeId: str = Field(..., description="Target E2 Node ID")
    ranFunctionId: int = Field(..., description="RAN Function ID")
    eventTrigger: EventTrigger = Field(..., description="Event trigger definition")
    actions: List[RicAction] = Field(..., description="List of actions")
    xappId: Optional[str] = Field(default=None, description="Requesting xApp ID")


class RicSubscription(BaseModel):
    """RIC Subscription state"""
    subscriptionId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ricRequestId: int = Field(..., description="RIC Request ID")
    e2NodeId: str = Field(..., description="E2 Node ID")
    ranFunctionId: int = Field(..., description="RAN Function ID")
    eventTrigger: EventTrigger = Field(..., description="Event trigger")
    actions: List[RicAction] = Field(..., description="Actions")
    state: SubscriptionState = Field(default=SubscriptionState.PENDING)
    xappId: Optional[str] = Field(default=None, description="Owning xApp")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    lastIndication: Optional[datetime] = Field(default=None)


class RicIndication(BaseModel):
    """RIC Indication from E2 Node per ETSI TS 104039 Section 8.2.5"""
    ricRequestId: int = Field(..., description="RIC Request ID")
    ranFunctionId: int = Field(..., description="RAN Function ID")
    indicationType: IndicationType = Field(..., description="REPORT or INSERT")
    indicationHeader: Dict = Field(..., description="Indication header")
    indicationMessage: Dict = Field(..., description="Indication message/payload")
    callProcessId: Optional[str] = Field(default=None, description="Call process ID for INSERT")


class RicControlRequest(BaseModel):
    """RIC Control Request per ETSI TS 104039 Section 8.2.6"""
    e2NodeId: str = Field(..., description="Target E2 Node ID")
    ranFunctionId: int = Field(..., description="RAN Function ID")
    controlHeader: Dict = Field(..., description="Control header")
    controlMessage: Dict = Field(..., description="Control message")
    controlAckRequest: bool = Field(default=True, description="Request acknowledgment")
    callProcessId: Optional[str] = Field(default=None, description="Call process ID")


class RicControlResponse(BaseModel):
    """RIC Control Response"""
    e2NodeId: str
    ranFunctionId: int
    controlOutcome: Optional[Dict] = None
    success: bool = True
    errorCause: Optional[str] = None


class RicQueryRequest(BaseModel):
    """RIC Query Request per ETSI TS 104039"""
    e2NodeId: str = Field(..., description="Target E2 Node ID")
    ranFunctionId: int = Field(..., description="RAN Function ID")
    queryHeader: Dict = Field(..., description="Query header")
    queryDefinition: Dict = Field(..., description="Query definition")


class XApp(BaseModel):
    """xApp registration and state"""
    xappId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    xappName: str = Field(..., description="xApp name")
    version: str = Field(default="1.0.0", description="xApp version")
    vendor: Optional[str] = Field(default=None, description="Vendor name")
    state: XAppState = Field(default=XAppState.REGISTERED)
    endpoint: Optional[str] = Field(default=None, description="xApp callback endpoint")
    subscribedE2Nodes: List[str] = Field(default=[], description="E2 Nodes subscribed to")
    registeredAt: datetime = Field(default_factory=datetime.utcnow)


class A1Policy(BaseModel):
    """A1 Policy received from Non-RT RIC"""
    policyId: str = Field(..., description="Policy instance ID")
    policyTypeId: str = Field(..., description="Policy type ID")
    policyData: Dict = Field(..., description="Policy parameters")
    scope: Optional[Dict] = Field(default=None, description="Policy scope (cells, UEs)")
    status: str = Field(default="ACTIVE", description="Policy status")
    receivedAt: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Near-RT RIC Core Class
# =============================================================================

class NearRtRic:
    """
    Near-RT RIC Implementation

    Implements E2 interface to E2 Nodes and A1 receiver from Non-RT RIC.
    Provides near-real-time control loop (10ms - 1s latency).

    Per ETSI TS 104038 Section 6:
    - E2 Setup procedure handling
    - RIC Subscription management
    - RIC Indication processing
    - RIC Control execution
    - RIC Query handling
    """

    def __init__(
        self,
        ric_id: str = "near-rt-ric-001",
        rnis_url: str = "http://127.0.0.1:8092",
        non_rt_ric_url: str = "http://127.0.0.1:8096"
    ):
        self.ric_id = ric_id
        self.rnis_url = rnis_url
        self.non_rt_ric_url = non_rt_ric_url

        # E2 Node registry
        self.e2_nodes: Dict[str, E2Node] = {}

        # Subscription management
        self.subscriptions: Dict[str, RicSubscription] = {}
        self.ric_request_id_counter = 1

        # xApp registry
        self.xapps: Dict[str, XApp] = {}

        # A1 Policy store
        self.a1_policies: Dict[str, A1Policy] = {}

        # Enrichment information cache
        self.enrichment_info: Dict[str, Any] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

        logger.info(f"Near-RT RIC initialized: {self.ric_id}")

    # -------------------------------------------------------------------------
    # E2 Setup Procedures (ETSI TS 104039 Section 8.2.1)
    # -------------------------------------------------------------------------

    def handle_e2_setup(self, config: E2NodeConfig, endpoint: str) -> Dict:
        """
        Handle E2 Setup Request from E2 Node

        Per ETSI TS 104039 Section 8.2.1.2:
        - Validates E2 Node configuration
        - Registers RAN functions
        - Returns E2 Setup Response with accepted functions
        """
        with tracer.start_as_current_span("e2_setup") as span:
            span.set_attribute("e2.node_id", config.globalE2NodeId)
            span.set_attribute("e2.node_type", config.nodeType.value)

            e2_node_id = config.globalE2NodeId

            # Create E2 Node entry
            e2_node = E2Node(
                e2NodeId=e2_node_id,
                nodeType=config.nodeType,
                globalE2NodeId=config.globalE2NodeId,
                ranFunctions=config.ranFunctions,
                endpoint=endpoint
            )

            self.e2_nodes[e2_node_id] = e2_node

            # Build accepted RAN functions list
            accepted_functions = []
            for rf in config.ranFunctions:
                accepted_functions.append({
                    "ranFunctionId": rf.ranFunctionId,
                    "ranFunctionRevision": rf.ranFunctionRevision
                })

            logger.info(f"E2 Setup completed for {e2_node_id}, "
                       f"accepted {len(accepted_functions)} RAN functions")

            return {
                "ricId": self.ric_id,
                "e2NodeId": e2_node_id,
                "ranFunctionsAccepted": accepted_functions,
                "ranFunctionsRejected": [],
                "transactionId": str(uuid.uuid4())
            }

    # -------------------------------------------------------------------------
    # RIC Subscription Procedures (ETSI TS 104039 Section 8.2.3)
    # -------------------------------------------------------------------------

    async def create_subscription(self, request: RicSubscriptionRequest) -> RicSubscription:
        """
        Create RIC Subscription

        Per ETSI TS 104039 Section 8.2.3.2:
        - Validates E2 Node and RAN function
        - Creates subscription with event trigger
        - Sends RIC Subscription Request to E2 Node
        """
        with tracer.start_as_current_span("ric_subscription_create") as span:
            span.set_attribute("e2.node_id", request.e2NodeId)
            span.set_attribute("ric.ran_function_id", request.ranFunctionId)

            # Validate E2 Node exists
            if request.e2NodeId not in self.e2_nodes:
                raise HTTPException(
                    status_code=404,
                    detail=f"E2 Node {request.e2NodeId} not found"
                )

            e2_node = self.e2_nodes[request.e2NodeId]

            # Validate RAN function is supported
            supported_func_ids = [rf.ranFunctionId for rf in e2_node.ranFunctions]
            if request.ranFunctionId not in supported_func_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"RAN Function {request.ranFunctionId} not supported by E2 Node"
                )

            # Generate RIC Request ID
            ric_request_id = self.ric_request_id_counter
            self.ric_request_id_counter += 1

            # Create subscription
            subscription = RicSubscription(
                ricRequestId=ric_request_id,
                e2NodeId=request.e2NodeId,
                ranFunctionId=request.ranFunctionId,
                eventTrigger=request.eventTrigger,
                actions=request.actions,
                xappId=request.xappId
            )

            # Send subscription request to E2 Node
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{e2_node.endpoint}/e2/subscription",
                        json={
                            "ricRequestId": ric_request_id,
                            "ranFunctionId": request.ranFunctionId,
                            "eventTrigger": request.eventTrigger.model_dump(),
                            "actions": [a.model_dump() for a in request.actions]
                        },
                        timeout=5.0
                    )

                    if response.status_code in (200, 201, 202):
                        subscription.state = SubscriptionState.ACTIVE
                    else:
                        subscription.state = SubscriptionState.FAILED
                        logger.warning(f"Subscription request failed: {response.text}")

            except httpx.RequestError as e:
                # E2 Node not reachable, but store subscription for retry
                subscription.state = SubscriptionState.PENDING
                logger.warning(f"E2 Node unreachable, subscription pending: {e}")

            self.subscriptions[subscription.subscriptionId] = subscription

            logger.info(f"Subscription {subscription.subscriptionId} created, "
                       f"state: {subscription.state}")

            return subscription

    async def delete_subscription(self, subscription_id: str) -> bool:
        """Delete RIC Subscription"""
        if subscription_id not in self.subscriptions:
            raise HTTPException(status_code=404, detail="Subscription not found")

        subscription = self.subscriptions[subscription_id]

        # Send delete request to E2 Node
        if subscription.e2NodeId in self.e2_nodes:
            e2_node = self.e2_nodes[subscription.e2NodeId]
            try:
                async with httpx.AsyncClient() as client:
                    await client.delete(
                        f"{e2_node.endpoint}/e2/subscription/{subscription.ricRequestId}",
                        timeout=5.0
                    )
            except httpx.RequestError:
                pass  # Best effort

        subscription.state = SubscriptionState.DELETED
        del self.subscriptions[subscription_id]

        logger.info(f"Subscription {subscription_id} deleted")
        return True

    # -------------------------------------------------------------------------
    # RIC Indication Handling (ETSI TS 104039 Section 8.2.5)
    # -------------------------------------------------------------------------

    async def handle_indication(self, indication: RicIndication) -> Dict:
        """
        Handle RIC Indication from E2 Node

        Per ETSI TS 104039 Section 8.2.5:
        - Processes REPORT indications (measurements, events)
        - Processes INSERT indications (decision points)
        - Routes to appropriate xApp
        """
        with tracer.start_as_current_span("ric_indication") as span:
            span.set_attribute("ric.request_id", indication.ricRequestId)
            span.set_attribute("ric.indication_type", indication.indicationType.value)

            # Find subscription by RIC Request ID
            subscription = None
            for sub in self.subscriptions.values():
                if sub.ricRequestId == indication.ricRequestId:
                    subscription = sub
                    break

            if subscription:
                subscription.lastIndication = datetime.utcnow()

                # Route to xApp if registered
                if subscription.xappId and subscription.xappId in self.xapps:
                    xapp = self.xapps[subscription.xappId]
                    if xapp.endpoint:
                        try:
                            async with httpx.AsyncClient() as client:
                                await client.post(
                                    f"{xapp.endpoint}/indication",
                                    json=indication.model_dump(),
                                    timeout=1.0  # Near-RT latency requirement
                                )
                        except httpx.RequestError as e:
                            logger.warning(f"Failed to route indication to xApp: {e}")

            logger.debug(f"Processed {indication.indicationType} indication "
                        f"for RIC Request ID {indication.ricRequestId}")

            return {"status": "processed", "ricRequestId": indication.ricRequestId}

    # -------------------------------------------------------------------------
    # RIC Control Procedures (ETSI TS 104039 Section 8.2.6)
    # -------------------------------------------------------------------------

    async def send_control(self, request: RicControlRequest) -> RicControlResponse:
        """
        Send RIC Control Request to E2 Node

        Per ETSI TS 104039 Section 8.2.6:
        - Validates target E2 Node
        - Sends control message
        - Optionally waits for acknowledgment
        """
        with tracer.start_as_current_span("ric_control") as span:
            span.set_attribute("e2.node_id", request.e2NodeId)
            span.set_attribute("ric.ran_function_id", request.ranFunctionId)

            if request.e2NodeId not in self.e2_nodes:
                raise HTTPException(
                    status_code=404,
                    detail=f"E2 Node {request.e2NodeId} not found"
                )

            e2_node = self.e2_nodes[request.e2NodeId]

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{e2_node.endpoint}/e2/control",
                        json={
                            "ranFunctionId": request.ranFunctionId,
                            "controlHeader": request.controlHeader,
                            "controlMessage": request.controlMessage,
                            "callProcessId": request.callProcessId
                        },
                        timeout=1.0  # Near-RT latency requirement
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return RicControlResponse(
                            e2NodeId=request.e2NodeId,
                            ranFunctionId=request.ranFunctionId,
                            controlOutcome=result.get("controlOutcome"),
                            success=True
                        )
                    else:
                        return RicControlResponse(
                            e2NodeId=request.e2NodeId,
                            ranFunctionId=request.ranFunctionId,
                            success=False,
                            errorCause=response.text
                        )

            except httpx.RequestError as e:
                logger.error(f"Control request failed: {e}")
                return RicControlResponse(
                    e2NodeId=request.e2NodeId,
                    ranFunctionId=request.ranFunctionId,
                    success=False,
                    errorCause=str(e)
                )

    # -------------------------------------------------------------------------
    # RIC Query Procedures (ETSI TS 104039 Section 8.2.7)
    # -------------------------------------------------------------------------

    async def send_query(self, request: RicQueryRequest) -> Dict:
        """Send RIC Query to E2 Node"""
        if request.e2NodeId not in self.e2_nodes:
            raise HTTPException(status_code=404, detail="E2 Node not found")

        e2_node = self.e2_nodes[request.e2NodeId]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{e2_node.endpoint}/e2/query",
                    json={
                        "ranFunctionId": request.ranFunctionId,
                        "queryHeader": request.queryHeader,
                        "queryDefinition": request.queryDefinition
                    },
                    timeout=2.0
                )
                return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=str(e))

    # -------------------------------------------------------------------------
    # xApp Management
    # -------------------------------------------------------------------------

    def register_xapp(self, xapp: XApp) -> XApp:
        """Register an xApp with the Near-RT RIC"""
        self.xapps[xapp.xappId] = xapp
        logger.info(f"xApp registered: {xapp.xappName} ({xapp.xappId})")
        return xapp

    def unregister_xapp(self, xapp_id: str) -> bool:
        """Unregister an xApp"""
        if xapp_id not in self.xapps:
            raise HTTPException(status_code=404, detail="xApp not found")

        # Clean up subscriptions owned by this xApp
        for sub_id, sub in list(self.subscriptions.items()):
            if sub.xappId == xapp_id:
                asyncio.create_task(self.delete_subscription(sub_id))

        del self.xapps[xapp_id]
        logger.info(f"xApp unregistered: {xapp_id}")
        return True

    # -------------------------------------------------------------------------
    # A1 Interface (Receiver from Non-RT RIC)
    # -------------------------------------------------------------------------

    def receive_a1_policy(self, policy: A1Policy) -> A1Policy:
        """
        Receive A1 Policy from Non-RT RIC

        Per ETSI TS 103983:
        - Store policy for enforcement
        - Apply to relevant xApps/control decisions
        """
        self.a1_policies[policy.policyId] = policy
        logger.info(f"A1 Policy received: {policy.policyId} (type: {policy.policyTypeId})")
        return policy

    def delete_a1_policy(self, policy_id: str) -> bool:
        """Delete A1 Policy"""
        if policy_id not in self.a1_policies:
            raise HTTPException(status_code=404, detail="Policy not found")
        del self.a1_policies[policy_id]
        logger.info(f"A1 Policy deleted: {policy_id}")
        return True

    def receive_enrichment_info(self, ei_type: str, ei_data: Dict) -> Dict:
        """Receive Enrichment Information from Non-RT RIC"""
        self.enrichment_info[ei_type] = {
            "data": ei_data,
            "receivedAt": datetime.utcnow().isoformat()
        }
        logger.info(f"Enrichment info received: {ei_type}")
        return {"status": "received", "eiType": ei_type}

    # -------------------------------------------------------------------------
    # RNIS Integration
    # -------------------------------------------------------------------------

    async def get_radio_measurements(self, ue_id: str) -> Dict:
        """Query RNIS for radio measurements"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.rnis_url}/rni/v2/ue/{ue_id}/measurements",
                    timeout=2.0
                )
                return response.json()
        except httpx.RequestError as e:
            logger.warning(f"RNIS query failed: {e}")
            return {"error": str(e)}

    async def get_cell_info(self) -> List[Dict]:
        """Query RNIS for cell information"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.rnis_url}/rni/v2/cells",
                    timeout=2.0
                )
                return response.json()
        except httpx.RequestError as e:
            logger.warning(f"RNIS cell query failed: {e}")
            return []

    # -------------------------------------------------------------------------
    # Status and Metrics
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Get Near-RT RIC status"""
        return {
            "ricId": self.ric_id,
            "e2NodesConnected": len(self.e2_nodes),
            "activeSubscriptions": len([s for s in self.subscriptions.values()
                                       if s.state == SubscriptionState.ACTIVE]),
            "registeredXApps": len(self.xapps),
            "activePolicies": len(self.a1_policies),
            "status": "RUNNING"
        }


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Near-RT RIC",
    description="Near-Real-Time RAN Intelligent Controller - O-RAN Compliant",
    version="1.0.0",
    docs_url="/ric/docs",
    redoc_url="/ric/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Near-RT RIC
ric = NearRtRic()


# =============================================================================
# E2 Interface Endpoints
# =============================================================================

@app.post("/e2/setup", response_model=Dict)
async def e2_setup(request: Request, config: E2NodeConfig):
    """
    E2 Setup Request from E2 Node
    Per ETSI TS 104039 Section 8.2.1
    """
    # Extract endpoint from request headers or body
    endpoint = request.headers.get("X-E2-Node-Endpoint", "http://127.0.0.1:38472")
    return ric.handle_e2_setup(config, endpoint)


@app.post("/e2/subscription", response_model=RicSubscription, status_code=201)
async def create_subscription(request: RicSubscriptionRequest):
    """
    Create RIC Subscription
    Per ETSI TS 104039 Section 8.2.3
    """
    return await ric.create_subscription(request)


@app.get("/e2/subscriptions", response_model=List[RicSubscription])
async def list_subscriptions():
    """List all RIC Subscriptions"""
    return list(ric.subscriptions.values())


@app.get("/e2/subscriptions/{subscription_id}", response_model=RicSubscription)
async def get_subscription(subscription_id: str):
    """Get specific subscription"""
    if subscription_id not in ric.subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return ric.subscriptions[subscription_id]


@app.delete("/e2/subscriptions/{subscription_id}")
async def delete_subscription(subscription_id: str):
    """Delete RIC Subscription"""
    await ric.delete_subscription(subscription_id)
    return {"status": "deleted", "subscriptionId": subscription_id}


@app.post("/e2/indication")
async def receive_indication(indication: RicIndication):
    """
    Receive RIC Indication from E2 Node
    Per ETSI TS 104039 Section 8.2.5
    """
    return await ric.handle_indication(indication)


@app.post("/e2/control", response_model=RicControlResponse)
async def send_control(request: RicControlRequest):
    """
    Send RIC Control to E2 Node
    Per ETSI TS 104039 Section 8.2.6
    """
    return await ric.send_control(request)


@app.post("/e2/query")
async def send_query(request: RicQueryRequest):
    """
    Send RIC Query to E2 Node
    Per ETSI TS 104039 Section 8.2.7
    """
    return await ric.send_query(request)


# =============================================================================
# E2 Node Management Endpoints
# =============================================================================

@app.get("/ric/e2-nodes", response_model=List[E2Node])
async def list_e2_nodes():
    """List connected E2 Nodes"""
    return list(ric.e2_nodes.values())


@app.get("/ric/e2-nodes/{e2_node_id}", response_model=E2Node)
async def get_e2_node(e2_node_id: str):
    """Get specific E2 Node"""
    if e2_node_id not in ric.e2_nodes:
        raise HTTPException(status_code=404, detail="E2 Node not found")
    return ric.e2_nodes[e2_node_id]


# =============================================================================
# xApp Management Endpoints
# =============================================================================

@app.get("/ric/xapps", response_model=List[XApp])
async def list_xapps():
    """List registered xApps"""
    return list(ric.xapps.values())


@app.post("/ric/xapps", response_model=XApp, status_code=201)
async def register_xapp(xapp: XApp):
    """Register an xApp"""
    return ric.register_xapp(xapp)


@app.delete("/ric/xapps/{xapp_id}")
async def unregister_xapp(xapp_id: str):
    """Unregister an xApp"""
    ric.unregister_xapp(xapp_id)
    return {"status": "unregistered", "xappId": xapp_id}


# =============================================================================
# A1 Interface Endpoints (Receiver from Non-RT RIC)
# =============================================================================

@app.post("/a1/policies", response_model=A1Policy, status_code=201)
async def receive_policy(policy: A1Policy):
    """
    Receive A1 Policy from Non-RT RIC
    Per ETSI TS 103983
    """
    return ric.receive_a1_policy(policy)


@app.get("/a1/policies", response_model=List[A1Policy])
async def list_policies():
    """List active A1 Policies"""
    return list(ric.a1_policies.values())


@app.delete("/a1/policies/{policy_id}")
async def delete_policy(policy_id: str):
    """Delete A1 Policy"""
    ric.delete_a1_policy(policy_id)
    return {"status": "deleted", "policyId": policy_id}


@app.post("/a1/enrichment")
async def receive_enrichment(ei_type: str, ei_data: Dict):
    """Receive Enrichment Information from Non-RT RIC"""
    return ric.receive_enrichment_info(ei_type, ei_data)


# =============================================================================
# RNIS Integration Endpoints
# =============================================================================

@app.get("/ric/radio/ue/{ue_id}")
async def get_ue_measurements(ue_id: str):
    """Get radio measurements for UE via RNIS"""
    return await ric.get_radio_measurements(ue_id)


@app.get("/ric/radio/cells")
async def get_cells():
    """Get cell information via RNIS"""
    return await ric.get_cell_info()


# =============================================================================
# Health and Metrics
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "near-rt-ric",
        "version": "1.0.0",
        "spec": "ETSI TS 104038/104039/104040"
    }


@app.get("/ric/status")
async def get_status():
    """Get Near-RT RIC status"""
    return ric.get_status()


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics"""
    status = ric.get_status()
    metrics = f"""# HELP near_rt_ric_e2_nodes_connected Number of connected E2 nodes
# TYPE near_rt_ric_e2_nodes_connected gauge
near_rt_ric_e2_nodes_connected {status['e2NodesConnected']}

# HELP near_rt_ric_subscriptions_active Number of active subscriptions
# TYPE near_rt_ric_subscriptions_active gauge
near_rt_ric_subscriptions_active {status['activeSubscriptions']}

# HELP near_rt_ric_xapps_registered Number of registered xApps
# TYPE near_rt_ric_xapps_registered gauge
near_rt_ric_xapps_registered {status['registeredXApps']}

# HELP near_rt_ric_policies_active Number of active A1 policies
# TYPE near_rt_ric_policies_active gauge
near_rt_ric_policies_active {status['activePolicies']}
"""
    return metrics


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8095)
