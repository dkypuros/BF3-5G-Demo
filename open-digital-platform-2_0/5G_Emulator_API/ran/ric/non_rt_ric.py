#!/usr/bin/env python3
"""
Non-RT RIC (Non-Real-Time RAN Intelligent Controller)
ETSI TS 103983 Compliant Implementation

This module implements the Non-RT RIC which provides:
- A1 interface to Near-RT RIC (policy, enrichment info, ML models)
- O1 integration with SMO (ZSM, VNFM)
- rApp hosting and lifecycle management
- Long-term RAN optimization (>1 second timescale)

Port: 8096
"""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)


# =============================================================================
# Enumerations (per ETSI TS 103983)
# =============================================================================

class PolicyStatus(str, Enum):
    """A1 Policy status per ETSI TS 103983"""
    NOT_ENFORCED = "NOT_ENFORCED"
    ENFORCED = "ENFORCED"
    DELETED = "DELETED"


class PolicyTypeStatus(str, Enum):
    """Policy type status"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class EiJobStatus(str, Enum):
    """Enrichment Information job status"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    DELIVERING = "DELIVERING"
    STOPPED = "STOPPED"


class RAppState(str, Enum):
    """rApp lifecycle states"""
    REGISTERED = "REGISTERED"
    DEPLOYING = "DEPLOYING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"


# =============================================================================
# Data Models (Pydantic)
# =============================================================================

# A1-P Policy Management Models

class PolicyTypeSchema(BaseModel):
    """Policy type JSON schema definition"""
    type: str = "object"
    properties: Dict = Field(default={})
    required: List[str] = Field(default=[])


class PolicyType(BaseModel):
    """
    A1 Policy Type per ETSI TS 103983 Section 7.2

    Defines a category of policies with a schema for policy instances.
    """
    policyTypeId: str = Field(..., description="Policy type identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(default="", description="Description")
    policySchema: PolicyTypeSchema = Field(..., description="JSON Schema for policy instances")
    status: PolicyTypeStatus = Field(default=PolicyTypeStatus.ACTIVE)
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class PolicyInstance(BaseModel):
    """
    A1 Policy Instance per ETSI TS 103983 Section 7.3

    A specific policy based on a policy type.
    """
    policyId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    policyTypeId: str = Field(..., description="Parent policy type ID")
    policyData: Dict = Field(..., description="Policy parameters per schema")
    scope: Optional[Dict] = Field(default=None, description="Policy scope (cells, UEs)")
    status: PolicyStatus = Field(default=PolicyStatus.NOT_ENFORCED)
    nearRtRicStatus: Optional[str] = Field(default=None, description="Status from Near-RT RIC")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    lastUpdated: datetime = Field(default_factory=datetime.utcnow)


class PolicyStatusResponse(BaseModel):
    """Policy status response from Near-RT RIC"""
    policyId: str
    status: PolicyStatus
    reason: Optional[str] = None
    lastEnforced: Optional[datetime] = None


# A1-EI Enrichment Information Models

class EiType(BaseModel):
    """Enrichment Information Type"""
    eiTypeId: str = Field(..., description="EI type identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(default="")
    schema: Dict = Field(default={}, description="Data schema")


class EiJob(BaseModel):
    """
    Enrichment Information Job per ETSI TS 103983 Section 8

    Defines a subscription for enrichment information delivery.
    """
    eiJobId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    eiTypeId: str = Field(..., description="EI type")
    targetUri: str = Field(..., description="Delivery endpoint (Near-RT RIC)")
    jobDefinition: Dict = Field(default={}, description="Job parameters")
    status: EiJobStatus = Field(default=EiJobStatus.PENDING)
    createdAt: datetime = Field(default_factory=datetime.utcnow)


# rApp Models

class RApp(BaseModel):
    """rApp registration and state"""
    rappId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rappName: str = Field(..., description="rApp name")
    version: str = Field(default="1.0.0")
    vendor: Optional[str] = Field(default=None)
    state: RAppState = Field(default=RAppState.REGISTERED)
    endpoint: Optional[str] = Field(default=None, description="rApp callback endpoint")
    managedPolicies: List[str] = Field(default=[], description="Managed policy types")
    registeredAt: datetime = Field(default_factory=datetime.utcnow)


# O1 Integration Models

class ZsmIntent(BaseModel):
    """ZSM Intent for closed-loop automation"""
    intentId: str
    intentType: str
    target: str
    objective: str
    constraints: Optional[Dict] = None
    priority: int = Field(default=5, ge=1, le=10)


class VnfInstance(BaseModel):
    """VNF Instance from VNFM"""
    vnfInstanceId: str
    vnfdId: str
    vnfInstanceName: str
    instantiationState: str
    vnfState: Optional[str] = None


# =============================================================================
# Non-RT RIC Core Class
# =============================================================================

class NonRtRic:
    """
    Non-RT RIC Implementation

    Implements A1 interface to Near-RT RIC and O1 integration with SMO.
    Provides non-real-time control loop (>1 second latency).

    Per ETSI TS 103983:
    - A1-P: Policy Management Service
    - A1-EI: Enrichment Information Service
    - A1-ML: ML Model Management Service (future)
    """

    def __init__(
        self,
        ric_id: str = "non-rt-ric-001",
        near_rt_ric_url: str = "http://127.0.0.1:8095",
        zsm_url: str = "http://127.0.0.1:8094",
        vnfm_url: str = "http://127.0.0.1:8093"
    ):
        self.ric_id = ric_id
        self.near_rt_ric_url = near_rt_ric_url
        self.zsm_url = zsm_url
        self.vnfm_url = vnfm_url

        # A1-P Policy Management
        self.policy_types: Dict[str, PolicyType] = {}
        self.policies: Dict[str, PolicyInstance] = {}

        # A1-EI Enrichment Information
        self.ei_types: Dict[str, EiType] = {}
        self.ei_jobs: Dict[str, EiJob] = {}

        # rApp Registry
        self.rapps: Dict[str, RApp] = {}

        # Analytics data store
        self.analytics_data: Dict[str, Any] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

        # Initialize default policy types
        self._initialize_default_policy_types()

        logger.info(f"Non-RT RIC initialized: {self.ric_id}")

    def _initialize_default_policy_types(self):
        """Initialize standard O-RAN policy types"""

        # QoS Target Policy (ORAN-WG2)
        qos_policy_type = PolicyType(
            policyTypeId="ORAN_QoSTarget_1.0.0",
            name="QoS Target Policy",
            description="Policy for QoS optimization objectives",
            policySchema=PolicyTypeSchema(
                properties={
                    "qosObjective": {"type": "string", "enum": ["maximize_throughput", "minimize_latency", "balance"]},
                    "targetKpi": {"type": "object"},
                    "scope": {"type": "object"}
                },
                required=["qosObjective"]
            )
        )
        self.policy_types[qos_policy_type.policyTypeId] = qos_policy_type

        # Traffic Steering Policy
        ts_policy_type = PolicyType(
            policyTypeId="ORAN_TrafficSteering_1.0.0",
            name="Traffic Steering Policy",
            description="Policy for traffic distribution across cells",
            policySchema=PolicyTypeSchema(
                properties={
                    "steeringObjective": {"type": "string"},
                    "targetCells": {"type": "array", "items": {"type": "string"}},
                    "loadThreshold": {"type": "number", "minimum": 0, "maximum": 100}
                },
                required=["steeringObjective"]
            )
        )
        self.policy_types[ts_policy_type.policyTypeId] = ts_policy_type

        # Load Balancing Policy
        lb_policy_type = PolicyType(
            policyTypeId="ORAN_LoadBalancing_1.0.0",
            name="Load Balancing Policy",
            description="Policy for cell load balancing",
            policySchema=PolicyTypeSchema(
                properties={
                    "balancingMode": {"type": "string", "enum": ["proactive", "reactive"]},
                    "loadDifferenceThreshold": {"type": "number"},
                    "handoverHysteresis": {"type": "number"}
                },
                required=["balancingMode"]
            )
        )
        self.policy_types[lb_policy_type.policyTypeId] = lb_policy_type

        logger.info(f"Initialized {len(self.policy_types)} default policy types")

    # -------------------------------------------------------------------------
    # A1-P Policy Management (ETSI TS 103983 Section 7)
    # -------------------------------------------------------------------------

    def create_policy_type(self, policy_type: PolicyType) -> PolicyType:
        """Create a new policy type"""
        if policy_type.policyTypeId in self.policy_types:
            raise HTTPException(status_code=409, detail="Policy type already exists")

        self.policy_types[policy_type.policyTypeId] = policy_type
        logger.info(f"Created policy type: {policy_type.policyTypeId}")
        return policy_type

    def get_policy_type(self, policy_type_id: str) -> PolicyType:
        """Get policy type by ID"""
        if policy_type_id not in self.policy_types:
            raise HTTPException(status_code=404, detail="Policy type not found")
        return self.policy_types[policy_type_id]

    async def create_policy(self, policy: PolicyInstance) -> PolicyInstance:
        """
        Create policy instance and deploy to Near-RT RIC

        Per ETSI TS 103983 Section 7.3:
        - Validate against policy type schema
        - Send to Near-RT RIC via A1 interface
        - Track enforcement status
        """
        with tracer.start_as_current_span("create_policy") as span:
            span.set_attribute("policy.type_id", policy.policyTypeId)

            # Validate policy type exists
            if policy.policyTypeId not in self.policy_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown policy type: {policy.policyTypeId}"
                )

            # Store policy
            self.policies[policy.policyId] = policy

            # Deploy to Near-RT RIC
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.near_rt_ric_url}/a1/policies",
                        json={
                            "policyId": policy.policyId,
                            "policyTypeId": policy.policyTypeId,
                            "policyData": policy.policyData,
                            "scope": policy.scope
                        },
                        timeout=5.0
                    )

                    if response.status_code in (200, 201):
                        policy.status = PolicyStatus.ENFORCED
                        logger.info(f"Policy {policy.policyId} deployed to Near-RT RIC")
                    else:
                        policy.status = PolicyStatus.NOT_ENFORCED
                        policy.nearRtRicStatus = f"Deployment failed: {response.text}"
                        logger.warning(f"Policy deployment failed: {response.text}")

            except httpx.RequestError as e:
                policy.status = PolicyStatus.NOT_ENFORCED
                policy.nearRtRicStatus = f"Connection failed: {str(e)}"
                logger.warning(f"Near-RT RIC unreachable: {e}")

            return policy

    async def update_policy(self, policy_id: str, policy_data: Dict) -> PolicyInstance:
        """Update existing policy"""
        if policy_id not in self.policies:
            raise HTTPException(status_code=404, detail="Policy not found")

        policy = self.policies[policy_id]
        policy.policyData = policy_data
        policy.lastUpdated = datetime.utcnow()

        # Update in Near-RT RIC
        try:
            async with httpx.AsyncClient() as client:
                await client.put(
                    f"{self.near_rt_ric_url}/a1/policies/{policy_id}",
                    json={"policyData": policy_data},
                    timeout=5.0
                )
        except httpx.RequestError as e:
            logger.warning(f"Failed to update policy in Near-RT RIC: {e}")

        return policy

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete policy and remove from Near-RT RIC"""
        if policy_id not in self.policies:
            raise HTTPException(status_code=404, detail="Policy not found")

        # Delete from Near-RT RIC
        try:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"{self.near_rt_ric_url}/a1/policies/{policy_id}",
                    timeout=5.0
                )
        except httpx.RequestError as e:
            logger.warning(f"Failed to delete policy from Near-RT RIC: {e}")

        del self.policies[policy_id]
        logger.info(f"Policy {policy_id} deleted")
        return True

    async def get_policy_status(self, policy_id: str) -> PolicyStatusResponse:
        """Get policy enforcement status from Near-RT RIC"""
        if policy_id not in self.policies:
            raise HTTPException(status_code=404, detail="Policy not found")

        policy = self.policies[policy_id]

        # Query Near-RT RIC for current status
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.near_rt_ric_url}/a1/policies/{policy_id}/status",
                    timeout=5.0
                )
                if response.status_code == 200:
                    status_data = response.json()
                    return PolicyStatusResponse(
                        policyId=policy_id,
                        status=PolicyStatus(status_data.get("status", "NOT_ENFORCED")),
                        reason=status_data.get("reason"),
                        lastEnforced=status_data.get("lastEnforced")
                    )
        except httpx.RequestError:
            pass

        return PolicyStatusResponse(
            policyId=policy_id,
            status=policy.status
        )

    # -------------------------------------------------------------------------
    # A1-EI Enrichment Information (ETSI TS 103983 Section 8)
    # -------------------------------------------------------------------------

    def create_ei_type(self, ei_type: EiType) -> EiType:
        """Create enrichment information type"""
        self.ei_types[ei_type.eiTypeId] = ei_type
        logger.info(f"Created EI type: {ei_type.eiTypeId}")
        return ei_type

    async def create_ei_job(self, ei_job: EiJob) -> EiJob:
        """
        Create EI job for delivering enrichment data to Near-RT RIC

        Per ETSI TS 103983 Section 8:
        - Register EI producer
        - Start data delivery to target URI
        """
        if ei_job.eiTypeId not in self.ei_types:
            raise HTTPException(status_code=400, detail="Unknown EI type")

        ei_job.status = EiJobStatus.ACTIVE
        self.ei_jobs[ei_job.eiJobId] = ei_job

        logger.info(f"Created EI job: {ei_job.eiJobId} for type {ei_job.eiTypeId}")
        return ei_job

    async def deliver_enrichment_info(self, ei_job_id: str, ei_data: Dict) -> Dict:
        """Deliver enrichment information to Near-RT RIC"""
        if ei_job_id not in self.ei_jobs:
            raise HTTPException(status_code=404, detail="EI job not found")

        ei_job = self.ei_jobs[ei_job_id]
        ei_job.status = EiJobStatus.DELIVERING

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.near_rt_ric_url}/a1/enrichment",
                    json={
                        "eiTypeId": ei_job.eiTypeId,
                        "eiData": ei_data
                    },
                    params={"ei_type": ei_job.eiTypeId},
                    timeout=5.0
                )

                if response.status_code == 200:
                    logger.info(f"Delivered EI data for job {ei_job_id}")
                    return {"status": "delivered", "eiJobId": ei_job_id}
                else:
                    return {"status": "failed", "error": response.text}

        except httpx.RequestError as e:
            logger.warning(f"EI delivery failed: {e}")
            return {"status": "failed", "error": str(e)}

    # -------------------------------------------------------------------------
    # rApp Management
    # -------------------------------------------------------------------------

    def register_rapp(self, rapp: RApp) -> RApp:
        """Register an rApp with the Non-RT RIC"""
        self.rapps[rapp.rappId] = rapp
        logger.info(f"rApp registered: {rapp.rappName} ({rapp.rappId})")
        return rapp

    def unregister_rapp(self, rapp_id: str) -> bool:
        """Unregister an rApp"""
        if rapp_id not in self.rapps:
            raise HTTPException(status_code=404, detail="rApp not found")

        del self.rapps[rapp_id]
        logger.info(f"rApp unregistered: {rapp_id}")
        return True

    # -------------------------------------------------------------------------
    # O1 Integration (SMO: ZSM + VNFM)
    # -------------------------------------------------------------------------

    async def get_zsm_intents(self) -> List[Dict]:
        """Query ZSM for active intents"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.zsm_url}/zsm/v1/intents",
                    timeout=5.0
                )
                return response.json()
        except httpx.RequestError as e:
            logger.warning(f"ZSM query failed: {e}")
            return []

    async def create_zsm_intent(self, intent: ZsmIntent) -> Dict:
        """
        Create ZSM intent for closed-loop automation

        Enables Non-RT RIC to trigger network-level actions via ZSM.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.zsm_url}/zsm/v1/intents",
                    json=intent.model_dump(),
                    timeout=5.0
                )
                return response.json()
        except httpx.RequestError as e:
            logger.warning(f"ZSM intent creation failed: {e}")
            raise HTTPException(status_code=503, detail=str(e))

    async def get_vnf_instances(self) -> List[Dict]:
        """Query VNFM for VNF instances"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.vnfm_url}/vnflcm/v1/vnf_instances",
                    timeout=5.0
                )
                return response.json()
        except httpx.RequestError as e:
            logger.warning(f"VNFM query failed: {e}")
            return []

    async def scale_vnf(self, vnf_id: str, scale_type: str, aspect_id: str, steps: int = 1) -> Dict:
        """
        Request VNF scaling via VNFM

        Per ETSI GS NFV-SOL 003:
        - scale_type: SCALE_OUT or SCALE_IN
        - aspect_id: Scaling aspect identifier
        - steps: Number of scaling steps
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.vnfm_url}/vnflcm/v1/vnf_instances/{vnf_id}/scale",
                    json={
                        "type": scale_type,
                        "aspectId": aspect_id,
                        "numberOfSteps": steps
                    },
                    timeout=10.0
                )
                return response.json()
        except httpx.RequestError as e:
            logger.warning(f"VNF scaling failed: {e}")
            raise HTTPException(status_code=503, detail=str(e))

    # -------------------------------------------------------------------------
    # Analytics and Data Management
    # -------------------------------------------------------------------------

    async def collect_ran_analytics(self) -> Dict:
        """
        Collect RAN analytics from Near-RT RIC

        Used for generating insights and policy recommendations.
        """
        analytics = {
            "timestamp": datetime.utcnow().isoformat(),
            "e2Nodes": [],
            "subscriptions": [],
            "measurements": []
        }

        try:
            async with httpx.AsyncClient() as client:
                # Get E2 nodes
                e2_response = await client.get(
                    f"{self.near_rt_ric_url}/ric/e2-nodes",
                    timeout=5.0
                )
                if e2_response.status_code == 200:
                    analytics["e2Nodes"] = e2_response.json()

                # Get subscriptions
                sub_response = await client.get(
                    f"{self.near_rt_ric_url}/e2/subscriptions",
                    timeout=5.0
                )
                if sub_response.status_code == 200:
                    analytics["subscriptions"] = sub_response.json()

                # Get RIC status
                status_response = await client.get(
                    f"{self.near_rt_ric_url}/ric/status",
                    timeout=5.0
                )
                if status_response.status_code == 200:
                    analytics["ricStatus"] = status_response.json()

        except httpx.RequestError as e:
            logger.warning(f"Analytics collection failed: {e}")
            analytics["error"] = str(e)

        self.analytics_data["latest"] = analytics
        return analytics

    # -------------------------------------------------------------------------
    # Status and Metrics
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Get Non-RT RIC status"""
        return {
            "ricId": self.ric_id,
            "policyTypes": len(self.policy_types),
            "activePolicies": len([p for p in self.policies.values()
                                  if p.status == PolicyStatus.ENFORCED]),
            "totalPolicies": len(self.policies),
            "eiJobs": len(self.ei_jobs),
            "registeredRApps": len(self.rapps),
            "status": "RUNNING",
            "nearRtRicUrl": self.near_rt_ric_url
        }


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Non-RT RIC",
    description="Non-Real-Time RAN Intelligent Controller - O-RAN Compliant",
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

# Initialize Non-RT RIC
ric = NonRtRic()


# =============================================================================
# A1-P Policy Management Endpoints
# =============================================================================

@app.get("/a1-p/policytypes", response_model=List[str])
async def list_policy_types():
    """List all policy type IDs"""
    return list(ric.policy_types.keys())


@app.get("/a1-p/policytypes/{policy_type_id}", response_model=PolicyType)
async def get_policy_type(policy_type_id: str):
    """Get policy type details"""
    return ric.get_policy_type(policy_type_id)


@app.put("/a1-p/policytypes/{policy_type_id}", response_model=PolicyType, status_code=201)
async def create_policy_type(policy_type_id: str, policy_type: PolicyType):
    """Create or update policy type"""
    policy_type.policyTypeId = policy_type_id
    return ric.create_policy_type(policy_type)


@app.get("/a1-p/policytypes/{policy_type_id}/policies", response_model=List[str])
async def list_policies_for_type(policy_type_id: str):
    """List policy IDs for a policy type"""
    return [p.policyId for p in ric.policies.values()
            if p.policyTypeId == policy_type_id]


@app.put("/a1-p/policytypes/{policy_type_id}/policies/{policy_id}",
         response_model=PolicyInstance, status_code=201)
async def create_policy(policy_type_id: str, policy_id: str, policy_data: Dict):
    """
    Create policy instance
    Per ETSI TS 103983 Section 7.3
    """
    policy = PolicyInstance(
        policyId=policy_id,
        policyTypeId=policy_type_id,
        policyData=policy_data
    )
    return await ric.create_policy(policy)


@app.get("/a1-p/policytypes/{policy_type_id}/policies/{policy_id}",
         response_model=PolicyInstance)
async def get_policy(policy_type_id: str, policy_id: str):
    """Get policy instance"""
    if policy_id not in ric.policies:
        raise HTTPException(status_code=404, detail="Policy not found")
    policy = ric.policies[policy_id]
    if policy.policyTypeId != policy_type_id:
        raise HTTPException(status_code=404, detail="Policy not found for this type")
    return policy


@app.delete("/a1-p/policytypes/{policy_type_id}/policies/{policy_id}")
async def delete_policy(policy_type_id: str, policy_id: str):
    """Delete policy instance"""
    await ric.delete_policy(policy_id)
    return {"status": "deleted", "policyId": policy_id}


@app.get("/a1-p/policytypes/{policy_type_id}/policies/{policy_id}/status",
         response_model=PolicyStatusResponse)
async def get_policy_status(policy_type_id: str, policy_id: str):
    """Get policy enforcement status"""
    return await ric.get_policy_status(policy_id)


# =============================================================================
# A1-EI Enrichment Information Endpoints
# =============================================================================

@app.get("/a1-ei/eitypes", response_model=List[str])
async def list_ei_types():
    """List enrichment information types"""
    return list(ric.ei_types.keys())


@app.put("/a1-ei/eitypes/{ei_type_id}", response_model=EiType, status_code=201)
async def create_ei_type(ei_type_id: str, ei_type: EiType):
    """Create enrichment information type"""
    ei_type.eiTypeId = ei_type_id
    return ric.create_ei_type(ei_type)


@app.get("/a1-ei/eijobs", response_model=List[EiJob])
async def list_ei_jobs():
    """List enrichment information jobs"""
    return list(ric.ei_jobs.values())


@app.put("/a1-ei/eijobs/{ei_job_id}", response_model=EiJob, status_code=201)
async def create_ei_job(ei_job_id: str, ei_job: EiJob):
    """Create enrichment information job"""
    ei_job.eiJobId = ei_job_id
    return await ric.create_ei_job(ei_job)


@app.delete("/a1-ei/eijobs/{ei_job_id}")
async def delete_ei_job(ei_job_id: str):
    """Delete EI job"""
    if ei_job_id not in ric.ei_jobs:
        raise HTTPException(status_code=404, detail="EI job not found")
    del ric.ei_jobs[ei_job_id]
    return {"status": "deleted", "eiJobId": ei_job_id}


@app.post("/a1-ei/eijobs/{ei_job_id}/deliver")
async def deliver_ei(ei_job_id: str, ei_data: Dict):
    """Deliver enrichment information"""
    return await ric.deliver_enrichment_info(ei_job_id, ei_data)


# =============================================================================
# rApp Management Endpoints
# =============================================================================

@app.get("/ric/rapps", response_model=List[RApp])
async def list_rapps():
    """List registered rApps"""
    return list(ric.rapps.values())


@app.post("/ric/rapps", response_model=RApp, status_code=201)
async def register_rapp(rapp: RApp):
    """Register an rApp"""
    return ric.register_rapp(rapp)


@app.delete("/ric/rapps/{rapp_id}")
async def unregister_rapp(rapp_id: str):
    """Unregister an rApp"""
    ric.unregister_rapp(rapp_id)
    return {"status": "unregistered", "rappId": rapp_id}


# =============================================================================
# O1 Integration Endpoints (SMO)
# =============================================================================

@app.get("/o1/intents")
async def get_intents():
    """Get ZSM intents"""
    return await ric.get_zsm_intents()


@app.post("/o1/intents")
async def create_intent(intent: ZsmIntent):
    """Create ZSM intent"""
    return await ric.create_zsm_intent(intent)


@app.get("/o1/vnf-instances")
async def get_vnf_instances():
    """Get VNF instances from VNFM"""
    return await ric.get_vnf_instances()


@app.post("/o1/vnf-instances/{vnf_id}/scale")
async def scale_vnf(vnf_id: str, scale_type: str, aspect_id: str, steps: int = 1):
    """Request VNF scaling"""
    return await ric.scale_vnf(vnf_id, scale_type, aspect_id, steps)


# =============================================================================
# Analytics Endpoints
# =============================================================================

@app.get("/ric/analytics")
async def get_analytics():
    """Get RAN analytics"""
    return await ric.collect_ran_analytics()


@app.get("/ric/analytics/latest")
async def get_latest_analytics():
    """Get latest cached analytics"""
    return ric.analytics_data.get("latest", {})


# =============================================================================
# Health and Metrics
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "non-rt-ric",
        "version": "1.0.0",
        "spec": "ETSI TS 103983"
    }


@app.get("/ric/status")
async def get_status():
    """Get Non-RT RIC status"""
    return ric.get_status()


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics"""
    status = ric.get_status()
    metrics = f"""# HELP non_rt_ric_policy_types Number of policy types
# TYPE non_rt_ric_policy_types gauge
non_rt_ric_policy_types {status['policyTypes']}

# HELP non_rt_ric_policies_active Number of active policies
# TYPE non_rt_ric_policies_active gauge
non_rt_ric_policies_active {status['activePolicies']}

# HELP non_rt_ric_policies_total Total number of policies
# TYPE non_rt_ric_policies_total gauge
non_rt_ric_policies_total {status['totalPolicies']}

# HELP non_rt_ric_ei_jobs Number of EI jobs
# TYPE non_rt_ric_ei_jobs gauge
non_rt_ric_ei_jobs {status['eiJobs']}

# HELP non_rt_ric_rapps_registered Number of registered rApps
# TYPE non_rt_ric_rapps_registered gauge
non_rt_ric_rapps_registered {status['registeredRApps']}
"""
    return metrics


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8096)
