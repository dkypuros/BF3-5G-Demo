# File location: 5G_Emulator_API/etsi/o2/smo.py
# O-RAN Service Management & Orchestration (SMO)
# Coordinates O2 IMS, O2 DMS, RIC, ZSM, and VNFM

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import argparse
from datetime import datetime
import uuid
import httpx
import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

from .models import (
    Intent,
    DeployNfRequest,
    ComponentStatus,
    DeploymentRequest,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry setup
resource = Resource.create({"service.name": "smo"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Service URLs
O2_IMS_URL = "http://127.0.0.1:8098"
O2_DMS_URL = "http://127.0.0.1:8099"
NON_RT_RIC_URL = "http://127.0.0.1:8096"
NEAR_RT_RIC_URL = "http://127.0.0.1:8095"
ZSM_URL = "http://127.0.0.1:8094"
VNFM_URL = "http://127.0.0.1:8093"
NRF_URL = "http://127.0.0.1:8000"

# In-memory state
intents: Dict[str, Intent] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("SMO starting up...")

    # Register with NRF
    try:
        async with httpx.AsyncClient() as client:
            registration = {
                "nf_type": "SMO",
                "ip": "127.0.0.1",
                "port": 8097
            }
            response = await client.post(f"{NRF_URL}/register", json=registration, timeout=5)
            if response.status_code == 200:
                logger.info("SMO registered with NRF")
            else:
                logger.warning(f"NRF registration returned {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not register with NRF: {e}")

    logger.info("SMO ready - Service Management & Orchestration")
    yield

    # Shutdown
    logger.info("SMO shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SMO - Service Management & Orchestration",
    description="O-RAN SMO for coordinating O2 IMS/DMS, RIC, ZSM, and VNFM",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

async def check_component_health(name: str, url: str) -> ComponentStatus:
    """Check health of a component"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/health", timeout=3)
            if response.status_code == 200:
                return ComponentStatus(
                    componentType="service",
                    name=name,
                    url=url,
                    status="HEALTHY",
                    lastChecked=datetime.now()
                )
    except Exception:
        pass

    return ComponentStatus(
        componentType="service",
        name=name,
        url=url,
        status="UNHEALTHY",
        lastChecked=datetime.now()
    )


# =============================================================================
# Health & Metrics
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "smo",
        "version": "1.0.0",
        "intents": {
            "total": len(intents),
            "pending": len([i for i in intents.values() if i.state == "PENDING"]),
            "completed": len([i for i in intents.values() if i.state == "COMPLETED"])
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    metrics_text = f"""# HELP smo_intents_total Total number of intents
# TYPE smo_intents_total gauge
smo_intents_total {len(intents)}

# HELP smo_intents_pending Pending intents
# TYPE smo_intents_pending gauge
smo_intents_pending {len([i for i in intents.values() if i.state == "PENDING"])}

# HELP smo_intents_completed Completed intents
# TYPE smo_intents_completed gauge
smo_intents_completed {len([i for i in intents.values() if i.state == "COMPLETED"])}

# HELP smo_intents_failed Failed intents
# TYPE smo_intents_failed gauge
smo_intents_failed {len([i for i in intents.values() if i.state == "FAILED"])}
"""
    return metrics_text


# =============================================================================
# Component Status
# =============================================================================

@app.get("/smo/v1/components", response_model=List[ComponentStatus])
async def get_components():
    """Get status of all managed components"""
    with tracer.start_as_current_span("get_components"):
        components = [
            ("O2-IMS", O2_IMS_URL),
            ("O2-DMS", O2_DMS_URL),
            ("Non-RT-RIC", NON_RT_RIC_URL),
            ("Near-RT-RIC", NEAR_RT_RIC_URL),
            ("ZSM", ZSM_URL),
            ("VNFM", VNFM_URL),
        ]

        # Check all components in parallel
        tasks = [check_component_health(name, url) for name, url in components]
        results = await asyncio.gather(*tasks)

        return results


@app.get("/smo/v1/components/{component_type}", response_model=List[ComponentStatus])
async def get_components_by_type(component_type: str = Path(..., description="Component type to filter")):
    """Get components filtered by type"""
    all_components = await get_components()

    type_mapping = {
        "o2": ["O2-IMS", "O2-DMS"],
        "ric": ["Non-RT-RIC", "Near-RT-RIC"],
        "management": ["ZSM", "VNFM"]
    }

    if component_type.lower() in type_mapping:
        names = type_mapping[component_type.lower()]
        return [c for c in all_components if c.name in names]

    return [c for c in all_components if component_type.lower() in c.name.lower()]


# =============================================================================
# Intents
# =============================================================================

@app.get("/smo/v1/intents", response_model=List[Intent])
async def get_intents(
    state: Optional[str] = Query(None, description="Filter by state"),
    objective: Optional[str] = Query(None, description="Filter by objective")
):
    """List all intents"""
    with tracer.start_as_current_span("get_intents"):
        result = list(intents.values())

        if state:
            result = [i for i in result if i.state == state]
        if objective:
            result = [i for i in result if i.objective == objective]

        return result


@app.get("/smo/v1/intents/{intent_id}", response_model=Intent)
async def get_intent(intent_id: str = Path(..., description="Intent ID")):
    """Get specific intent"""
    with tracer.start_as_current_span("get_intent") as span:
        span.set_attribute("intent_id", intent_id)
        intent = intents.get(intent_id)
        if not intent:
            raise HTTPException(status_code=404, detail=f"Intent {intent_id} not found")
        return intent


@app.post("/smo/v1/intents", response_model=Intent, status_code=201)
async def create_intent(intent: Intent):
    """Create a new intent"""
    with tracer.start_as_current_span("create_intent") as span:
        intent_id = str(uuid.uuid4())
        intent.intentId = intent_id
        intent.state = "PENDING"
        intent.createdAt = datetime.now()

        span.set_attribute("intent_id", intent_id)
        span.set_attribute("objective", intent.objective)

        intents[intent_id] = intent
        logger.info(f"Created intent: {intent_id} ({intent.objective})")

        return intent


@app.delete("/smo/v1/intents/{intent_id}", status_code=204)
async def delete_intent(intent_id: str = Path(..., description="Intent ID")):
    """Delete an intent"""
    with tracer.start_as_current_span("delete_intent") as span:
        span.set_attribute("intent_id", intent_id)
        if intent_id not in intents:
            raise HTTPException(status_code=404, detail=f"Intent {intent_id} not found")
        del intents[intent_id]
        return None


# =============================================================================
# High-Level Operations
# =============================================================================

@app.post("/smo/v1/deployNf")
async def deploy_nf(request: DeployNfRequest):
    """
    High-level NF deployment orchestration

    This endpoint:
    1. Queries O2 IMS for available resources
    2. Deploys NF via O2 DMS
    3. Optionally creates QoS policy via Non-RT RIC
    """
    with tracer.start_as_current_span("deploy_nf") as span:
        span.set_attribute("nf_type", request.nfType)
        span.set_attribute("name", request.name)

        result = {
            "steps": [],
            "deployment": None,
            "policy": None,
            "success": False
        }

        async with httpx.AsyncClient() as client:
            # Step 1: Check O2 IMS for resources
            try:
                logger.info(f"Step 1: Checking O2 IMS for resources...")
                response = await client.get(
                    f"{O2_IMS_URL}/o2ims-infrastructureInventory/v1/resourcePools/compute-pool/resources",
                    timeout=10
                )
                if response.status_code == 200:
                    resources = response.json()
                    available_cpu = sum(r.get("extensions", {}).get("available_cpu", 0) for r in resources)
                    available_ram = sum(r.get("extensions", {}).get("available_ram_gb", 0) for r in resources)
                    result["steps"].append({
                        "step": "check_resources",
                        "status": "success",
                        "details": {
                            "available_cpu": available_cpu,
                            "available_ram_gb": available_ram
                        }
                    })
                else:
                    result["steps"].append({
                        "step": "check_resources",
                        "status": "failed",
                        "error": f"O2 IMS returned {response.status_code}"
                    })
                    return result
            except Exception as e:
                result["steps"].append({
                    "step": "check_resources",
                    "status": "failed",
                    "error": str(e)
                })
                return result

            # Step 2: Deploy via O2 DMS
            try:
                logger.info(f"Step 2: Deploying {request.nfType} via O2 DMS...")
                descriptor_id = f"{request.nfType.lower()}-descriptor"
                deploy_request = {
                    "descriptorId": descriptor_id,
                    "name": request.name,
                    "replicas": request.replicas
                }
                response = await client.post(
                    f"{O2_DMS_URL}/o2dms/v1/deployments",
                    json=deploy_request,
                    timeout=30
                )
                if response.status_code == 201:
                    deployment = response.json()
                    result["deployment"] = deployment
                    result["steps"].append({
                        "step": "deploy_nf",
                        "status": "success",
                        "details": {
                            "deployment_id": deployment.get("deploymentId"),
                            "state": deployment.get("state")
                        }
                    })
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    result["steps"].append({
                        "step": "deploy_nf",
                        "status": "failed",
                        "error": error_detail
                    })
                    return result
            except Exception as e:
                result["steps"].append({
                    "step": "deploy_nf",
                    "status": "failed",
                    "error": str(e)
                })
                return result

            # Step 3: Create QoS policy via Non-RT RIC (if requested)
            if request.qosPolicy:
                try:
                    logger.info(f"Step 3: Creating QoS policy via Non-RT RIC...")
                    policy_id = f"policy-{request.name}"
                    policy_data = request.qosPolicy
                    response = await client.put(
                        f"{NON_RT_RIC_URL}/a1-p/policytypes/ORAN_QoSTarget_1.0.0/policies/{policy_id}",
                        json=policy_data,
                        timeout=10
                    )
                    if response.status_code in [200, 201]:
                        policy = response.json()
                        result["policy"] = policy
                        result["steps"].append({
                            "step": "create_policy",
                            "status": "success",
                            "details": {
                                "policy_id": policy.get("policyId"),
                                "status": policy.get("status")
                            }
                        })
                    else:
                        result["steps"].append({
                            "step": "create_policy",
                            "status": "failed",
                            "error": f"Non-RT RIC returned {response.status_code}"
                        })
                except Exception as e:
                    result["steps"].append({
                        "step": "create_policy",
                        "status": "skipped",
                        "error": f"Non-RT RIC not available: {e}"
                    })
            else:
                result["steps"].append({
                    "step": "create_policy",
                    "status": "skipped",
                    "reason": "No QoS policy requested"
                })

        result["success"] = True
        logger.info(f"NF deployment completed: {request.name}")
        return result


@app.post("/smo/v1/scaleNf")
async def scale_nf(
    deployment_id: str = Query(..., description="Deployment ID to scale"),
    replicas: int = Query(..., ge=1, description="Target replicas")
):
    """Scale an existing NF deployment via O2 DMS"""
    with tracer.start_as_current_span("scale_nf") as span:
        span.set_attribute("deployment_id", deployment_id)
        span.set_attribute("replicas", replicas)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.patch(
                    f"{O2_DMS_URL}/o2dms/v1/deployments/{deployment_id}",
                    json={"replicas": replicas},
                    timeout=30
                )
                if response.status_code == 200:
                    deployment = response.json()
                    return {
                        "success": True,
                        "deployment": deployment
                    }
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=response.json().get("detail", "Scale failed")
                    )
            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"O2 DMS not available: {e}")


@app.post("/smo/v1/createPolicy")
async def create_policy(
    policy_type: str = Query("ORAN_QoSTarget_1.0.0", description="Policy type"),
    policy_id: str = Query(..., description="Policy ID"),
    policy_data: Dict[str, Any] = None
):
    """Create an A1 policy via Non-RT RIC"""
    with tracer.start_as_current_span("create_policy") as span:
        span.set_attribute("policy_type", policy_type)
        span.set_attribute("policy_id", policy_id)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{NON_RT_RIC_URL}/a1-p/policytypes/{policy_type}/policies/{policy_id}",
                    json=policy_data or {},
                    timeout=10
                )
                if response.status_code in [200, 201]:
                    return {
                        "success": True,
                        "policy": response.json()
                    }
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=response.json().get("detail", "Policy creation failed")
                    )
            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"Non-RT RIC not available: {e}")


@app.get("/smo/v1/analytics")
async def get_analytics():
    """Get aggregated analytics from O2 IMS and RIC"""
    with tracer.start_as_current_span("get_analytics"):
        analytics = {
            "timestamp": datetime.now().isoformat(),
            "o2_ims": None,
            "o2_dms": None,
            "ric": None
        }

        async with httpx.AsyncClient() as client:
            # O2 IMS stats
            try:
                response = await client.get(f"{O2_IMS_URL}/health", timeout=5)
                if response.status_code == 200:
                    analytics["o2_ims"] = response.json()
            except Exception:
                analytics["o2_ims"] = {"status": "unavailable"}

            # O2 DMS stats
            try:
                response = await client.get(f"{O2_DMS_URL}/o2dms/v1/stats", timeout=5)
                if response.status_code == 200:
                    analytics["o2_dms"] = response.json()
            except Exception:
                analytics["o2_dms"] = {"status": "unavailable"}

            # RIC analytics
            try:
                response = await client.get(f"{NON_RT_RIC_URL}/ric/analytics", timeout=5)
                if response.status_code == 200:
                    analytics["ric"] = response.json()
            except Exception:
                analytics["ric"] = {"status": "unavailable"}

        return analytics


# =============================================================================
# Workflow Endpoints
# =============================================================================

@app.post("/smo/v1/workflow/deploy-and-configure")
async def deploy_and_configure_workflow(
    nf_type: str = Query(..., description="NF type (CU, DU, AMF, etc.)"),
    name: str = Query(..., description="Deployment name"),
    replicas: int = Query(1, ge=1, description="Number of replicas"),
    qos_objective: Optional[str] = Query(None, description="QoS objective"),
    throughput_target: Optional[int] = Query(None, description="Throughput target Mbps"),
    latency_target: Optional[int] = Query(None, description="Latency target ms")
):
    """
    Complete workflow: Deploy NF + Configure QoS Policy

    This is a convenience endpoint that combines deployNf and createPolicy
    into a single atomic workflow.
    """
    with tracer.start_as_current_span("deploy_and_configure_workflow") as span:
        span.set_attribute("nf_type", nf_type)
        span.set_attribute("name", name)

        # Build QoS policy if targets provided
        qos_policy = None
        if qos_objective or throughput_target or latency_target:
            qos_policy = {
                "qosObjective": qos_objective or "maximize_throughput",
                "targetKpi": {}
            }
            if throughput_target:
                qos_policy["targetKpi"]["throughput"] = throughput_target
            if latency_target:
                qos_policy["targetKpi"]["latency"] = latency_target

        # Call deployNf
        deploy_request = DeployNfRequest(
            nfType=nf_type,
            name=name,
            replicas=replicas,
            qosPolicy=qos_policy
        )

        result = await deploy_nf(deploy_request)
        return result


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMO - Service Management & Orchestration")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8097, help="Port to bind to")
    args = parser.parse_args()

    logger.info(f"Starting SMO on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
