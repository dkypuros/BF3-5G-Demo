# File location: 5G_Emulator_API/etsi/o2/o2_dms.py
# O-RAN O2 Deployment Management Service (DMS)
# Spec: O-RAN.WG6.O2DMS-INTERFACE

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
import uvicorn
import logging
import argparse
from datetime import datetime
import requests

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

from .fake_cluster import fake_cluster
from .models import (
    NfDeploymentDescriptor,
    Deployment,
    DeploymentRequest,
    DeploymentOperation,
    DeploymentState,
    ScaleRequest,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry setup
resource = Resource.create({"service.name": "o2-dms"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# NRF URL for service registration
NRF_URL = "http://127.0.0.1:8000"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("O2 DMS starting up...")

    # Register with NRF
    try:
        registration = {
            "nf_type": "O2_DMS",
            "ip": "127.0.0.1",
            "port": 8099
        }
        response = requests.post(f"{NRF_URL}/register", json=registration, timeout=5)
        if response.status_code == 200:
            logger.info("O2 DMS registered with NRF")
        else:
            logger.warning(f"NRF registration returned {response.status_code}")
    except requests.RequestException as e:
        logger.warning(f"Could not register with NRF: {e}")

    logger.info("O2 DMS ready - Deployment Management Service")
    yield

    # Shutdown
    logger.info("O2 DMS shutting down...")


# Create FastAPI app
app = FastAPI(
    title="O2 DMS - Deployment Management Service",
    description="O-RAN O2 Interface for NF Deployment Management",
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
# Health & Metrics
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    deployments = fake_cluster.get_deployments()
    running = len([d for d in deployments if d.state == DeploymentState.RUNNING])
    return {
        "status": "healthy",
        "service": "o2-dms",
        "version": "1.0.0",
        "spec": "O-RAN.WG6.O2DMS-INTERFACE",
        "deployments": {
            "total": len(deployments),
            "running": running
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    deployments = fake_cluster.get_deployments()
    running = len([d for d in deployments if d.state == DeploymentState.RUNNING])
    pending = len([d for d in deployments if d.state == DeploymentState.PENDING])
    failed = len([d for d in deployments if d.state == DeploymentState.FAILED])

    metrics_text = f"""# HELP o2dms_deployments_total Total number of deployments
# TYPE o2dms_deployments_total gauge
o2dms_deployments_total {len(deployments)}

# HELP o2dms_deployments_running Running deployments
# TYPE o2dms_deployments_running gauge
o2dms_deployments_running {running}

# HELP o2dms_deployments_pending Pending deployments
# TYPE o2dms_deployments_pending gauge
o2dms_deployments_pending {pending}

# HELP o2dms_deployments_failed Failed deployments
# TYPE o2dms_deployments_failed gauge
o2dms_deployments_failed {failed}

# HELP o2dms_descriptors_total Total NF descriptors
# TYPE o2dms_descriptors_total gauge
o2dms_descriptors_total {len(fake_cluster.get_nf_descriptors())}
"""
    return metrics_text


# =============================================================================
# NF Deployment Descriptors
# =============================================================================

@app.get("/o2dms/v1/nfDeploymentDescriptors", response_model=List[NfDeploymentDescriptor])
async def get_nf_descriptors():
    """List all NF deployment descriptors"""
    with tracer.start_as_current_span("get_nf_descriptors"):
        return fake_cluster.get_nf_descriptors()


@app.get("/o2dms/v1/nfDeploymentDescriptors/{descriptor_id}", response_model=NfDeploymentDescriptor)
async def get_nf_descriptor(descriptor_id: str = Path(..., description="Descriptor ID")):
    """Get specific NF descriptor"""
    with tracer.start_as_current_span("get_nf_descriptor") as span:
        span.set_attribute("descriptor_id", descriptor_id)
        descriptor = fake_cluster.get_nf_descriptor(descriptor_id)
        if not descriptor:
            raise HTTPException(status_code=404, detail=f"Descriptor {descriptor_id} not found")
        return descriptor


@app.post("/o2dms/v1/nfDeploymentDescriptors", response_model=NfDeploymentDescriptor, status_code=201)
async def create_nf_descriptor(descriptor: NfDeploymentDescriptor):
    """Create new NF deployment descriptor"""
    with tracer.start_as_current_span("create_nf_descriptor") as span:
        span.set_attribute("descriptor_id", descriptor.descriptorId)

        # Check if already exists
        existing = fake_cluster.get_nf_descriptor(descriptor.descriptorId)
        if existing:
            raise HTTPException(status_code=409, detail=f"Descriptor {descriptor.descriptorId} already exists")

        return fake_cluster.create_nf_descriptor(descriptor)


# =============================================================================
# Deployments
# =============================================================================

@app.get("/o2dms/v1/deployments", response_model=List[Deployment])
async def get_deployments(
    state: Optional[str] = Query(None, description="Filter by state"),
    descriptor_id: Optional[str] = Query(None, description="Filter by descriptor ID")
):
    """List all deployments"""
    with tracer.start_as_current_span("get_deployments"):
        deployments = fake_cluster.get_deployments()

        # Apply filters
        if state:
            deployments = [d for d in deployments if d.state.value == state]
        if descriptor_id:
            deployments = [d for d in deployments if d.descriptorId == descriptor_id]

        return deployments


@app.get("/o2dms/v1/deployments/{deployment_id}", response_model=Deployment)
async def get_deployment(deployment_id: str = Path(..., description="Deployment ID")):
    """Get specific deployment"""
    with tracer.start_as_current_span("get_deployment") as span:
        span.set_attribute("deployment_id", deployment_id)
        deployment = fake_cluster.get_deployment(deployment_id)
        if not deployment:
            raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")
        return deployment


@app.post("/o2dms/v1/deployments", response_model=Deployment, status_code=201)
async def create_deployment(request: DeploymentRequest):
    """Deploy an NF to the O-Cloud"""
    with tracer.start_as_current_span("create_deployment") as span:
        span.set_attribute("descriptor_id", request.descriptorId)
        span.set_attribute("name", request.name)
        span.set_attribute("replicas", request.replicas)

        # Validate descriptor exists
        descriptor = fake_cluster.get_nf_descriptor(request.descriptorId)
        if not descriptor:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown descriptor: {request.descriptorId}"
            )

        try:
            deployment = fake_cluster.deploy_workload(
                descriptor_id=request.descriptorId,
                name=request.name,
                replicas=request.replicas,
                pool_id=request.resourcePoolId or "compute-pool"
            )
            logger.info(f"Created deployment: {deployment.deploymentId} ({request.name})")
            return deployment
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.patch("/o2dms/v1/deployments/{deployment_id}", response_model=Deployment)
async def scale_deployment(
    deployment_id: str = Path(..., description="Deployment ID"),
    request: ScaleRequest = None
):
    """Scale a deployment"""
    with tracer.start_as_current_span("scale_deployment") as span:
        span.set_attribute("deployment_id", deployment_id)
        span.set_attribute("replicas", request.replicas)

        deployment = fake_cluster.get_deployment(deployment_id)
        if not deployment:
            raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")

        if deployment.state == DeploymentState.TERMINATED:
            raise HTTPException(status_code=400, detail="Cannot scale terminated deployment")

        try:
            deployment = fake_cluster.scale_deployment(deployment_id, request.replicas)
            logger.info(f"Scaled deployment {deployment_id} to {request.replicas} replicas")
            return deployment
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.delete("/o2dms/v1/deployments/{deployment_id}", status_code=204)
async def terminate_deployment(deployment_id: str = Path(..., description="Deployment ID")):
    """Terminate a deployment"""
    with tracer.start_as_current_span("terminate_deployment") as span:
        span.set_attribute("deployment_id", deployment_id)

        deployment = fake_cluster.get_deployment(deployment_id)
        if not deployment:
            raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")

        if deployment.state == DeploymentState.TERMINATED:
            raise HTTPException(status_code=400, detail="Deployment already terminated")

        if fake_cluster.terminate_deployment(deployment_id):
            logger.info(f"Terminated deployment {deployment_id}")
            return None
        else:
            raise HTTPException(status_code=500, detail="Failed to terminate deployment")


# =============================================================================
# Deployment Operations
# =============================================================================

@app.get("/o2dms/v1/deployments/{deployment_id}/operations", response_model=List[DeploymentOperation])
async def get_deployment_operations(deployment_id: str = Path(..., description="Deployment ID")):
    """Get operation history for a deployment"""
    with tracer.start_as_current_span("get_deployment_operations") as span:
        span.set_attribute("deployment_id", deployment_id)

        deployment = fake_cluster.get_deployment(deployment_id)
        if not deployment:
            raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")

        return fake_cluster.get_deployment_operations(deployment_id)


@app.get("/o2dms/v1/deployments/{deployment_id}/operations/{operation_id}", response_model=DeploymentOperation)
async def get_deployment_operation(
    deployment_id: str = Path(..., description="Deployment ID"),
    operation_id: str = Path(..., description="Operation ID")
):
    """Get specific operation"""
    with tracer.start_as_current_span("get_deployment_operation") as span:
        span.set_attribute("deployment_id", deployment_id)
        span.set_attribute("operation_id", operation_id)

        deployment = fake_cluster.get_deployment(deployment_id)
        if not deployment:
            raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")

        operations = fake_cluster.get_deployment_operations(deployment_id)
        for op in operations:
            if op.operationId == operation_id:
                return op

        raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found")


# =============================================================================
# Convenience Endpoints
# =============================================================================

@app.get("/o2dms/v1/stats")
async def get_dms_stats():
    """Get DMS statistics"""
    stats = fake_cluster.get_cluster_stats()
    deployments = fake_cluster.get_deployments()

    by_state = {}
    for d in deployments:
        state = d.state.value
        by_state[state] = by_state.get(state, 0) + 1

    by_type = {}
    for d in deployments:
        desc = fake_cluster.get_nf_descriptor(d.descriptorId)
        if desc:
            nf_type = desc.nfType
            by_type[nf_type] = by_type.get(nf_type, 0) + 1

    return {
        "cluster": stats,
        "deployments": {
            "total": len(deployments),
            "by_state": by_state,
            "by_nf_type": by_type
        },
        "descriptors": {
            "total": len(fake_cluster.get_nf_descriptors())
        }
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="O2 DMS - Deployment Management Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8099, help="Port to bind to")
    args = parser.parse_args()

    logger.info(f"Starting O2 DMS on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
