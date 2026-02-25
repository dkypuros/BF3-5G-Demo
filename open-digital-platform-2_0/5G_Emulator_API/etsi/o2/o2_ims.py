# File location: 5G_Emulator_API/etsi/o2/o2_ims.py
# O-RAN O2 Infrastructure Management Service (IMS)
# Spec: O-RAN.WG6.O2IMS-INTERFACE

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
    OCloudInfo,
    DeploymentManager,
    ResourceType,
    ResourcePool,
    Resource as O2Resource,
    AlarmEventRecord,
    AlarmSubscriptionInfo,
    InventorySubscription,
    PerceivedSeverity,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry setup
resource = Resource.create({"service.name": "o2-ims"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# NRF URL for service registration
NRF_URL = "http://127.0.0.1:8000"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("O2 IMS starting up...")

    # Register with NRF
    try:
        registration = {
            "nf_type": "O2_IMS",
            "ip": "127.0.0.1",
            "port": 8098
        }
        response = requests.post(f"{NRF_URL}/register", json=registration, timeout=5)
        if response.status_code == 200:
            logger.info("O2 IMS registered with NRF")
        else:
            logger.warning(f"NRF registration returned {response.status_code}")
    except requests.RequestException as e:
        logger.warning(f"Could not register with NRF: {e}")

    logger.info("O2 IMS ready - Infrastructure Management Service")
    yield

    # Shutdown
    logger.info("O2 IMS shutting down...")


# Create FastAPI app
app = FastAPI(
    title="O2 IMS - Infrastructure Management Service",
    description="O-RAN O2 Interface for O-Cloud Infrastructure Management",
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
    stats = fake_cluster.get_cluster_stats()
    return {
        "status": "healthy",
        "service": "o2-ims",
        "version": "1.0.0",
        "spec": "O-RAN.WG6.O2IMS-INTERFACE",
        "cluster_stats": stats
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    stats = fake_cluster.get_cluster_stats()
    metrics_text = f"""# HELP o2ims_nodes_total Total number of nodes
# TYPE o2ims_nodes_total gauge
o2ims_nodes_total {stats['nodes']}

# HELP o2ims_cpu_total Total CPU cores
# TYPE o2ims_cpu_total gauge
o2ims_cpu_total {stats['total_cpu']}

# HELP o2ims_cpu_allocated Allocated CPU cores
# TYPE o2ims_cpu_allocated gauge
o2ims_cpu_allocated {stats['allocated_cpu']}

# HELP o2ims_ram_total_gb Total RAM in GB
# TYPE o2ims_ram_total_gb gauge
o2ims_ram_total_gb {stats['total_ram_gb']}

# HELP o2ims_ram_allocated_gb Allocated RAM in GB
# TYPE o2ims_ram_allocated_gb gauge
o2ims_ram_allocated_gb {stats['allocated_ram_gb']}

# HELP o2ims_deployments_active Active deployments
# TYPE o2ims_deployments_active gauge
o2ims_deployments_active {stats['active_deployments']}

# HELP o2ims_alarms_active Active alarms
# TYPE o2ims_alarms_active gauge
o2ims_alarms_active {stats['active_alarms']}
"""
    return metrics_text


# =============================================================================
# Infrastructure Inventory - O-Cloud Info
# =============================================================================

@app.get("/o2ims-infrastructureInventory/v1", response_model=OCloudInfo)
async def get_ocloud_info():
    """Get O-Cloud information"""
    with tracer.start_as_current_span("get_ocloud_info"):
        return fake_cluster.get_ocloud_info()


@app.get("/o2ims-infrastructureInventory/api_versions")
async def get_api_versions():
    """Get supported API versions"""
    return {
        "apiVersions": [
            {"version": "v1", "isDeprecated": False}
        ]
    }


# =============================================================================
# Infrastructure Inventory - Deployment Managers
# =============================================================================

@app.get("/o2ims-infrastructureInventory/v1/deploymentManagers", response_model=List[DeploymentManager])
async def get_deployment_managers():
    """List all deployment managers"""
    with tracer.start_as_current_span("get_deployment_managers"):
        return fake_cluster.get_deployment_managers()


@app.get("/o2ims-infrastructureInventory/v1/deploymentManagers/{dm_id}", response_model=DeploymentManager)
async def get_deployment_manager(dm_id: str = Path(..., description="Deployment Manager ID")):
    """Get specific deployment manager"""
    with tracer.start_as_current_span("get_deployment_manager") as span:
        span.set_attribute("dm_id", dm_id)
        dm = fake_cluster.get_deployment_manager(dm_id)
        if not dm:
            raise HTTPException(status_code=404, detail=f"Deployment manager {dm_id} not found")
        return dm


# =============================================================================
# Infrastructure Inventory - Resource Types
# =============================================================================

@app.get("/o2ims-infrastructureInventory/v1/resourceTypes", response_model=List[ResourceType])
async def get_resource_types():
    """List all resource types"""
    with tracer.start_as_current_span("get_resource_types"):
        return fake_cluster.get_resource_types()


@app.get("/o2ims-infrastructureInventory/v1/resourceTypes/{type_id}", response_model=ResourceType)
async def get_resource_type(type_id: str = Path(..., description="Resource Type ID")):
    """Get specific resource type"""
    with tracer.start_as_current_span("get_resource_type") as span:
        span.set_attribute("type_id", type_id)
        rt = fake_cluster.get_resource_type(type_id)
        if not rt:
            raise HTTPException(status_code=404, detail=f"Resource type {type_id} not found")
        return rt


# =============================================================================
# Infrastructure Inventory - Resource Pools
# =============================================================================

@app.get("/o2ims-infrastructureInventory/v1/resourcePools", response_model=List[ResourcePool])
async def get_resource_pools():
    """List all resource pools"""
    with tracer.start_as_current_span("get_resource_pools"):
        return fake_cluster.get_resource_pools()


@app.get("/o2ims-infrastructureInventory/v1/resourcePools/{pool_id}", response_model=ResourcePool)
async def get_resource_pool(pool_id: str = Path(..., description="Resource Pool ID")):
    """Get specific resource pool"""
    with tracer.start_as_current_span("get_resource_pool") as span:
        span.set_attribute("pool_id", pool_id)
        pool = fake_cluster.get_resource_pool(pool_id)
        if not pool:
            raise HTTPException(status_code=404, detail=f"Resource pool {pool_id} not found")
        return pool


# =============================================================================
# Infrastructure Inventory - Resources
# =============================================================================

@app.get("/o2ims-infrastructureInventory/v1/resourcePools/{pool_id}/resources", response_model=List[O2Resource])
async def get_resources(pool_id: str = Path(..., description="Resource Pool ID")):
    """List resources in a pool"""
    with tracer.start_as_current_span("get_resources") as span:
        span.set_attribute("pool_id", pool_id)
        pool = fake_cluster.get_resource_pool(pool_id)
        if not pool:
            raise HTTPException(status_code=404, detail=f"Resource pool {pool_id} not found")
        return fake_cluster.get_resources(pool_id)


@app.get("/o2ims-infrastructureInventory/v1/resourcePools/{pool_id}/resources/{resource_id}", response_model=O2Resource)
async def get_resource(
    pool_id: str = Path(..., description="Resource Pool ID"),
    resource_id: str = Path(..., description="Resource ID")
):
    """Get specific resource"""
    with tracer.start_as_current_span("get_resource") as span:
        span.set_attribute("pool_id", pool_id)
        span.set_attribute("resource_id", resource_id)
        resource = fake_cluster.get_resource(pool_id, resource_id)
        if not resource:
            raise HTTPException(status_code=404, detail=f"Resource {resource_id} not found in pool {pool_id}")
        return resource


# =============================================================================
# Infrastructure Inventory - Subscriptions
# =============================================================================

@app.get("/o2ims-infrastructureInventory/v1/subscriptions")
async def get_inventory_subscriptions():
    """List inventory change subscriptions"""
    return fake_cluster.get_inventory_subscriptions()


@app.post("/o2ims-infrastructureInventory/v1/subscriptions", status_code=201)
async def create_inventory_subscription(subscription: InventorySubscription):
    """Create inventory change subscription"""
    with tracer.start_as_current_span("create_inventory_subscription"):
        result = fake_cluster.create_inventory_subscription(
            callback=subscription.callback,
            filter_str=subscription.filter
        )
        return result


@app.get("/o2ims-infrastructureInventory/v1/subscriptions/{sub_id}")
async def get_inventory_subscription(sub_id: str = Path(..., description="Subscription ID")):
    """Get specific subscription"""
    subs = fake_cluster.get_inventory_subscriptions()
    for s in subs:
        if s.get("subscriptionId") == sub_id:
            return s
    raise HTTPException(status_code=404, detail=f"Subscription {sub_id} not found")


@app.delete("/o2ims-infrastructureInventory/v1/subscriptions/{sub_id}", status_code=204)
async def delete_inventory_subscription(sub_id: str = Path(..., description="Subscription ID")):
    """Delete subscription"""
    if not fake_cluster.delete_inventory_subscription(sub_id):
        raise HTTPException(status_code=404, detail=f"Subscription {sub_id} not found")
    return None


# =============================================================================
# Infrastructure Monitoring - Alarms
# =============================================================================

@app.get("/o2ims-infrastructureMonitoring/v1/alarms", response_model=List[AlarmEventRecord])
async def get_alarms(
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment status"),
    severity: Optional[int] = Query(None, ge=0, le=5, description="Filter by severity")
):
    """List all alarms"""
    with tracer.start_as_current_span("get_alarms"):
        alarms = fake_cluster.get_alarms()

        # Apply filters
        if acknowledged is not None:
            alarms = [a for a in alarms if a.alarmAcknowledged == acknowledged]
        if severity is not None:
            alarms = [a for a in alarms if a.perceivedSeverity.value == severity]

        return alarms


@app.get("/o2ims-infrastructureMonitoring/v1/alarms/{alarm_id}", response_model=AlarmEventRecord)
async def get_alarm(alarm_id: str = Path(..., description="Alarm ID")):
    """Get specific alarm"""
    with tracer.start_as_current_span("get_alarm") as span:
        span.set_attribute("alarm_id", alarm_id)
        alarm = fake_cluster.get_alarm(alarm_id)
        if not alarm:
            raise HTTPException(status_code=404, detail=f"Alarm {alarm_id} not found")
        return alarm


@app.patch("/o2ims-infrastructureMonitoring/v1/alarms/{alarm_id}", response_model=AlarmEventRecord)
async def patch_alarm(
    alarm_id: str = Path(..., description="Alarm ID"),
    acknowledged: Optional[bool] = Query(None, description="Set acknowledgment status"),
    clear: Optional[bool] = Query(None, description="Clear the alarm")
):
    """Acknowledge or clear an alarm"""
    with tracer.start_as_current_span("patch_alarm") as span:
        span.set_attribute("alarm_id", alarm_id)

        alarm = fake_cluster.get_alarm(alarm_id)
        if not alarm:
            raise HTTPException(status_code=404, detail=f"Alarm {alarm_id} not found")

        if acknowledged:
            alarm = fake_cluster.acknowledge_alarm(alarm_id)
        if clear:
            fake_cluster.clear_alarm(alarm_id)
            alarm = fake_cluster.get_alarm(alarm_id)

        return alarm


@app.post("/o2ims-infrastructureMonitoring/v1/alarms", response_model=AlarmEventRecord, status_code=201)
async def create_alarm(
    severity: int = Query(..., ge=0, le=4, description="Severity (0=CRITICAL, 4=INDETERMINATE)"),
    resource_id: str = Query(..., description="Affected resource ID"),
    message: str = Query(..., description="Alarm message")
):
    """Create a new alarm (for testing)"""
    with tracer.start_as_current_span("create_alarm") as span:
        span.set_attribute("severity", severity)
        span.set_attribute("resource_id", resource_id)

        alarm = fake_cluster.create_alarm(
            severity=PerceivedSeverity(severity),
            resource_id=resource_id,
            message=message
        )
        return alarm


# =============================================================================
# Infrastructure Monitoring - Alarm Subscriptions
# =============================================================================

@app.get("/o2ims-infrastructureMonitoring/v1/alarmSubscriptions")
async def get_alarm_subscriptions():
    """List alarm subscriptions"""
    return fake_cluster.get_alarm_subscriptions()


@app.post("/o2ims-infrastructureMonitoring/v1/alarmSubscriptions", status_code=201)
async def create_alarm_subscription(subscription: AlarmSubscriptionInfo):
    """Create alarm subscription"""
    with tracer.start_as_current_span("create_alarm_subscription"):
        result = fake_cluster.create_alarm_subscription(
            callback=subscription.callback,
            filter_str=subscription.filter
        )
        return result


@app.get("/o2ims-infrastructureMonitoring/v1/alarmSubscriptions/{sub_id}")
async def get_alarm_subscription(sub_id: str = Path(..., description="Subscription ID")):
    """Get specific alarm subscription"""
    subs = fake_cluster.get_alarm_subscriptions()
    for s in subs:
        if s.get("alarmSubscriptionId") == sub_id:
            return s
    raise HTTPException(status_code=404, detail=f"Subscription {sub_id} not found")


@app.delete("/o2ims-infrastructureMonitoring/v1/alarmSubscriptions/{sub_id}", status_code=204)
async def delete_alarm_subscription(sub_id: str = Path(..., description="Subscription ID")):
    """Delete alarm subscription"""
    if not fake_cluster.delete_alarm_subscription(sub_id):
        raise HTTPException(status_code=404, detail=f"Subscription {sub_id} not found")
    return None


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="O2 IMS - Infrastructure Management Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8098, help="Port to bind to")
    args = parser.parse_args()

    logger.info(f"Starting O2 IMS on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
