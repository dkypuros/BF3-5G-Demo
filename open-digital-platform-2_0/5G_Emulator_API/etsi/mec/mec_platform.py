# File location: 5G_Emulator_API/etsi/mec/mec_platform.py
# ETSI GS MEC 003, GS MEC 011 - MEC Platform Implementation
# Multi-access Edge Computing Platform with Mp1 Interface

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import uvicorn
import uuid
import json
import logging
from datetime import datetime, timedelta, timezone
import os
import asyncio
import httpx
from opentelemetry import trace
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

# =============================================================================
# ETSI GS MEC 011 Data Models - MEC Application Support API
# =============================================================================

class SerializerType(str, Enum):
    """ETSI GS MEC 011 - SerializerType"""
    JSON = "JSON"
    XML = "XML"
    PROTOBUF3 = "PROTOBUF3"

class LocalityType(str, Enum):
    """ETSI GS MEC 011 - LocalityType for service discovery"""
    MEC_SYSTEM = "MEC_SYSTEM"
    MEC_HOST = "MEC_HOST"
    NFVI_POP = "NFVI_POP"
    ZONE = "ZONE"
    ZONE_GROUP = "ZONE_GROUP"
    NFVI_NODE = "NFVI_NODE"

class ServiceState(str, Enum):
    """ETSI GS MEC 011 - ServiceState"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"

class TransportType(str, Enum):
    """ETSI GS MEC 011 - TransportType"""
    REST_HTTP = "REST_HTTP"
    MB_TOPIC_BASED = "MB_TOPIC_BASED"
    MB_ROUTING = "MB_ROUTING"
    MB_PUBSUB = "MB_PUBSUB"
    RPC = "RPC"
    RPC_STREAMING = "RPC_STREAMING"
    WEBSOCKET = "WEBSOCKET"

class SecurityInfo(BaseModel):
    """ETSI GS MEC 011 - SecurityInfo"""
    oAuth2Info: Optional[Dict[str, Any]] = Field(None, description="OAuth 2.0 token endpoint")

class TransportInfo(BaseModel):
    """ETSI GS MEC 011 - TransportInfo (Section 8.1.2.4)"""
    id: str = Field(..., description="Transport identifier")
    name: str = Field(..., description="Transport name")
    description: Optional[str] = Field(None, description="Human-readable description")
    type: TransportType = Field(..., description="Type of transport")
    protocol: str = Field(..., description="Protocol name (e.g., HTTP)")
    version: str = Field(..., description="Protocol version")
    endpoint: Dict[str, Any] = Field(..., description="Endpoint information")
    security: Optional[SecurityInfo] = Field(None, description="Security info")
    implSpecificInfo: Optional[Dict[str, Any]] = Field(None, description="Implementation-specific info")

class CategoryRef(BaseModel):
    """ETSI GS MEC 011 - CategoryRef"""
    href: str = Field(..., description="Reference to category")
    id: str = Field(..., description="Category identifier")
    name: str = Field(..., description="Category name")
    version: str = Field(..., description="Category version")

class ServiceInfo(BaseModel):
    """ETSI GS MEC 011 - ServiceInfo (Section 8.1.2.2)

    Represents a MEC service that can be registered and discovered via Mp1.
    """
    serInstanceId: Optional[str] = Field(None, description="Service instance identifier")
    serName: str = Field(..., description="Service name")
    serCategory: Optional[CategoryRef] = Field(None, description="Service category")
    version: str = Field(..., description="Service version")
    state: ServiceState = Field(ServiceState.ACTIVE, description="Service state")
    transportInfo: TransportInfo = Field(..., description="Transport information")
    serializer: SerializerType = Field(SerializerType.JSON, description="Serializer type")
    scopeOfLocality: Optional[LocalityType] = Field(None, description="Scope of locality")
    consumedLocalOnly: bool = Field(False, description="Local consumption only")
    isLocal: bool = Field(True, description="Is local service")
    livenessInterval: Optional[int] = Field(None, description="Liveness interval in seconds")

class AppReadyConfirmation(BaseModel):
    """ETSI GS MEC 011 - AppReadyConfirmation"""
    indication: str = Field("READY", description="Ready indication")

class AppTerminationConfirmation(BaseModel):
    """ETSI GS MEC 011 - AppTerminationConfirmation"""
    operationAction: str = Field(..., description="Termination action")

class DnsRule(BaseModel):
    """ETSI GS MEC 011 - DnsRule (Section 8.1.2.8)"""
    dnsRuleId: str = Field(..., description="DNS rule identifier")
    domainName: str = Field(..., description="Domain name")
    ipAddressType: str = Field("IP_V4", description="IP address type")
    ipAddress: str = Field(..., description="IP address")
    ttl: Optional[int] = Field(300, description="TTL in seconds")
    state: str = Field("ACTIVE", description="Rule state")

class TrafficRule(BaseModel):
    """ETSI GS MEC 011 - TrafficRule (Section 8.1.2.9)"""
    trafficRuleId: str = Field(..., description="Traffic rule identifier")
    filterType: str = Field("FLOW", description="Filter type")
    priority: int = Field(0, ge=0, le=255, description="Rule priority")
    trafficFilter: List[Dict[str, Any]] = Field(..., description="Traffic filters")
    action: str = Field("FORWARD_DECAPSULATED", description="Traffic action")
    state: str = Field("ACTIVE", description="Rule state")

# =============================================================================
# ETSI GS MEC 003 - MEC Platform Architecture Components
# =============================================================================

class MecAppInstance(BaseModel):
    """ETSI GS MEC 003 - MEC Application Instance"""
    appInstanceId: str = Field(..., description="Application instance identifier")
    appName: str = Field(..., description="Application name")
    appProvider: str = Field(..., description="Application provider")
    appSoftVersion: Optional[str] = Field(None, description="Software version")
    appDId: Optional[str] = Field(None, description="Application descriptor ID")
    instantiationState: str = Field("NOT_INSTANTIATED", description="Instantiation state")
    instantiatedVnfInfo: Optional[Dict[str, Any]] = Field(None, description="VNF info")

class MecHostInfo(BaseModel):
    """ETSI GS MEC 003 - MEC Host Information"""
    hostId: str = Field(..., description="MEC host identifier")
    hostName: str = Field(..., description="MEC host name")
    capabilities: Optional[Dict[str, Any]] = Field(None, description="Host capabilities")
    hwResources: Optional[Dict[str, Any]] = Field(None, description="Hardware resources")

# =============================================================================
# MEC Platform Core Class
# =============================================================================

class MECPlatform:
    """ETSI GS MEC 003, GS MEC 011 - MEC Platform Manager

    Implements the Mp1 reference point for MEC service registration,
    discovery, and lifecycle management.
    """

    def __init__(self, host_id: str = None, upf_url: str = None):
        """Initialize MEC Platform

        Args:
            host_id: MEC host identifier
            upf_url: URL of the integrated UPF for N6 interface
        """
        self.host_id = host_id or str(uuid.uuid4())
        self.platform_id = str(uuid.uuid4())
        self.upf_url = upf_url or os.environ.get("UPF_URL", "http://127.0.0.1:8005")

        # Service registry (Mp1 service registration)
        self.services: Dict[str, ServiceInfo] = {}

        # Application instances
        self.app_instances: Dict[str, MecAppInstance] = {}

        # DNS and Traffic rules
        self.dns_rules: Dict[str, DnsRule] = {}
        self.traffic_rules: Dict[str, TrafficRule] = {}

        # Subscriptions for notifications
        self.subscriptions: Dict[str, Dict[str, Any]] = {}

        logger.info(f"MEC Platform initialized: host_id={self.host_id}")

    def register_service(self, service: ServiceInfo) -> ServiceInfo:
        """ETSI GS MEC 011 - Register a MEC service via Mp1

        Spec: Section 8.2.2 - POST /services
        """
        with tracer.start_as_current_span("mec_register_service") as span:
            span.set_attribute("etsi.service", "Mx1")
            span.set_attribute("etsi.operation", "RegisterService")

            # Generate service instance ID if not provided
            if not service.serInstanceId:
                service.serInstanceId = str(uuid.uuid4())

            self.services[service.serInstanceId] = service

            span.set_attribute("service.instance_id", service.serInstanceId)
            span.set_attribute("service.name", service.serName)

            logger.info(f"Registered MEC service: {service.serName} ({service.serInstanceId})")
            return service

    def discover_services(
        self,
        ser_name: str = None,
        ser_category_id: str = None,
        scope_of_locality: LocalityType = None,
        consumed_local_only: bool = None,
        is_local: bool = None
    ) -> List[ServiceInfo]:
        """ETSI GS MEC 011 - Discover MEC services via Mp1

        Spec: Section 8.2.3 - GET /services
        """
        with tracer.start_as_current_span("mec_discover_services") as span:
            span.set_attribute("etsi.service", "Mx1")
            span.set_attribute("etsi.operation", "DiscoverServices")

            results = list(self.services.values())

            # Filter by service name
            if ser_name:
                results = [s for s in results if s.serName == ser_name]

            # Filter by category
            if ser_category_id:
                results = [s for s in results if s.serCategory and s.serCategory.id == ser_category_id]

            # Filter by locality scope
            if scope_of_locality:
                results = [s for s in results if s.scopeOfLocality == scope_of_locality]

            # Filter by local consumption
            if consumed_local_only is not None:
                results = [s for s in results if s.consumedLocalOnly == consumed_local_only]

            # Filter by local flag
            if is_local is not None:
                results = [s for s in results if s.isLocal == is_local]

            # Only return active services
            results = [s for s in results if s.state == ServiceState.ACTIVE]

            span.set_attribute("discovered.count", len(results))
            return results

    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """ETSI GS MEC 011 - Get specific service by ID

        Spec: Section 8.2.4 - GET /services/{serviceId}
        """
        return self.services.get(service_id)

    def update_service(self, service_id: str, service: ServiceInfo) -> Optional[ServiceInfo]:
        """ETSI GS MEC 011 - Update service registration

        Spec: Section 8.2.5 - PUT /services/{serviceId}
        """
        if service_id not in self.services:
            return None

        service.serInstanceId = service_id
        self.services[service_id] = service
        logger.info(f"Updated MEC service: {service.serName} ({service_id})")
        return service

    def deregister_service(self, service_id: str) -> bool:
        """ETSI GS MEC 011 - Deregister a MEC service

        Spec: Section 8.2.6 - DELETE /services/{serviceId}
        """
        if service_id in self.services:
            service = self.services.pop(service_id)
            logger.info(f"Deregistered MEC service: {service.serName} ({service_id})")
            return True
        return False

    def register_app_instance(self, app: MecAppInstance) -> MecAppInstance:
        """Register a MEC application instance"""
        self.app_instances[app.appInstanceId] = app
        logger.info(f"Registered MEC app instance: {app.appName} ({app.appInstanceId})")
        return app

    def confirm_app_ready(self, app_instance_id: str, confirmation: AppReadyConfirmation) -> bool:
        """ETSI GS MEC 011 - Confirm application ready

        Spec: Section 8.3.2 - POST /applications/{appInstanceId}/confirm_ready
        """
        with tracer.start_as_current_span("mec_app_ready") as span:
            span.set_attribute("etsi.service", "Mx1")
            span.set_attribute("etsi.operation", "AppReadyConfirmation")

            if app_instance_id in self.app_instances:
                self.app_instances[app_instance_id].instantiationState = "INSTANTIATED"
                logger.info(f"App confirmed ready: {app_instance_id}")
                return True
            return False

    def add_dns_rule(self, app_instance_id: str, rule: DnsRule) -> DnsRule:
        """ETSI GS MEC 011 - Add DNS rule for application

        Spec: Section 8.4.2 - PUT /applications/{appInstanceId}/dns_rules/{dnsRuleId}
        """
        self.dns_rules[rule.dnsRuleId] = rule
        logger.info(f"Added DNS rule: {rule.dnsRuleId} -> {rule.domainName}")
        return rule

    def add_traffic_rule(self, app_instance_id: str, rule: TrafficRule) -> TrafficRule:
        """ETSI GS MEC 011 - Add traffic rule for application

        Spec: Section 8.5.2 - PUT /applications/{appInstanceId}/traffic_rules/{trafficRuleId}
        """
        self.traffic_rules[rule.trafficRuleId] = rule
        logger.info(f"Added traffic rule: {rule.trafficRuleId} (priority={rule.priority})")
        return rule

    async def notify_upf_traffic_steering(self, rule: TrafficRule):
        """Integrate with UPF for N6 traffic steering

        This connects the MEC Platform to the 5G Core UPF for traffic routing.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.upf_url}/traffic-steering",
                    json={
                        "ruleId": rule.trafficRuleId,
                        "priority": rule.priority,
                        "filters": rule.trafficFilter,
                        "action": rule.action
                    }
                )
                logger.info(f"UPF notified of traffic rule: {rule.trafficRuleId}")
        except Exception as e:
            logger.warning(f"Could not notify UPF: {e}")

# =============================================================================
# FastAPI Application - Mp1 REST API
# =============================================================================

app = FastAPI(
    title="ETSI MEC Platform - Mp1 Interface",
    description="Multi-access Edge Computing Platform implementing ETSI GS MEC 011",
    version="2.2.1",
    docs_url="/mp1/docs",
    openapi_url="/mp1/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global MEC Platform instance
mec_platform = MECPlatform()

# -----------------------------------------------------------------------------
# ETSI GS MEC 011 - MEC Service Management API (Mp1)
# -----------------------------------------------------------------------------

@app.post("/mec_service_mgmt/v1/applications/{appInstanceId}/services",
          response_model=ServiceInfo,
          status_code=201,
          tags=["MEC Service Management"])
async def register_service(
    appInstanceId: str = Path(..., description="Application instance ID"),
    service: ServiceInfo = ...
):
    """ETSI GS MEC 011 - Register MEC Service

    Spec: Section 8.2.2 - Service Registration
    This endpoint allows a MEC application to register a service that
    it provides to other MEC applications.
    """
    with tracer.start_as_current_span("mp1_register_service") as span:
        span.set_attribute("etsi.spec", "GS MEC 011")
        span.set_attribute("app.instance_id", appInstanceId)

        registered = mec_platform.register_service(service)
        return registered


@app.get("/mec_service_mgmt/v1/applications/{appInstanceId}/services",
         response_model=List[ServiceInfo],
         tags=["MEC Service Management"])
async def get_app_services(
    appInstanceId: str = Path(..., description="Application instance ID"),
    ser_name: Optional[str] = Query(None, alias="ser_name", description="Service name filter"),
    ser_category_id: Optional[str] = Query(None, alias="ser_category_id", description="Category filter")
):
    """ETSI GS MEC 011 - Get Application Services

    Spec: Section 8.2.3 - Service Discovery (application-specific)
    """
    return mec_platform.discover_services(
        ser_name=ser_name,
        ser_category_id=ser_category_id
    )


@app.get("/mec_service_mgmt/v1/services",
         response_model=List[ServiceInfo],
         tags=["MEC Service Management"])
async def discover_services(
    ser_name: Optional[str] = Query(None, description="Service name filter"),
    ser_category_id: Optional[str] = Query(None, description="Category ID filter"),
    scope_of_locality: Optional[LocalityType] = Query(None, description="Locality scope filter"),
    consumed_local_only: Optional[bool] = Query(None, description="Local consumption filter"),
    is_local: Optional[bool] = Query(None, description="Local service filter")
):
    """ETSI GS MEC 011 - Discover MEC Services

    Spec: Section 8.2.3 - Service Discovery (platform-wide)
    This allows MEC applications to discover available services.
    """
    with tracer.start_as_current_span("mp1_discover_services") as span:
        span.set_attribute("etsi.spec", "GS MEC 011")

        services = mec_platform.discover_services(
            ser_name=ser_name,
            ser_category_id=ser_category_id,
            scope_of_locality=scope_of_locality,
            consumed_local_only=consumed_local_only,
            is_local=is_local
        )

        span.set_attribute("discovered.count", len(services))
        return services


@app.get("/mec_service_mgmt/v1/services/{serviceId}",
         response_model=ServiceInfo,
         tags=["MEC Service Management"])
async def get_service(
    serviceId: str = Path(..., description="Service ID")
):
    """ETSI GS MEC 011 - Get Service by ID

    Spec: Section 8.2.4
    """
    service = mec_platform.get_service(serviceId)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    return service


@app.delete("/mec_service_mgmt/v1/applications/{appInstanceId}/services/{serviceId}",
            status_code=204,
            tags=["MEC Service Management"])
async def deregister_service(
    appInstanceId: str = Path(..., description="Application instance ID"),
    serviceId: str = Path(..., description="Service ID")
):
    """ETSI GS MEC 011 - Deregister Service

    Spec: Section 8.2.6
    """
    if not mec_platform.deregister_service(serviceId):
        raise HTTPException(status_code=404, detail="Service not found")


# -----------------------------------------------------------------------------
# ETSI GS MEC 011 - Application Lifecycle
# -----------------------------------------------------------------------------

@app.post("/mec_app_support/v1/applications/{appInstanceId}/confirm_ready",
          status_code=204,
          tags=["MEC Application Support"])
async def confirm_app_ready(
    appInstanceId: str = Path(..., description="Application instance ID"),
    confirmation: AppReadyConfirmation = ...
):
    """ETSI GS MEC 011 - Confirm Application Ready

    Spec: Section 8.3.2
    MEC application indicates it is ready to provide services.
    """
    if not mec_platform.confirm_app_ready(appInstanceId, confirmation):
        raise HTTPException(status_code=404, detail="Application instance not found")


@app.post("/mec_app_support/v1/applications/{appInstanceId}/confirm_termination",
          status_code=204,
          tags=["MEC Application Support"])
async def confirm_termination(
    appInstanceId: str = Path(..., description="Application instance ID"),
    confirmation: AppTerminationConfirmation = ...
):
    """ETSI GS MEC 011 - Confirm Application Termination

    Spec: Section 8.3.3
    """
    if appInstanceId in mec_platform.app_instances:
        mec_platform.app_instances[appInstanceId].instantiationState = "TERMINATED"
        logger.info(f"App termination confirmed: {appInstanceId}")


# -----------------------------------------------------------------------------
# ETSI GS MEC 011 - DNS Rules
# -----------------------------------------------------------------------------

@app.get("/mec_app_support/v1/applications/{appInstanceId}/dns_rules",
         response_model=List[DnsRule],
         tags=["DNS Rules"])
async def get_dns_rules(
    appInstanceId: str = Path(..., description="Application instance ID")
):
    """ETSI GS MEC 011 - Get DNS Rules

    Spec: Section 8.4.3
    """
    return list(mec_platform.dns_rules.values())


@app.put("/mec_app_support/v1/applications/{appInstanceId}/dns_rules/{dnsRuleId}",
         response_model=DnsRule,
         tags=["DNS Rules"])
async def update_dns_rule(
    appInstanceId: str = Path(..., description="Application instance ID"),
    dnsRuleId: str = Path(..., description="DNS rule ID"),
    rule: DnsRule = ...
):
    """ETSI GS MEC 011 - Create/Update DNS Rule

    Spec: Section 8.4.2
    """
    rule.dnsRuleId = dnsRuleId
    return mec_platform.add_dns_rule(appInstanceId, rule)


# -----------------------------------------------------------------------------
# ETSI GS MEC 011 - Traffic Rules
# -----------------------------------------------------------------------------

@app.get("/mec_app_support/v1/applications/{appInstanceId}/traffic_rules",
         response_model=List[TrafficRule],
         tags=["Traffic Rules"])
async def get_traffic_rules(
    appInstanceId: str = Path(..., description="Application instance ID")
):
    """ETSI GS MEC 011 - Get Traffic Rules

    Spec: Section 8.5.3
    """
    return list(mec_platform.traffic_rules.values())


@app.put("/mec_app_support/v1/applications/{appInstanceId}/traffic_rules/{trafficRuleId}",
         response_model=TrafficRule,
         tags=["Traffic Rules"])
async def update_traffic_rule(
    appInstanceId: str = Path(..., description="Application instance ID"),
    trafficRuleId: str = Path(..., description="Traffic rule ID"),
    rule: TrafficRule = ...
):
    """ETSI GS MEC 011 - Create/Update Traffic Rule

    Spec: Section 8.5.2
    Traffic rules steer traffic at the N6 interface.
    """
    rule.trafficRuleId = trafficRuleId
    result = mec_platform.add_traffic_rule(appInstanceId, rule)

    # Notify UPF of traffic steering rule
    asyncio.create_task(mec_platform.notify_upf_traffic_steering(rule))

    return result


# -----------------------------------------------------------------------------
# Health and Metrics
# -----------------------------------------------------------------------------

@app.get("/health", tags=["Operations"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "platform_id": mec_platform.platform_id,
        "host_id": mec_platform.host_id,
        "registered_services": len(mec_platform.services),
        "app_instances": len(mec_platform.app_instances),
        "dns_rules": len(mec_platform.dns_rules),
        "traffic_rules": len(mec_platform.traffic_rules)
    }


@app.get("/metrics", tags=["Operations"])
async def get_metrics():
    """Prometheus-compatible metrics"""
    metrics = []
    metrics.append(f'mec_services_registered_total {len(mec_platform.services)}')
    metrics.append(f'mec_app_instances_total {len(mec_platform.app_instances)}')
    metrics.append(f'mec_dns_rules_total {len(mec_platform.dns_rules)}')
    metrics.append(f'mec_traffic_rules_total {len(mec_platform.traffic_rules)}')
    return "\n".join(metrics)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("MEC_PLATFORM_PORT", 8090))
    logger.info(f"Starting MEC Platform on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
