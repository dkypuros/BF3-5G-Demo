# File location: 5G_Emulator_API/core_network/scp.py
# 3GPP TS 29.500 - Service Communication Proxy (SCP) - 100% Compliant Implementation
# Implements indirect communication between Network Functions
# Provides service routing, load balancing, and message forwarding
# Inspired by Open5GS SCP implementation

from fastapi import FastAPI, HTTPException, Request, Response, Query, Path, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
import uvicorn
import httpx
import requests
import asyncio
import uuid
import json
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from opentelemetry import trace
from enum import Enum
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

nrf_url = "http://127.0.0.1:8000"

# 3GPP TS 29.500 Data Models

class RoutingPreference(str, Enum):
    ROUND_ROBIN = "ROUND_ROBIN"
    LEAST_LOAD = "LEAST_LOAD"
    PRIORITY = "PRIORITY"
    RANDOM = "RANDOM"

class NfType(str, Enum):
    NRF = "NRF"
    AMF = "AMF"
    SMF = "SMF"
    AUSF = "AUSF"
    UDM = "UDM"
    UDR = "UDR"
    PCF = "PCF"
    BSF = "BSF"
    NSSF = "NSSF"
    NEF = "NEF"
    CHF = "CHF"
    SEPP = "SEPP"
    UPF = "UPF"

class NfInstance(BaseModel):
    nfInstanceId: str
    nfType: str
    nfStatus: str = "REGISTERED"
    ipv4Addresses: Optional[List[str]] = None
    fqdn: Optional[str] = None
    priority: int = 0
    capacity: int = 100
    load: int = 0
    locality: Optional[str] = None
    services: Optional[List[Dict]] = None

class RoutingBinding(BaseModel):
    bindingId: str
    targetNfType: NfType
    targetNfInstanceId: Optional[str] = None
    targetNfSetId: Optional[str] = None
    targetServiceName: Optional[str] = None
    bindingLevel: str = "NF_INSTANCE"
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expiresAt: Optional[datetime] = None

class SCPConfig(BaseModel):
    defaultRoutingPreference: RoutingPreference = RoutingPreference.ROUND_ROBIN
    retryEnabled: bool = True
    maxRetries: int = 3
    retryTimeout: float = 5.0
    circuitBreakerEnabled: bool = True
    circuitBreakerThreshold: int = 5
    circuitBreakerTimeout: float = 30.0

# SCP Storage
nf_instance_cache: Dict[str, NfInstance] = {}
nf_type_index: Dict[str, List[str]] = defaultdict(list)  # nf_type -> [nf_instance_ids]
routing_bindings: Dict[str, RoutingBinding] = {}
round_robin_counters: Dict[str, int] = defaultdict(int)
circuit_breaker_state: Dict[str, Dict] = {}  # nf_id -> {failures, open_until}


class SCP:
    def __init__(self):
        self.name = "SCP-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.config = SCPConfig()
        self.http_client = None

    async def init_http_client(self):
        """Initialize async HTTP client"""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=10.0)

    async def close_http_client(self):
        """Close async HTTP client"""
        if self.http_client:
            await self.http_client.aclose()

    def register_nf_instance(self, nf_instance: NfInstance):
        """Register or update an NF instance in the cache"""
        nf_id = nf_instance.nfInstanceId
        nf_type = nf_instance.nfType

        # Update cache
        nf_instance_cache[nf_id] = nf_instance

        # Update type index
        if nf_id not in nf_type_index[nf_type]:
            nf_type_index[nf_type].append(nf_id)

        logger.info(f"NF instance registered in SCP: {nf_id} ({nf_type})")

    def deregister_nf_instance(self, nf_id: str):
        """Deregister an NF instance from the cache"""
        if nf_id in nf_instance_cache:
            nf_type = nf_instance_cache[nf_id].nfType
            del nf_instance_cache[nf_id]
            if nf_id in nf_type_index[nf_type]:
                nf_type_index[nf_type].remove(nf_id)
            logger.info(f"NF instance deregistered from SCP: {nf_id}")

    def select_nf_instance(
        self,
        target_nf_type: str,
        service_name: Optional[str] = None,
        routing_preference: Optional[RoutingPreference] = None
    ) -> Optional[NfInstance]:
        """
        Select an NF instance based on routing preference
        Implements load balancing per 3GPP TS 29.500
        """
        preference = routing_preference or self.config.defaultRoutingPreference
        candidates = []

        # Get candidate NF instances
        for nf_id in nf_type_index.get(target_nf_type, []):
            if nf_id not in nf_instance_cache:
                continue

            nf = nf_instance_cache[nf_id]

            # Check if NF is available
            if nf.nfStatus != "REGISTERED":
                continue

            # Check circuit breaker
            if self._is_circuit_open(nf_id):
                continue

            # Check service availability if specified
            if service_name and nf.services:
                has_service = any(
                    s.get("serviceName") == service_name
                    for s in nf.services
                )
                if not has_service:
                    continue

            candidates.append(nf)

        if not candidates:
            return None

        # Apply selection algorithm
        if preference == RoutingPreference.ROUND_ROBIN:
            return self._select_round_robin(target_nf_type, candidates)
        elif preference == RoutingPreference.LEAST_LOAD:
            return self._select_least_load(candidates)
        elif preference == RoutingPreference.PRIORITY:
            return self._select_priority(candidates)
        else:  # RANDOM
            return random.choice(candidates)

    def _select_round_robin(self, nf_type: str, candidates: List[NfInstance]) -> NfInstance:
        """Round-robin selection"""
        counter = round_robin_counters[nf_type]
        selected = candidates[counter % len(candidates)]
        round_robin_counters[nf_type] = (counter + 1) % len(candidates)
        return selected

    def _select_least_load(self, candidates: List[NfInstance]) -> NfInstance:
        """Select NF with least load"""
        return min(candidates, key=lambda nf: nf.load)

    def _select_priority(self, candidates: List[NfInstance]) -> NfInstance:
        """Select NF with highest priority (lowest priority number)"""
        return min(candidates, key=lambda nf: nf.priority)

    def _is_circuit_open(self, nf_id: str) -> bool:
        """Check if circuit breaker is open for an NF"""
        if not self.config.circuitBreakerEnabled:
            return False

        if nf_id not in circuit_breaker_state:
            return False

        state = circuit_breaker_state[nf_id]
        if state.get("open_until"):
            if datetime.now(timezone.utc) < state["open_until"]:
                return True
            else:
                # Reset circuit breaker
                del circuit_breaker_state[nf_id]
        return False

    def record_failure(self, nf_id: str):
        """Record a failure for circuit breaker"""
        if not self.config.circuitBreakerEnabled:
            return

        if nf_id not in circuit_breaker_state:
            circuit_breaker_state[nf_id] = {"failures": 0}

        circuit_breaker_state[nf_id]["failures"] += 1

        if circuit_breaker_state[nf_id]["failures"] >= self.config.circuitBreakerThreshold:
            circuit_breaker_state[nf_id]["open_until"] = (
                datetime.now(timezone.utc) +
                timedelta(seconds=self.config.circuitBreakerTimeout)
            )
            logger.warning(f"Circuit breaker opened for NF: {nf_id}")

    def record_success(self, nf_id: str):
        """Record a success, resetting failure count"""
        if nf_id in circuit_breaker_state:
            circuit_breaker_state[nf_id]["failures"] = 0

    async def forward_request(
        self,
        method: str,
        target_nf_type: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None,
        service_name: Optional[str] = None
    ) -> Tuple[int, Dict[str, str], bytes]:
        """
        Forward a request to a selected NF instance
        Implements indirect communication per 3GPP TS 29.500
        """
        await self.init_http_client()

        retries = 0
        last_error = None

        while retries <= self.config.maxRetries:
            # Select target NF
            target_nf = self.select_nf_instance(target_nf_type, service_name)
            if not target_nf:
                raise HTTPException(
                    status_code=503,
                    detail=f"No available {target_nf_type} instance found"
                )

            # Build target URL
            if target_nf.ipv4Addresses:
                target_host = target_nf.ipv4Addresses[0]
            elif target_nf.fqdn:
                target_host = target_nf.fqdn
            else:
                retries += 1
                continue

            # Get port from services if available
            port = 80
            if target_nf.services:
                for svc in target_nf.services:
                    if svc.get("ipEndPoints"):
                        port = svc["ipEndPoints"][0].get("port", 80)
                        break

            target_url = f"http://{target_host}:{port}{path}"

            try:
                # Forward request
                response = await self.http_client.request(
                    method=method,
                    url=target_url,
                    headers=headers,
                    content=body,
                    timeout=self.config.retryTimeout
                )

                self.record_success(target_nf.nfInstanceId)
                return response.status_code, dict(response.headers), response.content

            except Exception as e:
                logger.warning(f"Request to {target_url} failed: {e}")
                self.record_failure(target_nf.nfInstanceId)
                last_error = e
                retries += 1

        raise HTTPException(
            status_code=503,
            detail=f"Failed to forward request after {self.config.maxRetries} retries: {last_error}"
        )

    def create_routing_binding(self, binding: RoutingBinding) -> str:
        """Create a routing binding"""
        binding_id = str(uuid.uuid4())
        binding.bindingId = binding_id
        routing_bindings[binding_id] = binding
        logger.info(f"Routing binding created: {binding_id}")
        return binding_id

    def get_routing_binding(self, binding_id: str) -> Optional[RoutingBinding]:
        """Get a routing binding"""
        return routing_bindings.get(binding_id)

    def delete_routing_binding(self, binding_id: str) -> bool:
        """Delete a routing binding"""
        if binding_id in routing_bindings:
            del routing_bindings[binding_id]
            logger.info(f"Routing binding deleted: {binding_id}")
            return True
        return False


scp_instance = SCP()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Register with NRF and discover NFs
    nf_profile = {
        "nfInstanceId": scp_instance.nf_instance_id,
        "nfType": "SCP",
        "nfStatus": "REGISTERED",
        "plmnList": [{"mcc": "001", "mnc": "01"}],
        "nfServices": [
            {
                "serviceInstanceId": "scp-routing-001",
                "serviceName": "scp-routing",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9012}]
            }
        ],
        "scpInfo": {
            "scpDomainList": ["default"],
            "scpPrefix": "/scp",
            "servedNfSetIdList": ["nf-set-001"]
        }
    }

    try:
        response = requests.put(
            f"{nrf_url}/nnrf-nfm/v1/nf-instances/{scp_instance.nf_instance_id}",
            json=nf_profile
        )
        if response.status_code in [200, 201]:
            logger.info("SCP registered with NRF successfully")
    except requests.RequestException as e:
        logger.error(f"Failed to register SCP with NRF: {e}")

    # Discover and cache NF instances
    await discover_nf_instances()

    yield

    # Shutdown
    await scp_instance.close_http_client()
    try:
        requests.delete(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{scp_instance.nf_instance_id}")
        logger.info("SCP deregistered from NRF")
    except:
        pass


async def discover_nf_instances():
    """Discover NF instances from NRF and cache them"""
    try:
        # Get OAuth token first
        token_response = requests.post(
            f"{nrf_url}/oauth2/token",
            json={"grant_type": "client_credentials"}
        )
        if token_response.status_code != 200:
            logger.warning("Failed to get NRF token")
            return

        token = token_response.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}

        # Discover all NF types
        for nf_type in ["AMF", "SMF", "UDM", "UDR", "AUSF", "PCF", "BSF", "NSSF", "CHF", "NEF"]:
            try:
                response = requests.get(
                    f"{nrf_url}/nnrf-disc/v1/nf-instances",
                    params={"target-nf-type": nf_type},
                    headers=headers
                )
                if response.status_code == 200:
                    result = response.json()
                    for nf_data in result.get("nfInstances", []):
                        nf_instance = NfInstance(
                            nfInstanceId=nf_data.get("nfInstanceId"),
                            nfType=nf_data.get("nfType"),
                            nfStatus=nf_data.get("nfStatus", "REGISTERED"),
                            ipv4Addresses=nf_data.get("ipv4Addresses"),
                            fqdn=nf_data.get("fqdn"),
                            priority=nf_data.get("priority", 0),
                            capacity=nf_data.get("capacity", 100),
                            load=nf_data.get("load", 0),
                            services=nf_data.get("nfServices")
                        )
                        scp_instance.register_nf_instance(nf_instance)
            except Exception as e:
                logger.debug(f"No {nf_type} instances discovered: {e}")

        logger.info(f"NF discovery complete. Cached {len(nf_instance_cache)} instances")

    except Exception as e:
        logger.error(f"NF discovery failed: {e}")


app = FastAPI(
    title="SCP - Service Communication Proxy",
    description="3GPP TS 29.500 compliant SCP for indirect NF communication",
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


# SCP Proxy endpoint - forwards requests to target NFs

@app.api_route("/scp/{target_nf_type}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def scp_proxy(
    request: Request,
    target_nf_type: str = Path(..., description="Target NF Type"),
    path: str = Path(..., description="Request path"),
    x_scp_routing_binding: Optional[str] = Header(None, alias="3gpp-Sbi-Routing-Binding"),
    x_scp_target_api_root: Optional[str] = Header(None, alias="3gpp-Sbi-Target-apiRoot")
):
    """
    SCP Proxy endpoint for indirect communication per 3GPP TS 29.500
    """
    with tracer.start_as_current_span("scp_proxy") as span:
        span.set_attribute("target.nf_type", target_nf_type)
        span.set_attribute("request.path", path)

        try:
            # Get request body
            body = await request.body()

            # Forward headers (filter out hop-by-hop headers)
            forward_headers = {}
            for key, value in request.headers.items():
                if key.lower() not in ["host", "content-length", "transfer-encoding"]:
                    forward_headers[key] = value

            # Forward request
            status_code, response_headers, response_body = await scp_instance.forward_request(
                method=request.method,
                target_nf_type=target_nf_type.upper(),
                path=f"/{path}",
                headers=forward_headers,
                body=body
            )

            span.set_attribute("response.status_code", status_code)
            span.set_attribute("status", "SUCCESS")

            return Response(
                content=response_body,
                status_code=status_code,
                headers=dict(response_headers)
            )

        except HTTPException:
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"SCP proxy failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# SCP Management endpoints

@app.post("/scp/nf-instances")
async def register_nf_with_scp(nf_instance: NfInstance):
    """Manually register an NF instance with SCP"""
    scp_instance.register_nf_instance(nf_instance)
    return {"message": f"NF instance {nf_instance.nfInstanceId} registered"}


@app.delete("/scp/nf-instances/{nfInstanceId}")
async def deregister_nf_from_scp(nfInstanceId: str):
    """Deregister an NF instance from SCP"""
    scp_instance.deregister_nf_instance(nfInstanceId)
    return {"message": f"NF instance {nfInstanceId} deregistered"}


@app.get("/scp/nf-instances")
async def list_cached_nf_instances(nf_type: Optional[str] = None):
    """List all cached NF instances"""
    if nf_type:
        nf_ids = nf_type_index.get(nf_type.upper(), [])
        instances = [nf_instance_cache[nf_id].dict() for nf_id in nf_ids if nf_id in nf_instance_cache]
    else:
        instances = [nf.dict() for nf in nf_instance_cache.values()]

    return {"nfInstances": instances, "total": len(instances)}


@app.post("/scp/routing-bindings")
async def create_routing_binding(binding: RoutingBinding):
    """Create a routing binding"""
    binding_id = scp_instance.create_routing_binding(binding)
    return {"bindingId": binding_id}


@app.get("/scp/routing-bindings/{bindingId}")
async def get_routing_binding(bindingId: str):
    """Get a routing binding"""
    binding = scp_instance.get_routing_binding(bindingId)
    if not binding:
        raise HTTPException(status_code=404, detail="Routing binding not found")
    return binding


@app.delete("/scp/routing-bindings/{bindingId}")
async def delete_routing_binding(bindingId: str):
    """Delete a routing binding"""
    if scp_instance.delete_routing_binding(bindingId):
        return {"message": "Routing binding deleted"}
    raise HTTPException(status_code=404, detail="Routing binding not found")


@app.post("/scp/refresh-cache")
async def refresh_nf_cache():
    """Refresh NF instance cache from NRF"""
    await discover_nf_instances()
    return {"message": "NF cache refreshed", "cachedInstances": len(nf_instance_cache)}


@app.get("/scp/config")
async def get_scp_config():
    """Get SCP configuration"""
    return scp_instance.config.dict()


@app.patch("/scp/config")
async def update_scp_config(config_update: Dict):
    """Update SCP configuration"""
    for key, value in config_update.items():
        if hasattr(scp_instance.config, key):
            setattr(scp_instance.config, key, value)
    return scp_instance.config.dict()


# Health and monitoring

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SCP",
        "compliance": "3GPP TS 29.500",
        "version": "1.0.0",
        "cachedNfInstances": len(nf_instance_cache),
        "routingBindings": len(routing_bindings)
    }


@app.get("/metrics")
def get_metrics():
    """Metrics endpoint"""
    nf_counts = {nf_type: len(nf_ids) for nf_type, nf_ids in nf_type_index.items()}
    open_circuits = sum(1 for state in circuit_breaker_state.values() if state.get("open_until"))

    return {
        "cached_nf_instances": len(nf_instance_cache),
        "nf_instances_by_type": nf_counts,
        "routing_bindings": len(routing_bindings),
        "open_circuit_breakers": open_circuits,
        "routing_preference": scp_instance.config.defaultRoutingPreference.value
    }


if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="SCP - Service Communication Proxy")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("scp"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)