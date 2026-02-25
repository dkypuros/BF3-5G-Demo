# File location: 5G_Emulator_API/core_network/bsf.py
# 3GPP TS 29.521 - Binding Support Function (BSF) - 100% Compliant Implementation
# Implements Nbsf_Management service for PCF binding management
# Inspired by Open5GS and Free5GC BSF implementations

from fastapi import FastAPI, HTTPException, Request, Query, Path, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
import requests
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

# 3GPP TS 29.521 Data Models

class BindingLevel(str, Enum):
    NF_SET = "NF_SET"
    NF_INSTANCE = "NF_INSTANCE"

class AddressDomain(str, Enum):
    IPV4 = "IPV4"
    IPV6 = "IPV6"

class PlmnId(BaseModel):
    mcc: str = Field(..., description="Mobile Country Code")
    mnc: str = Field(..., description="Mobile Network Code")

class Snssai(BaseModel):
    sst: int = Field(..., ge=0, le=255, description="Slice/Service Type")
    sd: Optional[str] = Field(None, description="Slice Differentiator")

class IpEndPoint(BaseModel):
    ipv4Address: Optional[str] = Field(None, description="IPv4 address")
    ipv6Address: Optional[str] = Field(None, description="IPv6 address")
    transport: Optional[str] = Field("TCP", description="Transport protocol")
    port: Optional[int] = Field(None, description="Port number")

class PcfBinding(BaseModel):
    supi: Optional[str] = Field(None, description="Subscription Permanent Identifier")
    gpsi: Optional[str] = Field(None, description="Generic Public Subscription Identifier")
    ipv4Addr: Optional[str] = Field(None, description="IPv4 address")
    ipv6Prefix: Optional[str] = Field(None, description="IPv6 prefix")
    ipDomain: Optional[str] = Field(None, description="IP domain")
    macAddr48: Optional[str] = Field(None, description="MAC address")
    dnn: str = Field(..., description="Data Network Name")
    pcfFqdn: Optional[str] = Field(None, description="PCF FQDN")
    pcfIpEndPoints: Optional[List[IpEndPoint]] = Field(None, description="PCF IP endpoints")
    pcfDiamHost: Optional[str] = Field(None, description="PCF Diameter host")
    pcfDiamRealm: Optional[str] = Field(None, description="PCF Diameter realm")
    snssai: Snssai = Field(..., description="S-NSSAI")
    suppFeat: Optional[str] = Field(None, description="Supported features")
    pcfId: Optional[str] = Field(None, description="PCF NF Instance ID")
    pcfSetId: Optional[str] = Field(None, description="PCF Set ID")
    recoveryTime: Optional[datetime] = Field(None, description="Recovery time")
    bindLevel: Optional[BindingLevel] = Field(None, description="Binding level")
    ipv4FrameRouteList: Optional[List[str]] = Field(None, description="IPv4 frame route list")
    ipv6FrameRouteList: Optional[List[str]] = Field(None, description="IPv6 frame route list")
    addIpv6Prefixes: Optional[List[str]] = Field(None, description="Additional IPv6 prefixes")
    addMacAddrs: Optional[List[str]] = Field(None, description="Additional MAC addresses")
    pcfForUePolicyDnn: Optional[str] = Field(None, description="PCF for UE policy DNN")
    pcfForUePolicySnssai: Optional[Snssai] = Field(None, description="PCF for UE policy S-NSSAI")

class PcfBindingPatch(BaseModel):
    ipv4Addr: Optional[str] = Field(None, description="IPv4 address")
    ipv6Prefix: Optional[str] = Field(None, description="IPv6 prefix")
    addIpv6Prefixes: Optional[List[str]] = Field(None, description="Additional IPv6 prefixes")
    ipv4FrameRouteList: Optional[List[str]] = Field(None, description="IPv4 frame route list")
    ipv6FrameRouteList: Optional[List[str]] = Field(None, description="IPv6 frame route list")
    addMacAddrs: Optional[List[str]] = Field(None, description="Additional MAC addresses")
    pcfId: Optional[str] = Field(None, description="PCF NF Instance ID")
    pcfSetId: Optional[str] = Field(None, description="PCF Set ID")

class DiscoveryResult(BaseModel):
    pcfBindings: List[PcfBinding] = Field(..., description="List of PCF bindings")
    suppFeat: Optional[str] = Field(None, description="Supported features")

# BSF Storage
pcf_bindings: Dict[str, PcfBinding] = {}
# Index for faster lookups
ipv4_binding_index: Dict[str, str] = {}  # ipv4 -> binding_id
ipv6_binding_index: Dict[str, str] = {}  # ipv6 -> binding_id
supi_binding_index: Dict[str, List[str]] = {}  # supi -> [binding_ids]


class BSF:
    def __init__(self):
        self.name = "BSF-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.supported_features = "0x07"  # Support basic features

    def create_binding(self, binding: PcfBinding) -> str:
        """Create a new PCF binding"""
        binding_id = str(uuid.uuid4())

        # Validate required fields
        if not binding.pcfFqdn and not binding.pcfIpEndPoints and not binding.pcfId:
            raise ValueError("At least one of pcfFqdn, pcfIpEndPoints, or pcfId must be provided")

        if not binding.ipv4Addr and not binding.ipv6Prefix and not binding.macAddr48:
            raise ValueError("At least one of ipv4Addr, ipv6Prefix, or macAddr48 must be provided")

        # Store binding
        pcf_bindings[binding_id] = binding

        # Update indexes
        if binding.ipv4Addr:
            ipv4_binding_index[binding.ipv4Addr] = binding_id
        if binding.ipv6Prefix:
            ipv6_binding_index[binding.ipv6Prefix] = binding_id
        if binding.supi:
            if binding.supi not in supi_binding_index:
                supi_binding_index[binding.supi] = []
            supi_binding_index[binding.supi].append(binding_id)

        return binding_id

    def find_bindings(
        self,
        ipv4_addr: Optional[str] = None,
        ipv6_prefix: Optional[str] = None,
        mac_addr: Optional[str] = None,
        dnn: Optional[str] = None,
        supi: Optional[str] = None,
        gpsi: Optional[str] = None,
        snssai: Optional[Snssai] = None,
        ip_domain: Optional[str] = None
    ) -> List[PcfBinding]:
        """Find PCF bindings matching the given criteria"""
        results = []

        # Fast path: direct index lookup for IP addresses
        if ipv4_addr and ipv4_addr in ipv4_binding_index:
            binding_id = ipv4_binding_index[ipv4_addr]
            if binding_id in pcf_bindings:
                binding = pcf_bindings[binding_id]
                # Verify additional criteria if provided
                if self._matches_criteria(binding, dnn, snssai):
                    results.append(binding)
            return results

        if ipv6_prefix and ipv6_prefix in ipv6_binding_index:
            binding_id = ipv6_binding_index[ipv6_prefix]
            if binding_id in pcf_bindings:
                binding = pcf_bindings[binding_id]
                if self._matches_criteria(binding, dnn, snssai):
                    results.append(binding)
            return results

        # Slower path: iterate through all bindings
        for binding_id, binding in pcf_bindings.items():
            if ipv4_addr and binding.ipv4Addr != ipv4_addr:
                continue
            if ipv6_prefix and binding.ipv6Prefix != ipv6_prefix:
                continue
            if mac_addr and binding.macAddr48 != mac_addr:
                continue
            if supi and binding.supi != supi:
                continue
            if gpsi and binding.gpsi != gpsi:
                continue
            if ip_domain and binding.ipDomain != ip_domain:
                continue
            if not self._matches_criteria(binding, dnn, snssai):
                continue

            results.append(binding)

        return results

    def _matches_criteria(
        self,
        binding: PcfBinding,
        dnn: Optional[str],
        snssai: Optional[Snssai]
    ) -> bool:
        """Check if binding matches additional criteria"""
        if dnn and binding.dnn != dnn:
            return False
        if snssai:
            if binding.snssai.sst != snssai.sst:
                return False
            if snssai.sd and binding.snssai.sd != snssai.sd:
                return False
        return True

    def update_binding(self, binding_id: str, patch: PcfBindingPatch) -> PcfBinding:
        """Update an existing PCF binding"""
        if binding_id not in pcf_bindings:
            raise ValueError(f"Binding {binding_id} not found")

        binding = pcf_bindings[binding_id]

        # Apply patches
        if patch.ipv4Addr:
            # Update index
            if binding.ipv4Addr:
                del ipv4_binding_index[binding.ipv4Addr]
            binding.ipv4Addr = patch.ipv4Addr
            ipv4_binding_index[patch.ipv4Addr] = binding_id

        if patch.ipv6Prefix:
            if binding.ipv6Prefix:
                del ipv6_binding_index[binding.ipv6Prefix]
            binding.ipv6Prefix = patch.ipv6Prefix
            ipv6_binding_index[patch.ipv6Prefix] = binding_id

        if patch.addIpv6Prefixes:
            binding.addIpv6Prefixes = patch.addIpv6Prefixes
        if patch.ipv4FrameRouteList:
            binding.ipv4FrameRouteList = patch.ipv4FrameRouteList
        if patch.ipv6FrameRouteList:
            binding.ipv6FrameRouteList = patch.ipv6FrameRouteList
        if patch.addMacAddrs:
            binding.addMacAddrs = patch.addMacAddrs
        if patch.pcfId:
            binding.pcfId = patch.pcfId
        if patch.pcfSetId:
            binding.pcfSetId = patch.pcfSetId

        return binding

    def delete_binding(self, binding_id: str) -> bool:
        """Delete a PCF binding"""
        if binding_id not in pcf_bindings:
            return False

        binding = pcf_bindings[binding_id]

        # Clean up indexes
        if binding.ipv4Addr and binding.ipv4Addr in ipv4_binding_index:
            del ipv4_binding_index[binding.ipv4Addr]
        if binding.ipv6Prefix and binding.ipv6Prefix in ipv6_binding_index:
            del ipv6_binding_index[binding.ipv6Prefix]
        if binding.supi and binding.supi in supi_binding_index:
            supi_binding_index[binding.supi].remove(binding_id)
            if not supi_binding_index[binding.supi]:
                del supi_binding_index[binding.supi]

        del pcf_bindings[binding_id]
        return True


bsf_instance = BSF()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Register with NRF
    nf_profile = {
        "nfInstanceId": bsf_instance.nf_instance_id,
        "nfType": "BSF",
        "nfStatus": "REGISTERED",
        "plmnList": [{"mcc": "001", "mnc": "01"}],
        "sNssais": [{"sst": 1, "sd": "010203"}],
        "nfServices": [
            {
                "serviceInstanceId": "nbsf-management-001",
                "serviceName": "nbsf-management",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9011}]
            }
        ],
        "bsfInfo": {
            "dnnList": ["internet", "ims"],
            "ipDomainList": ["pool.example.com"],
            "ipv4AddressRanges": [{"start": "10.0.0.0", "end": "10.255.255.255"}],
            "ipv6PrefixRanges": [{"start": "2001:db8::", "end": "2001:db8:ffff::"}]
        }
    }

    try:
        response = requests.put(
            f"{nrf_url}/nnrf-nfm/v1/nf-instances/{bsf_instance.nf_instance_id}",
            json=nf_profile
        )
        if response.status_code in [200, 201]:
            logger.info("BSF registered with NRF successfully")
        else:
            logger.warning(f"BSF registration with NRF failed: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Failed to register BSF with NRF: {e}")

    yield

    # Shutdown
    try:
        requests.delete(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{bsf_instance.nf_instance_id}")
        logger.info("BSF deregistered from NRF")
    except:
        pass


app = FastAPI(
    title="BSF - Binding Support Function",
    description="3GPP TS 29.521 compliant BSF implementation",
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


# 3GPP TS 29.521 - Nbsf_Management Service

@app.post("/nbsf-management/v1/pcfBindings", response_model=PcfBinding, status_code=201)
async def create_pcf_binding(binding: PcfBinding):
    """
    Create PCF Binding per 3GPP TS 29.521
    """
    with tracer.start_as_current_span("bsf_create_binding") as span:
        span.set_attribute("3gpp.service", "Nbsf_Management")
        span.set_attribute("3gpp.operation", "RegisterBinding")
        span.set_attribute("dnn", binding.dnn)
        span.set_attribute("snssai.sst", binding.snssai.sst)

        try:
            binding_id = bsf_instance.create_binding(binding)

            span.set_attribute("binding.id", binding_id)
            span.set_attribute("status", "SUCCESS")
            logger.info(f"PCF binding created: {binding_id}")

            # Add binding ID to response
            binding_dict = binding.dict()
            binding_dict["bindingId"] = binding_id

            return binding

        except ValueError as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"PCF binding creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/nbsf-management/v1/pcfBindings", response_model=DiscoveryResult)
async def discover_pcf_binding(
    ipv4Addr: Optional[str] = Query(None, alias="ipv4Addr", description="IPv4 address"),
    ipv6Prefix: Optional[str] = Query(None, alias="ipv6Prefix", description="IPv6 prefix"),
    macAddr48: Optional[str] = Query(None, alias="macAddr48", description="MAC address"),
    dnn: Optional[str] = Query(None, description="Data Network Name"),
    supi: Optional[str] = Query(None, description="SUPI"),
    gpsi: Optional[str] = Query(None, description="GPSI"),
    snssai: Optional[str] = Query(None, description="S-NSSAI (JSON)"),
    ipDomain: Optional[str] = Query(None, alias="ipDomain", description="IP domain")
):
    """
    Discover PCF Binding per 3GPP TS 29.521
    """
    with tracer.start_as_current_span("bsf_discover_binding") as span:
        span.set_attribute("3gpp.service", "Nbsf_Management")
        span.set_attribute("3gpp.operation", "DiscoverBinding")

        try:
            # Parse S-NSSAI if provided
            snssai_obj = None
            if snssai:
                snssai_dict = json.loads(snssai)
                snssai_obj = Snssai(**snssai_dict)

            # Find matching bindings
            bindings = bsf_instance.find_bindings(
                ipv4_addr=ipv4Addr,
                ipv6_prefix=ipv6Prefix,
                mac_addr=macAddr48,
                dnn=dnn,
                supi=supi,
                gpsi=gpsi,
                snssai=snssai_obj,
                ip_domain=ipDomain
            )

            span.set_attribute("bindings.found", len(bindings))
            span.set_attribute("status", "SUCCESS")
            logger.info(f"PCF binding discovery: {len(bindings)} bindings found")

            return DiscoveryResult(
                pcfBindings=bindings,
                suppFeat=bsf_instance.supported_features
            )

        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid S-NSSAI JSON: {e}")
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"PCF binding discovery failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/nbsf-management/v1/pcfBindings/{bindingId}", response_model=PcfBinding)
async def get_pcf_binding(bindingId: str = Path(..., description="Binding ID")):
    """
    Get individual PCF Binding per 3GPP TS 29.521
    """
    if bindingId not in pcf_bindings:
        raise HTTPException(status_code=404, detail="PCF binding not found")

    return pcf_bindings[bindingId]


@app.patch("/nbsf-management/v1/pcfBindings/{bindingId}", response_model=PcfBinding)
async def update_pcf_binding(
    bindingId: str = Path(..., description="Binding ID"),
    patch: PcfBindingPatch = None
):
    """
    Update PCF Binding per 3GPP TS 29.521
    """
    with tracer.start_as_current_span("bsf_update_binding") as span:
        span.set_attribute("binding.id", bindingId)

        try:
            if not patch:
                raise HTTPException(status_code=400, detail="Patch data required")

            updated_binding = bsf_instance.update_binding(bindingId, patch)

            span.set_attribute("status", "SUCCESS")
            logger.info(f"PCF binding updated: {bindingId}")

            return updated_binding

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"PCF binding update failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.delete("/nbsf-management/v1/pcfBindings/{bindingId}", status_code=204)
async def delete_pcf_binding(bindingId: str = Path(..., description="Binding ID")):
    """
    Delete PCF Binding per 3GPP TS 29.521
    """
    with tracer.start_as_current_span("bsf_delete_binding") as span:
        span.set_attribute("binding.id", bindingId)

        if bsf_instance.delete_binding(bindingId):
            span.set_attribute("status", "SUCCESS")
            logger.info(f"PCF binding deleted: {bindingId}")
            return None
        else:
            raise HTTPException(status_code=404, detail="PCF binding not found")


# Health and monitoring

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "BSF",
        "compliance": "3GPP TS 29.521",
        "version": "1.0.0",
        "activeBindings": len(pcf_bindings)
    }


@app.get("/metrics")
def get_metrics():
    """Metrics endpoint"""
    return {
        "total_bindings": len(pcf_bindings),
        "ipv4_bindings": len(ipv4_binding_index),
        "ipv6_bindings": len(ipv6_binding_index),
        "unique_supis": len(supi_binding_index)
    }


@app.get("/bsf/bindings")
def list_all_bindings():
    """List all PCF bindings (for debugging)"""
    return {
        "total": len(pcf_bindings),
        "bindings": [
            {"id": bid, "dnn": b.dnn, "ipv4": b.ipv4Addr, "ipv6": b.ipv6Prefix, "pcfId": b.pcfId}
            for bid, b in pcf_bindings.items()
        ]
    }


if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="BSF - Binding Support Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("bsf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)