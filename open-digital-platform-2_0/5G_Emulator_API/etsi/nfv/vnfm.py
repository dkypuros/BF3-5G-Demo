# File location: 5G_Emulator_API/etsi/nfv/vnfm.py
# ETSI GS NFV-SOL 003 - VNF Lifecycle Management Interface
# Manages lifecycle of Virtualized Network Functions

from fastapi import FastAPI, HTTPException, Query, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
import uuid
import logging
from datetime import datetime, timezone
import os
import asyncio
from opentelemetry import trace
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

# =============================================================================
# ETSI GS NFV-SOL 003 Data Models
# =============================================================================

class VnfOperationalStateType(str, Enum):
    """ETSI GS NFV-SOL 003 - VNF Operational State"""
    STARTED = "STARTED"
    STOPPED = "STOPPED"

class VnfState(str, Enum):
    """ETSI GS NFV-SOL 003 - VNF Instantiation State"""
    NOT_INSTANTIATED = "NOT_INSTANTIATED"
    INSTANTIATED = "INSTANTIATED"

class LcmOperationType(str, Enum):
    """ETSI GS NFV-SOL 003 - LCM Operation Types"""
    INSTANTIATE = "INSTANTIATE"
    SCALE = "SCALE"
    SCALE_TO_LEVEL = "SCALE_TO_LEVEL"
    CHANGE_FLAVOUR = "CHANGE_FLAVOUR"
    TERMINATE = "TERMINATE"
    HEAL = "HEAL"
    OPERATE = "OPERATE"
    MODIFY_INFO = "MODIFY_INFO"

class LcmOperationStateType(str, Enum):
    """ETSI GS NFV-SOL 003 - LCM Operation State"""
    STARTING = "STARTING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED_TEMP = "FAILED_TEMP"
    FAILED = "FAILED"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"

class VnfInstance(BaseModel):
    """ETSI GS NFV-SOL 003 - VnfInstance (Section 5.5.2.2)"""
    id: str = Field(..., description="VNF instance ID")
    vnfInstanceName: Optional[str] = Field(None, description="VNF instance name")
    vnfInstanceDescription: Optional[str] = Field(None, description="Description")
    vnfdId: str = Field(..., description="VNFD identifier")
    vnfProvider: str = Field(..., description="VNF provider")
    vnfProductName: str = Field(..., description="VNF product name")
    vnfSoftwareVersion: str = Field(..., description="Software version")
    vnfdVersion: str = Field(..., description="VNFD version")
    instantiationState: VnfState = Field(VnfState.NOT_INSTANTIATED, description="Instantiation state")
    instantiatedVnfInfo: Optional[Dict[str, Any]] = Field(None, description="Instantiated VNF info")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")
    _links: Optional[Dict[str, Any]] = Field(None, description="HATEOAS links")

class InstantiateVnfRequest(BaseModel):
    """ETSI GS NFV-SOL 003 - InstantiateVnfRequest (Section 5.5.2.4)"""
    flavourId: str = Field(..., description="Deployment flavour ID")
    instantiationLevelId: Optional[str] = Field(None, description="Instantiation level")
    extVirtualLinks: Optional[List[Dict[str, Any]]] = Field(None, description="External VLs")
    extManagedVirtualLinks: Optional[List[Dict[str, Any]]] = Field(None, description="External managed VLs")
    vimConnectionInfo: Optional[Dict[str, Any]] = Field(None, description="VIM connection info")
    localizationLanguage: Optional[str] = Field(None, description="Localization")
    additionalParams: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class TerminateVnfRequest(BaseModel):
    """ETSI GS NFV-SOL 003 - TerminateVnfRequest (Section 5.5.2.10)"""
    terminationType: str = Field("GRACEFUL", description="GRACEFUL or FORCEFUL")
    gracefulTerminationTimeout: Optional[int] = Field(None, description="Timeout in seconds")
    additionalParams: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class ScaleVnfRequest(BaseModel):
    """ETSI GS NFV-SOL 003 - ScaleVnfRequest (Section 5.5.2.5)"""
    type: str = Field(..., description="SCALE_OUT or SCALE_IN")
    aspectId: str = Field(..., description="Scaling aspect ID")
    numberOfSteps: Optional[int] = Field(1, ge=1, description="Number of scaling steps")
    additionalParams: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class VnfLcmOpOcc(BaseModel):
    """ETSI GS NFV-SOL 003 - VnfLcmOpOcc (Section 5.5.2.13)

    Represents a VNF lifecycle management operation occurrence.
    """
    id: str = Field(..., description="Operation occurrence ID")
    operationState: LcmOperationStateType = Field(..., description="Operation state")
    stateEnteredTime: datetime = Field(..., description="State entry time")
    startTime: datetime = Field(..., description="Operation start time")
    vnfInstanceId: str = Field(..., description="VNF instance ID")
    grantId: Optional[str] = Field(None, description="Grant ID")
    operation: LcmOperationType = Field(..., description="Operation type")
    isAutomaticInvocation: bool = Field(False, description="Auto invocation flag")
    operationParams: Optional[Dict[str, Any]] = Field(None, description="Operation parameters")
    isCancelPending: bool = Field(False, description="Cancel pending flag")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details")
    _links: Optional[Dict[str, Any]] = Field(None, description="HATEOAS links")

class CreateVnfRequest(BaseModel):
    """ETSI GS NFV-SOL 003 - CreateVnfRequest (Section 5.5.2.3)"""
    vnfdId: str = Field(..., description="VNFD identifier")
    vnfInstanceName: Optional[str] = Field(None, description="VNF instance name")
    vnfInstanceDescription: Optional[str] = Field(None, description="Description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")

# =============================================================================
# VNFM Core Class
# =============================================================================

class VNFM:
    """ETSI GS NFV-SOL 003 - VNF Manager

    Manages lifecycle of VNFs representing 5G Core network functions.
    Each NF (AMF, SMF, UPF, etc.) can be managed as a VNF.
    """

    def __init__(self):
        self.vnfm_id = str(uuid.uuid4())

        # VNF instances registry
        self.vnf_instances: Dict[str, VnfInstance] = {}

        # LCM operation occurrences
        self.lcm_op_occs: Dict[str, VnfLcmOpOcc] = {}

        # VNFD catalog (simplified)
        self.vnfd_catalog = {
            "vnfd-amf-v1": {"provider": "5G-Lab", "product": "AMF", "version": "1.0.0"},
            "vnfd-smf-v1": {"provider": "5G-Lab", "product": "SMF", "version": "1.0.0"},
            "vnfd-upf-v1": {"provider": "5G-Lab", "product": "UPF", "version": "1.0.0"},
            "vnfd-nrf-v1": {"provider": "5G-Lab", "product": "NRF", "version": "1.0.0"},
            "vnfd-ausf-v1": {"provider": "5G-Lab", "product": "AUSF", "version": "1.0.0"},
            "vnfd-udm-v1": {"provider": "5G-Lab", "product": "UDM", "version": "1.0.0"},
            "vnfd-pcf-v1": {"provider": "5G-Lab", "product": "PCF", "version": "1.0.0"},
        }

        logger.info(f"VNFM initialized: {self.vnfm_id}")

    def create_vnf_instance(self, request: CreateVnfRequest) -> VnfInstance:
        """ETSI GS NFV-SOL 003 - Create VNF Instance (Section 5.4.2)

        Creates a VNF instance resource in NOT_INSTANTIATED state.
        """
        with tracer.start_as_current_span("vnfm_create_instance") as span:
            span.set_attribute("etsi.spec", "GS NFV-SOL 003")
            span.set_attribute("etsi.operation", "CreateVnfInstance")

            vnfd = self.vnfd_catalog.get(request.vnfdId, {})
            if not vnfd:
                raise ValueError(f"VNFD not found: {request.vnfdId}")

            instance_id = str(uuid.uuid4())
            instance = VnfInstance(
                id=instance_id,
                vnfInstanceName=request.vnfInstanceName or f"vnf-{instance_id[:8]}",
                vnfInstanceDescription=request.vnfInstanceDescription,
                vnfdId=request.vnfdId,
                vnfProvider=vnfd.get("provider", "Unknown"),
                vnfProductName=vnfd.get("product", "Unknown"),
                vnfSoftwareVersion=vnfd.get("version", "1.0.0"),
                vnfdVersion="1.0.0",
                instantiationState=VnfState.NOT_INSTANTIATED,
                metadata=request.metadata,
                _links={
                    "self": {"href": f"/vnflcm/v1/vnf_instances/{instance_id}"},
                    "instantiate": {"href": f"/vnflcm/v1/vnf_instances/{instance_id}/instantiate"}
                }
            )

            self.vnf_instances[instance_id] = instance
            span.set_attribute("vnf.instance_id", instance_id)
            logger.info(f"Created VNF instance: {instance_id} ({instance.vnfProductName})")

            return instance

    async def instantiate_vnf(self, instance_id: str, request: InstantiateVnfRequest) -> VnfLcmOpOcc:
        """ETSI GS NFV-SOL 003 - Instantiate VNF (Section 5.4.4)

        Instantiates a VNF - starts the actual NF.
        """
        with tracer.start_as_current_span("vnfm_instantiate") as span:
            span.set_attribute("etsi.spec", "GS NFV-SOL 003")
            span.set_attribute("etsi.operation", "InstantiateVnf")

            instance = self.vnf_instances.get(instance_id)
            if not instance:
                raise ValueError(f"VNF instance not found: {instance_id}")

            if instance.instantiationState != VnfState.NOT_INSTANTIATED:
                raise ValueError(f"VNF already instantiated: {instance_id}")

            # Create LCM operation occurrence
            op_occ_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            op_occ = VnfLcmOpOcc(
                id=op_occ_id,
                operationState=LcmOperationStateType.STARTING,
                stateEnteredTime=now,
                startTime=now,
                vnfInstanceId=instance_id,
                operation=LcmOperationType.INSTANTIATE,
                operationParams=request.dict()
            )
            self.lcm_op_occs[op_occ_id] = op_occ

            # Simulate async instantiation
            asyncio.create_task(self._do_instantiate(instance_id, op_occ_id, request))

            span.set_attribute("vnf.instance_id", instance_id)
            span.set_attribute("lcm.op_occ_id", op_occ_id)

            return op_occ

    async def _do_instantiate(self, instance_id: str, op_occ_id: str, request: InstantiateVnfRequest):
        """Background task to perform instantiation"""
        try:
            op_occ = self.lcm_op_occs[op_occ_id]
            op_occ.operationState = LcmOperationStateType.PROCESSING
            op_occ.stateEnteredTime = datetime.now(timezone.utc)

            # Simulate instantiation time
            await asyncio.sleep(2)

            # Update instance state
            instance = self.vnf_instances[instance_id]
            instance.instantiationState = VnfState.INSTANTIATED
            instance.instantiatedVnfInfo = {
                "flavourId": request.flavourId,
                "vnfState": VnfOperationalStateType.STARTED.value,
                "scaleStatus": [],
                "extCpInfo": [],
                "vnfcResourceInfo": [{
                    "id": str(uuid.uuid4()),
                    "vduId": f"vdu-{instance.vnfProductName.lower()}",
                    "computeResource": {"resourceId": str(uuid.uuid4())}
                }]
            }

            # Complete operation
            op_occ.operationState = LcmOperationStateType.COMPLETED
            op_occ.stateEnteredTime = datetime.now(timezone.utc)

            logger.info(f"VNF instantiated: {instance_id}")

        except Exception as e:
            op_occ = self.lcm_op_occs[op_occ_id]
            op_occ.operationState = LcmOperationStateType.FAILED
            op_occ.stateEnteredTime = datetime.now(timezone.utc)
            op_occ.error = {"detail": str(e)}
            logger.error(f"VNF instantiation failed: {instance_id} - {e}")

    async def terminate_vnf(self, instance_id: str, request: TerminateVnfRequest) -> VnfLcmOpOcc:
        """ETSI GS NFV-SOL 003 - Terminate VNF (Section 5.4.8)"""
        with tracer.start_as_current_span("vnfm_terminate") as span:
            span.set_attribute("etsi.spec", "GS NFV-SOL 003")
            span.set_attribute("etsi.operation", "TerminateVnf")

            instance = self.vnf_instances.get(instance_id)
            if not instance:
                raise ValueError(f"VNF instance not found: {instance_id}")

            # Create LCM operation occurrence
            op_occ_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            op_occ = VnfLcmOpOcc(
                id=op_occ_id,
                operationState=LcmOperationStateType.PROCESSING,
                stateEnteredTime=now,
                startTime=now,
                vnfInstanceId=instance_id,
                operation=LcmOperationType.TERMINATE,
                operationParams=request.dict()
            )
            self.lcm_op_occs[op_occ_id] = op_occ

            # Simulate termination
            asyncio.create_task(self._do_terminate(instance_id, op_occ_id, request))

            return op_occ

    async def _do_terminate(self, instance_id: str, op_occ_id: str, request: TerminateVnfRequest):
        """Background task to perform termination"""
        try:
            if request.terminationType == "GRACEFUL" and request.gracefulTerminationTimeout:
                await asyncio.sleep(min(request.gracefulTerminationTimeout, 5))
            else:
                await asyncio.sleep(1)

            instance = self.vnf_instances[instance_id]
            instance.instantiationState = VnfState.NOT_INSTANTIATED
            instance.instantiatedVnfInfo = None

            op_occ = self.lcm_op_occs[op_occ_id]
            op_occ.operationState = LcmOperationStateType.COMPLETED
            op_occ.stateEnteredTime = datetime.now(timezone.utc)

            logger.info(f"VNF terminated: {instance_id}")

        except Exception as e:
            op_occ = self.lcm_op_occs[op_occ_id]
            op_occ.operationState = LcmOperationStateType.FAILED
            op_occ.error = {"detail": str(e)}

    def delete_vnf_instance(self, instance_id: str) -> bool:
        """ETSI GS NFV-SOL 003 - Delete VNF Instance (Section 5.4.3)"""
        instance = self.vnf_instances.get(instance_id)
        if not instance:
            return False

        if instance.instantiationState == VnfState.INSTANTIATED:
            raise ValueError("Cannot delete instantiated VNF - terminate first")

        del self.vnf_instances[instance_id]
        logger.info(f"Deleted VNF instance: {instance_id}")
        return True

    def get_vnf_instance(self, instance_id: str) -> Optional[VnfInstance]:
        """Get VNF instance by ID"""
        return self.vnf_instances.get(instance_id)

    def get_all_vnf_instances(self) -> List[VnfInstance]:
        """Get all VNF instances"""
        return list(self.vnf_instances.values())

    def get_lcm_op_occ(self, op_occ_id: str) -> Optional[VnfLcmOpOcc]:
        """Get LCM operation occurrence"""
        return self.lcm_op_occs.get(op_occ_id)

    def get_all_lcm_op_occs(self, vnf_instance_id: str = None) -> List[VnfLcmOpOcc]:
        """Get all LCM operation occurrences"""
        occs = list(self.lcm_op_occs.values())
        if vnf_instance_id:
            occs = [o for o in occs if o.vnfInstanceId == vnf_instance_id]
        return occs


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ETSI NFV-SOL 003 VNFM API",
    description="VNF Lifecycle Management implementing ETSI GS NFV-SOL 003",
    version="1.4.1",
    docs_url="/vnflcm/docs",
    openapi_url="/vnflcm/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vnfm = VNFM()


# -----------------------------------------------------------------------------
# VNF Instance Endpoints
# -----------------------------------------------------------------------------

@app.post("/vnflcm/v1/vnf_instances",
          response_model=VnfInstance,
          status_code=201,
          tags=["VNF Instances"])
async def create_vnf_instance(request: CreateVnfRequest):
    """ETSI GS NFV-SOL 003 - Create VNF Instance

    Spec: Section 5.4.2
    """
    try:
        return vnfm.create_vnf_instance(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/vnflcm/v1/vnf_instances",
         response_model=List[VnfInstance],
         tags=["VNF Instances"])
async def get_vnf_instances():
    """ETSI GS NFV-SOL 003 - Query VNF Instances

    Spec: Section 5.4.2
    """
    return vnfm.get_all_vnf_instances()


@app.get("/vnflcm/v1/vnf_instances/{vnfInstanceId}",
         response_model=VnfInstance,
         tags=["VNF Instances"])
async def get_vnf_instance(vnfInstanceId: str = Path(..., description="VNF Instance ID")):
    """ETSI GS NFV-SOL 003 - Read VNF Instance

    Spec: Section 5.4.3
    """
    instance = vnfm.get_vnf_instance(vnfInstanceId)
    if not instance:
        raise HTTPException(status_code=404, detail="VNF instance not found")
    return instance


@app.delete("/vnflcm/v1/vnf_instances/{vnfInstanceId}",
            status_code=204,
            tags=["VNF Instances"])
async def delete_vnf_instance(vnfInstanceId: str = Path(..., description="VNF Instance ID")):
    """ETSI GS NFV-SOL 003 - Delete VNF Instance

    Spec: Section 5.4.3
    """
    try:
        if not vnfm.delete_vnf_instance(vnfInstanceId):
            raise HTTPException(status_code=404, detail="VNF instance not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


# -----------------------------------------------------------------------------
# VNF Lifecycle Operations
# -----------------------------------------------------------------------------

@app.post("/vnflcm/v1/vnf_instances/{vnfInstanceId}/instantiate",
          status_code=202,
          tags=["VNF Lifecycle"])
async def instantiate_vnf(
    vnfInstanceId: str = Path(..., description="VNF Instance ID"),
    request: InstantiateVnfRequest = ...
):
    """ETSI GS NFV-SOL 003 - Instantiate VNF

    Spec: Section 5.4.4
    """
    try:
        op_occ = await vnfm.instantiate_vnf(vnfInstanceId, request)
        return {"id": op_occ.id, "_links": {"self": {"href": f"/vnflcm/v1/vnf_lcm_op_occs/{op_occ.id}"}}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/vnflcm/v1/vnf_instances/{vnfInstanceId}/terminate",
          status_code=202,
          tags=["VNF Lifecycle"])
async def terminate_vnf(
    vnfInstanceId: str = Path(..., description="VNF Instance ID"),
    request: TerminateVnfRequest = ...
):
    """ETSI GS NFV-SOL 003 - Terminate VNF

    Spec: Section 5.4.8
    """
    try:
        op_occ = await vnfm.terminate_vnf(vnfInstanceId, request)
        return {"id": op_occ.id, "_links": {"self": {"href": f"/vnflcm/v1/vnf_lcm_op_occs/{op_occ.id}"}}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------------------------------------------------------
# LCM Operation Occurrences
# -----------------------------------------------------------------------------

@app.get("/vnflcm/v1/vnf_lcm_op_occs",
         response_model=List[VnfLcmOpOcc],
         tags=["LCM Operations"])
async def get_lcm_op_occs(
    vnfInstanceId: Optional[str] = Query(None, description="Filter by VNF instance")
):
    """ETSI GS NFV-SOL 003 - Query LCM Operation Occurrences

    Spec: Section 5.4.12
    """
    return vnfm.get_all_lcm_op_occs(vnfInstanceId)


@app.get("/vnflcm/v1/vnf_lcm_op_occs/{vnfLcmOpOccId}",
         response_model=VnfLcmOpOcc,
         tags=["LCM Operations"])
async def get_lcm_op_occ(vnfLcmOpOccId: str = Path(..., description="LCM Operation Occurrence ID")):
    """ETSI GS NFV-SOL 003 - Read LCM Operation Occurrence

    Spec: Section 5.4.12
    """
    op_occ = vnfm.get_lcm_op_occ(vnfLcmOpOccId)
    if not op_occ:
        raise HTTPException(status_code=404, detail="LCM operation occurrence not found")
    return op_occ


# -----------------------------------------------------------------------------
# Health and Metrics
# -----------------------------------------------------------------------------

@app.get("/health", tags=["Operations"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vnfm_id": vnfm.vnfm_id,
        "vnf_instances": len(vnfm.vnf_instances),
        "lcm_op_occs": len(vnfm.lcm_op_occs),
        "vnfd_catalog": len(vnfm.vnfd_catalog)
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("VNFM_PORT", 8093))
    logger.info(f"Starting VNFM on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
