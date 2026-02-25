# File location: 5G_Emulator_API/etsi/o2/models.py
# O-RAN O2 Interface Data Models
# Based on O-RAN.WG6.O2IMS-INTERFACE and O-RAN.WG6.O2DMS-INTERFACE

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import uuid


# =============================================================================
# Enums
# =============================================================================

class ResourceKind(str, Enum):
    """O2 IMS Resource Kind"""
    UNDEFINED = "UNDEFINED"
    PHYSICAL = "PHYSICAL"
    LOGICAL = "LOGICAL"


class ResourceClass(str, Enum):
    """O2 IMS Resource Class"""
    UNDEFINED = "UNDEFINED"
    COMPUTE = "COMPUTE"
    NETWORKING = "NETWORKING"
    STORAGE = "STORAGE"


class PerceivedSeverity(int, Enum):
    """O2 IMS Alarm Severity (per O-RAN spec)"""
    CRITICAL = 0
    MAJOR = 1
    MINOR = 2
    WARNING = 3
    INDETERMINATE = 4
    CLEARED = 5


class AlarmNotificationType(int, Enum):
    """O2 IMS Alarm Notification Type"""
    NEW = 0
    CHANGE = 1
    CLEAR = 2
    ACKNOWLEDGE = 3


class DeploymentState(str, Enum):
    """O2 DMS Deployment State"""
    PENDING = "PENDING"
    DEPLOYING = "DEPLOYING"
    RUNNING = "RUNNING"
    SCALING = "SCALING"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"
    FAILED = "FAILED"


class OperationType(str, Enum):
    """O2 DMS Operation Type"""
    INSTANTIATE = "INSTANTIATE"
    SCALE = "SCALE"
    TERMINATE = "TERMINATE"
    MODIFY = "MODIFY"


class OperationState(str, Enum):
    """O2 DMS Operation State"""
    STARTING = "STARTING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


class NodeStatus(str, Enum):
    """Fake Cluster Node Status"""
    AVAILABLE = "AVAILABLE"
    IN_USE = "IN_USE"
    MAINTENANCE = "MAINTENANCE"
    FAILED = "FAILED"


# =============================================================================
# O2 IMS Infrastructure Inventory Models
# =============================================================================

class OCloudInfo(BaseModel):
    """O-RAN O2 IMS - O-Cloud Information"""
    oCloudId: str = Field(..., description="Internal O-Cloud identifier")
    globalCloudId: str = Field(..., description="SMO-assigned global identifier")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Description")
    serviceUri: str = Field(..., description="Root URI to all services")
    extensions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Vendor extensions")


class DeploymentManager(BaseModel):
    """O-RAN O2 IMS - Deployment Manager reference"""
    deploymentManagerId: str = Field(..., description="Server-allocated ID")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Description")
    oCloudId: str = Field(..., description="Parent O-Cloud ID")
    serviceUri: str = Field(..., description="DMS API endpoint")
    supportedLocations: Optional[List[str]] = Field(default_factory=list, description="Supported location IDs")
    capabilities: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Capabilities")
    capacity: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Available capacity")
    extensions: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ResourceType(BaseModel):
    """O-RAN O2 IMS - Resource Type definition"""
    resourceTypeId: str = Field(..., description="Server-allocated ID")
    name: str = Field(..., description="Resource type name")
    description: Optional[str] = Field(None, description="Description")
    vendor: Optional[str] = Field(None, description="Vendor name")
    model: Optional[str] = Field(None, description="Model")
    version: Optional[str] = Field(None, description="Version")
    resourceKind: ResourceKind = Field(ResourceKind.UNDEFINED, description="PHYSICAL/LOGICAL")
    resourceClass: ResourceClass = Field(ResourceClass.UNDEFINED, description="COMPUTE/STORAGE/NETWORKING")
    extensions: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ResourcePool(BaseModel):
    """O-RAN O2 IMS - Resource Pool"""
    resourcePoolId: str = Field(..., description="Server-allocated ID")
    name: str = Field(..., description="Pool name")
    description: Optional[str] = Field(None, description="Description")
    oCloudId: str = Field(..., description="Parent O-Cloud ID")
    globalLocationId: Optional[str] = Field(None, description="SMO-assigned location ID")
    location: Optional[str] = Field(None, description="Geographic location")
    extensions: Optional[Dict[str, Any]] = Field(default_factory=dict)


class Resource(BaseModel):
    """O-RAN O2 IMS - Individual Resource"""
    resourceId: str = Field(..., description="Server-allocated ID")
    resourcePoolId: str = Field(..., description="Parent pool ID")
    description: Optional[str] = Field(None, description="Description")
    resourceTypeId: str = Field(..., description="Type reference")
    globalAssetId: Optional[str] = Field(None, description="Serial number/identifier")
    elements: Optional[List["Resource"]] = Field(default_factory=list, description="Nested resources")
    tags: Optional[List[str]] = Field(default_factory=list, description="Classification tags")
    groups: Optional[List[str]] = Field(default_factory=list, description="Group membership")
    extensions: Optional[Dict[str, Any]] = Field(default_factory=dict)


# =============================================================================
# O2 IMS Alarm/Monitoring Models
# =============================================================================

class AlarmEventRecord(BaseModel):
    """O-RAN O2 IMS - Alarm Event Record"""
    alarmEventRecordId: str = Field(..., description="Alarm record ID")
    alarmDefinitionId: Optional[str] = Field(None, description="Alarm definition reference")
    probableCauseId: Optional[str] = Field(None, description="Probable cause reference")
    alarmRaisedTime: datetime = Field(..., description="When alarm was raised")
    alarmChangedTime: Optional[datetime] = Field(None, description="Last modification time")
    alarmClearedTime: Optional[datetime] = Field(None, description="When alarm was cleared")
    alarmAcknowledgedTime: Optional[datetime] = Field(None, description="When acknowledged")
    alarmAcknowledged: bool = Field(False, description="Acknowledgment status")
    resourceTypeId: Optional[str] = Field(None, description="Type of affected resource")
    resourceId: Optional[str] = Field(None, description="Affected resource ID")
    perceivedSeverity: PerceivedSeverity = Field(..., description="Severity level")
    alarmMessage: Optional[str] = Field(None, description="Human-readable message")
    extensions: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AlarmSubscriptionInfo(BaseModel):
    """O-RAN O2 IMS - Alarm Subscription"""
    alarmSubscriptionId: Optional[str] = Field(None, description="Server-allocated ID")
    consumerSubscriptionId: Optional[str] = Field(None, description="Consumer-provided ID")
    filter: Optional[str] = Field(None, description="Event type filter (NEW|CHANGE|CLEAR|ACKNOWLEDGE)")
    callback: str = Field(..., description="Webhook URL for notifications")


class InventorySubscription(BaseModel):
    """O-RAN O2 IMS - Inventory Change Subscription"""
    subscriptionId: Optional[str] = Field(None, description="Server-allocated ID")
    consumerSubscriptionId: Optional[str] = Field(None, description="Consumer-provided ID")
    filter: Optional[str] = Field(None, description="Filter criteria")
    callback: str = Field(..., description="Webhook URL for notifications")


# =============================================================================
# O2 DMS Deployment Models
# =============================================================================

class NfDeploymentDescriptor(BaseModel):
    """O-RAN O2 DMS - NF Deployment Descriptor"""
    descriptorId: str = Field(..., description="Descriptor ID")
    name: str = Field(..., description="NF name")
    version: str = Field(..., description="NF version")
    provider: Optional[str] = Field(None, description="NF provider")
    description: Optional[str] = Field(None, description="Description")
    nfType: str = Field(..., description="NF type (CU, DU, AMF, etc.)")
    requiredCpu: int = Field(1, ge=1, description="Required vCPUs")
    requiredMemoryGb: int = Field(1, ge=1, description="Required memory in GB")
    requiredStorageGb: int = Field(1, ge=1, description="Required storage in GB")
    containerImage: Optional[str] = Field(None, description="Container image reference")
    extensions: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DeploymentRequest(BaseModel):
    """O-RAN O2 DMS - Deployment Request"""
    descriptorId: str = Field(..., description="NF descriptor to deploy")
    name: str = Field(..., description="Deployment name")
    replicas: int = Field(1, ge=1, description="Number of replicas")
    resourcePoolId: Optional[str] = Field(None, description="Target resource pool")
    additionalParams: Optional[Dict[str, Any]] = Field(default_factory=dict)


class Deployment(BaseModel):
    """O-RAN O2 DMS - Deployment Instance"""
    deploymentId: str = Field(..., description="Server-allocated ID")
    name: str = Field(..., description="Deployment name")
    descriptorId: str = Field(..., description="NF descriptor reference")
    state: DeploymentState = Field(DeploymentState.PENDING, description="Current state")
    replicas: int = Field(1, description="Number of replicas")
    readyReplicas: int = Field(0, description="Ready replicas")
    resourcePoolId: Optional[str] = Field(None, description="Resource pool")
    allocatedResources: Optional[Dict[str, Any]] = Field(default_factory=dict)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(), description="Creation time")
    updatedAt: Optional[datetime] = Field(None, description="Last update time")
    extensions: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DeploymentOperation(BaseModel):
    """O-RAN O2 DMS - Deployment Operation Record"""
    operationId: str = Field(..., description="Operation ID")
    deploymentId: str = Field(..., description="Target deployment")
    operationType: OperationType = Field(..., description="Operation type")
    state: OperationState = Field(OperationState.STARTING, description="Operation state")
    startTime: datetime = Field(default_factory=lambda: datetime.now(), description="Start time")
    endTime: Optional[datetime] = Field(None, description="End time")
    errorMessage: Optional[str] = Field(None, description="Error details if failed")
    extensions: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ScaleRequest(BaseModel):
    """O-RAN O2 DMS - Scale Request"""
    replicas: int = Field(..., ge=1, description="Target number of replicas")
    additionalParams: Optional[Dict[str, Any]] = Field(default_factory=dict)


# =============================================================================
# SMO Models
# =============================================================================

class Intent(BaseModel):
    """SMO Intent for orchestration"""
    intentId: Optional[str] = Field(None, description="Server-allocated ID")
    name: str = Field(..., description="Intent name")
    objective: str = Field(..., description="Intent objective (e.g., deploy_nf, scale_nf)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Intent parameters")
    state: str = Field("PENDING", description="Intent state")
    createdAt: Optional[datetime] = Field(None, description="Creation time")
    completedAt: Optional[datetime] = Field(None, description="Completion time")
    result: Optional[Dict[str, Any]] = Field(None, description="Intent result")


class DeployNfRequest(BaseModel):
    """SMO High-level NF Deployment Request"""
    nfType: str = Field(..., description="NF type (CU, DU, AMF, etc.)")
    name: str = Field(..., description="Deployment name")
    replicas: int = Field(1, ge=1, description="Number of replicas")
    qosPolicy: Optional[Dict[str, Any]] = Field(None, description="QoS policy to apply via RIC")
    additionalParams: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ComponentStatus(BaseModel):
    """SMO Component Status"""
    componentType: str = Field(..., description="Component type")
    name: str = Field(..., description="Component name")
    url: str = Field(..., description="Component URL")
    status: str = Field("UNKNOWN", description="Health status")
    lastChecked: Optional[datetime] = Field(None, description="Last health check time")


# Enable forward references
Resource.model_rebuild()
