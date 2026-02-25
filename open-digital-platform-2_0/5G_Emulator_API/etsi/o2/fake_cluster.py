# File location: 5G_Emulator_API/etsi/o2/fake_cluster.py
# Simulated O-Cloud Cluster for Testing
# Provides in-memory infrastructure state without requiring real Kubernetes/OpenShift

from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import logging
from threading import Lock

from .models import (
    OCloudInfo,
    DeploymentManager,
    ResourceType,
    ResourcePool,
    Resource,
    AlarmEventRecord,
    NfDeploymentDescriptor,
    Deployment,
    DeploymentState,
    DeploymentOperation,
    OperationType,
    OperationState,
    ResourceKind,
    ResourceClass,
    PerceivedSeverity,
    NodeStatus,
)

logger = logging.getLogger(__name__)


class FakeCluster:
    """
    Simulates an O-Cloud infrastructure with in-memory state.

    Provides:
    - 3 compute nodes (8 vCPU, 32GB RAM each)
    - Resource pools (compute, storage, network)
    - Deployment tracking
    - Alarm management
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern for shared cluster state"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Initialize all state
        self._init_ocloud()
        self._init_resource_types()
        self._init_nodes()
        self._init_resource_pools()
        self._init_deployment_manager()
        self._init_nf_descriptors()

        # Runtime state
        self.deployments: Dict[str, Deployment] = {}
        self.operations: Dict[str, DeploymentOperation] = {}
        self.alarms: Dict[str, AlarmEventRecord] = {}
        self.alarm_subscriptions: Dict[str, Dict[str, Any]] = {}
        self.inventory_subscriptions: Dict[str, Dict[str, Any]] = {}

        logger.info("FakeCluster initialized with 3 nodes, 24 vCPU, 96GB RAM total")

    def _init_ocloud(self):
        """Initialize O-Cloud info"""
        self.ocloud = OCloudInfo(
            oCloudId="ocloud-001",
            globalCloudId="homelab-drcoopertbbt-001",
            name="HomeLabCloud",
            description="Simulated O-Cloud for O-RAN testing",
            serviceUri="http://127.0.0.1:8098",
            extensions={
                "vendor": "HomeLabCloud",
                "version": "1.0.0",
                "region": "us-west-2"
            }
        )

    def _init_resource_types(self):
        """Initialize resource types"""
        self.resource_types: Dict[str, ResourceType] = {
            "compute-node": ResourceType(
                resourceTypeId="compute-node",
                name="Compute Node",
                description="x86_64 compute node",
                vendor="Generic",
                model="VM-Standard",
                version="1.0",
                resourceKind=ResourceKind.LOGICAL,
                resourceClass=ResourceClass.COMPUTE
            ),
            "storage-volume": ResourceType(
                resourceTypeId="storage-volume",
                name="Storage Volume",
                description="Block storage volume",
                vendor="Generic",
                model="SSD-Standard",
                version="1.0",
                resourceKind=ResourceKind.LOGICAL,
                resourceClass=ResourceClass.STORAGE
            ),
            "network-interface": ResourceType(
                resourceTypeId="network-interface",
                name="Network Interface",
                description="Virtual network interface",
                vendor="Generic",
                model="VirtIO",
                version="1.0",
                resourceKind=ResourceKind.LOGICAL,
                resourceClass=ResourceClass.NETWORKING
            )
        }

    def _init_nodes(self):
        """Initialize fake compute nodes"""
        self.nodes: Dict[str, Dict[str, Any]] = {
            "node-001": {
                "id": "node-001",
                "name": "compute-1",
                "cpu": 8,
                "ram_gb": 32,
                "storage_gb": 100,
                "status": NodeStatus.AVAILABLE,
                "allocated_cpu": 0,
                "allocated_ram_gb": 0,
                "deployments": []
            },
            "node-002": {
                "id": "node-002",
                "name": "compute-2",
                "cpu": 8,
                "ram_gb": 32,
                "storage_gb": 100,
                "status": NodeStatus.AVAILABLE,
                "allocated_cpu": 0,
                "allocated_ram_gb": 0,
                "deployments": []
            },
            "node-003": {
                "id": "node-003",
                "name": "compute-3",
                "cpu": 8,
                "ram_gb": 32,
                "storage_gb": 100,
                "status": NodeStatus.AVAILABLE,
                "allocated_cpu": 0,
                "allocated_ram_gb": 0,
                "deployments": []
            }
        }

    def _init_resource_pools(self):
        """Initialize resource pools"""
        self.resource_pools: Dict[str, ResourcePool] = {
            "compute-pool": ResourcePool(
                resourcePoolId="compute-pool",
                name="Compute Pool",
                description="Pool of compute nodes",
                oCloudId="ocloud-001",
                globalLocationId="loc-us-west-2",
                location="US West",
                extensions={
                    "total_cpu": 24,
                    "total_ram_gb": 96,
                    "node_count": 3
                }
            ),
            "storage-pool": ResourcePool(
                resourcePoolId="storage-pool",
                name="Storage Pool",
                description="Block storage pool",
                oCloudId="ocloud-001",
                globalLocationId="loc-us-west-2",
                location="US West",
                extensions={
                    "total_capacity_gb": 500,
                    "available_capacity_gb": 500,
                    "storage_type": "SSD"
                }
            ),
            "network-pool": ResourcePool(
                resourcePoolId="network-pool",
                name="Network Pool",
                description="Network resources",
                oCloudId="ocloud-001",
                globalLocationId="loc-us-west-2",
                location="US West",
                extensions={
                    "cidr": "10.200.0.0/16",
                    "gateway": "10.200.0.1",
                    "dns": "10.96.0.10"
                }
            )
        }

    def _init_deployment_manager(self):
        """Initialize deployment manager reference"""
        self.deployment_manager = DeploymentManager(
            deploymentManagerId="dms-001",
            name="O2-DMS",
            description="O2 Deployment Management Service",
            oCloudId="ocloud-001",
            serviceUri="http://127.0.0.1:8099",
            supportedLocations=["loc-us-west-2"],
            capabilities={
                "kubernetes": True,
                "helm": True,
                "nf_types": ["CU", "DU", "AMF", "SMF", "UPF"]
            },
            capacity={
                "max_deployments": 100,
                "current_deployments": 0
            }
        )

    def _init_nf_descriptors(self):
        """Initialize pre-defined NF deployment descriptors"""
        self.nf_descriptors: Dict[str, NfDeploymentDescriptor] = {
            "cu-descriptor": NfDeploymentDescriptor(
                descriptorId="cu-descriptor",
                name="5G CU",
                version="1.0.0",
                provider="O-RAN",
                description="5G Central Unit",
                nfType="CU",
                requiredCpu=2,
                requiredMemoryGb=4,
                requiredStorageGb=10,
                containerImage="5g-emulator/cu:latest"
            ),
            "du-descriptor": NfDeploymentDescriptor(
                descriptorId="du-descriptor",
                name="5G DU",
                version="1.0.0",
                provider="O-RAN",
                description="5G Distributed Unit",
                nfType="DU",
                requiredCpu=2,
                requiredMemoryGb=4,
                requiredStorageGb=10,
                containerImage="5g-emulator/du:latest"
            ),
            "amf-descriptor": NfDeploymentDescriptor(
                descriptorId="amf-descriptor",
                name="5G AMF",
                version="1.0.0",
                provider="3GPP",
                description="Access and Mobility Management Function",
                nfType="AMF",
                requiredCpu=2,
                requiredMemoryGb=4,
                requiredStorageGb=5,
                containerImage="5g-emulator/amf:latest"
            ),
            "smf-descriptor": NfDeploymentDescriptor(
                descriptorId="smf-descriptor",
                name="5G SMF",
                version="1.0.0",
                provider="3GPP",
                description="Session Management Function",
                nfType="SMF",
                requiredCpu=1,
                requiredMemoryGb=2,
                requiredStorageGb=5,
                containerImage="5g-emulator/smf:latest"
            ),
            "upf-descriptor": NfDeploymentDescriptor(
                descriptorId="upf-descriptor",
                name="5G UPF",
                version="1.0.0",
                provider="3GPP",
                description="User Plane Function",
                nfType="UPF",
                requiredCpu=4,
                requiredMemoryGb=8,
                requiredStorageGb=10,
                containerImage="5g-emulator/upf:latest"
            )
        }

    # =========================================================================
    # O-Cloud Info
    # =========================================================================

    def get_ocloud_info(self) -> OCloudInfo:
        """Get O-Cloud information"""
        return self.ocloud

    def get_deployment_managers(self) -> List[DeploymentManager]:
        """Get list of deployment managers"""
        return [self.deployment_manager]

    def get_deployment_manager(self, dm_id: str) -> Optional[DeploymentManager]:
        """Get specific deployment manager"""
        if dm_id == self.deployment_manager.deploymentManagerId:
            return self.deployment_manager
        return None

    # =========================================================================
    # Resource Types
    # =========================================================================

    def get_resource_types(self) -> List[ResourceType]:
        """Get all resource types"""
        return list(self.resource_types.values())

    def get_resource_type(self, type_id: str) -> Optional[ResourceType]:
        """Get specific resource type"""
        return self.resource_types.get(type_id)

    # =========================================================================
    # Resource Pools
    # =========================================================================

    def get_resource_pools(self) -> List[ResourcePool]:
        """Get all resource pools"""
        return list(self.resource_pools.values())

    def get_resource_pool(self, pool_id: str) -> Optional[ResourcePool]:
        """Get specific resource pool"""
        return self.resource_pools.get(pool_id)

    # =========================================================================
    # Resources (Nodes)
    # =========================================================================

    def get_resources(self, pool_id: str) -> List[Resource]:
        """Get resources in a pool"""
        if pool_id == "compute-pool":
            return [
                Resource(
                    resourceId=node["id"],
                    resourcePoolId="compute-pool",
                    description=node["name"],
                    resourceTypeId="compute-node",
                    globalAssetId=f"asset-{node['id']}",
                    extensions={
                        "cpu": node["cpu"],
                        "ram_gb": node["ram_gb"],
                        "storage_gb": node["storage_gb"],
                        "status": node["status"].value,
                        "allocated_cpu": node["allocated_cpu"],
                        "allocated_ram_gb": node["allocated_ram_gb"],
                        "available_cpu": node["cpu"] - node["allocated_cpu"],
                        "available_ram_gb": node["ram_gb"] - node["allocated_ram_gb"]
                    }
                )
                for node in self.nodes.values()
            ]
        elif pool_id == "storage-pool":
            return [
                Resource(
                    resourceId="storage-001",
                    resourcePoolId="storage-pool",
                    description="Main storage volume",
                    resourceTypeId="storage-volume",
                    globalAssetId="asset-storage-001",
                    extensions=self.resource_pools["storage-pool"].extensions
                )
            ]
        elif pool_id == "network-pool":
            return [
                Resource(
                    resourceId="network-001",
                    resourcePoolId="network-pool",
                    description="Pod network",
                    resourceTypeId="network-interface",
                    globalAssetId="asset-network-001",
                    extensions=self.resource_pools["network-pool"].extensions
                )
            ]
        return []

    def get_resource(self, pool_id: str, resource_id: str) -> Optional[Resource]:
        """Get specific resource"""
        resources = self.get_resources(pool_id)
        for r in resources:
            if r.resourceId == resource_id:
                return r
        return None

    # =========================================================================
    # NF Descriptors
    # =========================================================================

    def get_nf_descriptors(self) -> List[NfDeploymentDescriptor]:
        """Get all NF deployment descriptors"""
        return list(self.nf_descriptors.values())

    def get_nf_descriptor(self, descriptor_id: str) -> Optional[NfDeploymentDescriptor]:
        """Get specific NF descriptor"""
        return self.nf_descriptors.get(descriptor_id)

    def create_nf_descriptor(self, descriptor: NfDeploymentDescriptor) -> NfDeploymentDescriptor:
        """Create new NF descriptor"""
        self.nf_descriptors[descriptor.descriptorId] = descriptor
        logger.info(f"Created NF descriptor: {descriptor.descriptorId}")
        return descriptor

    # =========================================================================
    # Deployments
    # =========================================================================

    def deploy_workload(
        self,
        descriptor_id: str,
        name: str,
        replicas: int = 1,
        pool_id: str = "compute-pool"
    ) -> Deployment:
        """Deploy a workload to the fake cluster"""
        descriptor = self.get_nf_descriptor(descriptor_id)
        if not descriptor:
            raise ValueError(f"Unknown descriptor: {descriptor_id}")

        # Check capacity
        required_cpu = descriptor.requiredCpu * replicas
        required_ram = descriptor.requiredMemoryGb * replicas

        available_cpu = sum(n["cpu"] - n["allocated_cpu"] for n in self.nodes.values())
        available_ram = sum(n["ram_gb"] - n["allocated_ram_gb"] for n in self.nodes.values())

        if required_cpu > available_cpu or required_ram > available_ram:
            raise ValueError(f"Insufficient resources: need {required_cpu} CPU, {required_ram}GB RAM")

        # Create deployment
        deployment_id = str(uuid.uuid4())
        deployment = Deployment(
            deploymentId=deployment_id,
            name=name,
            descriptorId=descriptor_id,
            state=DeploymentState.DEPLOYING,
            replicas=replicas,
            readyReplicas=0,
            resourcePoolId=pool_id,
            allocatedResources={
                "cpu": required_cpu,
                "ram_gb": required_ram,
                "storage_gb": descriptor.requiredStorageGb * replicas
            },
            createdAt=datetime.now()
        )

        # Allocate resources to nodes (simple round-robin)
        allocated_replicas = 0
        for node in self.nodes.values():
            while allocated_replicas < replicas:
                if (node["cpu"] - node["allocated_cpu"] >= descriptor.requiredCpu and
                    node["ram_gb"] - node["allocated_ram_gb"] >= descriptor.requiredMemoryGb):
                    node["allocated_cpu"] += descriptor.requiredCpu
                    node["allocated_ram_gb"] += descriptor.requiredMemoryGb
                    node["deployments"].append(deployment_id)
                    allocated_replicas += 1
                else:
                    break

        # Simulate deployment completing
        deployment.state = DeploymentState.RUNNING
        deployment.readyReplicas = replicas
        deployment.updatedAt = datetime.now()

        self.deployments[deployment_id] = deployment
        self.deployment_manager.capacity["current_deployments"] = len(self.deployments)

        # Create operation record
        operation = DeploymentOperation(
            operationId=str(uuid.uuid4()),
            deploymentId=deployment_id,
            operationType=OperationType.INSTANTIATE,
            state=OperationState.COMPLETED,
            startTime=deployment.createdAt,
            endTime=deployment.updatedAt
        )
        self.operations[operation.operationId] = operation

        logger.info(f"Deployed {name} ({descriptor_id}) with {replicas} replicas")
        return deployment

    def get_deployments(self) -> List[Deployment]:
        """Get all deployments"""
        return list(self.deployments.values())

    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get specific deployment"""
        return self.deployments.get(deployment_id)

    def scale_deployment(self, deployment_id: str, replicas: int) -> Deployment:
        """Scale a deployment"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Unknown deployment: {deployment_id}")

        descriptor = self.get_nf_descriptor(deployment.descriptorId)
        old_replicas = deployment.replicas
        delta = replicas - old_replicas

        if delta > 0:
            # Scale up - check capacity
            required_cpu = descriptor.requiredCpu * delta
            required_ram = descriptor.requiredMemoryGb * delta
            available_cpu = sum(n["cpu"] - n["allocated_cpu"] for n in self.nodes.values())
            available_ram = sum(n["ram_gb"] - n["allocated_ram_gb"] for n in self.nodes.values())

            if required_cpu > available_cpu or required_ram > available_ram:
                raise ValueError(f"Insufficient resources for scale up")

            # Allocate
            for node in self.nodes.values():
                while delta > 0:
                    if (node["cpu"] - node["allocated_cpu"] >= descriptor.requiredCpu and
                        node["ram_gb"] - node["allocated_ram_gb"] >= descriptor.requiredMemoryGb):
                        node["allocated_cpu"] += descriptor.requiredCpu
                        node["allocated_ram_gb"] += descriptor.requiredMemoryGb
                        delta -= 1
                    else:
                        break
        elif delta < 0:
            # Scale down - release resources
            to_release = abs(delta)
            for node in self.nodes.values():
                while to_release > 0 and deployment_id in node["deployments"]:
                    node["allocated_cpu"] -= descriptor.requiredCpu
                    node["allocated_ram_gb"] -= descriptor.requiredMemoryGb
                    to_release -= 1

        # Update deployment
        deployment.replicas = replicas
        deployment.readyReplicas = replicas
        deployment.allocatedResources = {
            "cpu": descriptor.requiredCpu * replicas,
            "ram_gb": descriptor.requiredMemoryGb * replicas,
            "storage_gb": descriptor.requiredStorageGb * replicas
        }
        deployment.updatedAt = datetime.now()

        # Create operation record
        operation = DeploymentOperation(
            operationId=str(uuid.uuid4()),
            deploymentId=deployment_id,
            operationType=OperationType.SCALE,
            state=OperationState.COMPLETED,
            startTime=datetime.now(),
            endTime=datetime.now(),
            extensions={"old_replicas": old_replicas, "new_replicas": replicas}
        )
        self.operations[operation.operationId] = operation

        logger.info(f"Scaled {deployment.name} from {old_replicas} to {replicas} replicas")
        return deployment

    def terminate_deployment(self, deployment_id: str) -> bool:
        """Terminate a deployment"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return False

        descriptor = self.get_nf_descriptor(deployment.descriptorId)

        # Release resources
        for node in self.nodes.values():
            count = node["deployments"].count(deployment_id)
            node["allocated_cpu"] -= descriptor.requiredCpu * count
            node["allocated_ram_gb"] -= descriptor.requiredMemoryGb * count
            node["deployments"] = [d for d in node["deployments"] if d != deployment_id]

        # Update deployment state
        deployment.state = DeploymentState.TERMINATED
        deployment.readyReplicas = 0
        deployment.updatedAt = datetime.now()

        # Create operation record
        operation = DeploymentOperation(
            operationId=str(uuid.uuid4()),
            deploymentId=deployment_id,
            operationType=OperationType.TERMINATE,
            state=OperationState.COMPLETED,
            startTime=datetime.now(),
            endTime=datetime.now()
        )
        self.operations[operation.operationId] = operation

        self.deployment_manager.capacity["current_deployments"] = len(
            [d for d in self.deployments.values() if d.state != DeploymentState.TERMINATED]
        )

        logger.info(f"Terminated deployment {deployment.name}")
        return True

    def get_deployment_operations(self, deployment_id: str) -> List[DeploymentOperation]:
        """Get operations for a deployment"""
        return [op for op in self.operations.values() if op.deploymentId == deployment_id]

    # =========================================================================
    # Alarms
    # =========================================================================

    def create_alarm(
        self,
        severity: PerceivedSeverity,
        resource_id: str,
        message: str,
        resource_type_id: str = "compute-node"
    ) -> AlarmEventRecord:
        """Create a new alarm"""
        alarm_id = str(uuid.uuid4())
        alarm = AlarmEventRecord(
            alarmEventRecordId=alarm_id,
            alarmRaisedTime=datetime.now(),
            perceivedSeverity=severity,
            resourceId=resource_id,
            resourceTypeId=resource_type_id,
            alarmMessage=message,
            alarmAcknowledged=False
        )
        self.alarms[alarm_id] = alarm
        logger.warning(f"Alarm created: {severity.name} - {message}")
        return alarm

    def get_alarms(self) -> List[AlarmEventRecord]:
        """Get all alarms"""
        return list(self.alarms.values())

    def get_alarm(self, alarm_id: str) -> Optional[AlarmEventRecord]:
        """Get specific alarm"""
        return self.alarms.get(alarm_id)

    def acknowledge_alarm(self, alarm_id: str) -> Optional[AlarmEventRecord]:
        """Acknowledge an alarm"""
        alarm = self.alarms.get(alarm_id)
        if alarm:
            alarm.alarmAcknowledged = True
            alarm.alarmAcknowledgedTime = datetime.now()
            alarm.alarmChangedTime = datetime.now()
            logger.info(f"Alarm acknowledged: {alarm_id}")
        return alarm

    def clear_alarm(self, alarm_id: str) -> bool:
        """Clear (resolve) an alarm"""
        alarm = self.alarms.get(alarm_id)
        if alarm:
            alarm.perceivedSeverity = PerceivedSeverity.CLEARED
            alarm.alarmClearedTime = datetime.now()
            alarm.alarmChangedTime = datetime.now()
            logger.info(f"Alarm cleared: {alarm_id}")
            return True
        return False

    # =========================================================================
    # Subscriptions
    # =========================================================================

    def create_alarm_subscription(self, callback: str, filter_str: Optional[str] = None) -> Dict[str, Any]:
        """Create alarm subscription"""
        sub_id = str(uuid.uuid4())
        subscription = {
            "alarmSubscriptionId": sub_id,
            "callback": callback,
            "filter": filter_str,
            "createdAt": datetime.now().isoformat()
        }
        self.alarm_subscriptions[sub_id] = subscription
        logger.info(f"Created alarm subscription: {sub_id}")
        return subscription

    def get_alarm_subscriptions(self) -> List[Dict[str, Any]]:
        """Get all alarm subscriptions"""
        return list(self.alarm_subscriptions.values())

    def delete_alarm_subscription(self, sub_id: str) -> bool:
        """Delete alarm subscription"""
        if sub_id in self.alarm_subscriptions:
            del self.alarm_subscriptions[sub_id]
            return True
        return False

    def create_inventory_subscription(self, callback: str, filter_str: Optional[str] = None) -> Dict[str, Any]:
        """Create inventory change subscription"""
        sub_id = str(uuid.uuid4())
        subscription = {
            "subscriptionId": sub_id,
            "callback": callback,
            "filter": filter_str,
            "createdAt": datetime.now().isoformat()
        }
        self.inventory_subscriptions[sub_id] = subscription
        logger.info(f"Created inventory subscription: {sub_id}")
        return subscription

    def get_inventory_subscriptions(self) -> List[Dict[str, Any]]:
        """Get all inventory subscriptions"""
        return list(self.inventory_subscriptions.values())

    def delete_inventory_subscription(self, sub_id: str) -> bool:
        """Delete inventory subscription"""
        if sub_id in self.inventory_subscriptions:
            del self.inventory_subscriptions[sub_id]
            return True
        return False

    # =========================================================================
    # Cluster Stats
    # =========================================================================

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get overall cluster statistics"""
        total_cpu = sum(n["cpu"] for n in self.nodes.values())
        allocated_cpu = sum(n["allocated_cpu"] for n in self.nodes.values())
        total_ram = sum(n["ram_gb"] for n in self.nodes.values())
        allocated_ram = sum(n["allocated_ram_gb"] for n in self.nodes.values())

        active_deployments = len([d for d in self.deployments.values()
                                   if d.state == DeploymentState.RUNNING])
        active_alarms = len([a for a in self.alarms.values()
                             if a.perceivedSeverity != PerceivedSeverity.CLEARED])

        return {
            "nodes": len(self.nodes),
            "total_cpu": total_cpu,
            "allocated_cpu": allocated_cpu,
            "available_cpu": total_cpu - allocated_cpu,
            "cpu_utilization_pct": round(allocated_cpu / total_cpu * 100, 1) if total_cpu > 0 else 0,
            "total_ram_gb": total_ram,
            "allocated_ram_gb": allocated_ram,
            "available_ram_gb": total_ram - allocated_ram,
            "ram_utilization_pct": round(allocated_ram / total_ram * 100, 1) if total_ram > 0 else 0,
            "active_deployments": active_deployments,
            "total_deployments": len(self.deployments),
            "active_alarms": active_alarms,
            "total_alarms": len(self.alarms)
        }


# Global singleton instance
fake_cluster = FakeCluster()
