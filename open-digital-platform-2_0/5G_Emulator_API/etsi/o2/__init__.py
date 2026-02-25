# File location: 5G_Emulator_API/etsi/o2/__init__.py
# O-RAN O2 Interface Implementation
# O2 IMS (Infrastructure Management Service) and O2 DMS (Deployment Management Service)

"""
O-RAN O2 Interface Components

This package implements the O-RAN O2 interface for managing O-Cloud infrastructure:
- O2 IMS: Infrastructure Management Service (port 8098)
- O2 DMS: Deployment Management Service (port 8099)
- SMO: Service Management & Orchestration (port 8097)
- Fake Cluster: Simulated O-Cloud for testing

Specs:
- O-RAN.WG6.O2IMS-INTERFACE
- O-RAN.WG6.O2DMS-INTERFACE
"""

from .fake_cluster import FakeCluster
from .models import (
    OCloudInfo,
    DeploymentManager,
    ResourceType,
    ResourcePool,
    Resource,
    AlarmEventRecord,
    NfDeploymentDescriptor,
    Deployment,
)

__all__ = [
    "FakeCluster",
    "OCloudInfo",
    "DeploymentManager",
    "ResourceType",
    "ResourcePool",
    "Resource",
    "AlarmEventRecord",
    "NfDeploymentDescriptor",
    "Deployment",
]
