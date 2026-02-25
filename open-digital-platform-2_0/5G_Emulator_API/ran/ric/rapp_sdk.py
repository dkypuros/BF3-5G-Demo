#!/usr/bin/env python3
"""
rApp SDK (Non-RT RIC Application Framework)
O-RAN Compliant Implementation

This module provides the framework for developing rApps that run on the Non-RT RIC.
rApps operate in the non-real-time control loop (>1 second latency).

Features:
- A1 Policy management
- Enrichment information delivery
- ML model deployment (future)
- Analytics integration
- O1 interface interaction

Usage:
    from rapp_sdk import RAppBase

    class MyRApp(RAppBase):
        async def on_analytics(self, analytics):
            # Analyze RAN data and create policies
            if analytics['avgLoad'] > 70:
                await self.create_policy(...)

    rapp = MyRApp("my-rapp", "http://127.0.0.1:8096")
    await rapp.start()
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

import httpx
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# rApp Configuration
# =============================================================================

class RAppConfig(BaseModel):
    """rApp configuration"""
    rappName: str
    rappId: Optional[str] = None
    version: str = "1.0.0"
    vendor: Optional[str] = None
    ricUrl: str = "http://127.0.0.1:8096"
    analyticsInterval: int = 60  # seconds
    enableAutoPolicy: bool = True


# =============================================================================
# rApp Base Class
# =============================================================================

class RAppBase(ABC):
    """
    Base class for rApp development

    Provides:
    - Registration with Non-RT RIC
    - A1 Policy management
    - Enrichment information delivery
    - Analytics collection
    - O1 integration (ZSM, VNFM)

    Subclass this and implement on_analytics() to create an rApp.
    """

    def __init__(
        self,
        rapp_name: str,
        ric_url: str = "http://127.0.0.1:8096",
        analytics_interval: int = 60
    ):
        self.rapp_name = rapp_name
        self.rapp_id: Optional[str] = None
        self.ric_url = ric_url
        self.analytics_interval = analytics_interval

        # Policy tracking
        self.managed_policies: Dict[str, Dict] = {}

        # State
        self.running = False
        self.registered = False

        # Metrics
        self.metrics = {
            "policies_created": 0,
            "policies_updated": 0,
            "enrichment_sent": 0,
            "analytics_processed": 0,
            "errors": 0,
            "start_time": None
        }

        logger.info(f"rApp initialized: {rapp_name}")

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    async def start(self):
        """Start the rApp"""
        logger.info(f"Starting rApp: {self.rapp_name}")

        # Register with Non-RT RIC
        await self._register()

        self.running = True
        self.metrics["start_time"] = datetime.utcnow().isoformat()

        # Start analytics collection loop
        asyncio.create_task(self._analytics_loop())

        # Call user initialization
        await self.on_start()

        logger.info(f"rApp {self.rapp_name} started successfully")

    async def stop(self):
        """Stop the rApp"""
        logger.info(f"Stopping rApp: {self.rapp_name}")

        # Delete managed policies
        for policy_id in list(self.managed_policies.keys()):
            await self.delete_policy(policy_id)

        # Unregister from RIC
        await self._unregister()

        self.running = False

        # Call user cleanup
        await self.on_stop()

        logger.info(f"rApp {self.rapp_name} stopped")

    async def _register(self):
        """Register rApp with Non-RT RIC"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ric_url}/ric/rapps",
                    json={
                        "rappName": self.rapp_name,
                        "version": "1.0.0"
                    },
                    timeout=5.0
                )

                if response.status_code in (200, 201):
                    data = response.json()
                    self.rapp_id = data.get("rappId")
                    self.registered = True
                    logger.info(f"rApp registered with ID: {self.rapp_id}")
                else:
                    logger.error(f"rApp registration failed: {response.text}")

        except httpx.RequestError as e:
            logger.error(f"Failed to register rApp: {e}")

    async def _unregister(self):
        """Unregister rApp from Non-RT RIC"""
        if not self.rapp_id:
            return

        try:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"{self.ric_url}/ric/rapps/{self.rapp_id}",
                    timeout=5.0
                )
                self.registered = False
                logger.info(f"rApp {self.rapp_id} unregistered")
        except httpx.RequestError as e:
            logger.error(f"Failed to unregister rApp: {e}")

    async def _analytics_loop(self):
        """Background loop for collecting and processing analytics"""
        while self.running:
            try:
                analytics = await self.get_analytics()
                self.metrics["analytics_processed"] += 1
                await self.on_analytics(analytics)
            except Exception as e:
                logger.error(f"Analytics processing failed: {e}")
                self.metrics["errors"] += 1

            await asyncio.sleep(self.analytics_interval)

    # -------------------------------------------------------------------------
    # A1 Policy Management
    # -------------------------------------------------------------------------

    async def create_policy(
        self,
        policy_type_id: str,
        policy_data: Dict,
        scope: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Create A1 policy

        Args:
            policy_type_id: Policy type identifier
            policy_data: Policy parameters
            scope: Optional scope (cells, UEs, slices)

        Returns:
            Policy ID if successful, None otherwise
        """
        import uuid
        policy_id = str(uuid.uuid4())

        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.ric_url}/a1-p/policytypes/{policy_type_id}/policies/{policy_id}",
                    json=policy_data,
                    timeout=5.0
                )

                if response.status_code in (200, 201):
                    self.managed_policies[policy_id] = {
                        "policyTypeId": policy_type_id,
                        "policyData": policy_data,
                        "scope": scope
                    }
                    self.metrics["policies_created"] += 1
                    logger.info(f"Policy created: {policy_id}")
                    return policy_id
                else:
                    logger.error(f"Policy creation failed: {response.text}")
                    return None

        except httpx.RequestError as e:
            logger.error(f"Policy creation failed: {e}")
            self.metrics["errors"] += 1
            return None

    async def update_policy(
        self,
        policy_id: str,
        policy_data: Dict
    ) -> bool:
        """Update existing policy"""
        if policy_id not in self.managed_policies:
            logger.warning(f"Policy {policy_id} not managed by this rApp")
            return False

        policy_type_id = self.managed_policies[policy_id]["policyTypeId"]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.ric_url}/a1-p/policytypes/{policy_type_id}/policies/{policy_id}",
                    json=policy_data,
                    timeout=5.0
                )

                if response.status_code == 200:
                    self.managed_policies[policy_id]["policyData"] = policy_data
                    self.metrics["policies_updated"] += 1
                    logger.info(f"Policy updated: {policy_id}")
                    return True
                else:
                    return False

        except httpx.RequestError as e:
            logger.error(f"Policy update failed: {e}")
            return False

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete policy"""
        if policy_id not in self.managed_policies:
            return False

        policy_type_id = self.managed_policies[policy_id]["policyTypeId"]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.ric_url}/a1-p/policytypes/{policy_type_id}/policies/{policy_id}",
                    timeout=5.0
                )

                if response.status_code in (200, 204):
                    del self.managed_policies[policy_id]
                    logger.info(f"Policy deleted: {policy_id}")
                    return True
                else:
                    return False

        except httpx.RequestError as e:
            logger.error(f"Policy deletion failed: {e}")
            return False

    async def get_policy_status(self, policy_id: str) -> Dict:
        """Get policy enforcement status"""
        if policy_id not in self.managed_policies:
            return {"error": "Policy not managed"}

        policy_type_id = self.managed_policies[policy_id]["policyTypeId"]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.ric_url}/a1-p/policytypes/{policy_type_id}/policies/{policy_id}/status",
                    timeout=5.0
                )
                return response.json() if response.status_code == 200 else {}
        except httpx.RequestError:
            return {}

    # -------------------------------------------------------------------------
    # Enrichment Information
    # -------------------------------------------------------------------------

    async def send_enrichment(
        self,
        ei_type: str,
        ei_data: Dict
    ) -> bool:
        """
        Send enrichment information to Near-RT RIC

        Args:
            ei_type: Enrichment information type
            ei_data: Enrichment data payload

        Returns:
            True if successful
        """
        # First create EI job if needed
        ei_job_id = f"ei-job-{ei_type}"

        try:
            async with httpx.AsyncClient() as client:
                # Deliver enrichment data
                response = await client.post(
                    f"{self.ric_url}/a1-ei/eijobs/{ei_job_id}/deliver",
                    json=ei_data,
                    timeout=5.0
                )

                if response.status_code == 200:
                    self.metrics["enrichment_sent"] += 1
                    logger.info(f"Enrichment info delivered: {ei_type}")
                    return True
                else:
                    return False

        except httpx.RequestError as e:
            logger.error(f"Enrichment delivery failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Analytics and Data Collection
    # -------------------------------------------------------------------------

    async def get_analytics(self) -> Dict:
        """Get RAN analytics from Non-RT RIC"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.ric_url}/ric/analytics",
                    timeout=10.0
                )
                return response.json() if response.status_code == 200 else {}
        except httpx.RequestError as e:
            logger.error(f"Analytics fetch failed: {e}")
            return {}

    async def get_near_rt_ric_status(self) -> Dict:
        """Get Near-RT RIC status via Non-RT RIC"""
        analytics = await self.get_analytics()
        return analytics.get("ricStatus", {})

    # -------------------------------------------------------------------------
    # O1 Integration (ZSM, VNFM)
    # -------------------------------------------------------------------------

    async def create_zsm_intent(
        self,
        intent_type: str,
        target: str,
        objective: str,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Create ZSM intent for closed-loop automation

        Args:
            intent_type: SCALING, HEALING, OPTIMIZATION
            target: Target NF (amf, smf, upf, etc.)
            objective: Intent objective description
            constraints: Optional constraints

        Returns:
            Created intent
        """
        import uuid

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ric_url}/o1/intents",
                    json={
                        "intentId": str(uuid.uuid4())[:8],
                        "intentType": intent_type,
                        "target": target,
                        "objective": objective,
                        "constraints": constraints or {},
                        "priority": 5
                    },
                    timeout=5.0
                )
                return response.json() if response.status_code in (200, 201) else {}

        except httpx.RequestError as e:
            logger.error(f"ZSM intent creation failed: {e}")
            return {}

    async def request_vnf_scaling(
        self,
        vnf_id: str,
        scale_type: str,
        aspect_id: str = "default"
    ) -> Dict:
        """Request VNF scaling via VNFM"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ric_url}/o1/vnf-instances/{vnf_id}/scale",
                    params={
                        "scale_type": scale_type,
                        "aspect_id": aspect_id,
                        "steps": 1
                    },
                    timeout=10.0
                )
                return response.json() if response.status_code == 200 else {}

        except httpx.RequestError as e:
            logger.error(f"VNF scaling request failed: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Abstract Methods (to be implemented by user)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def on_analytics(self, analytics: Dict):
        """
        Process RAN analytics

        Override this method to implement rApp logic.
        Called periodically based on analytics_interval.

        Args:
            analytics: RAN analytics data
        """
        pass

    async def on_start(self):
        """Called after rApp starts (optional override)"""
        pass

    async def on_stop(self):
        """Called before rApp stops (optional override)"""
        pass

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_metrics(self) -> Dict:
        """Get rApp metrics"""
        return {
            "rappName": self.rapp_name,
            "rappId": self.rapp_id,
            "running": self.running,
            "managedPolicies": len(self.managed_policies),
            **self.metrics
        }


# =============================================================================
# Example rApps
# =============================================================================

class TrafficSteeringRApp(RAppBase):
    """
    Example rApp: Traffic Steering

    Analyzes traffic patterns and creates steering policies.
    """

    def __init__(self, ric_url: str = "http://127.0.0.1:8096"):
        super().__init__("traffic-steering-rapp", ric_url, analytics_interval=60)
        self.load_threshold = 70

    async def on_analytics(self, analytics: Dict):
        """Analyze traffic and create steering policies"""
        e2_nodes = analytics.get("e2Nodes", [])

        for node in e2_nodes:
            # Check if traffic steering policy needed
            if self._needs_steering(node):
                await self._create_steering_policy(node)

    def _needs_steering(self, node: Dict) -> bool:
        """Determine if node needs traffic steering"""
        # Simplified logic - check load
        return node.get("load", 0) > self.load_threshold

    async def _create_steering_policy(self, node: Dict):
        """Create traffic steering policy"""
        policy_id = await self.create_policy(
            policy_type_id="ORAN_TrafficSteering_1.0.0",
            policy_data={
                "steeringObjective": "load_balance",
                "targetCells": [node.get("e2NodeId")],
                "loadThreshold": self.load_threshold
            },
            scope={"nodeId": node.get("e2NodeId")}
        )

        if policy_id:
            logger.info(f"Traffic steering policy created for {node.get('e2NodeId')}")


class EnergyEfficiencyRApp(RAppBase):
    """
    Example rApp: Energy Efficiency

    Monitors network load and creates energy saving policies.
    """

    def __init__(self, ric_url: str = "http://127.0.0.1:8096"):
        super().__init__("energy-efficiency-rapp", ric_url, analytics_interval=300)
        self.low_load_threshold = 20

    async def on_analytics(self, analytics: Dict):
        """Analyze load and optimize energy"""
        ric_status = analytics.get("ricStatus", {})
        e2_nodes = analytics.get("e2Nodes", [])

        # Check for low-load conditions
        low_load_nodes = [n for n in e2_nodes
                         if n.get("load", 100) < self.low_load_threshold]

        if len(low_load_nodes) > len(e2_nodes) / 2:
            # Most nodes are low load - trigger energy saving
            await self._enable_energy_saving()

    async def _enable_energy_saving(self):
        """Enable energy saving mode"""
        await self.create_policy(
            policy_type_id="ORAN_QoSTarget_1.0.0",
            policy_data={
                "qosObjective": "minimize_energy",
                "targetKpi": {
                    "energyEfficiency": "high"
                }
            }
        )

        # Also create ZSM intent
        await self.create_zsm_intent(
            intent_type="OPTIMIZATION",
            target="ran",
            objective="Minimize energy consumption during low-load period",
            constraints={"minQos": "maintained"}
        )
