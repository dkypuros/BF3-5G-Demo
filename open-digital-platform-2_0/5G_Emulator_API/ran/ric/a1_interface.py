#!/usr/bin/env python3
"""
A1 Interface Implementation
ETSI TS 103983 Compliant

This module provides the A1 interface between Non-RT RIC and Near-RT RIC:
- A1-P: Policy Management Service
- A1-EI: Enrichment Information Service
- A1-ML: ML Model Management Service (future)

The A1 interface operates over HTTP/REST with JSON payloads.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

import httpx
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# A1 Interface Constants
# =============================================================================

class A1ServiceType(str, Enum):
    """A1 Service types per ETSI TS 103983"""
    POLICY = "A1-P"
    ENRICHMENT_INFO = "A1-EI"
    ML_MODEL = "A1-ML"


# A1 API Paths
A1_POLICY_BASE = "/a1/policies"
A1_POLICY_TYPES = "/a1/policytypes"
A1_ENRICHMENT = "/a1/enrichment"
A1_ML_MODELS = "/a1/ml-models"


# =============================================================================
# A1 Client (Non-RT RIC side - sends to Near-RT RIC)
# =============================================================================

class A1Client:
    """
    A1 Interface Client

    Used by Non-RT RIC to communicate with Near-RT RIC.
    Implements A1-P, A1-EI, and A1-ML services.
    """

    def __init__(self, near_rt_ric_url: str = "http://127.0.0.1:8095"):
        self.near_rt_ric_url = near_rt_ric_url
        self.timeout = 5.0
        logger.info(f"A1 Client initialized, target: {near_rt_ric_url}")

    # -------------------------------------------------------------------------
    # A1-P: Policy Management
    # -------------------------------------------------------------------------

    async def create_policy(
        self,
        policy_type_id: str,
        policy_id: str,
        policy_data: Dict,
        scope: Optional[Dict] = None
    ) -> Dict:
        """
        Create policy instance in Near-RT RIC

        Per ETSI TS 103983 Section 7.3
        """
        payload = {
            "policyId": policy_id,
            "policyTypeId": policy_type_id,
            "policyData": policy_data,
            "scope": scope
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.near_rt_ric_url}{A1_POLICY_BASE}",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code in (200, 201):
                    logger.info(f"Policy {policy_id} created in Near-RT RIC")
                    return {"status": "created", "policyId": policy_id}
                else:
                    logger.warning(f"Policy creation failed: {response.text}")
                    return {"status": "failed", "error": response.text}

        except httpx.RequestError as e:
            logger.error(f"A1 policy creation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def update_policy(
        self,
        policy_type_id: str,
        policy_id: str,
        policy_data: Dict
    ) -> Dict:
        """Update existing policy in Near-RT RIC"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.near_rt_ric_url}{A1_POLICY_BASE}/{policy_id}",
                    json={"policyData": policy_data},
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    logger.info(f"Policy {policy_id} updated")
                    return {"status": "updated", "policyId": policy_id}
                else:
                    return {"status": "failed", "error": response.text}

        except httpx.RequestError as e:
            logger.error(f"A1 policy update failed: {e}")
            return {"status": "error", "error": str(e)}

    async def delete_policy(self, policy_id: str) -> Dict:
        """Delete policy from Near-RT RIC"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.near_rt_ric_url}{A1_POLICY_BASE}/{policy_id}",
                    timeout=self.timeout
                )

                if response.status_code in (200, 204):
                    logger.info(f"Policy {policy_id} deleted")
                    return {"status": "deleted", "policyId": policy_id}
                else:
                    return {"status": "failed", "error": response.text}

        except httpx.RequestError as e:
            logger.error(f"A1 policy deletion failed: {e}")
            return {"status": "error", "error": str(e)}

    async def get_policy_status(self, policy_id: str) -> Dict:
        """
        Get policy enforcement status from Near-RT RIC

        Per ETSI TS 103983 Section 7.4
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.near_rt_ric_url}{A1_POLICY_BASE}/{policy_id}/status",
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    return {"status": "unknown", "error": response.text}

        except httpx.RequestError as e:
            logger.error(f"A1 policy status query failed: {e}")
            return {"status": "error", "error": str(e)}

    async def list_policies(self) -> List[Dict]:
        """List all policies in Near-RT RIC"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.near_rt_ric_url}{A1_POLICY_BASE}",
                    timeout=self.timeout
                )
                return response.json() if response.status_code == 200 else []
        except httpx.RequestError:
            return []

    # -------------------------------------------------------------------------
    # A1-EI: Enrichment Information
    # -------------------------------------------------------------------------

    async def send_enrichment_info(
        self,
        ei_type: str,
        ei_data: Dict
    ) -> Dict:
        """
        Send enrichment information to Near-RT RIC

        Per ETSI TS 103983 Section 8
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.near_rt_ric_url}{A1_ENRICHMENT}",
                    json=ei_data,
                    params={"ei_type": ei_type},
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    logger.info(f"Enrichment info delivered: {ei_type}")
                    return {"status": "delivered", "eiType": ei_type}
                else:
                    return {"status": "failed", "error": response.text}

        except httpx.RequestError as e:
            logger.error(f"A1 EI delivery failed: {e}")
            return {"status": "error", "error": str(e)}

    async def register_ei_producer(
        self,
        ei_producer_id: str,
        supported_ei_types: List[str]
    ) -> Dict:
        """Register as EI producer with Near-RT RIC"""
        payload = {
            "eiProducerId": ei_producer_id,
            "supportedEiTypes": supported_ei_types
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.near_rt_ric_url}/a1/ei-producers",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code in (200, 201):
                    return {"status": "registered", "producerId": ei_producer_id}
                else:
                    return {"status": "failed", "error": response.text}

        except httpx.RequestError as e:
            return {"status": "error", "error": str(e)}

    # -------------------------------------------------------------------------
    # A1-ML: ML Model Management (Future)
    # -------------------------------------------------------------------------

    async def deploy_ml_model(
        self,
        model_id: str,
        model_data: Dict
    ) -> Dict:
        """
        Deploy ML model to Near-RT RIC

        Per ETSI TS 103983 Section 9 (when specified)
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.near_rt_ric_url}{A1_ML_MODELS}",
                    json={"modelId": model_id, "modelData": model_data},
                    timeout=self.timeout
                )

                if response.status_code in (200, 201):
                    logger.info(f"ML model {model_id} deployed")
                    return {"status": "deployed", "modelId": model_id}
                else:
                    return {"status": "failed", "error": response.text}

        except httpx.RequestError as e:
            return {"status": "error", "error": str(e)}

    # -------------------------------------------------------------------------
    # Health and Connectivity
    # -------------------------------------------------------------------------

    async def check_connectivity(self) -> bool:
        """Check if Near-RT RIC is reachable"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.near_rt_ric_url}/health",
                    timeout=2.0
                )
                return response.status_code == 200
        except httpx.RequestError:
            return False

    async def get_near_rt_ric_status(self) -> Dict:
        """Get Near-RT RIC status"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.near_rt_ric_url}/ric/status",
                    timeout=self.timeout
                )
                return response.json() if response.status_code == 200 else {}
        except httpx.RequestError as e:
            return {"error": str(e)}


# =============================================================================
# A1 Policy Feedback Handler
# =============================================================================

class A1PolicyFeedback(BaseModel):
    """A1 Policy feedback from Near-RT RIC"""
    policyId: str
    status: str
    reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class A1FeedbackHandler:
    """
    Handles A1 policy feedback from Near-RT RIC

    Used by Non-RT RIC to process enforcement status updates.
    """

    def __init__(self):
        self.feedback_queue: List[A1PolicyFeedback] = []
        self.callbacks: Dict[str, callable] = {}

    def register_callback(self, policy_id: str, callback: callable):
        """Register callback for policy feedback"""
        self.callbacks[policy_id] = callback

    async def process_feedback(self, feedback: A1PolicyFeedback):
        """Process feedback from Near-RT RIC"""
        self.feedback_queue.append(feedback)

        # Execute registered callback
        if feedback.policyId in self.callbacks:
            callback = self.callbacks[feedback.policyId]
            try:
                await callback(feedback) if asyncio.iscoroutinefunction(callback) else callback(feedback)
            except Exception as e:
                logger.error(f"Feedback callback failed: {e}")

        logger.info(f"Policy feedback received: {feedback.policyId} -> {feedback.status}")

    def get_recent_feedback(self, limit: int = 10) -> List[A1PolicyFeedback]:
        """Get recent feedback entries"""
        return self.feedback_queue[-limit:]


# =============================================================================
# A1 Message Types
# =============================================================================

class A1PolicyNotification(BaseModel):
    """A1 Policy status notification"""
    notificationType: str = "POLICY_STATUS"
    policyId: str
    policyTypeId: str
    status: str
    enforcementStatus: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class A1EnrichmentDelivery(BaseModel):
    """A1 Enrichment information delivery"""
    deliveryType: str = "EI_DELIVERY"
    eiTypeId: str
    eiData: Dict
    producerId: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# A1 Connection Manager
# =============================================================================

class A1ConnectionManager:
    """
    Manages A1 connection state between Non-RT RIC and Near-RT RIC

    Handles:
    - Connection establishment
    - Heartbeat monitoring
    - Reconnection
    """

    def __init__(self, near_rt_ric_url: str):
        self.near_rt_ric_url = near_rt_ric_url
        self.connected = False
        self.last_heartbeat: Optional[datetime] = None
        self.heartbeat_interval = 30  # seconds

    async def connect(self) -> bool:
        """Establish A1 connection"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.near_rt_ric_url}/health",
                    timeout=5.0
                )
                self.connected = response.status_code == 200
                if self.connected:
                    self.last_heartbeat = datetime.utcnow()
                    logger.info("A1 connection established")
                return self.connected
        except httpx.RequestError as e:
            logger.warning(f"A1 connection failed: {e}")
            self.connected = False
            return False

    async def heartbeat(self) -> bool:
        """Send heartbeat to maintain connection"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.near_rt_ric_url}/health",
                    timeout=2.0
                )
                if response.status_code == 200:
                    self.last_heartbeat = datetime.utcnow()
                    self.connected = True
                    return True
                else:
                    self.connected = False
                    return False
        except httpx.RequestError:
            self.connected = False
            return False

    async def start_heartbeat_loop(self):
        """Start background heartbeat loop"""
        while True:
            await self.heartbeat()
            await asyncio.sleep(self.heartbeat_interval)

    def get_connection_status(self) -> Dict:
        """Get current connection status"""
        return {
            "connected": self.connected,
            "lastHeartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "nearRtRicUrl": self.near_rt_ric_url
        }
