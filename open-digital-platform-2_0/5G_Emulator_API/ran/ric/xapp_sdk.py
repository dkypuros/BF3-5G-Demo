#!/usr/bin/env python3
"""
xApp SDK (Near-RT RIC Application Framework)
O-RAN Compliant Implementation

This module provides the framework for developing xApps that run on the Near-RT RIC.
xApps operate in the near-real-time control loop (10ms - 1s latency).

Features:
- E2 subscription management
- RIC Indication handling
- RIC Control execution
- Callback-based event handling
- Metrics and logging

Usage:
    from xapp_sdk import XAppBase

    class MyXApp(XAppBase):
        async def on_indication(self, indication):
            # Process indication
            if indication['indicationType'] == 'REPORT':
                await self.send_control(...)

    xapp = MyXApp("my-xapp", "http://127.0.0.1:8095")
    await xapp.start()
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

import httpx
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# xApp Configuration
# =============================================================================

class XAppConfig(BaseModel):
    """xApp configuration"""
    xappName: str
    xappId: Optional[str] = None
    version: str = "1.0.0"
    vendor: Optional[str] = None
    ricUrl: str = "http://127.0.0.1:8095"
    callbackPort: int = 9000
    enableMetrics: bool = True
    logLevel: str = "INFO"


class SubscriptionConfig(BaseModel):
    """E2 subscription configuration"""
    e2NodeId: str
    ranFunctionId: int
    reportingPeriodMs: int = 1000
    actionType: str = "REPORT"
    eventConditions: Optional[Dict] = None


# =============================================================================
# xApp Base Class
# =============================================================================

class XAppBase(ABC):
    """
    Base class for xApp development

    Provides:
    - Registration with Near-RT RIC
    - E2 subscription management
    - Indication handling with callbacks
    - Control message sending
    - Lifecycle management

    Subclass this and implement on_indication() to create an xApp.
    """

    def __init__(
        self,
        xapp_name: str,
        ric_url: str = "http://127.0.0.1:8095",
        callback_port: int = 9000
    ):
        self.xapp_name = xapp_name
        self.xapp_id: Optional[str] = None
        self.ric_url = ric_url
        self.callback_port = callback_port

        # Subscription tracking
        self.subscriptions: Dict[str, Dict] = {}

        # Callback registry
        self.indication_callbacks: Dict[str, Callable] = {}

        # State
        self.running = False
        self.registered = False

        # Metrics
        self.metrics = {
            "indications_received": 0,
            "controls_sent": 0,
            "errors": 0,
            "start_time": None
        }

        logger.info(f"xApp initialized: {xapp_name}")

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    async def start(self):
        """Start the xApp"""
        logger.info(f"Starting xApp: {self.xapp_name}")

        # Register with Near-RT RIC
        await self._register()

        # Start callback server
        asyncio.create_task(self._start_callback_server())

        self.running = True
        self.metrics["start_time"] = datetime.utcnow().isoformat()

        # Call user initialization
        await self.on_start()

        logger.info(f"xApp {self.xapp_name} started successfully")

    async def stop(self):
        """Stop the xApp"""
        logger.info(f"Stopping xApp: {self.xapp_name}")

        # Unsubscribe from all subscriptions
        for sub_id in list(self.subscriptions.keys()):
            await self.unsubscribe(sub_id)

        # Unregister from RIC
        await self._unregister()

        self.running = False

        # Call user cleanup
        await self.on_stop()

        logger.info(f"xApp {self.xapp_name} stopped")

    async def _register(self):
        """Register xApp with Near-RT RIC"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ric_url}/ric/xapps",
                    json={
                        "xappName": self.xapp_name,
                        "version": "1.0.0",
                        "endpoint": f"http://127.0.0.1:{self.callback_port}"
                    },
                    timeout=5.0
                )

                if response.status_code in (200, 201):
                    data = response.json()
                    self.xapp_id = data.get("xappId")
                    self.registered = True
                    logger.info(f"xApp registered with ID: {self.xapp_id}")
                else:
                    logger.error(f"xApp registration failed: {response.text}")

        except httpx.RequestError as e:
            logger.error(f"Failed to register xApp: {e}")

    async def _unregister(self):
        """Unregister xApp from Near-RT RIC"""
        if not self.xapp_id:
            return

        try:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"{self.ric_url}/ric/xapps/{self.xapp_id}",
                    timeout=5.0
                )
                self.registered = False
                logger.info(f"xApp {self.xapp_id} unregistered")
        except httpx.RequestError as e:
            logger.error(f"Failed to unregister xApp: {e}")

    async def _start_callback_server(self):
        """Start HTTP server for receiving indications"""
        from fastapi import FastAPI
        import uvicorn

        callback_app = FastAPI()

        @callback_app.post("/indication")
        async def receive_indication(indication: Dict):
            await self._handle_indication(indication)
            return {"status": "received"}

        config = uvicorn.Config(
            callback_app,
            host="0.0.0.0",
            port=self.callback_port,
            log_level="warning"
        )
        server = uvicorn.Server(config)
        await server.serve()

    # -------------------------------------------------------------------------
    # E2 Subscription Methods
    # -------------------------------------------------------------------------

    async def subscribe(
        self,
        e2_node_id: str,
        ran_function_id: int,
        reporting_period_ms: int = 1000,
        action_type: str = "REPORT",
        event_conditions: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Create E2 subscription

        Args:
            e2_node_id: Target E2 Node ID
            ran_function_id: RAN Function ID
            reporting_period_ms: Reporting period in milliseconds
            action_type: REPORT, INSERT, CONTROL, POLICY
            event_conditions: Optional event trigger conditions

        Returns:
            Subscription ID if successful, None otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ric_url}/e2/subscription",
                    json={
                        "e2NodeId": e2_node_id,
                        "ranFunctionId": ran_function_id,
                        "eventTrigger": {
                            "triggerType": "periodic",
                            "reportingPeriodMs": reporting_period_ms,
                            "eventConditions": event_conditions
                        },
                        "actions": [{
                            "actionId": 1,
                            "actionType": action_type,
                            "actionDefinition": {}
                        }],
                        "xappId": self.xapp_id
                    },
                    timeout=5.0
                )

                if response.status_code in (200, 201):
                    data = response.json()
                    sub_id = data.get("subscriptionId")
                    self.subscriptions[sub_id] = {
                        "e2NodeId": e2_node_id,
                        "ranFunctionId": ran_function_id,
                        "state": data.get("state", "ACTIVE")
                    }
                    logger.info(f"Subscription created: {sub_id}")
                    return sub_id
                else:
                    logger.error(f"Subscription failed: {response.text}")
                    return None

        except httpx.RequestError as e:
            logger.error(f"Subscription request failed: {e}")
            self.metrics["errors"] += 1
            return None

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Delete E2 subscription"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.ric_url}/e2/subscriptions/{subscription_id}",
                    timeout=5.0
                )

                if response.status_code in (200, 204):
                    if subscription_id in self.subscriptions:
                        del self.subscriptions[subscription_id]
                    logger.info(f"Subscription deleted: {subscription_id}")
                    return True
                else:
                    return False

        except httpx.RequestError as e:
            logger.error(f"Unsubscribe failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # RIC Control Methods
    # -------------------------------------------------------------------------

    async def send_control(
        self,
        e2_node_id: str,
        ran_function_id: int,
        control_header: Dict,
        control_message: Dict,
        call_process_id: Optional[str] = None
    ) -> Dict:
        """
        Send RIC Control message to E2 Node

        Args:
            e2_node_id: Target E2 Node ID
            ran_function_id: RAN Function ID
            control_header: Control header parameters
            control_message: Control message payload
            call_process_id: Optional call process ID for INSERT response

        Returns:
            Control response from E2 Node
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ric_url}/e2/control",
                    json={
                        "e2NodeId": e2_node_id,
                        "ranFunctionId": ran_function_id,
                        "controlHeader": control_header,
                        "controlMessage": control_message,
                        "callProcessId": call_process_id,
                        "controlAckRequest": True
                    },
                    timeout=1.0  # Near-RT latency requirement
                )

                self.metrics["controls_sent"] += 1

                if response.status_code == 200:
                    return response.json()
                else:
                    self.metrics["errors"] += 1
                    return {"success": False, "error": response.text}

        except httpx.RequestError as e:
            self.metrics["errors"] += 1
            logger.error(f"Control request failed: {e}")
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # Indication Handling
    # -------------------------------------------------------------------------

    async def _handle_indication(self, indication: Dict):
        """Internal indication handler"""
        self.metrics["indications_received"] += 1

        indication_type = indication.get("indicationType", "REPORT")

        # Execute registered callback
        if indication_type in self.indication_callbacks:
            callback = self.indication_callbacks[indication_type]
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(indication)
                else:
                    callback(indication)
            except Exception as e:
                logger.error(f"Indication callback failed: {e}")
                self.metrics["errors"] += 1

        # Call user handler
        await self.on_indication(indication)

    def register_callback(self, indication_type: str, callback: Callable):
        """Register callback for specific indication type"""
        self.indication_callbacks[indication_type] = callback
        logger.info(f"Callback registered for {indication_type}")

    # -------------------------------------------------------------------------
    # Abstract Methods (to be implemented by user)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def on_indication(self, indication: Dict):
        """
        Handle RIC Indication

        Override this method to implement xApp logic.
        Called for every indication received from E2 Nodes.

        Args:
            indication: RIC Indication with header and message
        """
        pass

    async def on_start(self):
        """Called after xApp starts (optional override)"""
        pass

    async def on_stop(self):
        """Called before xApp stops (optional override)"""
        pass

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    async def get_e2_nodes(self) -> List[Dict]:
        """Get list of connected E2 Nodes"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.ric_url}/ric/e2-nodes",
                    timeout=5.0
                )
                return response.json() if response.status_code == 200 else []
        except httpx.RequestError:
            return []

    async def get_radio_measurements(self, ue_id: str) -> Dict:
        """Get radio measurements for UE via RNIS"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.ric_url}/ric/radio/ue/{ue_id}",
                    timeout=2.0
                )
                return response.json() if response.status_code == 200 else {}
        except httpx.RequestError:
            return {}

    def get_metrics(self) -> Dict:
        """Get xApp metrics"""
        return {
            "xappName": self.xapp_name,
            "xappId": self.xapp_id,
            "running": self.running,
            "subscriptions": len(self.subscriptions),
            **self.metrics
        }


# =============================================================================
# Example xApps
# =============================================================================

class LoadBalancingXApp(XAppBase):
    """
    Example xApp: Load Balancing

    Monitors cell load and triggers handovers when threshold exceeded.
    """

    def __init__(self, ric_url: str = "http://127.0.0.1:8095"):
        super().__init__("load-balancing-xapp", ric_url)
        self.load_threshold = 80  # percent
        self.cell_loads: Dict[str, float] = {}

    async def on_start(self):
        """Subscribe to cell load reports"""
        e2_nodes = await self.get_e2_nodes()
        for node in e2_nodes:
            await self.subscribe(
                e2_node_id=node["e2NodeId"],
                ran_function_id=1,  # KPM
                reporting_period_ms=1000
            )

    async def on_indication(self, indication: Dict):
        """Process cell load indication"""
        if indication.get("indicationType") != "REPORT":
            return

        message = indication.get("indicationMessage", {})
        cell_id = message.get("cellId")
        load = message.get("prbUtilization", 0)

        if cell_id:
            self.cell_loads[cell_id] = load

            if load > self.load_threshold:
                logger.info(f"Cell {cell_id} overloaded ({load}%), triggering load balancing")
                # Find target cell with lower load
                target_cell = self._find_target_cell(cell_id)
                if target_cell:
                    await self._trigger_handover(cell_id, target_cell)

    def _find_target_cell(self, source_cell: str) -> Optional[str]:
        """Find cell with lowest load"""
        candidates = {k: v for k, v in self.cell_loads.items()
                     if k != source_cell and v < self.load_threshold}
        if candidates:
            return min(candidates, key=candidates.get)
        return None

    async def _trigger_handover(self, source_cell: str, target_cell: str):
        """Send handover control message"""
        for node_id, sub in self.subscriptions.items():
            await self.send_control(
                e2_node_id=sub["e2NodeId"],
                ran_function_id=2,  # RC
                control_header={"controlType": "HANDOVER"},
                control_message={
                    "sourceCell": source_cell,
                    "targetCell": target_cell,
                    "reason": "LOAD_BALANCING"
                }
            )


class QoSOptimizationXApp(XAppBase):
    """
    Example xApp: QoS Optimization

    Monitors QoS KPIs and adjusts scheduler parameters.
    """

    def __init__(self, ric_url: str = "http://127.0.0.1:8095"):
        super().__init__("qos-optimization-xapp", ric_url)
        self.latency_threshold_ms = 20
        self.throughput_threshold_mbps = 10

    async def on_indication(self, indication: Dict):
        """Process QoS indication"""
        message = indication.get("indicationMessage", {})
        ue_id = message.get("ueId")
        latency = message.get("latencyMs", 0)
        throughput = message.get("throughputMbps", 0)

        if latency > self.latency_threshold_ms:
            logger.info(f"UE {ue_id} high latency ({latency}ms), adjusting scheduler")
            await self._adjust_scheduler_priority(ue_id, "HIGH")

        elif throughput < self.throughput_threshold_mbps:
            logger.info(f"UE {ue_id} low throughput ({throughput}Mbps), increasing resources")
            await self._increase_resources(ue_id)

    async def _adjust_scheduler_priority(self, ue_id: str, priority: str):
        """Adjust scheduler priority for UE"""
        for node_id, sub in self.subscriptions.items():
            await self.send_control(
                e2_node_id=sub["e2NodeId"],
                ran_function_id=2,
                control_header={"controlType": "SCHEDULER_CONFIG"},
                control_message={
                    "ueId": ue_id,
                    "schedulerPriority": priority
                }
            )

    async def _increase_resources(self, ue_id: str):
        """Increase PRB allocation for UE"""
        for node_id, sub in self.subscriptions.items():
            await self.send_control(
                e2_node_id=sub["e2NodeId"],
                ran_function_id=2,
                control_header={"controlType": "RESOURCE_ALLOCATION"},
                control_message={
                    "ueId": ue_id,
                    "prbIncrease": 10
                }
            )
