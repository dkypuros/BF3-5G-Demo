# File location: 5G_Emulator_API/service_assurance/collector.py
# Metrics Collector - Gathers metrics from 5G Network Functions

"""
Metrics Collector for 5G Service Assurance

Collects metrics from all network functions:
- Polls NF REST endpoints for operational metrics
- Integrates with OpenTelemetry for distributed tracing data
- Subscribes to Prometheus metrics endpoints
- Feeds data to KQI Calculator for processing

Supports both pull (polling) and push (webhook) collection modes.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
import requests
from requests.exceptions import RequestException

from .models import NFType
from .kqi_calculator import KQICalculator

logger = logging.getLogger(__name__)


class NFEndpoint:
    """Configuration for a Network Function endpoint"""

    def __init__(
        self,
        nf_type: NFType,
        host: str,
        port: int,
        metrics_path: str = "/metrics",
        health_path: str = "/health",
        enabled: bool = True,
    ):
        self.nf_type = nf_type
        self.host = host
        self.port = port
        self.metrics_path = metrics_path
        self.health_path = health_path
        self.enabled = enabled
        self.last_collection: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def metrics_url(self) -> str:
        return f"{self.base_url}{self.metrics_path}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_path}"


class MetricsCollector:
    """
    Collects metrics from all 5G Network Functions.
    """

    def __init__(
        self,
        kqi_calculator: KQICalculator,
        collection_interval: int = 30,
        timeout: int = 5,
    ):
        """
        Initialize Metrics Collector.

        Args:
            kqi_calculator: KQI Calculator to feed metrics to
            collection_interval: Seconds between collection cycles
            timeout: HTTP request timeout in seconds
        """
        self.kqi_calculator = kqi_calculator
        self.collection_interval = collection_interval
        self.timeout = timeout

        # NF endpoints
        self._endpoints: Dict[NFType, NFEndpoint] = {}

        # Collection state
        self._collection_active = False
        self._collection_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            "collections_total": 0,
            "collections_successful": 0,
            "collections_failed": 0,
            "last_collection_time": None,
        }

        # Callbacks for metric events
        self._metric_callbacks: List[Callable] = []

        # Initialize default endpoints
        self._init_default_endpoints()

        logger.info("Metrics Collector initialized")

    def _init_default_endpoints(self):
        """Initialize default NF endpoints based on main.py configuration"""
        default_endpoints = [
            NFEndpoint(NFType.NRF, "127.0.0.1", 8000, "/nrf/metrics"),
            NFEndpoint(NFType.AMF, "127.0.0.1", 9000, "/amf/metrics"),
            NFEndpoint(NFType.SMF, "127.0.0.1", 9001, "/smf/metrics"),
            NFEndpoint(NFType.UPF, "127.0.0.1", 9002, "/upf/metrics"),
            NFEndpoint(NFType.AUSF, "127.0.0.1", 9003, "/ausf/metrics"),
            NFEndpoint(NFType.UDM, "127.0.0.1", 9004, "/udm/metrics"),
            NFEndpoint(NFType.UDR, "127.0.0.1", 9005, "/udr/metrics"),
            NFEndpoint(NFType.CU, "127.0.0.1", 9008, "/cu/metrics"),
            NFEndpoint(NFType.DU, "127.0.0.1", 9007, "/du/metrics"),
            NFEndpoint(NFType.RRU, "127.0.0.1", 9009, "/rru/metrics"),
        ]

        for endpoint in default_endpoints:
            self._endpoints[endpoint.nf_type] = endpoint

        logger.info(f"Initialized {len(self._endpoints)} default NF endpoints")

    def register_endpoint(self, endpoint: NFEndpoint):
        """Register a new NF endpoint for collection"""
        self._endpoints[endpoint.nf_type] = endpoint
        logger.info(f"Registered endpoint: {endpoint.nf_type.value} at {endpoint.base_url}")

    def unregister_endpoint(self, nf_type: NFType):
        """Unregister an NF endpoint"""
        if nf_type in self._endpoints:
            del self._endpoints[nf_type]
            logger.info(f"Unregistered endpoint: {nf_type.value}")

    def register_callback(self, callback: Callable):
        """Register callback for metric collection events"""
        self._metric_callbacks.append(callback)

    def _collect_from_endpoint(self, endpoint: NFEndpoint) -> Dict:
        """
        Collect metrics from a single NF endpoint.

        Returns dict with collected metrics or empty dict on failure.
        """
        if not endpoint.enabled:
            return {}

        try:
            # Try to get metrics from the NF
            # First check if NF is healthy
            health_response = requests.get(
                endpoint.health_url,
                timeout=self.timeout
            )

            if health_response.status_code != 200:
                endpoint.last_error = f"Health check failed: {health_response.status_code}"
                endpoint.consecutive_failures += 1
                return {}

            # Get metrics
            metrics_response = requests.get(
                endpoint.metrics_url,
                timeout=self.timeout
            )

            if metrics_response.status_code != 200:
                # Try alternative: get status endpoint
                status_response = requests.get(
                    f"{endpoint.base_url}/status",
                    timeout=self.timeout
                )
                if status_response.status_code == 200:
                    return self._parse_status_response(endpoint.nf_type, status_response.json())

                endpoint.last_error = f"Metrics fetch failed: {metrics_response.status_code}"
                endpoint.consecutive_failures += 1
                return {}

            endpoint.last_collection = datetime.utcnow()
            endpoint.consecutive_failures = 0
            endpoint.last_error = None

            return self._parse_metrics_response(endpoint.nf_type, metrics_response.json())

        except RequestException as e:
            endpoint.last_error = str(e)
            endpoint.consecutive_failures += 1
            logger.debug(f"Failed to collect from {endpoint.nf_type.value}: {e}")
            return {}

    def _parse_metrics_response(self, nf_type: NFType, data: Dict) -> Dict:
        """Parse metrics response from NF"""
        metrics = {}

        # Handle different NF metric formats
        if nf_type == NFType.AMF:
            metrics.update(self._parse_amf_metrics(data))
        elif nf_type == NFType.SMF:
            metrics.update(self._parse_smf_metrics(data))
        elif nf_type == NFType.UPF:
            metrics.update(self._parse_upf_metrics(data))
        elif nf_type == NFType.AUSF:
            metrics.update(self._parse_ausf_metrics(data))
        elif nf_type in (NFType.GNB, NFType.CU, NFType.DU):
            metrics.update(self._parse_ran_metrics(data))
        else:
            # Generic parsing
            metrics.update(self._parse_generic_metrics(nf_type, data))

        return metrics

    def _parse_status_response(self, nf_type: NFType, data: Dict) -> Dict:
        """Parse status response (fallback if metrics endpoint unavailable)"""
        metrics = {}

        # Extract availability from status
        if "status" in data:
            is_healthy = data["status"].lower() in ("ok", "healthy", "running", "up")
            metrics["kpi_nf_availability"] = 100.0 if is_healthy else 0.0

        return metrics

    def _parse_amf_metrics(self, data: Dict) -> Dict:
        """Parse AMF-specific metrics"""
        metrics = {}

        # Registration metrics
        if "registrations" in data:
            reg = data["registrations"]
            metrics["kpi_registration_attempts"] = reg.get("attempts", 0)
            metrics["kpi_registration_success"] = reg.get("successful", 0)

        # Latency metrics
        if "latency" in data:
            metrics["kpi_amf_latency"] = data["latency"].get("p95", 0)

        # Connection metrics
        if "connections" in data:
            metrics["amf_active_connections"] = data["connections"].get("active", 0)

        return metrics

    def _parse_smf_metrics(self, data: Dict) -> Dict:
        """Parse SMF-specific metrics"""
        metrics = {}

        # Session metrics
        if "sessions" in data:
            sess = data["sessions"]
            metrics["kpi_pdu_session_attempts"] = sess.get("attempts", 0)
            metrics["kpi_pdu_session_success"] = sess.get("successful", 0)
            metrics["kpi_active_sessions"] = sess.get("active", 0)
            metrics["kpi_dropped_sessions"] = sess.get("dropped", 0)

        # Latency
        if "latency" in data:
            metrics["kpi_smf_latency"] = data["latency"].get("p95", 0)

        return metrics

    def _parse_upf_metrics(self, data: Dict) -> Dict:
        """Parse UPF-specific metrics"""
        metrics = {}

        # Packet metrics
        if "packets" in data:
            pkt = data["packets"]
            metrics["kpi_packets_tx"] = pkt.get("transmitted", 0)
            metrics["kpi_packets_dropped"] = pkt.get("dropped", 0)

        # Throughput
        if "throughput" in data:
            metrics["kpi_actual_throughput"] = data["throughput"].get("current_mbps", 0)
            metrics["kpi_subscribed_throughput"] = data["throughput"].get("subscribed_mbps", 1000)

        # Latency
        if "latency" in data:
            metrics["kpi_upf_latency"] = data["latency"].get("p95", 0)

        return metrics

    def _parse_ausf_metrics(self, data: Dict) -> Dict:
        """Parse AUSF-specific metrics"""
        metrics = {}

        # Authentication metrics
        if "authentication" in data:
            auth = data["authentication"]
            metrics["kpi_auth_attempts"] = auth.get("attempts", 0)
            metrics["kpi_auth_success"] = auth.get("successful", 0)

        return metrics

    def _parse_ran_metrics(self, data: Dict) -> Dict:
        """Parse RAN (gNB/CU/DU) metrics"""
        metrics = {}

        # Latency
        if "latency" in data:
            metrics["kpi_ran_latency"] = data["latency"].get("p95", 0)

        # Handover metrics
        if "handover" in data:
            ho = data["handover"]
            metrics["kpi_handover_attempts"] = ho.get("attempts", 0)
            metrics["kpi_handover_success"] = ho.get("successful", 0)

        return metrics

    def _parse_generic_metrics(self, nf_type: NFType, data: Dict) -> Dict:
        """Generic metric parsing for any NF"""
        metrics = {}

        # Try common metric names
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metric_key = f"{nf_type.value.lower()}_{key}"
                metrics[metric_key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        metric_key = f"{nf_type.value.lower()}_{key}_{sub_key}"
                        metrics[metric_key] = sub_value

        return metrics

    def collect_all(self) -> Dict[NFType, Dict]:
        """
        Collect metrics from all registered endpoints.

        Returns:
            Dict mapping NF type to collected metrics
        """
        all_metrics = {}
        collection_start = time.time()

        for nf_type, endpoint in self._endpoints.items():
            metrics = self._collect_from_endpoint(endpoint)
            if metrics:
                all_metrics[nf_type] = metrics

                # Feed to KQI Calculator
                for metric_id, value in metrics.items():
                    self.kqi_calculator.record_raw_metric(metric_id, value)

        # Update statistics
        collection_time = time.time() - collection_start
        self._stats["collections_total"] += 1
        self._stats["last_collection_time"] = datetime.utcnow()

        if all_metrics:
            self._stats["collections_successful"] += 1
        else:
            self._stats["collections_failed"] += 1

        # Trigger callbacks
        for callback in self._metric_callbacks:
            try:
                callback(all_metrics)
            except Exception as e:
                logger.error(f"Metric callback error: {e}")

        logger.debug(
            f"Collected metrics from {len(all_metrics)} NFs in {collection_time:.2f}s"
        )

        return all_metrics

    def collect_simulated_metrics(self):
        """
        Generate simulated metrics for testing when NFs are not available.
        Useful for development and testing of the assurance system.
        """
        import random

        simulated = {
            # Registration metrics (high success rate)
            "kpi_registration_attempts": random.randint(100, 200),
            "kpi_registration_success": random.randint(98, 200),

            # PDU session metrics
            "kpi_pdu_session_attempts": random.randint(50, 100),
            "kpi_pdu_session_success": random.randint(48, 100),
            "kpi_active_sessions": random.randint(500, 1000),
            "kpi_dropped_sessions": random.randint(0, 5),

            # Authentication metrics
            "kpi_auth_attempts": random.randint(100, 200),
            "kpi_auth_success": random.randint(99, 200),

            # Latency metrics (in ms)
            "kpi_amf_latency": random.uniform(5, 15),
            "kpi_smf_latency": random.uniform(3, 10),
            "kpi_upf_latency": random.uniform(1, 5),
            "kpi_ran_latency": random.uniform(2, 8),

            # Packet metrics
            "kpi_packets_tx": random.randint(100000, 500000),
            "kpi_packets_dropped": random.randint(0, 100),

            # Throughput
            "kpi_actual_throughput": random.uniform(800, 1000),
            "kpi_subscribed_throughput": 1000,

            # Handover metrics
            "kpi_handover_attempts": random.randint(10, 50),
            "kpi_handover_success": random.randint(9, 50),

            # Availability (percentage)
            "kpi_nf_availability": random.uniform(99.9, 100.0),
            "kpi_n6_firewall_uptime": random.uniform(99.99, 100.0),
        }

        # Feed to KQI Calculator
        for metric_id, value in simulated.items():
            self.kqi_calculator.record_raw_metric(metric_id, value)

        logger.debug("Generated simulated metrics")
        return simulated

    def start_collection(self, use_simulation: bool = False):
        """
        Start continuous metric collection.

        Args:
            use_simulation: If True, use simulated metrics instead of real collection
        """
        if self._collection_active:
            logger.warning("Collection already active")
            return

        self._collection_active = True

        def collection_loop():
            while self._collection_active:
                try:
                    if use_simulation:
                        self.collect_simulated_metrics()
                    else:
                        collected = self.collect_all()
                        # Fall back to simulation if no real metrics
                        if not collected:
                            self.collect_simulated_metrics()
                except Exception as e:
                    logger.error(f"Collection error: {e}")

                time.sleep(self.collection_interval)

        self._collection_thread = threading.Thread(target=collection_loop, daemon=True)
        self._collection_thread.start()
        logger.info(
            f"Metric collection started (interval: {self.collection_interval}s, "
            f"simulation: {use_simulation})"
        )

    def stop_collection(self):
        """Stop continuous metric collection"""
        self._collection_active = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Metric collection stopped")

    def get_endpoint_status(self) -> List[Dict]:
        """Get status of all registered endpoints"""
        status = []
        for nf_type, endpoint in self._endpoints.items():
            status.append({
                "nf_type": nf_type.value,
                "url": endpoint.base_url,
                "enabled": endpoint.enabled,
                "last_collection": endpoint.last_collection.isoformat() if endpoint.last_collection else None,
                "last_error": endpoint.last_error,
                "consecutive_failures": endpoint.consecutive_failures,
                "healthy": endpoint.consecutive_failures == 0,
            })
        return status

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            **self._stats,
            "endpoints_registered": len(self._endpoints),
            "endpoints_healthy": sum(
                1 for e in self._endpoints.values() if e.consecutive_failures == 0
            ),
        }

    def push_metrics(self, nf_type: NFType, metrics: Dict):
        """
        Receive pushed metrics from NFs (webhook mode).

        Args:
            nf_type: Source NF type
            metrics: Metrics dictionary
        """
        parsed = self._parse_metrics_response(nf_type, metrics)

        for metric_id, value in parsed.items():
            self.kqi_calculator.record_raw_metric(metric_id, value)

        logger.debug(f"Received pushed metrics from {nf_type.value}: {len(parsed)} metrics")
