# File location: 5G_Emulator_API/service_assurance/kqi_calculator.py
# KQI/KPI Calculator - 3GPP TS 28.552/554 aligned metrics

"""
KQI/KPI Calculator for 5G Service Assurance

Implements Key Quality Indicators (customer-facing) and Key Performance Indicators
(network-facing) based on 3GPP specifications:
- TS 28.552: 5G Performance Measurements
- TS 28.554: End-to-end KPIs for 5G networks

KQI Categories:
- Accessibility: Registration success, PDU session establishment
- Retainability: Session continuity, handover success
- Integrity: Throughput, latency, packet loss
- Availability: NF availability, service availability
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import statistics
import uuid

from .models import (
    KQIMetric,
    KPIMetric,
    MetricCategory,
    NFType,
)

logger = logging.getLogger(__name__)


class KQICalculator:
    """
    Calculates Key Quality Indicators and Key Performance Indicators
    for 5G network service assurance.
    """

    def __init__(self, history_window_minutes: int = 60):
        """
        Initialize KQI Calculator.

        Args:
            history_window_minutes: Time window for historical metrics
        """
        self.history_window = timedelta(minutes=history_window_minutes)

        # Raw metric storage: metric_id -> deque of (timestamp, value)
        self._raw_metrics: Dict[str, deque] = {}

        # Computed KQIs and KPIs cache
        self._kqi_cache: Dict[str, KQIMetric] = {}
        self._kpi_cache: Dict[str, KPIMetric] = {}

        # Define KQI/KPI specifications
        self._init_kqi_definitions()
        self._init_kpi_definitions()

        logger.info("KQI Calculator initialized")

    def _init_kqi_definitions(self):
        """Initialize KQI definitions based on 3GPP TS 28.554"""
        self.kqi_definitions = {
            # Accessibility KQIs
            "kqi_registration_success_rate": {
                "name": "Registration Success Rate",
                "category": MetricCategory.ACCESSIBILITY,
                "unit": "%",
                "target": 99.9,
                "warning_threshold": 99.5,
                "critical_threshold": 99.0,
                "formula": "(successful_registrations / total_registration_attempts) * 100",
                "source_kpis": ["kpi_registration_attempts", "kpi_registration_success"],
            },
            "kqi_pdu_session_success_rate": {
                "name": "PDU Session Establishment Success Rate",
                "category": MetricCategory.ACCESSIBILITY,
                "unit": "%",
                "target": 99.9,
                "warning_threshold": 99.0,
                "critical_threshold": 98.0,
                "formula": "(successful_pdu_sessions / total_pdu_session_attempts) * 100",
                "source_kpis": ["kpi_pdu_session_attempts", "kpi_pdu_session_success"],
            },
            "kqi_authentication_success_rate": {
                "name": "Authentication Success Rate",
                "category": MetricCategory.ACCESSIBILITY,
                "unit": "%",
                "target": 99.99,
                "warning_threshold": 99.9,
                "critical_threshold": 99.5,
                "formula": "(successful_auth / total_auth_attempts) * 100",
                "source_kpis": ["kpi_auth_attempts", "kpi_auth_success"],
            },

            # Retainability KQIs
            "kqi_session_continuity_rate": {
                "name": "Session Continuity Rate",
                "category": MetricCategory.RETAINABILITY,
                "unit": "%",
                "target": 99.9,
                "warning_threshold": 99.5,
                "critical_threshold": 99.0,
                "formula": "(sessions_maintained / total_active_sessions) * 100",
                "source_kpis": ["kpi_active_sessions", "kpi_dropped_sessions"],
            },
            "kqi_handover_success_rate": {
                "name": "Handover Success Rate",
                "category": MetricCategory.MOBILITY,
                "unit": "%",
                "target": 99.5,
                "warning_threshold": 99.0,
                "critical_threshold": 98.0,
                "formula": "(successful_handovers / total_handover_attempts) * 100",
                "source_kpis": ["kpi_handover_attempts", "kpi_handover_success"],
            },

            # Integrity KQIs
            "kqi_user_plane_latency": {
                "name": "User Plane Latency (E2E)",
                "category": MetricCategory.INTEGRITY,
                "unit": "ms",
                "target": 10.0,  # eMBB target
                "warning_threshold": 15.0,
                "critical_threshold": 20.0,
                "formula": "p95(upf_latency + ran_latency)",
                "source_kpis": ["kpi_upf_latency", "kpi_ran_latency"],
            },
            "kqi_control_plane_latency": {
                "name": "Control Plane Latency",
                "category": MetricCategory.INTEGRITY,
                "unit": "ms",
                "target": 50.0,
                "warning_threshold": 75.0,
                "critical_threshold": 100.0,
                "formula": "p95(amf_processing_time + smf_processing_time)",
                "source_kpis": ["kpi_amf_latency", "kpi_smf_latency"],
            },
            "kqi_packet_loss_rate": {
                "name": "Packet Loss Rate",
                "category": MetricCategory.INTEGRITY,
                "unit": "%",
                "target": 0.01,
                "warning_threshold": 0.1,
                "critical_threshold": 1.0,
                "formula": "(dropped_packets / total_packets) * 100",
                "source_kpis": ["kpi_packets_tx", "kpi_packets_dropped"],
            },
            "kqi_throughput_achievement": {
                "name": "Throughput Achievement Rate",
                "category": MetricCategory.INTEGRITY,
                "unit": "%",
                "target": 95.0,
                "warning_threshold": 90.0,
                "critical_threshold": 80.0,
                "formula": "(actual_throughput / subscribed_throughput) * 100",
                "source_kpis": ["kpi_actual_throughput", "kpi_subscribed_throughput"],
            },

            # Availability KQIs
            "kqi_service_availability": {
                "name": "5G Service Availability",
                "category": MetricCategory.AVAILABILITY,
                "unit": "%",
                "target": 99.99,
                "warning_threshold": 99.9,
                "critical_threshold": 99.5,
                "formula": "(service_uptime / total_time) * 100",
                "source_kpis": ["kpi_nf_availability"],
            },
            "kqi_n6_firewall_availability": {
                "name": "N6 Firewall Availability",
                "category": MetricCategory.AVAILABILITY,
                "unit": "%",
                "target": 99.999,
                "warning_threshold": 99.99,
                "critical_threshold": 99.9,
                "formula": "(firewall_uptime / total_time) * 100",
                "source_kpis": ["kpi_n6_firewall_uptime"],
            },
        }

    def _init_kpi_definitions(self):
        """Initialize KPI definitions based on 3GPP TS 28.552"""
        self.kpi_definitions = {
            # AMF KPIs
            "kpi_registration_attempts": {
                "name": "Registration Attempts",
                "category": MetricCategory.ACCESSIBILITY,
                "nf_type": NFType.AMF,
                "unit": "count",
            },
            "kpi_registration_success": {
                "name": "Successful Registrations",
                "category": MetricCategory.ACCESSIBILITY,
                "nf_type": NFType.AMF,
                "unit": "count",
            },
            "kpi_amf_latency": {
                "name": "AMF Processing Latency",
                "category": MetricCategory.INTEGRITY,
                "nf_type": NFType.AMF,
                "unit": "ms",
            },

            # SMF KPIs
            "kpi_pdu_session_attempts": {
                "name": "PDU Session Establishment Attempts",
                "category": MetricCategory.ACCESSIBILITY,
                "nf_type": NFType.SMF,
                "unit": "count",
            },
            "kpi_pdu_session_success": {
                "name": "Successful PDU Sessions",
                "category": MetricCategory.ACCESSIBILITY,
                "nf_type": NFType.SMF,
                "unit": "count",
            },
            "kpi_smf_latency": {
                "name": "SMF Processing Latency",
                "category": MetricCategory.INTEGRITY,
                "nf_type": NFType.SMF,
                "unit": "ms",
            },

            # UPF KPIs
            "kpi_upf_latency": {
                "name": "UPF Processing Latency",
                "category": MetricCategory.INTEGRITY,
                "nf_type": NFType.UPF,
                "unit": "ms",
            },
            "kpi_packets_tx": {
                "name": "Packets Transmitted",
                "category": MetricCategory.INTEGRITY,
                "nf_type": NFType.UPF,
                "unit": "count",
            },
            "kpi_packets_dropped": {
                "name": "Packets Dropped",
                "category": MetricCategory.INTEGRITY,
                "nf_type": NFType.UPF,
                "unit": "count",
            },
            "kpi_actual_throughput": {
                "name": "Actual Throughput",
                "category": MetricCategory.INTEGRITY,
                "nf_type": NFType.UPF,
                "unit": "Mbps",
            },
            "kpi_subscribed_throughput": {
                "name": "Subscribed Throughput",
                "category": MetricCategory.INTEGRITY,
                "nf_type": NFType.UPF,
                "unit": "Mbps",
            },

            # AUSF KPIs
            "kpi_auth_attempts": {
                "name": "Authentication Attempts",
                "category": MetricCategory.ACCESSIBILITY,
                "nf_type": NFType.AUSF,
                "unit": "count",
            },
            "kpi_auth_success": {
                "name": "Successful Authentications",
                "category": MetricCategory.ACCESSIBILITY,
                "nf_type": NFType.AUSF,
                "unit": "count",
            },

            # RAN KPIs
            "kpi_ran_latency": {
                "name": "RAN Latency",
                "category": MetricCategory.INTEGRITY,
                "nf_type": NFType.GNB,
                "unit": "ms",
            },
            "kpi_handover_attempts": {
                "name": "Handover Attempts",
                "category": MetricCategory.MOBILITY,
                "nf_type": NFType.GNB,
                "unit": "count",
            },
            "kpi_handover_success": {
                "name": "Successful Handovers",
                "category": MetricCategory.MOBILITY,
                "nf_type": NFType.GNB,
                "unit": "count",
            },

            # Session KPIs
            "kpi_active_sessions": {
                "name": "Active PDU Sessions",
                "category": MetricCategory.RETAINABILITY,
                "nf_type": NFType.SMF,
                "unit": "count",
            },
            "kpi_dropped_sessions": {
                "name": "Dropped Sessions",
                "category": MetricCategory.RETAINABILITY,
                "nf_type": NFType.SMF,
                "unit": "count",
            },

            # Availability KPIs
            "kpi_nf_availability": {
                "name": "NF Availability",
                "category": MetricCategory.AVAILABILITY,
                "nf_type": NFType.NRF,
                "unit": "%",
            },
            "kpi_n6_firewall_uptime": {
                "name": "N6 Firewall Uptime",
                "category": MetricCategory.AVAILABILITY,
                "nf_type": NFType.N6_FIREWALL,
                "unit": "%",
            },
        }

    def record_raw_metric(self, metric_id: str, value: float, timestamp: Optional[datetime] = None):
        """
        Record a raw metric value for later computation.

        Args:
            metric_id: The metric identifier
            value: The metric value
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if metric_id not in self._raw_metrics:
            self._raw_metrics[metric_id] = deque(maxlen=10000)

        self._raw_metrics[metric_id].append((timestamp, value))
        self._cleanup_old_metrics(metric_id)

    def _cleanup_old_metrics(self, metric_id: str):
        """Remove metrics older than the history window"""
        if metric_id not in self._raw_metrics:
            return

        cutoff = datetime.utcnow() - self.history_window
        while self._raw_metrics[metric_id] and self._raw_metrics[metric_id][0][0] < cutoff:
            self._raw_metrics[metric_id].popleft()

    def _get_metric_values(self, metric_id: str,
                          window_minutes: Optional[int] = None) -> List[float]:
        """Get metric values within a time window"""
        if metric_id not in self._raw_metrics:
            return []

        if window_minutes is None:
            return [v for _, v in self._raw_metrics[metric_id]]

        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        return [v for t, v in self._raw_metrics[metric_id] if t >= cutoff]

    def _get_latest_value(self, metric_id: str) -> Optional[float]:
        """Get the most recent value for a metric"""
        if metric_id not in self._raw_metrics or not self._raw_metrics[metric_id]:
            return None
        return self._raw_metrics[metric_id][-1][1]

    def _compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute statistical measures for a list of values"""
        if not values:
            return {}

        result = {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
        }

        if len(values) >= 2:
            result["std_dev"] = statistics.stdev(values)
            result["median"] = statistics.median(values)

        # Percentiles for latency metrics
        if len(values) >= 10:
            sorted_values = sorted(values)
            result["p50"] = sorted_values[int(len(sorted_values) * 0.50)]
            result["p90"] = sorted_values[int(len(sorted_values) * 0.90)]
            result["p95"] = sorted_values[int(len(sorted_values) * 0.95)]
            result["p99"] = sorted_values[int(len(sorted_values) * 0.99)]

        return result

    def compute_kpi(self, kpi_id: str, window_minutes: int = 5) -> Optional[KPIMetric]:
        """
        Compute a KPI from raw metrics.

        Args:
            kpi_id: The KPI identifier
            window_minutes: Time window for computation

        Returns:
            Computed KPIMetric or None if insufficient data
        """
        if kpi_id not in self.kpi_definitions:
            logger.warning(f"Unknown KPI: {kpi_id}")
            return None

        definition = self.kpi_definitions[kpi_id]
        values = self._get_metric_values(kpi_id, window_minutes)

        if not values:
            # Return with zero/default value if no data
            return KPIMetric(
                kpi_id=kpi_id,
                name=definition["name"],
                category=definition["category"],
                nf_type=definition["nf_type"],
                value=0.0,
                unit=definition["unit"],
            )

        stats = self._compute_statistics(values)

        # For count metrics, use sum; for latency/rate metrics, use mean
        if definition["unit"] == "count":
            value = stats.get("sum", 0)
        elif definition["unit"] in ("ms", "us"):
            value = stats.get("p95", stats.get("mean", 0))  # Use P95 for latency
        else:
            value = stats.get("mean", 0)

        kpi = KPIMetric(
            kpi_id=kpi_id,
            name=definition["name"],
            category=definition["category"],
            nf_type=definition["nf_type"],
            value=value,
            unit=definition["unit"],
            baseline_value=stats.get("mean"),
            min_value=stats.get("min"),
            max_value=stats.get("max"),
            std_deviation=stats.get("std_dev"),
        )

        self._kpi_cache[kpi_id] = kpi
        return kpi

    def compute_kqi(self, kqi_id: str, window_minutes: int = 5) -> Optional[KQIMetric]:
        """
        Compute a KQI from underlying KPIs.

        Args:
            kqi_id: The KQI identifier
            window_minutes: Time window for computation

        Returns:
            Computed KQIMetric or None if insufficient data
        """
        if kqi_id not in self.kqi_definitions:
            logger.warning(f"Unknown KQI: {kqi_id}")
            return None

        definition = self.kqi_definitions[kqi_id]

        # Compute based on KQI type
        value = self._calculate_kqi_value(kqi_id, definition, window_minutes)

        kqi = KQIMetric(
            kqi_id=kqi_id,
            name=definition["name"],
            category=definition["category"],
            value=value,
            unit=definition["unit"],
            target_value=definition.get("target"),
            warning_threshold=definition.get("warning_threshold"),
            critical_threshold=definition.get("critical_threshold"),
            source_kpis=definition.get("source_kpis", []),
            calculation_formula=definition.get("formula"),
        )

        self._kqi_cache[kqi_id] = kqi
        return kqi

    def _calculate_kqi_value(self, kqi_id: str, definition: Dict,
                            window_minutes: int) -> float:
        """Calculate KQI value based on its formula"""

        # Success rate calculations
        if "success_rate" in kqi_id:
            return self._calculate_success_rate(definition, window_minutes)

        # Latency calculations
        if "latency" in kqi_id:
            return self._calculate_latency(definition, window_minutes)

        # Packet loss calculation
        if "packet_loss" in kqi_id:
            return self._calculate_packet_loss(window_minutes)

        # Throughput achievement
        if "throughput_achievement" in kqi_id:
            return self._calculate_throughput_achievement(window_minutes)

        # Availability calculations
        if "availability" in kqi_id:
            return self._calculate_availability(definition, window_minutes)

        # Default: return 0
        return 0.0

    def _calculate_success_rate(self, definition: Dict, window_minutes: int) -> float:
        """Calculate success rate KQI"""
        source_kpis = definition.get("source_kpis", [])
        if len(source_kpis) < 2:
            return 100.0  # Default to 100% if no data

        attempts_kpi = source_kpis[0]
        success_kpi = source_kpis[1]

        attempts = self._get_metric_values(attempts_kpi, window_minutes)
        successes = self._get_metric_values(success_kpi, window_minutes)

        total_attempts = sum(attempts) if attempts else 0
        total_successes = sum(successes) if successes else 0

        if total_attempts == 0:
            return 100.0  # No attempts = 100% success

        return (total_successes / total_attempts) * 100

    def _calculate_latency(self, definition: Dict, window_minutes: int) -> float:
        """Calculate latency KQI (P95)"""
        source_kpis = definition.get("source_kpis", [])
        all_latencies = []

        for kpi_id in source_kpis:
            latencies = self._get_metric_values(kpi_id, window_minutes)
            all_latencies.extend(latencies)

        if not all_latencies:
            return 0.0

        stats = self._compute_statistics(all_latencies)
        return stats.get("p95", stats.get("mean", 0))

    def _calculate_packet_loss(self, window_minutes: int) -> float:
        """Calculate packet loss rate"""
        tx_values = self._get_metric_values("kpi_packets_tx", window_minutes)
        dropped_values = self._get_metric_values("kpi_packets_dropped", window_minutes)

        total_tx = sum(tx_values) if tx_values else 0
        total_dropped = sum(dropped_values) if dropped_values else 0

        if total_tx == 0:
            return 0.0

        return (total_dropped / total_tx) * 100

    def _calculate_throughput_achievement(self, window_minutes: int) -> float:
        """Calculate throughput achievement rate"""
        actual = self._get_metric_values("kpi_actual_throughput", window_minutes)
        subscribed = self._get_metric_values("kpi_subscribed_throughput", window_minutes)

        avg_actual = statistics.mean(actual) if actual else 0
        avg_subscribed = statistics.mean(subscribed) if subscribed else 0

        if avg_subscribed == 0:
            return 100.0

        return min(100.0, (avg_actual / avg_subscribed) * 100)

    def _calculate_availability(self, definition: Dict, window_minutes: int) -> float:
        """Calculate availability KQI"""
        source_kpis = definition.get("source_kpis", [])

        availabilities = []
        for kpi_id in source_kpis:
            values = self._get_metric_values(kpi_id, window_minutes)
            if values:
                availabilities.append(statistics.mean(values))

        if not availabilities:
            return 100.0  # Default to 100% if no data

        return statistics.mean(availabilities)

    def compute_all_kqis(self, window_minutes: int = 5) -> List[KQIMetric]:
        """Compute all defined KQIs"""
        kqis = []
        for kqi_id in self.kqi_definitions:
            kqi = self.compute_kqi(kqi_id, window_minutes)
            if kqi:
                kqis.append(kqi)
        return kqis

    def compute_all_kpis(self, window_minutes: int = 5) -> List[KPIMetric]:
        """Compute all defined KPIs"""
        kpis = []
        for kpi_id in self.kpi_definitions:
            kpi = self.compute_kpi(kpi_id, window_minutes)
            if kpi:
                kpis.append(kpi)
        return kpis

    def get_kqi(self, kqi_id: str) -> Optional[KQIMetric]:
        """Get cached KQI or compute it"""
        if kqi_id in self._kqi_cache:
            return self._kqi_cache[kqi_id]
        return self.compute_kqi(kqi_id)

    def get_kpi(self, kpi_id: str) -> Optional[KPIMetric]:
        """Get cached KPI or compute it"""
        if kpi_id in self._kpi_cache:
            return self._kpi_cache[kpi_id]
        return self.compute_kpi(kpi_id)

    def get_health_score(self) -> Tuple[float, str]:
        """
        Calculate overall health score based on KQIs.

        Returns:
            Tuple of (score 0-100, health_state)
        """
        kqis = self.compute_all_kqis()
        if not kqis:
            return 100.0, "unknown"

        scores = []
        for kqi in kqis:
            if kqi.target_value is None:
                continue

            # Calculate how well we're meeting the target
            if kqi.unit == "%":
                # For percentage metrics, compare directly
                if kqi.critical_threshold and kqi.value < kqi.critical_threshold:
                    scores.append(50)  # Critical
                elif kqi.warning_threshold and kqi.value < kqi.warning_threshold:
                    scores.append(75)  # Warning
                else:
                    scores.append(100)  # Good
            elif kqi.unit == "ms":
                # For latency, lower is better
                if kqi.critical_threshold and kqi.value > kqi.critical_threshold:
                    scores.append(50)
                elif kqi.warning_threshold and kqi.value > kqi.warning_threshold:
                    scores.append(75)
                else:
                    scores.append(100)

        if not scores:
            return 100.0, "healthy"

        avg_score = statistics.mean(scores)

        if avg_score >= 90:
            state = "healthy"
        elif avg_score >= 70:
            state = "degraded"
        else:
            state = "critical"

        return avg_score, state
