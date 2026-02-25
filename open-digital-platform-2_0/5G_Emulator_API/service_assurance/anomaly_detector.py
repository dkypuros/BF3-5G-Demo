# File location: 5G_Emulator_API/service_assurance/anomaly_detector.py
# Anomaly Detector - Statistical anomaly detection for 5G metrics

"""
Anomaly Detector for 5G Service Assurance

Implements multiple anomaly detection algorithms:
- Z-Score based detection for outliers
- Moving average deviation detection
- Rate of change detection for spikes/drops
- Seasonal pattern detection

These methods detect issues before they become SLA violations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque
import statistics
import math
import uuid
import threading

from .models import (
    AnomalyEvent,
    AnomalyType,
    SeverityLevel,
    NFType,
)
from .kqi_calculator import KQICalculator

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in 5G network metrics using statistical methods.
    """

    def __init__(
        self,
        kqi_calculator: KQICalculator,
        zscore_threshold: float = 3.0,
        rate_change_threshold: float = 50.0,
        min_samples: int = 30,
    ):
        """
        Initialize Anomaly Detector.

        Args:
            kqi_calculator: KQI Calculator for metric access
            zscore_threshold: Z-score threshold for outlier detection
            rate_change_threshold: Percentage change threshold for spike/drop
            min_samples: Minimum samples needed for detection
        """
        self.kqi_calculator = kqi_calculator
        self.zscore_threshold = zscore_threshold
        self.rate_change_threshold = rate_change_threshold
        self.min_samples = min_samples

        # Metric baselines: metric_id -> {mean, std_dev, min, max}
        self._baselines: Dict[str, Dict] = {}

        # Moving averages: metric_id -> deque of values
        self._moving_averages: Dict[str, deque] = {}

        # Active anomalies
        self._active_anomalies: Dict[str, AnomalyEvent] = {}

        # Anomaly history
        self._anomaly_history: List[AnomalyEvent] = []

        # Callbacks
        self._anomaly_callbacks: List[Callable[[AnomalyEvent], None]] = []

        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Configure metric-specific thresholds
        self._metric_config = self._init_metric_config()

        logger.info("Anomaly Detector initialized")

    def _init_metric_config(self) -> Dict[str, Dict]:
        """Initialize metric-specific detection configuration"""
        return {
            # Latency metrics - sensitive to increases
            "kqi_user_plane_latency": {
                "zscore_threshold": 2.5,
                "rate_threshold": 30.0,
                "direction": "high",  # Anomaly when value is high
                "severity_map": {2.5: SeverityLevel.WARNING, 3.0: SeverityLevel.MAJOR, 4.0: SeverityLevel.CRITICAL},
            },
            "kqi_control_plane_latency": {
                "zscore_threshold": 2.5,
                "rate_threshold": 40.0,
                "direction": "high",
                "severity_map": {2.5: SeverityLevel.WARNING, 3.0: SeverityLevel.MAJOR, 4.0: SeverityLevel.CRITICAL},
            },
            # Success rate metrics - sensitive to decreases
            "kqi_registration_success_rate": {
                "zscore_threshold": 2.0,
                "rate_threshold": 5.0,  # Even small drops are significant
                "direction": "low",  # Anomaly when value is low
                "severity_map": {2.0: SeverityLevel.WARNING, 2.5: SeverityLevel.MAJOR, 3.0: SeverityLevel.CRITICAL},
            },
            "kqi_pdu_session_success_rate": {
                "zscore_threshold": 2.0,
                "rate_threshold": 5.0,
                "direction": "low",
                "severity_map": {2.0: SeverityLevel.WARNING, 2.5: SeverityLevel.MAJOR, 3.0: SeverityLevel.CRITICAL},
            },
            "kqi_authentication_success_rate": {
                "zscore_threshold": 2.0,
                "rate_threshold": 2.0,  # Very sensitive
                "direction": "low",
                "severity_map": {2.0: SeverityLevel.WARNING, 2.5: SeverityLevel.MAJOR, 3.0: SeverityLevel.CRITICAL},
            },
            # Packet loss - sensitive to increases
            "kqi_packet_loss_rate": {
                "zscore_threshold": 2.0,
                "rate_threshold": 100.0,  # Packet loss can spike significantly
                "direction": "high",
                "severity_map": {2.0: SeverityLevel.WARNING, 2.5: SeverityLevel.MAJOR, 3.0: SeverityLevel.CRITICAL},
            },
            # Availability - very sensitive
            "kqi_service_availability": {
                "zscore_threshold": 1.5,
                "rate_threshold": 1.0,  # Any drop is significant
                "direction": "low",
                "severity_map": {1.5: SeverityLevel.WARNING, 2.0: SeverityLevel.MAJOR, 2.5: SeverityLevel.CRITICAL},
            },
        }

    def register_anomaly_callback(self, callback: Callable[[AnomalyEvent], None]):
        """Register callback for anomaly events"""
        self._anomaly_callbacks.append(callback)

    def update_baseline(self, metric_id: str, values: List[float]):
        """
        Update baseline statistics for a metric.

        Args:
            metric_id: Metric identifier
            values: Historical values for baseline calculation
        """
        if len(values) < self.min_samples:
            logger.debug(f"Insufficient samples for baseline: {metric_id}")
            return

        self._baselines[metric_id] = {
            "mean": statistics.mean(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "updated_at": datetime.utcnow(),
        }

        logger.debug(f"Updated baseline for {metric_id}: mean={self._baselines[metric_id]['mean']:.2f}")

    def _get_metric_config(self, metric_id: str) -> Dict:
        """Get configuration for a specific metric"""
        return self._metric_config.get(metric_id, {
            "zscore_threshold": self.zscore_threshold,
            "rate_threshold": self.rate_change_threshold,
            "direction": "both",
            "severity_map": {3.0: SeverityLevel.WARNING, 4.0: SeverityLevel.MAJOR, 5.0: SeverityLevel.CRITICAL},
        })

    def calculate_zscore(self, metric_id: str, value: float) -> Optional[float]:
        """
        Calculate Z-score for a value against baseline.

        Args:
            metric_id: Metric identifier
            value: Current value

        Returns:
            Z-score or None if no baseline
        """
        baseline = self._baselines.get(metric_id)
        if not baseline or baseline["std_dev"] == 0:
            return None

        zscore = (value - baseline["mean"]) / baseline["std_dev"]
        return zscore

    def detect_zscore_anomaly(
        self, metric_id: str, value: float, nf_type: Optional[NFType] = None
    ) -> Optional[AnomalyEvent]:
        """
        Detect anomaly using Z-score method.

        Args:
            metric_id: Metric identifier
            value: Current value
            nf_type: Optional network function type

        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        zscore = self.calculate_zscore(metric_id, value)
        if zscore is None:
            return None

        config = self._get_metric_config(metric_id)
        threshold = config["zscore_threshold"]
        direction = config["direction"]

        # Check if anomaly based on direction
        is_anomaly = False
        if direction == "high" and zscore > threshold:
            is_anomaly = True
        elif direction == "low" and zscore < -threshold:
            is_anomaly = True
        elif direction == "both" and abs(zscore) > threshold:
            is_anomaly = True

        if not is_anomaly:
            return None

        # Determine severity
        severity = SeverityLevel.WARNING
        abs_zscore = abs(zscore)
        for z_thresh, sev in sorted(config["severity_map"].items()):
            if abs_zscore >= z_thresh:
                severity = sev

        # Determine anomaly type
        anomaly_type = AnomalyType.SPIKE if zscore > 0 else AnomalyType.DROP

        baseline = self._baselines[metric_id]
        anomaly = AnomalyEvent(
            anomaly_id=str(uuid.uuid4()),
            anomaly_type=anomaly_type,
            metric_id=metric_id,
            metric_name=metric_id.replace("kqi_", "").replace("kpi_", "").replace("_", " ").title(),
            nf_type=nf_type,
            current_value=value,
            expected_value=baseline["mean"],
            deviation_score=abs_zscore,
            confidence=min(0.99, 0.5 + (abs_zscore - threshold) * 0.1),
            severity=severity,
            description=f"Z-score anomaly detected: {metric_id} value {value:.2f} "
                       f"(z-score: {zscore:.2f}, baseline: {baseline['mean']:.2f})",
        )

        return anomaly

    def detect_rate_change_anomaly(
        self,
        metric_id: str,
        current_value: float,
        previous_value: float,
        nf_type: Optional[NFType] = None,
    ) -> Optional[AnomalyEvent]:
        """
        Detect anomaly based on rate of change.

        Args:
            metric_id: Metric identifier
            current_value: Current value
            previous_value: Previous value
            nf_type: Optional network function type

        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        if previous_value == 0:
            return None

        rate_change = ((current_value - previous_value) / previous_value) * 100
        config = self._get_metric_config(metric_id)
        threshold = config["rate_threshold"]
        direction = config["direction"]

        # Check if anomaly based on direction
        is_anomaly = False
        if direction == "high" and rate_change > threshold:
            is_anomaly = True
        elif direction == "low" and rate_change < -threshold:
            is_anomaly = True
        elif direction == "both" and abs(rate_change) > threshold:
            is_anomaly = True

        if not is_anomaly:
            return None

        # Determine severity based on magnitude
        abs_change = abs(rate_change)
        if abs_change > threshold * 3:
            severity = SeverityLevel.CRITICAL
        elif abs_change > threshold * 2:
            severity = SeverityLevel.MAJOR
        else:
            severity = SeverityLevel.WARNING

        anomaly_type = AnomalyType.SPIKE if rate_change > 0 else AnomalyType.DROP

        anomaly = AnomalyEvent(
            anomaly_id=str(uuid.uuid4()),
            anomaly_type=anomaly_type,
            metric_id=metric_id,
            metric_name=metric_id.replace("kqi_", "").replace("kpi_", "").replace("_", " ").title(),
            nf_type=nf_type,
            current_value=current_value,
            expected_value=previous_value,
            deviation_score=abs_change / threshold,
            confidence=min(0.99, 0.5 + (abs_change - threshold) / threshold * 0.2),
            severity=severity,
            description=f"Rate change anomaly: {metric_id} changed {rate_change:.1f}% "
                       f"(from {previous_value:.2f} to {current_value:.2f})",
        )

        return anomaly

    def detect_moving_average_anomaly(
        self,
        metric_id: str,
        value: float,
        window_size: int = 10,
        nf_type: Optional[NFType] = None,
    ) -> Optional[AnomalyEvent]:
        """
        Detect anomaly using moving average deviation.

        Args:
            metric_id: Metric identifier
            value: Current value
            window_size: Moving average window size
            nf_type: Optional network function type

        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        if metric_id not in self._moving_averages:
            self._moving_averages[metric_id] = deque(maxlen=window_size * 2)

        self._moving_averages[metric_id].append(value)

        if len(self._moving_averages[metric_id]) < window_size:
            return None

        # Calculate moving average
        recent_values = list(self._moving_averages[metric_id])[-window_size:]
        ma = statistics.mean(recent_values)

        # Calculate deviation from MA
        if ma == 0:
            return None

        deviation_pct = abs((value - ma) / ma) * 100
        config = self._get_metric_config(metric_id)

        # Use rate threshold as deviation threshold
        if deviation_pct < config["rate_threshold"]:
            return None

        # Determine severity
        if deviation_pct > config["rate_threshold"] * 3:
            severity = SeverityLevel.CRITICAL
        elif deviation_pct > config["rate_threshold"] * 2:
            severity = SeverityLevel.MAJOR
        else:
            severity = SeverityLevel.WARNING

        anomaly_type = AnomalyType.SPIKE if value > ma else AnomalyType.DROP

        anomaly = AnomalyEvent(
            anomaly_id=str(uuid.uuid4()),
            anomaly_type=anomaly_type,
            metric_id=metric_id,
            metric_name=metric_id.replace("kqi_", "").replace("kpi_", "").replace("_", " ").title(),
            nf_type=nf_type,
            current_value=value,
            expected_value=ma,
            deviation_score=deviation_pct / config["rate_threshold"],
            confidence=min(0.95, 0.4 + deviation_pct / 100),
            severity=severity,
            description=f"Moving average deviation: {metric_id} value {value:.2f} "
                       f"deviates {deviation_pct:.1f}% from MA {ma:.2f}",
        )

        return anomaly

    def analyze_metric(
        self,
        metric_id: str,
        value: float,
        previous_value: Optional[float] = None,
        nf_type: Optional[NFType] = None,
    ) -> List[AnomalyEvent]:
        """
        Run all anomaly detection methods on a metric.

        Args:
            metric_id: Metric identifier
            value: Current value
            previous_value: Optional previous value for rate change detection
            nf_type: Optional network function type

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Z-score detection
        zscore_anomaly = self.detect_zscore_anomaly(metric_id, value, nf_type)
        if zscore_anomaly:
            anomalies.append(zscore_anomaly)

        # Rate change detection
        if previous_value is not None:
            rate_anomaly = self.detect_rate_change_anomaly(
                metric_id, value, previous_value, nf_type
            )
            if rate_anomaly:
                anomalies.append(rate_anomaly)

        # Moving average detection
        ma_anomaly = self.detect_moving_average_anomaly(metric_id, value, nf_type=nf_type)
        if ma_anomaly:
            anomalies.append(ma_anomaly)

        # Deduplicate and merge similar anomalies
        if len(anomalies) > 1:
            anomalies = self._merge_anomalies(anomalies)

        # Process detected anomalies
        for anomaly in anomalies:
            self._process_anomaly(anomaly)

        return anomalies

    def _merge_anomalies(self, anomalies: List[AnomalyEvent]) -> List[AnomalyEvent]:
        """Merge similar anomalies detected by different methods"""
        if len(anomalies) <= 1:
            return anomalies

        # Take the highest severity anomaly and boost confidence
        sorted_anomalies = sorted(
            anomalies,
            key=lambda a: (
                {"critical": 4, "major": 3, "minor": 2, "warning": 1, "info": 0}[a.severity.value],
                a.confidence,
            ),
            reverse=True,
        )

        primary = sorted_anomalies[0]

        # Boost confidence based on multiple detections
        primary.confidence = min(0.99, primary.confidence + 0.1 * (len(anomalies) - 1))
        primary.description += f" (confirmed by {len(anomalies)} detection methods)"

        return [primary]

    def _process_anomaly(self, anomaly: AnomalyEvent):
        """Process a detected anomaly"""
        anomaly_key = f"{anomaly.metric_id}_{anomaly.anomaly_type.value}"

        # Check if this is a new anomaly or continuation
        if anomaly_key in self._active_anomalies:
            # Update existing anomaly
            existing = self._active_anomalies[anomaly_key]
            existing.current_value = anomaly.current_value
            existing.deviation_score = anomaly.deviation_score
            if anomaly.severity.value > existing.severity.value:
                existing.severity = anomaly.severity
        else:
            # New anomaly
            self._active_anomalies[anomaly_key] = anomaly

            # Trigger callbacks
            for callback in self._anomaly_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    logger.error(f"Anomaly callback error: {e}")

            logger.warning(f"Anomaly detected: {anomaly.description}")

    def clear_anomaly(self, metric_id: str, anomaly_type: AnomalyType):
        """Clear an active anomaly when it resolves"""
        anomaly_key = f"{metric_id}_{anomaly_type.value}"

        if anomaly_key in self._active_anomalies:
            anomaly = self._active_anomalies.pop(anomaly_key)
            self._anomaly_history.append(anomaly)
            logger.info(f"Anomaly cleared: {metric_id}")

    def get_active_anomalies(self) -> List[AnomalyEvent]:
        """Get all currently active anomalies"""
        return list(self._active_anomalies.values())

    def get_anomaly_history(
        self,
        start_time: Optional[datetime] = None,
        metric_id: Optional[str] = None,
        severity: Optional[SeverityLevel] = None,
    ) -> List[AnomalyEvent]:
        """Get anomaly history with optional filters"""
        anomalies = self._anomaly_history.copy()

        if metric_id:
            anomalies = [a for a in anomalies if a.metric_id == metric_id]

        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]

        if start_time:
            anomalies = [a for a in anomalies if a.detected_at >= start_time]

        return anomalies

    def analyze_all_kqis(self) -> List[AnomalyEvent]:
        """Analyze all KQIs for anomalies"""
        all_anomalies = []

        kqis = self.kqi_calculator.compute_all_kqis()
        for kqi in kqis:
            # Build baseline from historical data if not exists
            if kqi.kqi_id not in self._baselines:
                values = self.kqi_calculator._get_metric_values(kqi.kqi_id)
                if len(values) >= self.min_samples:
                    self.update_baseline(kqi.kqi_id, values)

            anomalies = self.analyze_metric(kqi.kqi_id, kqi.value)
            all_anomalies.extend(anomalies)

        return all_anomalies

    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous anomaly monitoring"""
        if self._monitoring_active:
            logger.warning("Anomaly monitoring already active")
            return

        self._monitoring_active = True

        def monitor_loop():
            while self._monitoring_active:
                try:
                    self.analyze_all_kqis()
                except Exception as e:
                    logger.error(f"Anomaly monitoring error: {e}")
                threading.Event().wait(interval_seconds)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Anomaly monitoring started (interval: {interval_seconds}s)")

    def stop_monitoring(self):
        """Stop continuous anomaly monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Anomaly monitoring stopped")

    def get_anomaly_summary(self) -> Dict:
        """Get summary of current anomaly state"""
        active = self.get_active_anomalies()

        severity_counts = {
            "critical": 0,
            "major": 0,
            "minor": 0,
            "warning": 0,
        }

        for a in active:
            if a.severity.value in severity_counts:
                severity_counts[a.severity.value] += 1

        return {
            "total_active": len(active),
            "by_severity": severity_counts,
            "affected_metrics": list(set(a.metric_id for a in active)),
            "highest_severity": max((a.severity.value for a in active), default="none"),
        }
