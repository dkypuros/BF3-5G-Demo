# File location: 5G_Emulator_API/service_assurance/sla_manager.py
# SLA Manager - Service Level Agreement monitoring and violation tracking

"""
SLA Manager for 5G Service Assurance

Manages Service Level Agreements with:
- Pre-defined SLA templates for eMBB, URLLC, mMTC service types
- Real-time threshold monitoring
- Violation detection and tracking
- Compliance reporting

Reference: TMF GB917 - SLA Management Handbook
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import uuid
import threading

from .models import (
    SLADefinition,
    SLAThreshold,
    SLAViolation,
    SeverityLevel,
    NFType,
    MetricCategory,
)
from .kqi_calculator import KQICalculator

logger = logging.getLogger(__name__)


class SLAManager:
    """
    Manages SLA definitions and monitors for violations.
    """

    def __init__(self, kqi_calculator: KQICalculator):
        """
        Initialize SLA Manager.

        Args:
            kqi_calculator: KQI Calculator instance for metric access
        """
        self.kqi_calculator = kqi_calculator

        # SLA storage
        self._slas: Dict[str, SLADefinition] = {}

        # Active violations
        self._active_violations: Dict[str, SLAViolation] = {}

        # Violation history
        self._violation_history: List[SLAViolation] = []

        # Callbacks for violation events
        self._violation_callbacks: List[Callable[[SLAViolation], None]] = []

        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Initialize default SLAs
        self._init_default_slas()

        logger.info("SLA Manager initialized")

    def _init_default_slas(self):
        """Initialize default SLA definitions for 5G services"""

        # eMBB SLA - Enhanced Mobile Broadband
        embb_sla = SLADefinition(
            sla_id="sla_embb_standard",
            name="eMBB Standard Service SLA",
            description="SLA for Enhanced Mobile Broadband services",
            service_type="eMBB",
            thresholds=[
                SLAThreshold(
                    metric_id="kqi_registration_success_rate",
                    operator=">=",
                    value=99.9,
                    duration_seconds=60
                ),
                SLAThreshold(
                    metric_id="kqi_pdu_session_success_rate",
                    operator=">=",
                    value=99.0,
                    duration_seconds=60
                ),
                SLAThreshold(
                    metric_id="kqi_user_plane_latency",
                    operator="<=",
                    value=20.0,  # 20ms max
                    duration_seconds=30
                ),
                SLAThreshold(
                    metric_id="kqi_throughput_achievement",
                    operator=">=",
                    value=90.0,  # 90% of subscribed rate
                    duration_seconds=60
                ),
                SLAThreshold(
                    metric_id="kqi_packet_loss_rate",
                    operator="<=",
                    value=0.1,  # 0.1% max
                    duration_seconds=60
                ),
                SLAThreshold(
                    metric_id="kqi_service_availability",
                    operator=">=",
                    value=99.9,
                    duration_seconds=0
                ),
            ],
            target_compliance_pct=99.9,
            measurement_window_hours=24,
            violation_severity=SeverityLevel.MAJOR,
        )
        self.register_sla(embb_sla)

        # URLLC SLA - Ultra-Reliable Low-Latency Communications
        urllc_sla = SLADefinition(
            sla_id="sla_urllc_standard",
            name="URLLC Standard Service SLA",
            description="SLA for Ultra-Reliable Low-Latency services",
            service_type="URLLC",
            thresholds=[
                SLAThreshold(
                    metric_id="kqi_registration_success_rate",
                    operator=">=",
                    value=99.999,
                    duration_seconds=30
                ),
                SLAThreshold(
                    metric_id="kqi_pdu_session_success_rate",
                    operator=">=",
                    value=99.99,
                    duration_seconds=30
                ),
                SLAThreshold(
                    metric_id="kqi_user_plane_latency",
                    operator="<=",
                    value=1.0,  # 1ms max for URLLC
                    duration_seconds=10
                ),
                SLAThreshold(
                    metric_id="kqi_packet_loss_rate",
                    operator="<=",
                    value=0.00001,  # 10^-5 for URLLC
                    duration_seconds=30
                ),
                SLAThreshold(
                    metric_id="kqi_service_availability",
                    operator=">=",
                    value=99.9999,  # Six nines
                    duration_seconds=0
                ),
            ],
            target_compliance_pct=99.999,
            measurement_window_hours=24,
            violation_severity=SeverityLevel.CRITICAL,
        )
        self.register_sla(urllc_sla)

        # mMTC SLA - Massive Machine-Type Communications
        mmtc_sla = SLADefinition(
            sla_id="sla_mmtc_standard",
            name="mMTC Standard Service SLA",
            description="SLA for Massive Machine-Type Communications",
            service_type="mMTC",
            thresholds=[
                SLAThreshold(
                    metric_id="kqi_registration_success_rate",
                    operator=">=",
                    value=99.0,
                    duration_seconds=120
                ),
                SLAThreshold(
                    metric_id="kqi_pdu_session_success_rate",
                    operator=">=",
                    value=98.0,
                    duration_seconds=120
                ),
                SLAThreshold(
                    metric_id="kqi_service_availability",
                    operator=">=",
                    value=99.5,
                    duration_seconds=0
                ),
            ],
            target_compliance_pct=99.0,
            measurement_window_hours=24,
            violation_severity=SeverityLevel.MINOR,
        )
        self.register_sla(mmtc_sla)

        # N6 Firewall SLA
        n6_firewall_sla = SLADefinition(
            sla_id="sla_n6_firewall",
            name="N6 Firewall Security SLA",
            description="SLA for N6 Interface Firewall performance",
            service_type="Security",
            thresholds=[
                SLAThreshold(
                    metric_id="kqi_n6_firewall_availability",
                    operator=">=",
                    value=99.999,
                    duration_seconds=0
                ),
            ],
            target_compliance_pct=99.999,
            measurement_window_hours=24,
            violation_severity=SeverityLevel.CRITICAL,
        )
        self.register_sla(n6_firewall_sla)

        logger.info(f"Initialized {len(self._slas)} default SLAs")

    def register_sla(self, sla: SLADefinition) -> str:
        """
        Register a new SLA definition.

        Args:
            sla: SLA definition to register

        Returns:
            SLA ID
        """
        self._slas[sla.sla_id] = sla
        logger.info(f"Registered SLA: {sla.sla_id} - {sla.name}")
        return sla.sla_id

    def unregister_sla(self, sla_id: str) -> bool:
        """Remove an SLA definition"""
        if sla_id in self._slas:
            del self._slas[sla_id]
            logger.info(f"Unregistered SLA: {sla_id}")
            return True
        return False

    def get_sla(self, sla_id: str) -> Optional[SLADefinition]:
        """Get an SLA by ID"""
        return self._slas.get(sla_id)

    def list_slas(self) -> List[SLADefinition]:
        """List all registered SLAs"""
        return list(self._slas.values())

    def register_violation_callback(self, callback: Callable[[SLAViolation], None]):
        """Register a callback for violation events"""
        self._violation_callbacks.append(callback)

    def check_threshold(self, threshold: SLAThreshold) -> tuple[bool, float]:
        """
        Check if a threshold is being met.

        Args:
            threshold: The threshold to check

        Returns:
            Tuple of (is_violated, actual_value)
        """
        kqi = self.kqi_calculator.get_kqi(threshold.metric_id)
        if kqi is None:
            return False, 0.0

        actual_value = kqi.value
        threshold_value = threshold.value
        operator = threshold.operator

        is_violated = False
        if operator == ">=" and actual_value < threshold_value:
            is_violated = True
        elif operator == ">" and actual_value <= threshold_value:
            is_violated = True
        elif operator == "<=" and actual_value > threshold_value:
            is_violated = True
        elif operator == "<" and actual_value >= threshold_value:
            is_violated = True
        elif operator == "==" and actual_value != threshold_value:
            is_violated = True

        return is_violated, actual_value

    def evaluate_sla(self, sla_id: str) -> List[SLAViolation]:
        """
        Evaluate an SLA and return any violations.

        Args:
            sla_id: SLA ID to evaluate

        Returns:
            List of violations found
        """
        sla = self._slas.get(sla_id)
        if not sla:
            logger.warning(f"Unknown SLA: {sla_id}")
            return []

        violations = []
        for threshold in sla.thresholds:
            is_violated, actual_value = self.check_threshold(threshold)

            if is_violated:
                violation = self._create_or_update_violation(
                    sla, threshold, actual_value
                )
                if violation:
                    violations.append(violation)
            else:
                # Check if there was an active violation that is now resolved
                self._resolve_violation_if_exists(sla_id, threshold.metric_id)

        return violations

    def _create_or_update_violation(
        self, sla: SLADefinition, threshold: SLAThreshold, actual_value: float
    ) -> Optional[SLAViolation]:
        """Create a new violation or update an existing one"""
        violation_key = f"{sla.sla_id}_{threshold.metric_id}"

        # Check for existing active violation
        if violation_key in self._active_violations:
            existing = self._active_violations[violation_key]
            existing.actual_value = actual_value
            existing.duration_seconds = int(
                (datetime.utcnow() - existing.start_time).total_seconds()
            )
            deviation = abs(actual_value - threshold.value) / threshold.value * 100
            existing.deviation_pct = deviation
            return existing

        # Check duration threshold
        # For now, create immediately (duration handling could be enhanced)

        # Calculate deviation
        deviation = abs(actual_value - threshold.value) / threshold.value * 100

        violation = SLAViolation(
            violation_id=str(uuid.uuid4()),
            sla_id=sla.sla_id,
            sla_name=sla.name,
            threshold=threshold,
            actual_value=actual_value,
            expected_value=threshold.value,
            deviation_pct=deviation,
            start_time=datetime.utcnow(),
            severity=sla.violation_severity,
        )

        self._active_violations[violation_key] = violation

        # Trigger callbacks
        for callback in self._violation_callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error(f"Violation callback error: {e}")

        logger.warning(
            f"SLA Violation: {sla.name} - {threshold.metric_id} "
            f"(actual: {actual_value}, threshold: {threshold.operator} {threshold.value})"
        )

        return violation

    def _resolve_violation_if_exists(self, sla_id: str, metric_id: str):
        """Mark a violation as resolved if it exists"""
        violation_key = f"{sla_id}_{metric_id}"

        if violation_key in self._active_violations:
            violation = self._active_violations.pop(violation_key)
            violation.end_time = datetime.utcnow()
            violation.duration_seconds = int(
                (violation.end_time - violation.start_time).total_seconds()
            )
            violation.resolved = True

            self._violation_history.append(violation)
            logger.info(
                f"Violation resolved: {violation.sla_name} - {metric_id} "
                f"(duration: {violation.duration_seconds}s)"
            )

    def evaluate_all_slas(self) -> List[SLAViolation]:
        """Evaluate all registered SLAs"""
        all_violations = []
        for sla_id in self._slas:
            violations = self.evaluate_sla(sla_id)
            all_violations.extend(violations)
        return all_violations

    def get_active_violations(self) -> List[SLAViolation]:
        """Get all currently active violations"""
        return list(self._active_violations.values())

    def get_violation_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        sla_id: Optional[str] = None,
    ) -> List[SLAViolation]:
        """
        Get violation history with optional filters.

        Args:
            start_time: Filter violations after this time
            end_time: Filter violations before this time
            sla_id: Filter by specific SLA

        Returns:
            List of historical violations
        """
        violations = self._violation_history.copy()

        if sla_id:
            violations = [v for v in violations if v.sla_id == sla_id]

        if start_time:
            violations = [v for v in violations if v.start_time >= start_time]

        if end_time:
            violations = [v for v in violations if v.start_time <= end_time]

        return violations

    def acknowledge_violation(self, violation_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a violation"""
        for violation in self._active_violations.values():
            if violation.violation_id == violation_id:
                violation.acknowledged = True
                violation.acknowledged_by = acknowledged_by
                logger.info(f"Violation {violation_id} acknowledged by {acknowledged_by}")
                return True
        return False

    def get_compliance_summary(
        self, sla_id: Optional[str] = None, window_hours: int = 24
    ) -> Dict[str, float]:
        """
        Get SLA compliance summary.

        Args:
            sla_id: Specific SLA ID or None for all
            window_hours: Time window for compliance calculation

        Returns:
            Dict of SLA ID -> compliance percentage
        """
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        compliance = {}

        slas_to_check = [sla_id] if sla_id else list(self._slas.keys())

        for sid in slas_to_check:
            sla = self._slas.get(sid)
            if not sla:
                continue

            # Get violations in the window
            violations = self.get_violation_history(start_time=cutoff, sla_id=sid)

            # Calculate total violation time
            total_violation_seconds = sum(v.duration_seconds for v in violations)

            # Add active violations
            for v in self._active_violations.values():
                if v.sla_id == sid:
                    total_violation_seconds += int(
                        (datetime.utcnow() - v.start_time).total_seconds()
                    )

            total_seconds = window_hours * 3600
            compliance_pct = (
                (total_seconds - total_violation_seconds) / total_seconds * 100
            )
            compliance[sid] = max(0, min(100, compliance_pct))

        return compliance

    def get_sla_status(self, sla_id: str) -> Dict:
        """Get comprehensive status for an SLA"""
        sla = self._slas.get(sla_id)
        if not sla:
            return {"error": "SLA not found"}

        active_violations = [
            v for v in self._active_violations.values() if v.sla_id == sla_id
        ]

        compliance = self.get_compliance_summary(sla_id)

        return {
            "sla_id": sla_id,
            "sla_name": sla.name,
            "service_type": sla.service_type,
            "status": "violated" if active_violations else "compliant",
            "compliance_pct": compliance.get(sla_id, 100.0),
            "active_violations": len(active_violations),
            "violations": [v.dict() for v in active_violations],
            "threshold_count": len(sla.thresholds),
            "target_compliance_pct": sla.target_compliance_pct,
        }

    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous SLA monitoring"""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return

        self._monitoring_active = True

        def monitor_loop():
            while self._monitoring_active:
                try:
                    self.evaluate_all_slas()
                except Exception as e:
                    logger.error(f"SLA monitoring error: {e}")
                threading.Event().wait(interval_seconds)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"SLA monitoring started (interval: {interval_seconds}s)")

    def stop_monitoring(self):
        """Stop continuous SLA monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("SLA monitoring stopped")
