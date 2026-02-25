# File location: 5G_Emulator_API/service_assurance/rca_engine.py
# Root Cause Analysis Engine - Intelligent fault correlation and diagnosis

"""
Root Cause Analysis Engine for 5G Service Assurance

Provides intelligent analysis of service degradations by:
- Correlating related anomalies and violations
- Analyzing NF dependency chains
- Identifying probable root causes with confidence scores
- Generating actionable recommendations

Uses a knowledge base of 5G network fault patterns and dependencies.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid

from .models import (
    RCAResult,
    RCACause,
    SLAViolation,
    AnomalyEvent,
    NFType,
    SeverityLevel,
)
from .kqi_calculator import KQICalculator
from .sla_manager import SLAManager
from .anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class RCAEngine:
    """
    Root Cause Analysis Engine for diagnosing 5G network issues.
    """

    def __init__(
        self,
        kqi_calculator: KQICalculator,
        sla_manager: SLAManager,
        anomaly_detector: AnomalyDetector,
    ):
        """
        Initialize RCA Engine.

        Args:
            kqi_calculator: KQI Calculator instance
            sla_manager: SLA Manager instance
            anomaly_detector: Anomaly Detector instance
        """
        self.kqi_calculator = kqi_calculator
        self.sla_manager = sla_manager
        self.anomaly_detector = anomaly_detector

        # RCA history
        self._rca_history: List[RCAResult] = []

        # Initialize knowledge base
        self._nf_dependencies = self._init_nf_dependencies()
        self._fault_patterns = self._init_fault_patterns()
        self._metric_to_nf = self._init_metric_nf_mapping()

        logger.info("RCA Engine initialized")

    def _init_nf_dependencies(self) -> Dict[NFType, List[NFType]]:
        """
        Initialize 5G NF dependency graph.
        Key depends on values (downstream dependencies).
        """
        return {
            # User plane depends on SMF and N6 firewall
            NFType.UPF: [NFType.SMF, NFType.N6_FIREWALL],
            # SMF depends on AMF, PCF, UDM
            NFType.SMF: [NFType.AMF, NFType.PCF, NFType.UDM],
            # AMF depends on AUSF, UDM, NRF
            NFType.AMF: [NFType.AUSF, NFType.UDM, NFType.NRF],
            # AUSF depends on UDM
            NFType.AUSF: [NFType.UDM],
            # UDM depends on UDR
            NFType.UDM: [NFType.UDR],
            # PCF depends on UDR
            NFType.PCF: [NFType.UDR],
            # RAN depends on core
            NFType.GNB: [NFType.AMF],
            NFType.CU: [NFType.GNB, NFType.AMF],
            NFType.DU: [NFType.CU],
            NFType.RRU: [NFType.DU],
            # All NFs depend on NRF for discovery
            NFType.UDR: [NFType.NRF],
            NFType.AUSF: [NFType.NRF, NFType.UDM],
        }

    def _init_fault_patterns(self) -> List[Dict]:
        """Initialize fault pattern knowledge base"""
        return [
            # Registration failures
            {
                "pattern_id": "reg_failure_auth",
                "name": "Registration Failure - Authentication",
                "symptoms": ["kqi_registration_success_rate", "kqi_authentication_success_rate"],
                "primary_nf": NFType.AUSF,
                "related_nfs": [NFType.UDM, NFType.AMF],
                "causes": [
                    {"desc": "AUSF authentication algorithm failure", "probability": 0.4},
                    {"desc": "UDM subscription data unavailable", "probability": 0.3},
                    {"desc": "AMF-AUSF communication failure", "probability": 0.2},
                    {"desc": "Invalid UE credentials", "probability": 0.1},
                ],
                "recommendations": [
                    "Check AUSF service health and logs",
                    "Verify UDM database connectivity",
                    "Inspect AMF-AUSF N12 interface",
                ],
            },
            # Session establishment failures
            {
                "pattern_id": "session_failure_smf",
                "name": "PDU Session Failure - SMF",
                "symptoms": ["kqi_pdu_session_success_rate"],
                "primary_nf": NFType.SMF,
                "related_nfs": [NFType.UPF, NFType.PCF, NFType.AMF],
                "causes": [
                    {"desc": "SMF session context exhaustion", "probability": 0.3},
                    {"desc": "UPF N4 interface failure", "probability": 0.25},
                    {"desc": "PCF policy retrieval failure", "probability": 0.25},
                    {"desc": "IP address pool exhaustion", "probability": 0.2},
                ],
                "recommendations": [
                    "Check SMF session capacity and memory",
                    "Verify PFCP connectivity to UPF",
                    "Inspect PCF policy configuration",
                    "Monitor IP address pool utilization",
                ],
            },
            # Latency issues
            {
                "pattern_id": "latency_user_plane",
                "name": "User Plane Latency Degradation",
                "symptoms": ["kqi_user_plane_latency"],
                "primary_nf": NFType.UPF,
                "related_nfs": [NFType.GNB, NFType.N6_FIREWALL],
                "causes": [
                    {"desc": "UPF processing congestion", "probability": 0.35},
                    {"desc": "N6 firewall inspection delay", "probability": 0.25},
                    {"desc": "RAN backhaul congestion", "probability": 0.2},
                    {"desc": "GTP tunnel overhead", "probability": 0.2},
                ],
                "recommendations": [
                    "Check UPF CPU and memory utilization",
                    "Review N6 firewall rule complexity",
                    "Monitor transport network metrics",
                    "Verify QoS flow configuration",
                ],
            },
            {
                "pattern_id": "latency_control_plane",
                "name": "Control Plane Latency Degradation",
                "symptoms": ["kqi_control_plane_latency"],
                "primary_nf": NFType.AMF,
                "related_nfs": [NFType.SMF, NFType.NRF],
                "causes": [
                    {"desc": "AMF processing overload", "probability": 0.3},
                    {"desc": "NRF service discovery delays", "probability": 0.25},
                    {"desc": "SMF session processing delays", "probability": 0.25},
                    {"desc": "HTTP/2 connection pool exhaustion", "probability": 0.2},
                ],
                "recommendations": [
                    "Check AMF instance scaling",
                    "Verify NRF response times",
                    "Monitor inter-NF HTTP latencies",
                    "Review connection pool settings",
                ],
            },
            # Packet loss
            {
                "pattern_id": "packet_loss",
                "name": "Packet Loss Detection",
                "symptoms": ["kqi_packet_loss_rate"],
                "primary_nf": NFType.UPF,
                "related_nfs": [NFType.GNB, NFType.N6_FIREWALL],
                "causes": [
                    {"desc": "UPF buffer overflow", "probability": 0.3},
                    {"desc": "N6 firewall dropping packets", "probability": 0.25},
                    {"desc": "QoS enforcement dropping low-priority traffic", "probability": 0.25},
                    {"desc": "Transport network congestion", "probability": 0.2},
                ],
                "recommendations": [
                    "Check UPF buffer utilization",
                    "Review N6 firewall block statistics",
                    "Verify QoS policy configuration",
                    "Monitor transport interface errors",
                ],
            },
            # Availability issues
            {
                "pattern_id": "availability_core",
                "name": "Core Network Availability Issue",
                "symptoms": ["kqi_service_availability"],
                "primary_nf": NFType.NRF,
                "related_nfs": [NFType.AMF, NFType.SMF, NFType.UPF],
                "causes": [
                    {"desc": "NRF service registry unavailable", "probability": 0.3},
                    {"desc": "Multiple NF instance failures", "probability": 0.25},
                    {"desc": "Network connectivity issue", "probability": 0.25},
                    {"desc": "Database backend failure", "probability": 0.2},
                ],
                "recommendations": [
                    "Check NRF health and registered NFs",
                    "Verify all NF instance statuses",
                    "Test network connectivity between NFs",
                    "Check database backend health",
                ],
            },
            # Handover failures
            {
                "pattern_id": "handover_failure",
                "name": "Handover Failure",
                "symptoms": ["kqi_handover_success_rate"],
                "primary_nf": NFType.GNB,
                "related_nfs": [NFType.AMF, NFType.CU],
                "causes": [
                    {"desc": "NGAP handover procedure failure", "probability": 0.3},
                    {"desc": "Target cell unavailable", "probability": 0.25},
                    {"desc": "AMF context transfer failure", "probability": 0.25},
                    {"desc": "Xn interface issues between gNBs", "probability": 0.2},
                ],
                "recommendations": [
                    "Check gNB NGAP connection status",
                    "Verify target cell configurations",
                    "Monitor AMF handover logs",
                    "Inspect Xn interface connectivity",
                ],
            },
            # N6 Firewall issues
            {
                "pattern_id": "n6_firewall",
                "name": "N6 Firewall Availability/Performance",
                "symptoms": ["kqi_n6_firewall_availability"],
                "primary_nf": NFType.N6_FIREWALL,
                "related_nfs": [NFType.UPF],
                "causes": [
                    {"desc": "BlueField-3 DPU hardware issue", "probability": 0.25},
                    {"desc": "DOCA flow rule overflow", "probability": 0.25},
                    {"desc": "Firewall configuration error", "probability": 0.25},
                    {"desc": "High packet rate overwhelming firewall", "probability": 0.25},
                ],
                "recommendations": [
                    "Check DPU health status",
                    "Monitor flow table utilization",
                    "Review firewall rule configuration",
                    "Verify hardware acceleration status",
                ],
            },
        ]

    def _init_metric_nf_mapping(self) -> Dict[str, List[NFType]]:
        """Map metrics to responsible NFs"""
        return {
            "kqi_registration_success_rate": [NFType.AMF, NFType.AUSF, NFType.UDM],
            "kqi_authentication_success_rate": [NFType.AUSF, NFType.UDM],
            "kqi_pdu_session_success_rate": [NFType.SMF, NFType.UPF, NFType.PCF],
            "kqi_session_continuity_rate": [NFType.SMF, NFType.UPF],
            "kqi_handover_success_rate": [NFType.GNB, NFType.AMF, NFType.CU],
            "kqi_user_plane_latency": [NFType.UPF, NFType.GNB, NFType.N6_FIREWALL],
            "kqi_control_plane_latency": [NFType.AMF, NFType.SMF, NFType.NRF],
            "kqi_packet_loss_rate": [NFType.UPF, NFType.N6_FIREWALL, NFType.GNB],
            "kqi_throughput_achievement": [NFType.UPF, NFType.PCF],
            "kqi_service_availability": [NFType.NRF, NFType.AMF, NFType.SMF],
            "kqi_n6_firewall_availability": [NFType.N6_FIREWALL],
        }

    def _find_matching_patterns(
        self, metrics: List[str]
    ) -> List[Tuple[Dict, float]]:
        """Find fault patterns that match the affected metrics"""
        matches = []

        for pattern in self._fault_patterns:
            symptom_match = 0
            for symptom in pattern["symptoms"]:
                if symptom in metrics:
                    symptom_match += 1

            if symptom_match > 0:
                # Calculate match score
                match_score = symptom_match / len(pattern["symptoms"])
                matches.append((pattern, match_score))

        # Sort by match score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _get_dependency_chain(self, nf_type: NFType, visited: Optional[set] = None) -> List[NFType]:
        """Get the dependency chain for an NF"""
        if visited is None:
            visited = set()

        if nf_type in visited:
            return []

        visited.add(nf_type)
        chain = [nf_type]

        dependencies = self._nf_dependencies.get(nf_type, [])
        for dep in dependencies:
            chain.extend(self._get_dependency_chain(dep, visited))

        return chain

    def _correlate_events(
        self,
        violations: List[SLAViolation],
        anomalies: List[AnomalyEvent],
        time_window_minutes: int = 30,
    ) -> Dict:
        """Correlate violations and anomalies within a time window"""
        correlation = {
            "affected_metrics": set(),
            "affected_nfs": set(),
            "event_timeline": [],
            "severity_max": SeverityLevel.INFO,
        }

        # Process violations
        for v in violations:
            correlation["affected_metrics"].add(v.threshold.metric_id)
            correlation["affected_nfs"].update(v.affected_nfs)
            correlation["event_timeline"].append({
                "type": "violation",
                "time": v.start_time,
                "metric": v.threshold.metric_id,
                "severity": v.severity,
            })
            if v.severity.value > correlation["severity_max"].value:
                correlation["severity_max"] = v.severity

        # Process anomalies
        for a in anomalies:
            correlation["affected_metrics"].add(a.metric_id)
            if a.nf_type:
                correlation["affected_nfs"].add(a.nf_type)
            correlation["event_timeline"].append({
                "type": "anomaly",
                "time": a.detected_at,
                "metric": a.metric_id,
                "severity": a.severity,
            })
            if a.severity.value > correlation["severity_max"].value:
                correlation["severity_max"] = a.severity

        # Map metrics to NFs if not already mapped
        for metric in correlation["affected_metrics"]:
            nfs = self._metric_to_nf.get(metric, [])
            correlation["affected_nfs"].update(nfs)

        # Sort timeline
        correlation["event_timeline"].sort(key=lambda x: x["time"])

        # Convert sets to lists for JSON serialization
        correlation["affected_metrics"] = list(correlation["affected_metrics"])
        correlation["affected_nfs"] = list(correlation["affected_nfs"])

        return correlation

    def analyze(
        self,
        trigger_id: str,
        time_range_minutes: int = 30,
    ) -> RCAResult:
        """
        Perform root cause analysis.

        Args:
            trigger_id: ID of the violation or anomaly that triggered RCA
            time_range_minutes: Time range to analyze

        Returns:
            RCAResult with probable causes and recommendations
        """
        logger.info(f"Starting RCA for trigger: {trigger_id}")

        # Gather current state
        violations = self.sla_manager.get_active_violations()
        anomalies = self.anomaly_detector.get_active_anomalies()

        # Correlate events
        correlation = self._correlate_events(violations, anomalies, time_range_minutes)

        # Find matching fault patterns
        pattern_matches = self._find_matching_patterns(correlation["affected_metrics"])

        # Build root causes
        root_causes = []
        for pattern, match_score in pattern_matches[:3]:  # Top 3 matches
            for cause_info in pattern["causes"]:
                # Adjust probability based on match score
                adjusted_prob = cause_info["probability"] * match_score

                cause = RCACause(
                    cause_id=f"{pattern['pattern_id']}_{cause_info['desc'][:10]}",
                    description=cause_info["desc"],
                    probability=adjusted_prob,
                    nf_type=pattern["primary_nf"],
                    evidence=[
                        f"Pattern match: {pattern['name']} ({match_score*100:.0f}%)",
                        f"Affected metrics: {', '.join(correlation['affected_metrics'][:3])}",
                    ],
                    recommended_actions=pattern["recommendations"],
                )
                root_causes.append(cause)

        # Sort by probability
        root_causes.sort(key=lambda c: c.probability, reverse=True)
        root_causes = root_causes[:5]  # Top 5 causes

        # Build dependency chain
        dependency_chain = []
        for nf in correlation["affected_nfs"]:
            if isinstance(nf, NFType):
                chain = self._get_dependency_chain(nf)
                dependency_chain.extend([str(n.value) for n in chain])
        dependency_chain = list(dict.fromkeys(dependency_chain))  # Remove duplicates, preserve order

        # Generate impact summary
        impact_summary = self._generate_impact_summary(correlation, root_causes)

        # Compile recommendations
        immediate_actions = []
        long_term_recommendations = []
        for cause in root_causes[:3]:
            immediate_actions.extend(cause.recommended_actions[:2])
        immediate_actions = list(dict.fromkeys(immediate_actions))  # Deduplicate

        # Add general recommendations
        long_term_recommendations = [
            "Review capacity planning for affected NFs",
            "Consider implementing predictive monitoring",
            "Update runbooks for this failure pattern",
            "Schedule post-incident review",
        ]

        # Create RCA result
        rca_result = RCAResult(
            rca_id=str(uuid.uuid4()),
            triggered_by=trigger_id,
            analysis_end=datetime.utcnow(),
            time_range_analyzed_minutes=time_range_minutes,
            root_causes=root_causes,
            impact_summary=impact_summary,
            affected_services=self._get_affected_services(correlation["affected_metrics"]),
            affected_nfs=[nf for nf in correlation["affected_nfs"] if isinstance(nf, NFType)],
            correlated_events=[
                f"{e['type']}: {e['metric']}" for e in correlation["event_timeline"][:10]
            ],
            dependency_chain=dependency_chain,
            immediate_actions=immediate_actions,
            long_term_recommendations=long_term_recommendations,
        )

        # Store in history
        self._rca_history.append(rca_result)

        logger.info(f"RCA completed: {rca_result.rca_id} - {len(root_causes)} causes identified")
        return rca_result

    def _generate_impact_summary(self, correlation: Dict, root_causes: List[RCACause]) -> str:
        """Generate human-readable impact summary"""
        affected_count = len(correlation["affected_metrics"])
        nf_count = len(correlation["affected_nfs"])
        severity = correlation["severity_max"].value

        summary_parts = [
            f"Impact: {severity.upper()} severity issue affecting {affected_count} metrics across {nf_count} network functions."
        ]

        if root_causes:
            top_cause = root_causes[0]
            summary_parts.append(
                f"Most probable cause: {top_cause.description} ({top_cause.probability*100:.0f}% confidence)."
            )

        # Add specific impact based on affected metrics
        metrics = correlation["affected_metrics"]
        if any("registration" in m for m in metrics):
            summary_parts.append("User registration may be impacted.")
        if any("session" in m for m in metrics):
            summary_parts.append("PDU session establishment may be impacted.")
        if any("latency" in m for m in metrics):
            summary_parts.append("Service latency is degraded.")
        if any("availability" in m for m in metrics):
            summary_parts.append("Service availability is affected.")

        return " ".join(summary_parts)

    def _get_affected_services(self, metrics: List[str]) -> List[str]:
        """Determine affected services from metrics"""
        services = set()

        for metric in metrics:
            if "registration" in metric or "authentication" in metric:
                services.add("User Registration")
            if "session" in metric:
                services.add("PDU Session Management")
            if "user_plane" in metric or "packet" in metric or "throughput" in metric:
                services.add("User Plane Data")
            if "control_plane" in metric:
                services.add("Control Plane Signaling")
            if "handover" in metric:
                services.add("Mobility Management")
            if "availability" in metric:
                services.add("Overall Service Availability")
            if "firewall" in metric:
                services.add("N6 Security")

        return list(services)

    def analyze_violation(self, violation: SLAViolation) -> RCAResult:
        """Convenience method to analyze a specific violation"""
        return self.analyze(violation.violation_id)

    def analyze_anomaly(self, anomaly: AnomalyEvent) -> RCAResult:
        """Convenience method to analyze a specific anomaly"""
        return self.analyze(anomaly.anomaly_id)

    def get_rca_history(
        self,
        start_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[RCAResult]:
        """Get RCA history"""
        results = self._rca_history.copy()

        if start_time:
            results = [r for r in results if r.analysis_start >= start_time]

        return results[-limit:]

    def get_rca(self, rca_id: str) -> Optional[RCAResult]:
        """Get a specific RCA by ID"""
        for rca in self._rca_history:
            if rca.rca_id == rca_id:
                return rca
        return None
