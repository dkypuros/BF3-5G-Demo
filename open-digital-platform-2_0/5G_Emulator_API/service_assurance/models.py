# File location: 5G_Emulator_API/service_assurance/models.py
# Data models for Service Assurance - 3GPP and TMF-aligned definitions

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


# =============================================================================
# Enumerations
# =============================================================================

class MetricCategory(str, Enum):
    """Categories of metrics aligned with 3GPP TS 28.552/554"""
    ACCESSIBILITY = "accessibility"      # Can the user access the service?
    RETAINABILITY = "retainability"      # Can the service be maintained?
    INTEGRITY = "integrity"              # Is the service quality acceptable?
    AVAILABILITY = "availability"        # Is the service available?
    MOBILITY = "mobility"                # Handover/mobility performance
    UTILIZATION = "utilization"          # Resource utilization
    SECURITY = "security"                # Security-related metrics


class SeverityLevel(str, Enum):
    """Severity levels for violations and anomalies"""
    CRITICAL = "critical"    # Service impacting, immediate action required
    MAJOR = "major"          # Significant degradation, urgent attention
    MINOR = "minor"          # Minor degradation, scheduled attention
    WARNING = "warning"      # Trending toward violation
    INFO = "info"            # Informational only


class NFType(str, Enum):
    """Network Function types"""
    AMF = "AMF"
    SMF = "SMF"
    UPF = "UPF"
    AUSF = "AUSF"
    UDM = "UDM"
    UDR = "UDR"
    PCF = "PCF"
    NRF = "NRF"
    NSSF = "NSSF"
    NEF = "NEF"
    GNB = "gNodeB"
    CU = "CU"
    DU = "DU"
    RRU = "RRU"
    N6_FIREWALL = "N6_FIREWALL"
    SERVICE_ASSURANCE = "SERVICE_ASSURANCE"


class ServiceHealthState(str, Enum):
    """Overall service health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AnomalyType(str, Enum):
    """Types of detected anomalies"""
    SPIKE = "spike"                      # Sudden increase
    DROP = "drop"                        # Sudden decrease
    TREND = "trend"                      # Gradual change over time
    PATTERN = "pattern"                  # Recurring pattern anomaly
    OUTLIER = "outlier"                  # Statistical outlier
    CORRELATION = "correlation"          # Unusual correlation between metrics


# =============================================================================
# KQI/KPI Models
# =============================================================================

class KPIMetric(BaseModel):
    """
    Key Performance Indicator - Low-level technical metric
    Reference: 3GPP TS 28.552 - 5G Performance Measurements
    """
    kpi_id: str = Field(..., description="Unique KPI identifier")
    name: str = Field(..., description="Human-readable KPI name")
    description: Optional[str] = Field(None, description="KPI description")
    category: MetricCategory = Field(..., description="Metric category")
    nf_type: NFType = Field(..., description="Source network function")
    value: float = Field(..., description="Current KPI value")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Statistical context
    baseline_value: Optional[float] = Field(None, description="Expected baseline")
    min_value: Optional[float] = Field(None, description="Minimum observed")
    max_value: Optional[float] = Field(None, description="Maximum observed")
    std_deviation: Optional[float] = Field(None, description="Standard deviation")

    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict, description="Additional labels")


class KQIMetric(BaseModel):
    """
    Key Quality Indicator - Customer/service experience metric
    Reference: TMF GB923 - Quality of Experience Metrics
    """
    kqi_id: str = Field(..., description="Unique KQI identifier")
    name: str = Field(..., description="Human-readable KQI name")
    description: Optional[str] = Field(None, description="KQI description")
    category: MetricCategory = Field(..., description="Metric category")
    value: float = Field(..., description="Current KQI value")
    unit: str = Field(..., description="Unit of measurement (%, ms, etc.)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Thresholds
    target_value: Optional[float] = Field(None, description="Target/SLA value")
    warning_threshold: Optional[float] = Field(None, description="Warning threshold")
    critical_threshold: Optional[float] = Field(None, description="Critical threshold")

    # Derived from KPIs
    source_kpis: List[str] = Field(default_factory=list, description="Contributing KPI IDs")
    calculation_formula: Optional[str] = Field(None, description="How KQI is calculated")

    # Service context
    service_type: Optional[str] = Field(None, description="e.g., eMBB, URLLC, mMTC")
    slice_id: Optional[str] = Field(None, description="Network slice identifier")


# =============================================================================
# SLA Models
# =============================================================================

class SLAThreshold(BaseModel):
    """Threshold definition for SLA monitoring"""
    metric_id: str = Field(..., description="KQI/KPI ID to monitor")
    operator: str = Field(..., description="Comparison operator: <, >, <=, >=, ==")
    value: float = Field(..., description="Threshold value")
    duration_seconds: int = Field(0, description="Duration threshold must be exceeded")


class SLADefinition(BaseModel):
    """
    Service Level Agreement definition
    Reference: TMF GB917 - SLA Management Handbook
    """
    sla_id: str = Field(..., description="Unique SLA identifier")
    name: str = Field(..., description="SLA name")
    description: Optional[str] = Field(None, description="SLA description")
    service_type: str = Field(..., description="Service type (eMBB, URLLC, mMTC)")

    # Thresholds
    thresholds: List[SLAThreshold] = Field(..., description="SLA thresholds")

    # Compliance targets
    target_compliance_pct: float = Field(99.9, description="Target compliance percentage")
    measurement_window_hours: int = Field(24, description="Measurement window")

    # Severity and actions
    violation_severity: SeverityLevel = Field(SeverityLevel.MAJOR)
    auto_remediation: bool = Field(False, description="Enable auto-remediation")

    # Metadata
    effective_from: datetime = Field(default_factory=datetime.utcnow)
    effective_to: Optional[datetime] = Field(None)
    owner: Optional[str] = Field(None, description="SLA owner/stakeholder")


class SLAViolation(BaseModel):
    """Record of an SLA violation event"""
    violation_id: str = Field(..., description="Unique violation identifier")
    sla_id: str = Field(..., description="Violated SLA ID")
    sla_name: str = Field(..., description="Violated SLA name")
    threshold: SLAThreshold = Field(..., description="Violated threshold")

    # Violation details
    actual_value: float = Field(..., description="Actual metric value")
    expected_value: float = Field(..., description="Expected/threshold value")
    deviation_pct: float = Field(..., description="Percentage deviation")

    # Timing
    start_time: datetime = Field(..., description="When violation started")
    end_time: Optional[datetime] = Field(None, description="When violation ended")
    duration_seconds: int = Field(0, description="Violation duration")

    # Context
    severity: SeverityLevel = Field(..., description="Violation severity")
    affected_nfs: List[NFType] = Field(default_factory=list)
    affected_subscribers: int = Field(0, description="Estimated affected subscribers")

    # Resolution
    acknowledged: bool = Field(False)
    acknowledged_by: Optional[str] = Field(None)
    resolved: bool = Field(False)
    resolution_notes: Optional[str] = Field(None)
    rca_id: Optional[str] = Field(None, description="Linked RCA ID")


# =============================================================================
# Anomaly Detection Models
# =============================================================================

class AnomalyEvent(BaseModel):
    """Detected anomaly event"""
    anomaly_id: str = Field(..., description="Unique anomaly identifier")
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")

    # Affected metric
    metric_id: str = Field(..., description="Affected metric ID")
    metric_name: str = Field(..., description="Affected metric name")
    nf_type: Optional[NFType] = Field(None, description="Affected NF")

    # Anomaly details
    current_value: float = Field(..., description="Current metric value")
    expected_value: float = Field(..., description="Expected value")
    deviation_score: float = Field(..., description="Statistical deviation score")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")

    # Timing
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None)

    # Context
    severity: SeverityLevel = Field(..., description="Anomaly severity")
    description: str = Field(..., description="Human-readable description")

    # Correlation
    correlated_anomalies: List[str] = Field(default_factory=list)
    potential_causes: List[str] = Field(default_factory=list)


# =============================================================================
# Root Cause Analysis Models
# =============================================================================

class RCACause(BaseModel):
    """A potential root cause"""
    cause_id: str = Field(..., description="Cause identifier")
    description: str = Field(..., description="Cause description")
    probability: float = Field(..., ge=0, le=1, description="Probability 0-1")
    nf_type: Optional[NFType] = Field(None, description="Affected NF")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    recommended_actions: List[str] = Field(default_factory=list)


class RCAResult(BaseModel):
    """Root Cause Analysis result"""
    rca_id: str = Field(..., description="Unique RCA identifier")
    triggered_by: str = Field(..., description="What triggered RCA (violation_id, anomaly_id)")

    # Analysis scope
    analysis_start: datetime = Field(default_factory=datetime.utcnow)
    analysis_end: Optional[datetime] = Field(None)
    time_range_analyzed_minutes: int = Field(30, description="Historical time range analyzed")

    # Results
    root_causes: List[RCACause] = Field(default_factory=list)
    impact_summary: str = Field(..., description="Summary of service impact")
    affected_services: List[str] = Field(default_factory=list)
    affected_nfs: List[NFType] = Field(default_factory=list)

    # Correlation data
    correlated_events: List[str] = Field(default_factory=list)
    dependency_chain: List[str] = Field(default_factory=list, description="NF dependency chain")

    # Recommendations
    immediate_actions: List[str] = Field(default_factory=list)
    long_term_recommendations: List[str] = Field(default_factory=list)


# =============================================================================
# Health and Reporting Models
# =============================================================================

class NFHealthStatus(BaseModel):
    """Health status of a single Network Function"""
    nf_type: NFType = Field(..., description="Network function type")
    nf_instance_id: Optional[str] = Field(None, description="Instance identifier")
    state: ServiceHealthState = Field(..., description="Current health state")

    # Metrics summary
    active_violations: int = Field(0)
    active_anomalies: int = Field(0)
    compliance_pct: float = Field(100.0, description="SLA compliance percentage")

    # Key metrics snapshot
    latency_ms: Optional[float] = Field(None)
    success_rate_pct: Optional[float] = Field(None)
    throughput: Optional[float] = Field(None)
    error_rate_pct: Optional[float] = Field(None)

    last_updated: datetime = Field(default_factory=datetime.utcnow)


class ServiceHealthStatus(BaseModel):
    """Overall service health status"""
    overall_state: ServiceHealthState = Field(..., description="Overall health")
    health_score: float = Field(..., ge=0, le=100, description="Health score 0-100")

    # Component health
    nf_health: Dict[str, NFHealthStatus] = Field(default_factory=dict)

    # Active issues
    active_violations: int = Field(0)
    active_anomalies: int = Field(0)
    active_rcas: int = Field(0)

    # Trends
    health_trend: str = Field("stable", description="improving, stable, degrading")

    # Summary
    summary: str = Field(..., description="Health summary")
    top_issues: List[str] = Field(default_factory=list)

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AssuranceReport(BaseModel):
    """Comprehensive assurance report"""
    report_id: str = Field(..., description="Unique report identifier")
    report_type: str = Field("periodic", description="periodic, on-demand, incident")

    # Time range
    start_time: datetime = Field(...)
    end_time: datetime = Field(...)

    # Health summary
    health_status: ServiceHealthStatus = Field(...)

    # KQI/KPI summary
    kqi_summary: List[KQIMetric] = Field(default_factory=list)
    kpi_summary: List[KPIMetric] = Field(default_factory=list)

    # SLA compliance
    sla_compliance_summary: Dict[str, float] = Field(default_factory=dict)
    violations_in_period: List[SLAViolation] = Field(default_factory=list)

    # Anomalies
    anomalies_in_period: List[AnomalyEvent] = Field(default_factory=list)

    # RCAs
    rca_results: List[RCAResult] = Field(default_factory=list)

    # Generated
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: str = Field("service_assurance")
