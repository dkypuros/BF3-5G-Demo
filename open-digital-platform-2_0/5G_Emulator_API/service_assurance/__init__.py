# File location: 5G_Emulator_API/service_assurance/__init__.py
# Service Assurance Module for 5G Network Emulator
# Provides SLA monitoring, anomaly detection, KQI/KPI calculation, and root cause analysis

"""
Service Assurance Module - Enterprise-Grade 5G Network Assurance

This module provides comprehensive service assurance capabilities for the 5G network emulator:

- KQI/KPI Calculator: Computes Key Quality Indicators and Key Performance Indicators
- SLA Manager: Monitors Service Level Agreement compliance
- Anomaly Detector: Identifies abnormal patterns using statistical methods
- RCA Engine: Root Cause Analysis for service degradations
- Metrics Collector: Gathers metrics from all Network Functions

Integration: Registers with NRF as a supporting network function and subscribes
to OpenTelemetry metrics from all 5G core and RAN components.
"""

__version__ = "1.0.0"
__author__ = "5G Lab Team"

from .models import (
    KQIMetric,
    KPIMetric,
    SLADefinition,
    SLAViolation,
    AnomalyEvent,
    RCAResult,
    ServiceHealthStatus,
    AssuranceReport
)

from .kqi_calculator import KQICalculator
from .sla_manager import SLAManager
from .anomaly_detector import AnomalyDetector
from .rca_engine import RCAEngine
from .collector import MetricsCollector

__all__ = [
    # Models
    "KQIMetric",
    "KPIMetric",
    "SLADefinition",
    "SLAViolation",
    "AnomalyEvent",
    "RCAResult",
    "ServiceHealthStatus",
    "AssuranceReport",
    # Core Components
    "KQICalculator",
    "SLAManager",
    "AnomalyDetector",
    "RCAEngine",
    "MetricsCollector",
]
