# File location: 5G_Emulator_API/test_service_assurance.py
# Test suite for Service Assurance module

"""
Service Assurance Test Suite

Tests all components of the service assurance system:
- KQI/KPI Calculator
- SLA Manager
- Anomaly Detector
- RCA Engine
- Metrics Collector

Run with: python test_service_assurance.py
"""

import sys
import time
import random
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, '.')

from service_assurance.models import (
    MetricCategory,
    SeverityLevel,
    NFType,
    AnomalyType,
)
from service_assurance.kqi_calculator import KQICalculator
from service_assurance.sla_manager import SLAManager
from service_assurance.anomaly_detector import AnomalyDetector
from service_assurance.rca_engine import RCAEngine
from service_assurance.collector import MetricsCollector


class TestResult:
    """Simple test result tracker"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def add(self, name: str, passed: bool, detail: str = ""):
        self.tests.append((name, passed, detail))
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self):
        print("\n" + "=" * 60)
        print("SERVICE ASSURANCE TEST RESULTS")
        print("=" * 60)
        for name, passed, detail in self.tests:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}")
            if detail and not passed:
                print(f"       {detail}")
        print("-" * 60)
        print(f"Total: {self.passed + self.failed} | Passed: {self.passed} | Failed: {self.failed}")
        print("=" * 60)
        return self.failed == 0


def test_kqi_calculator():
    """Test KQI Calculator functionality"""
    results = TestResult()
    print("\n--- Testing KQI Calculator ---")

    # Initialize calculator
    calc = KQICalculator()
    results.add("KQI Calculator initialization", calc is not None)

    # Test metric recording
    for i in range(50):
        calc.record_raw_metric("kpi_registration_attempts", random.randint(90, 110))
        calc.record_raw_metric("kpi_registration_success", random.randint(88, 110))
        calc.record_raw_metric("kpi_amf_latency", random.uniform(5, 15))
        calc.record_raw_metric("kpi_upf_latency", random.uniform(1, 5))

    results.add("Metric recording", True)

    # Test KPI computation
    kpi = calc.compute_kpi("kpi_registration_attempts")
    results.add("KPI computation", kpi is not None and kpi.value > 0,
                f"Value: {kpi.value if kpi else 'None'}")

    # Test KQI computation
    kqi = calc.compute_kqi("kqi_registration_success_rate")
    results.add("KQI computation", kqi is not None,
                f"Value: {kqi.value if kqi else 'None'}%")

    # Test all KQIs
    all_kqis = calc.compute_all_kqis()
    results.add("Compute all KQIs", len(all_kqis) > 0,
                f"Count: {len(all_kqis)}")

    # Test health score
    score, state = calc.get_health_score()
    results.add("Health score calculation", 0 <= score <= 100,
                f"Score: {score:.1f}, State: {state}")

    return results


def test_sla_manager():
    """Test SLA Manager functionality"""
    results = TestResult()
    print("\n--- Testing SLA Manager ---")

    # Initialize with KQI Calculator
    calc = KQICalculator()
    sla = SLAManager(calc)
    results.add("SLA Manager initialization", sla is not None)

    # Test default SLAs loaded
    slas = sla.list_slas()
    results.add("Default SLAs loaded", len(slas) >= 3,
                f"Count: {len(slas)}")

    # Test get SLA
    embb_sla = sla.get_sla("sla_embb_standard")
    results.add("Get eMBB SLA", embb_sla is not None and embb_sla.service_type == "eMBB")

    # Populate metrics for SLA evaluation
    for i in range(30):
        calc.record_raw_metric("kpi_registration_attempts", 100)
        calc.record_raw_metric("kpi_registration_success", random.randint(95, 100))
        calc.record_raw_metric("kqi_registration_success_rate", random.uniform(99.0, 100.0))
        calc.record_raw_metric("kqi_user_plane_latency", random.uniform(5, 15))

    # Test SLA evaluation
    violations = sla.evaluate_sla("sla_embb_standard")
    results.add("SLA evaluation runs", True,
                f"Violations: {len(violations)}")

    # Test compliance summary
    compliance = sla.get_compliance_summary()
    results.add("Compliance summary", len(compliance) > 0,
                f"SLAs tracked: {len(compliance)}")

    # Test SLA status
    status = sla.get_sla_status("sla_embb_standard")
    results.add("SLA status retrieval", "sla_id" in status)

    return results


def test_anomaly_detector():
    """Test Anomaly Detector functionality"""
    results = TestResult()
    print("\n--- Testing Anomaly Detector ---")

    # Initialize
    calc = KQICalculator()
    detector = AnomalyDetector(calc)
    results.add("Anomaly Detector initialization", detector is not None)

    # Build baseline with normal values
    normal_values = [random.uniform(99.0, 100.0) for _ in range(50)]
    detector.update_baseline("kqi_registration_success_rate", normal_values)
    results.add("Baseline update", "kqi_registration_success_rate" in detector._baselines)

    # Test Z-score calculation
    zscore = detector.calculate_zscore("kqi_registration_success_rate", 99.5)
    results.add("Z-score calculation", zscore is not None,
                f"Z-score: {zscore:.2f}" if zscore else "")

    # Test anomaly detection with normal value
    anomaly = detector.detect_zscore_anomaly("kqi_registration_success_rate", 99.5)
    results.add("Normal value detection (no anomaly)", anomaly is None)

    # Test anomaly detection with abnormal value
    anomaly = detector.detect_zscore_anomaly("kqi_registration_success_rate", 85.0)
    results.add("Abnormal value detection (anomaly)", anomaly is not None,
                f"Type: {anomaly.anomaly_type.value if anomaly else ''}")

    # Test rate change detection
    rate_anomaly = detector.detect_rate_change_anomaly(
        "kqi_registration_success_rate", 80.0, 99.5
    )
    results.add("Rate change detection", rate_anomaly is not None)

    # Test anomaly summary
    summary = detector.get_anomaly_summary()
    results.add("Anomaly summary", "total_active" in summary)

    return results


def test_rca_engine():
    """Test RCA Engine functionality"""
    results = TestResult()
    print("\n--- Testing RCA Engine ---")

    # Initialize all components
    calc = KQICalculator()
    sla = SLAManager(calc)
    detector = AnomalyDetector(calc)
    rca = RCAEngine(calc, sla, detector)
    results.add("RCA Engine initialization", rca is not None)

    # Test dependency chain
    chain = rca._get_dependency_chain(NFType.UPF)
    results.add("Dependency chain retrieval", len(chain) > 0,
                f"Chain: {[n.value for n in chain]}")

    # Populate some data and create a violation
    for i in range(30):
        calc.record_raw_metric("kpi_registration_attempts", 100)
        calc.record_raw_metric("kpi_registration_success", 70)  # Low success rate
        calc.record_raw_metric("kqi_registration_success_rate", 70.0)  # Below SLA

    # Evaluate SLA to generate violation
    sla.evaluate_all_slas()

    # Test RCA analysis
    rca_result = rca.analyze("test_trigger_001")
    results.add("RCA analysis execution", rca_result is not None)
    results.add("RCA identifies causes", len(rca_result.root_causes) > 0,
                f"Causes: {len(rca_result.root_causes)}")
    results.add("RCA provides recommendations", len(rca_result.immediate_actions) > 0,
                f"Actions: {len(rca_result.immediate_actions)}")

    # Test RCA history
    history = rca.get_rca_history()
    results.add("RCA history tracking", len(history) > 0)

    return results


def test_metrics_collector():
    """Test Metrics Collector functionality"""
    results = TestResult()
    print("\n--- Testing Metrics Collector ---")

    # Initialize
    calc = KQICalculator()
    collector = MetricsCollector(calc)
    results.add("Metrics Collector initialization", collector is not None)

    # Test default endpoints
    endpoints = collector.get_endpoint_status()
    results.add("Default endpoints registered", len(endpoints) > 0,
                f"Endpoints: {len(endpoints)}")

    # Test simulated metrics collection
    simulated = collector.collect_simulated_metrics()
    results.add("Simulated metrics generation", len(simulated) > 0,
                f"Metrics: {len(simulated)}")

    # Verify metrics fed to KQI Calculator
    kpi = calc.compute_kpi("kpi_registration_attempts")
    results.add("Metrics fed to calculator", kpi is not None and kpi.value > 0)

    # Test stats
    stats = collector.get_stats()
    results.add("Collection stats", "collections_total" in stats)

    return results


def test_integration():
    """Test integrated service assurance workflow"""
    results = TestResult()
    print("\n--- Testing Integration ---")

    # Initialize full stack
    calc = KQICalculator()
    sla = SLAManager(calc)
    detector = AnomalyDetector(calc)
    rca = RCAEngine(calc, sla, detector)
    collector = MetricsCollector(calc)

    results.add("Full stack initialization", True)

    # Simulate normal operation
    print("  Simulating normal operation...")
    for _ in range(20):
        collector.collect_simulated_metrics()

    # Evaluate SLAs
    violations = sla.evaluate_all_slas()
    results.add("Normal operation - low violations", len(violations) <= 2,
                f"Violations: {len(violations)}")

    # Simulate degraded operation
    print("  Simulating degraded operation...")
    for _ in range(10):
        calc.record_raw_metric("kpi_registration_attempts", 100)
        calc.record_raw_metric("kpi_registration_success", 50)  # 50% success
        calc.record_raw_metric("kqi_registration_success_rate", 50.0)
        calc.record_raw_metric("kqi_user_plane_latency", 100.0)  # High latency

    # Evaluate again
    violations = sla.evaluate_all_slas()
    results.add("Degraded operation - violations detected", len(violations) > 0,
                f"Violations: {len(violations)}")

    # Detect anomalies
    detector.update_baseline("kqi_registration_success_rate", [99.5] * 50)
    anomalies = detector.analyze_metric("kqi_registration_success_rate", 50.0)
    results.add("Anomalies detected", len(anomalies) > 0,
                f"Anomalies: {len(anomalies)}")

    # Run RCA
    if violations:
        rca_result = rca.analyze_violation(violations[0])
        results.add("RCA triggered and completed", rca_result is not None)
        results.add("RCA impact summary generated", len(rca_result.impact_summary) > 0)

    # Get final health score
    score, state = calc.get_health_score()
    results.add("Health score reflects degradation", score < 90 or state != "healthy",
                f"Score: {score:.1f}, State: {state}")

    return results


def test_api_models():
    """Test Pydantic models"""
    results = TestResult()
    print("\n--- Testing API Models ---")

    from service_assurance.models import (
        KQIMetric, KPIMetric, SLADefinition, SLAThreshold,
        SLAViolation, AnomalyEvent, RCAResult, ServiceHealthStatus
    )

    # Test KQI model
    kqi = KQIMetric(
        kqi_id="test_kqi",
        name="Test KQI",
        category=MetricCategory.ACCESSIBILITY,
        value=99.5,
        unit="%",
    )
    results.add("KQI model creation", kqi.kqi_id == "test_kqi")

    # Test SLA model
    threshold = SLAThreshold(
        metric_id="test_metric",
        operator=">=",
        value=99.0,
    )
    sla_def = SLADefinition(
        sla_id="test_sla",
        name="Test SLA",
        service_type="eMBB",
        thresholds=[threshold],
    )
    results.add("SLA definition model", sla_def.sla_id == "test_sla")

    # Test violation model
    violation = SLAViolation(
        violation_id="viol_001",
        sla_id="test_sla",
        sla_name="Test SLA",
        threshold=threshold,
        actual_value=98.0,
        expected_value=99.0,
        deviation_pct=1.0,
        start_time=datetime.utcnow(),
        severity=SeverityLevel.MAJOR,
    )
    results.add("Violation model", violation.violation_id == "viol_001")

    # Test anomaly model
    anomaly = AnomalyEvent(
        anomaly_id="anom_001",
        anomaly_type=AnomalyType.DROP,
        metric_id="test_metric",
        metric_name="Test Metric",
        current_value=80.0,
        expected_value=99.0,
        deviation_score=3.5,
        confidence=0.95,
        severity=SeverityLevel.MAJOR,
        description="Test anomaly",
    )
    results.add("Anomaly model", anomaly.anomaly_id == "anom_001")

    return results


def main():
    """Run all tests"""
    print("=" * 60)
    print("5G SERVICE ASSURANCE TEST SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    all_results = []

    # Run test suites
    all_results.append(("KQI Calculator", test_kqi_calculator()))
    all_results.append(("SLA Manager", test_sla_manager()))
    all_results.append(("Anomaly Detector", test_anomaly_detector()))
    all_results.append(("RCA Engine", test_rca_engine()))
    all_results.append(("Metrics Collector", test_metrics_collector()))
    all_results.append(("API Models", test_api_models()))
    all_results.append(("Integration", test_integration()))

    # Print individual results
    for name, result in all_results:
        result.summary()

    # Overall summary
    total_passed = sum(r.passed for _, r in all_results)
    total_failed = sum(r.failed for _, r in all_results)

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    for name, result in all_results:
        status = "PASS" if result.failed == 0 else "FAIL"
        print(f"[{status}] {name}: {result.passed}/{result.passed + result.failed}")

    print("-" * 60)
    print(f"TOTAL: {total_passed + total_failed} tests")
    print(f"PASSED: {total_passed}")
    print(f"FAILED: {total_failed}")

    if total_failed == 0:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print(f"\n{total_failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
