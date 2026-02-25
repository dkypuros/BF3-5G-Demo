# File location: 5G_Emulator_API/service_assurance/assurance_api.py
# Service Assurance FastAPI Application

"""
Service Assurance REST API

Provides REST endpoints for:
- Health status and service health dashboard
- KQI/KPI queries
- SLA management and compliance
- Anomaly events
- Root cause analysis
- Metric ingestion (push mode)

Integrates with NRF for service discovery.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import uvicorn

from .models import (
    KQIMetric,
    KPIMetric,
    SLADefinition,
    SLAViolation,
    AnomalyEvent,
    RCAResult,
    ServiceHealthStatus,
    ServiceHealthState,
    NFHealthStatus,
    AssuranceReport,
    NFType,
    SeverityLevel,
)
from .kqi_calculator import KQICalculator
from .sla_manager import SLAManager
from .anomaly_detector import AnomalyDetector
from .rca_engine import RCAEngine
from .collector import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
kqi_calculator: Optional[KQICalculator] = None
sla_manager: Optional[SLAManager] = None
anomaly_detector: Optional[AnomalyDetector] = None
rca_engine: Optional[RCAEngine] = None
metrics_collector: Optional[MetricsCollector] = None

# Configuration
NRF_URL = os.environ.get("NRF_URL", "http://127.0.0.1:8000")
SERVICE_PORT = int(os.environ.get("ASSURANCE_PORT", "9011"))


# Request/Response Models
class MetricPushRequest(BaseModel):
    """Request model for pushing metrics"""
    nf_type: str = Field(..., description="Network function type")
    metrics: dict = Field(..., description="Metrics dictionary")
    timestamp: Optional[datetime] = None


class RCARequest(BaseModel):
    """Request model for triggering RCA"""
    trigger_id: str = Field(..., description="Violation or anomaly ID that triggers RCA")
    time_range_minutes: int = Field(30, description="Time range to analyze")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global kqi_calculator, sla_manager, anomaly_detector, rca_engine, metrics_collector

    # Startup
    logger.info("Starting Service Assurance...")

    # Initialize components
    kqi_calculator = KQICalculator()
    sla_manager = SLAManager(kqi_calculator)
    anomaly_detector = AnomalyDetector(kqi_calculator)
    rca_engine = RCAEngine(kqi_calculator, sla_manager, anomaly_detector)
    metrics_collector = MetricsCollector(kqi_calculator)

    # Register violation callback to trigger RCA
    def on_violation(violation: SLAViolation):
        if violation.severity in (SeverityLevel.CRITICAL, SeverityLevel.MAJOR):
            logger.info(f"Auto-triggering RCA for violation: {violation.violation_id}")
            rca_engine.analyze(violation.violation_id)

    sla_manager.register_violation_callback(on_violation)

    # Register anomaly callback
    def on_anomaly(anomaly: AnomalyEvent):
        if anomaly.severity == SeverityLevel.CRITICAL:
            logger.info(f"Auto-triggering RCA for anomaly: {anomaly.anomaly_id}")
            rca_engine.analyze(anomaly.anomaly_id)

    anomaly_detector.register_anomaly_callback(on_anomaly)

    # Register with NRF
    try:
        nf_registration = {
            "nf_type": "SERVICE_ASSURANCE",
            "ip": "127.0.0.1",
            "port": SERVICE_PORT,
        }
        response = requests.post(f"{NRF_URL}/register", json=nf_registration, timeout=5)
        if response.status_code == 200:
            logger.info("Service Assurance registered with NRF")
        else:
            logger.warning(f"NRF registration returned: {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not register with NRF: {e}")

    # Start monitoring services
    metrics_collector.start_collection(use_simulation=True)  # Start with simulation
    sla_manager.start_monitoring(interval_seconds=30)
    anomaly_detector.start_monitoring(interval_seconds=30)

    logger.info("Service Assurance started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Service Assurance...")
    metrics_collector.stop_collection()
    sla_manager.stop_monitoring()
    anomaly_detector.stop_monitoring()
    logger.info("Service Assurance stopped")


# Create FastAPI app
app = FastAPI(
    title="5G Service Assurance API",
    description="Service Assurance for 5G Network - KQI/KPI, SLA, Anomaly Detection, RCA",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
    )


@app.get("/status", tags=["Health"])
async def get_status():
    """Get comprehensive service status"""
    return {
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "kqi_calculator": "active" if kqi_calculator else "inactive",
            "sla_manager": "active" if sla_manager else "inactive",
            "anomaly_detector": "active" if anomaly_detector else "inactive",
            "rca_engine": "active" if rca_engine else "inactive",
            "metrics_collector": "active" if metrics_collector else "inactive",
        },
        "metrics_collection": metrics_collector.get_stats() if metrics_collector else {},
    }


@app.get("/dashboard", tags=["Health"])
async def get_dashboard():
    """Get dashboard data - comprehensive view of network health"""
    if not all([kqi_calculator, sla_manager, anomaly_detector]):
        raise HTTPException(status_code=503, detail="Service not fully initialized")

    # Calculate health score
    health_score, health_state = kqi_calculator.get_health_score()

    # Get active issues
    active_violations = sla_manager.get_active_violations()
    active_anomalies = anomaly_detector.get_active_anomalies()

    # Get compliance summary
    compliance = sla_manager.get_compliance_summary()

    # Get top KQIs
    kqis = kqi_calculator.compute_all_kqis()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "health": {
            "score": round(health_score, 1),
            "state": health_state,
        },
        "issues": {
            "active_violations": len(active_violations),
            "active_anomalies": len(active_anomalies),
            "critical_count": sum(1 for v in active_violations if v.severity == SeverityLevel.CRITICAL),
        },
        "compliance": compliance,
        "kqis": [
            {
                "id": kqi.kqi_id,
                "name": kqi.name,
                "value": round(kqi.value, 2),
                "unit": kqi.unit,
                "status": "ok" if kqi.target_value and kqi.value >= kqi.target_value * 0.95 else "degraded",
            }
            for kqi in kqis[:10]
        ],
    }


# =============================================================================
# KQI/KPI Endpoints
# =============================================================================

@app.get("/kqi", response_model=List[KQIMetric], tags=["KQI/KPI"])
async def list_kqis(
    category: Optional[str] = Query(None, description="Filter by category"),
    window_minutes: int = Query(5, description="Computation window in minutes"),
):
    """List all KQIs"""
    if not kqi_calculator:
        raise HTTPException(status_code=503, detail="KQI Calculator not initialized")

    kqis = kqi_calculator.compute_all_kqis(window_minutes)

    if category:
        kqis = [k for k in kqis if k.category.value == category]

    return kqis


@app.get("/kqi/{kqi_id}", response_model=KQIMetric, tags=["KQI/KPI"])
async def get_kqi(
    kqi_id: str = Path(..., description="KQI identifier"),
    window_minutes: int = Query(5, description="Computation window"),
):
    """Get a specific KQI"""
    if not kqi_calculator:
        raise HTTPException(status_code=503, detail="KQI Calculator not initialized")

    kqi = kqi_calculator.compute_kqi(kqi_id, window_minutes)
    if not kqi:
        raise HTTPException(status_code=404, detail=f"KQI not found: {kqi_id}")

    return kqi


@app.get("/kpi", response_model=List[KPIMetric], tags=["KQI/KPI"])
async def list_kpis(
    nf_type: Optional[str] = Query(None, description="Filter by NF type"),
    window_minutes: int = Query(5, description="Computation window"),
):
    """List all KPIs"""
    if not kqi_calculator:
        raise HTTPException(status_code=503, detail="KQI Calculator not initialized")

    kpis = kqi_calculator.compute_all_kpis(window_minutes)

    if nf_type:
        kpis = [k for k in kpis if k.nf_type.value == nf_type]

    return kpis


@app.get("/kpi/{kpi_id}", response_model=KPIMetric, tags=["KQI/KPI"])
async def get_kpi(
    kpi_id: str = Path(..., description="KPI identifier"),
    window_minutes: int = Query(5, description="Computation window"),
):
    """Get a specific KPI"""
    if not kqi_calculator:
        raise HTTPException(status_code=503, detail="KQI Calculator not initialized")

    kpi = kqi_calculator.compute_kpi(kpi_id, window_minutes)
    if not kpi:
        raise HTTPException(status_code=404, detail=f"KPI not found: {kpi_id}")

    return kpi


# =============================================================================
# SLA Endpoints
# =============================================================================

@app.get("/sla", response_model=List[SLADefinition], tags=["SLA"])
async def list_slas():
    """List all registered SLAs"""
    if not sla_manager:
        raise HTTPException(status_code=503, detail="SLA Manager not initialized")

    return sla_manager.list_slas()


@app.get("/sla/{sla_id}", tags=["SLA"])
async def get_sla_status(sla_id: str = Path(..., description="SLA identifier")):
    """Get SLA status and compliance"""
    if not sla_manager:
        raise HTTPException(status_code=503, detail="SLA Manager not initialized")

    status = sla_manager.get_sla_status(sla_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])

    return status


@app.get("/sla/compliance/summary", tags=["SLA"])
async def get_compliance_summary(
    window_hours: int = Query(24, description="Compliance window in hours"),
):
    """Get SLA compliance summary"""
    if not sla_manager:
        raise HTTPException(status_code=503, detail="SLA Manager not initialized")

    return sla_manager.get_compliance_summary(window_hours=window_hours)


@app.get("/violations", response_model=List[SLAViolation], tags=["SLA"])
async def list_violations(
    active_only: bool = Query(True, description="Only active violations"),
    sla_id: Optional[str] = Query(None, description="Filter by SLA ID"),
):
    """List SLA violations"""
    if not sla_manager:
        raise HTTPException(status_code=503, detail="SLA Manager not initialized")

    if active_only:
        violations = sla_manager.get_active_violations()
    else:
        violations = sla_manager.get_violation_history(sla_id=sla_id)

    if sla_id and active_only:
        violations = [v for v in violations if v.sla_id == sla_id]

    return violations


@app.post("/violations/{violation_id}/acknowledge", tags=["SLA"])
async def acknowledge_violation(
    violation_id: str = Path(..., description="Violation ID"),
    acknowledged_by: str = Query(..., description="Acknowledger name"),
):
    """Acknowledge a violation"""
    if not sla_manager:
        raise HTTPException(status_code=503, detail="SLA Manager not initialized")

    success = sla_manager.acknowledge_violation(violation_id, acknowledged_by)
    if not success:
        raise HTTPException(status_code=404, detail="Violation not found")

    return {"status": "acknowledged", "violation_id": violation_id}


# =============================================================================
# Anomaly Endpoints
# =============================================================================

@app.get("/anomalies", response_model=List[AnomalyEvent], tags=["Anomaly"])
async def list_anomalies(
    active_only: bool = Query(True, description="Only active anomalies"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
):
    """List detected anomalies"""
    if not anomaly_detector:
        raise HTTPException(status_code=503, detail="Anomaly Detector not initialized")

    if active_only:
        anomalies = anomaly_detector.get_active_anomalies()
    else:
        anomalies = anomaly_detector.get_anomaly_history()

    if severity:
        anomalies = [a for a in anomalies if a.severity.value == severity]

    return anomalies


@app.get("/anomalies/summary", tags=["Anomaly"])
async def get_anomaly_summary():
    """Get anomaly summary"""
    if not anomaly_detector:
        raise HTTPException(status_code=503, detail="Anomaly Detector not initialized")

    return anomaly_detector.get_anomaly_summary()


# =============================================================================
# RCA Endpoints
# =============================================================================

@app.post("/rca", response_model=RCAResult, tags=["RCA"])
async def trigger_rca(request: RCARequest):
    """Trigger root cause analysis"""
    if not rca_engine:
        raise HTTPException(status_code=503, detail="RCA Engine not initialized")

    result = rca_engine.analyze(
        trigger_id=request.trigger_id,
        time_range_minutes=request.time_range_minutes,
    )

    return result


@app.get("/rca", tags=["RCA"])
async def list_rcas(limit: int = Query(10, description="Maximum results")):
    """List recent RCA results"""
    if not rca_engine:
        raise HTTPException(status_code=503, detail="RCA Engine not initialized")

    return rca_engine.get_rca_history(limit=limit)


@app.get("/rca/{rca_id}", response_model=RCAResult, tags=["RCA"])
async def get_rca(rca_id: str = Path(..., description="RCA identifier")):
    """Get specific RCA result"""
    if not rca_engine:
        raise HTTPException(status_code=503, detail="RCA Engine not initialized")

    result = rca_engine.get_rca(rca_id)
    if not result:
        raise HTTPException(status_code=404, detail="RCA not found")

    return result


# =============================================================================
# Metrics Push Endpoint
# =============================================================================

@app.post("/metrics/push", tags=["Metrics"])
async def push_metrics(request: MetricPushRequest):
    """Push metrics from NFs (webhook mode)"""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics Collector not initialized")

    try:
        nf_type = NFType(request.nf_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid NF type: {request.nf_type}")

    metrics_collector.push_metrics(nf_type, request.metrics)

    return {"status": "accepted", "metrics_count": len(request.metrics)}


@app.get("/metrics/endpoints", tags=["Metrics"])
async def get_metric_endpoints():
    """Get status of metric collection endpoints"""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics Collector not initialized")

    return metrics_collector.get_endpoint_status()


# =============================================================================
# Report Endpoint
# =============================================================================

@app.get("/report", response_model=AssuranceReport, tags=["Reports"])
async def generate_report(
    hours: int = Query(24, description="Report time range in hours"),
):
    """Generate comprehensive assurance report"""
    if not all([kqi_calculator, sla_manager, anomaly_detector, rca_engine]):
        raise HTTPException(status_code=503, detail="Service not fully initialized")

    from datetime import timedelta
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)

    # Build health status
    health_score, health_state = kqi_calculator.get_health_score()

    health_status = ServiceHealthStatus(
        overall_state=ServiceHealthState(health_state),
        health_score=health_score,
        active_violations=len(sla_manager.get_active_violations()),
        active_anomalies=len(anomaly_detector.get_active_anomalies()),
        active_rcas=len([r for r in rca_engine.get_rca_history() if r.analysis_start >= start_time]),
        health_trend="stable",
        summary=f"Network health score: {health_score:.1f}/100",
        top_issues=[v.sla_name for v in sla_manager.get_active_violations()[:5]],
    )

    report = AssuranceReport(
        report_id=f"report_{int(datetime.utcnow().timestamp())}",
        report_type="on-demand",
        start_time=start_time,
        end_time=end_time,
        health_status=health_status,
        kqi_summary=kqi_calculator.compute_all_kqis(),
        kpi_summary=kqi_calculator.compute_all_kpis(),
        sla_compliance_summary=sla_manager.get_compliance_summary(window_hours=hours),
        violations_in_period=sla_manager.get_violation_history(start_time=start_time),
        anomalies_in_period=anomaly_detector.get_anomaly_history(start_time=start_time),
        rca_results=rca_engine.get_rca_history(start_time=start_time),
    )

    return report


# =============================================================================
# Main Entry Point
# =============================================================================

def create_app() -> FastAPI:
    """Factory function to create the app"""
    return app


def main():
    """Run the service assurance API"""
    import argparse

    parser = argparse.ArgumentParser(description="5G Service Assurance API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=SERVICE_PORT, help="Port to bind to")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
