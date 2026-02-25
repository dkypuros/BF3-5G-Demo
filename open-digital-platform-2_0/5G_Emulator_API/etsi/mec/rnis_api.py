# File location: 5G_Emulator_API/etsi/mec/rnis_api.py
# ETSI GS MEC 012 - Radio Network Information Service API
# Provides radio network information to MEC applications

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
import uuid
import logging
from datetime import datetime, timezone
import os
import httpx
from opentelemetry import trace
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

# =============================================================================
# ETSI GS MEC 012 Data Models - RNIS API
# =============================================================================

class Ecgi(BaseModel):
    """ETSI GS MEC 012 - E-UTRAN Cell Global Identifier"""
    plmn: Dict[str, str] = Field(..., description="PLMN identity (mcc, mnc)")
    cellId: str = Field(..., description="Cell ID")

class NrCgi(BaseModel):
    """ETSI GS MEC 012 - NR Cell Global Identifier (5G)"""
    plmn: Dict[str, str] = Field(..., description="PLMN identity")
    nrCellId: str = Field(..., description="NR Cell ID (36 bits)")

class AssociateId(BaseModel):
    """ETSI GS MEC 012 - Associate ID for UE identification"""
    type: str = Field(..., description="ID type: UE_IPV4_ADDRESS, UE_IPV6_ADDRESS, NATED_IP_ADDRESS, GTP_TEID")
    value: str = Field(..., description="ID value")

class RsrpRsrq(BaseModel):
    """ETSI GS MEC 012 - RSRP/RSRQ measurements"""
    rsrp: int = Field(..., ge=-156, le=-44, description="RSRP in dBm")
    rsrq: int = Field(..., ge=-34, le=3, description="RSRQ in dB")
    sinr: Optional[int] = Field(None, ge=-23, le=40, description="SINR in dB")

class MeasRepUeNotification(BaseModel):
    """ETSI GS MEC 012 - UE Measurement Report (Section 6.2.2)"""
    timestamp: datetime = Field(..., description="Timestamp of measurement")
    associateId: List[AssociateId] = Field(..., description="UE identifiers")
    ecgi: Optional[Ecgi] = Field(None, description="E-UTRAN cell (4G)")
    nrCgi: Optional[NrCgi] = Field(None, description="NR cell (5G)")
    rsrp: int = Field(..., description="Reference Signal Received Power")
    rsrq: int = Field(..., description="Reference Signal Received Quality")
    rsrpResultsNCell: Optional[List[Dict[str, Any]]] = Field(None, description="Neighbor cell RSRP")
    triggerType: Optional[str] = Field(None, description="Trigger type")

class L2MeasInfo(BaseModel):
    """ETSI GS MEC 012 - L2 Measurement Info"""
    cellId: str = Field(..., description="Cell ID")
    dlGbrPrbUsage: Optional[int] = Field(None, description="DL GBR PRB usage (%)")
    ulGbrPrbUsage: Optional[int] = Field(None, description="UL GBR PRB usage (%)")
    dlNonGbrPrbUsage: Optional[int] = Field(None, description="DL non-GBR PRB usage (%)")
    ulNonGbrPrbUsage: Optional[int] = Field(None, description="UL non-GBR PRB usage (%)")
    dlTotalPrbUsage: Optional[int] = Field(None, description="DL total PRB usage (%)")
    ulTotalPrbUsage: Optional[int] = Field(None, description="UL total PRB usage (%)")
    receivedDedicatedPreambles: Optional[int] = Field(None, description="RACH preambles")
    receivedRandomlySelectedPreambles: Optional[int] = Field(None, description="Random preambles")

class CellInfo(BaseModel):
    """ETSI GS MEC 012 - Cell Information"""
    cellId: str = Field(..., description="Cell identifier")
    nrCgi: Optional[NrCgi] = Field(None, description="NR CGI")
    ecgi: Optional[Ecgi] = Field(None, description="E-UTRAN CGI")
    dl_earfcn: Optional[int] = Field(None, description="DL EARFCN")
    ul_earfcn: Optional[int] = Field(None, description="UL EARFCN")
    dl_bandwidth: Optional[int] = Field(None, description="DL bandwidth (MHz)")
    ul_bandwidth: Optional[int] = Field(None, description="UL bandwidth (MHz)")
    pci: Optional[int] = Field(None, description="Physical Cell ID")
    numberOfUes: Optional[int] = Field(None, description="Connected UEs")

class PlmnInfo(BaseModel):
    """ETSI GS MEC 012 - PLMN Information"""
    appInstanceId: str = Field(..., description="App instance requesting info")
    plmn: List[Dict[str, str]] = Field(..., description="List of PLMNs")
    timeStamp: Optional[datetime] = Field(None, description="Timestamp")

class RabInfo(BaseModel):
    """ETSI GS MEC 012 - Radio Access Bearer Info"""
    appInstanceId: str = Field(..., description="App instance ID")
    requestId: str = Field(..., description="Request ID")
    cellUserInfo: Optional[List[Dict[str, Any]]] = Field(None, description="Cell/user info")
    timeStamp: Optional[datetime] = Field(None, description="Timestamp")

class RnisSubscription(BaseModel):
    """ETSI GS MEC 012 - RNIS Subscription Base"""
    subscriptionType: str = Field(..., description="Type of subscription")
    callbackReference: str = Field(..., description="Callback URL")
    filterCriteriaAssocQci: Optional[Dict[str, Any]] = Field(None, description="QCI filter")
    filterCriteriaAssocTri: Optional[Dict[str, Any]] = Field(None, description="Trigger filter")
    expiryDeadline: Optional[datetime] = Field(None, description="Expiry time")

# =============================================================================
# RNIS Service Core Class
# =============================================================================

class RnisService:
    """ETSI GS MEC 012 - Radio Network Information Service

    Provides radio network information from gNodeB/CU to MEC applications.
    """

    def __init__(self, gnb_url: str = None, cu_url: str = None):
        """Initialize RNIS

        Args:
            gnb_url: URL of gNodeB for RAN info
            cu_url: URL of CU for radio measurements
        """
        self.service_id = str(uuid.uuid4())
        self.gnb_url = gnb_url or os.environ.get("GNB_URL", "http://127.0.0.1:8010")
        self.cu_url = cu_url or os.environ.get("CU_URL", "http://127.0.0.1:8011")

        # Cell registry
        self.cells: Dict[str, CellInfo] = {}

        # UE measurement cache
        self.ue_measurements: Dict[str, MeasRepUeNotification] = {}

        # L2 measurements cache
        self.l2_measurements: Dict[str, L2MeasInfo] = {}

        # Subscriptions
        self.subscriptions: Dict[str, RnisSubscription] = {}

        # Initialize sample cells
        self._init_sample_cells()

        logger.info(f"RNIS Service initialized: {self.service_id}")

    def _init_sample_cells(self):
        """Initialize sample cell data"""
        cells = [
            {"id": "cell001", "pci": 1, "bw": 100},
            {"id": "cell002", "pci": 2, "bw": 100},
            {"id": "cell003", "pci": 3, "bw": 50},
        ]

        for cell in cells:
            self.cells[cell["id"]] = CellInfo(
                cellId=cell["id"],
                nrCgi=NrCgi(plmn={"mcc": "310", "mnc": "260"}, nrCellId=cell["id"]),
                dl_bandwidth=cell["bw"],
                ul_bandwidth=cell["bw"],
                pci=cell["pci"],
                numberOfUes=0
            )

            # Initialize L2 measurements
            self.l2_measurements[cell["id"]] = L2MeasInfo(
                cellId=cell["id"],
                dlTotalPrbUsage=25,
                ulTotalPrbUsage=15,
                dlGbrPrbUsage=5,
                ulGbrPrbUsage=3
            )

    async def get_ue_measurement(self, ue_id: str) -> Optional[MeasRepUeNotification]:
        """Get UE radio measurements from CU"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.cu_url}/measurements/{ue_id}")
                if response.status_code == 200:
                    data = response.json()
                    meas = MeasRepUeNotification(
                        timestamp=datetime.now(timezone.utc),
                        associateId=[AssociateId(type="UE_IPV4_ADDRESS", value=ue_id)],
                        nrCgi=NrCgi(plmn={"mcc": "310", "mnc": "260"}, nrCellId=data.get("cellId", "cell001")),
                        rsrp=data.get("rsrp", -85),
                        rsrq=data.get("rsrq", -10)
                    )
                    self.ue_measurements[ue_id] = meas
                    return meas
        except Exception as e:
            logger.warning(f"Could not query CU for measurements: {e}")

        # Return simulated measurement
        return self._simulate_measurement(ue_id)

    def _simulate_measurement(self, ue_id: str) -> MeasRepUeNotification:
        """Simulate UE measurement for demo"""
        import random

        cell_id = random.choice(list(self.cells.keys()))
        meas = MeasRepUeNotification(
            timestamp=datetime.now(timezone.utc),
            associateId=[AssociateId(type="UE_IPV4_ADDRESS", value=ue_id)],
            nrCgi=NrCgi(plmn={"mcc": "310", "mnc": "260"}, nrCellId=cell_id),
            rsrp=random.randint(-110, -70),
            rsrq=random.randint(-20, -5)
        )
        self.ue_measurements[ue_id] = meas
        return meas

    def get_cell_info(self, cell_id: str) -> Optional[CellInfo]:
        """Get cell information"""
        return self.cells.get(cell_id)

    def get_all_cells(self) -> List[CellInfo]:
        """Get all cells"""
        return list(self.cells.values())

    def get_l2_measurement(self, cell_id: str) -> Optional[L2MeasInfo]:
        """Get L2 measurements for a cell"""
        return self.l2_measurements.get(cell_id)

    def create_subscription(self, sub: RnisSubscription) -> RnisSubscription:
        """Create RNIS subscription"""
        sub_id = str(uuid.uuid4())
        self.subscriptions[sub_id] = sub
        logger.info(f"Created RNIS subscription: {sub_id}")
        return sub


# =============================================================================
# FastAPI Application - RNIS API
# =============================================================================

app = FastAPI(
    title="ETSI MEC Radio Network Information Service API",
    description="RNIS implementing ETSI GS MEC 012",
    version="2.2.1",
    docs_url="/rni/docs",
    openapi_url="/rni/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RNIS instance
rnis_service = RnisService()


# -----------------------------------------------------------------------------
# ETSI GS MEC 012 - Query Endpoints
# -----------------------------------------------------------------------------

@app.get("/rni/v2/queries/rab_info",
         response_model=RabInfo,
         tags=["RNI Queries"])
async def get_rab_info(
    app_instance_id: str = Query(..., alias="app_ins_id", description="App instance ID"),
    cell_id: Optional[List[str]] = Query(None, description="Cell IDs"),
    ue_ipv4_address: Optional[List[str]] = Query(None, description="UE IPv4 addresses")
):
    """ETSI GS MEC 012 - Query RAB Information

    Spec: Section 7.3.2 - GET /queries/rab_info
    """
    with tracer.start_as_current_span("rnis_rab_query") as span:
        span.set_attribute("etsi.spec", "GS MEC 012")

        return RabInfo(
            appInstanceId=app_instance_id,
            requestId=str(uuid.uuid4()),
            timeStamp=datetime.now(timezone.utc)
        )


@app.get("/rni/v2/queries/plmn_info",
         response_model=PlmnInfo,
         tags=["RNI Queries"])
async def get_plmn_info(
    app_instance_id: str = Query(..., alias="app_ins_id", description="App instance ID")
):
    """ETSI GS MEC 012 - Query PLMN Information

    Spec: Section 7.4.2
    """
    return PlmnInfo(
        appInstanceId=app_instance_id,
        plmn=[{"mcc": "310", "mnc": "260"}],
        timeStamp=datetime.now(timezone.utc)
    )


@app.get("/rni/v2/queries/layer2_meas",
         response_model=Dict[str, Any],
         tags=["RNI Queries"])
async def get_l2_measurements(
    app_instance_id: str = Query(..., alias="app_ins_id", description="App instance ID"),
    cell_id: Optional[List[str]] = Query(None, description="Cell IDs")
):
    """ETSI GS MEC 012 - Query L2 Measurements

    Spec: Section 7.5.2
    """
    measurements = []
    cells_to_query = cell_id if cell_id else list(rnis_service.cells.keys())

    for cid in cells_to_query:
        meas = rnis_service.get_l2_measurement(cid)
        if meas:
            measurements.append(meas.dict())

    return {
        "l2Meas": {
            "timeStamp": datetime.now(timezone.utc).isoformat(),
            "cellInfo": measurements
        }
    }


# -----------------------------------------------------------------------------
# Cell Information
# -----------------------------------------------------------------------------

@app.get("/rni/v2/cells",
         response_model=Dict[str, Any],
         tags=["Cell Information"])
async def get_cells():
    """Get all cell information"""
    cells = rnis_service.get_all_cells()
    return {
        "cellInfo": [c.dict() for c in cells]
    }


@app.get("/rni/v2/cells/{cellId}",
         response_model=CellInfo,
         tags=["Cell Information"])
async def get_cell(cellId: str = Path(..., description="Cell ID")):
    """Get specific cell information"""
    cell = rnis_service.get_cell_info(cellId)
    if not cell:
        raise HTTPException(status_code=404, detail="Cell not found")
    return cell


# -----------------------------------------------------------------------------
# UE Measurements
# -----------------------------------------------------------------------------

@app.get("/rni/v2/ue/{ueId}/measurements",
         response_model=MeasRepUeNotification,
         tags=["UE Measurements"])
async def get_ue_measurements(ueId: str = Path(..., description="UE ID")):
    """Get UE radio measurements

    Integration point with RAN (gNodeB/CU).
    """
    meas = await rnis_service.get_ue_measurement(ueId)
    return meas


# -----------------------------------------------------------------------------
# Subscriptions
# -----------------------------------------------------------------------------

@app.post("/rni/v2/subscriptions",
          response_model=RnisSubscription,
          status_code=201,
          tags=["Subscriptions"])
async def create_subscription(subscription: RnisSubscription):
    """ETSI GS MEC 012 - Create RNIS Subscription

    Spec: Section 7.6
    """
    return rnis_service.create_subscription(subscription)


# -----------------------------------------------------------------------------
# Health and Metrics
# -----------------------------------------------------------------------------

@app.get("/health", tags=["Operations"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service_id": rnis_service.service_id,
        "cells": len(rnis_service.cells),
        "ue_measurements_cached": len(rnis_service.ue_measurements),
        "subscriptions": len(rnis_service.subscriptions)
    }


@app.get("/metrics", tags=["Operations"])
async def get_metrics():
    """Prometheus-compatible metrics"""
    metrics = []
    metrics.append(f'rnis_cells_total {len(rnis_service.cells)}')
    metrics.append(f'rnis_ue_measurements_cached {len(rnis_service.ue_measurements)}')
    metrics.append(f'rnis_subscriptions_total {len(rnis_service.subscriptions)}')

    # Per-cell metrics
    for cell_id, l2 in rnis_service.l2_measurements.items():
        if l2.dlTotalPrbUsage is not None:
            metrics.append(f'rnis_cell_dl_prb_usage{{cell="{cell_id}"}} {l2.dlTotalPrbUsage}')
        if l2.ulTotalPrbUsage is not None:
            metrics.append(f'rnis_cell_ul_prb_usage{{cell="{cell_id}"}} {l2.ulTotalPrbUsage}')

    return "\n".join(metrics)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("MEC_RNIS_PORT", 8092))
    logger.info(f"Starting MEC RNIS API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
