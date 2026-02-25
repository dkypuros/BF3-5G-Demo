# File location: 5G_Emulator_API/etsi/mec/location_api.py
# ETSI GS MEC 013 - Location Service API
# Provides UE location information to MEC applications

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
# ETSI GS MEC 013 Data Models - Location API
# =============================================================================

class LocationType(str, Enum):
    """ETSI GS MEC 013 - LocationInfo type"""
    CURRENT = "Current"
    LAST_KNOWN = "LastKnown"

class Shape(str, Enum):
    """ETSI GS MEC 013 - Geographic shape types"""
    ELLIPSOID_POINT = "ellipsoidPoint"
    ELLIPSOID_POINT_UNCERTAINTY_CIRCLE = "ellipsoidPointUncertaintyCircle"
    ELLIPSOID_POINT_UNCERTAINTY_ELLIPSE = "ellipsoidPointUncertaintyEllipse"
    POLYGON = "polygon"

class Coordinates(BaseModel):
    """ETSI GS MEC 013 - Geographic coordinates"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    altitude: Optional[float] = Field(None, description="Altitude in meters")

class LocationInfo(BaseModel):
    """ETSI GS MEC 013 - LocationInfo (Section 6.2.2)

    Represents the geographic location of a terminal/UE.
    """
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    altitude: Optional[float] = Field(None, description="Altitude in meters")
    accuracy: Optional[int] = Field(None, ge=0, description="Accuracy in meters")
    accuracyAltitude: Optional[int] = Field(None, description="Altitude accuracy")
    accuracySemiMajor: Optional[int] = Field(None, description="Semi-major axis accuracy")
    accuracySemiMinor: Optional[int] = Field(None, description="Semi-minor axis accuracy")
    orientationMajorAxis: Optional[int] = Field(None, description="Orientation of major axis")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of location")
    shape: Optional[Shape] = Field(Shape.ELLIPSOID_POINT_UNCERTAINTY_CIRCLE, description="Shape type")

class CivicAddress(BaseModel):
    """ETSI GS MEC 013 - CivicAddress (Section 6.2.4)"""
    country: Optional[str] = Field(None, description="Country code (ISO 3166)")
    a1: Optional[str] = Field(None, description="State/Province")
    a2: Optional[str] = Field(None, description="County")
    a3: Optional[str] = Field(None, description="City")
    a4: Optional[str] = Field(None, description="District")
    a6: Optional[str] = Field(None, description="Street")
    prd: Optional[str] = Field(None, description="Street prefix")
    pod: Optional[str] = Field(None, description="Street suffix")
    sts: Optional[str] = Field(None, description="Street name")
    hno: Optional[str] = Field(None, description="House number")
    pc: Optional[str] = Field(None, description="Postal code")

class TerminalLocation(BaseModel):
    """ETSI GS MEC 013 - TerminalLocation (Section 6.2.6)

    Location information for a specific terminal/UE.
    """
    address: str = Field(..., description="Terminal address (MSISDN or acr:)")
    locationInfo: Optional[LocationInfo] = Field(None, description="Geographic location")
    civicInfo: Optional[CivicAddress] = Field(None, description="Civic address")
    timestamp: Optional[datetime] = Field(None, description="Timestamp")
    currentAccessPointId: Optional[str] = Field(None, description="Current cell/AP ID")
    previousAccessPointId: Optional[str] = Field(None, description="Previous cell/AP ID")
    enteringLeavingCriteria: Optional[str] = Field(None, description="Enter/Leave zone")
    resourceURL: Optional[str] = Field(None, description="Self URL")

class AccessPointInfo(BaseModel):
    """ETSI GS MEC 013 - AccessPointInfo (Section 6.4.2)

    Information about an access point (cell site).
    """
    accessPointId: str = Field(..., description="Access point identifier (Cell ID)")
    locationInfo: Optional[LocationInfo] = Field(None, description="AP location")
    connectionType: str = Field("5G_NR", description="Connection type")
    operatorId: Optional[str] = Field(None, description="Operator ID")
    timezone: Optional[str] = Field(None, description="Timezone")
    numberOfUsers: Optional[int] = Field(None, ge=0, description="Connected users")
    interestRealm: Optional[str] = Field(None, description="Interest realm")
    resourceURL: Optional[str] = Field(None, description="Self URL")

class ZoneInfo(BaseModel):
    """ETSI GS MEC 013 - ZoneInfo (Section 6.3.2)

    Represents a geographic zone for location tracking.
    """
    zoneId: str = Field(..., description="Zone identifier")
    numberOfAccessPoints: int = Field(0, ge=0, description="Number of APs in zone")
    numberOfUnserviceableAccessPoints: int = Field(0, ge=0, description="Unavailable APs")
    numberOfUsers: int = Field(0, ge=0, description="Users in zone")
    resourceURL: Optional[str] = Field(None, description="Self URL")

class UserTrackingSubscription(BaseModel):
    """ETSI GS MEC 013 - UserTrackingSubscription (Section 6.5.2)"""
    clientCorrelator: Optional[str] = Field(None, description="Client correlator")
    callbackReference: str = Field(..., description="Callback URL for notifications")
    address: str = Field(..., description="User address to track")
    userEventCriteria: Optional[List[str]] = Field(None, description="Event criteria")
    resourceURL: Optional[str] = Field(None, description="Self URL")

class ZonalTrafficSubscription(BaseModel):
    """ETSI GS MEC 013 - ZonalTrafficSubscription (Section 6.5.3)"""
    clientCorrelator: Optional[str] = Field(None, description="Client correlator")
    callbackReference: str = Field(..., description="Callback URL")
    zoneId: str = Field(..., description="Zone to monitor")
    interestRealm: Optional[List[str]] = Field(None, description="Interest realms")
    userEventCriteria: Optional[List[str]] = Field(None, description="Event criteria")
    resourceURL: Optional[str] = Field(None, description="Self URL")

# =============================================================================
# Location Service Core Class
# =============================================================================

class LocationService:
    """ETSI GS MEC 013 - Location Service

    Provides location information for UEs to MEC applications.
    Integrates with 5G Core AMF for UE context and location data.
    """

    def __init__(self, amf_url: str = None):
        """Initialize Location Service

        Args:
            amf_url: URL of the AMF for UE location queries
        """
        self.service_id = str(uuid.uuid4())
        self.amf_url = amf_url or os.environ.get("AMF_URL", "http://127.0.0.1:8001")

        # Zone registry
        self.zones: Dict[str, ZoneInfo] = {}

        # Access point registry (cells)
        self.access_points: Dict[str, AccessPointInfo] = {}

        # Cached UE locations (from AMF)
        self.ue_locations: Dict[str, TerminalLocation] = {}

        # Subscriptions
        self.user_subscriptions: Dict[str, UserTrackingSubscription] = {}
        self.zonal_subscriptions: Dict[str, ZonalTrafficSubscription] = {}

        # Initialize default zone
        self._init_default_zone()

        logger.info(f"Location Service initialized: {self.service_id}")

    def _init_default_zone(self):
        """Initialize default zone with simulated cells"""
        # Create default zone
        zone = ZoneInfo(
            zoneId="zone001",
            numberOfAccessPoints=3,
            numberOfUnserviceableAccessPoints=0,
            numberOfUsers=0,
            resourceURL="/location/v2/zones/zone001"
        )
        self.zones["zone001"] = zone

        # Create simulated access points (cells)
        cells = [
            {"id": "cell001", "lat": 37.7749, "lon": -122.4194, "name": "San Francisco"},
            {"id": "cell002", "lat": 37.7850, "lon": -122.4094, "name": "SF Downtown"},
            {"id": "cell003", "lat": 37.7650, "lon": -122.4294, "name": "SF South"},
        ]

        for cell in cells:
            ap = AccessPointInfo(
                accessPointId=cell["id"],
                locationInfo=LocationInfo(
                    latitude=cell["lat"],
                    longitude=cell["lon"],
                    accuracy=50,
                    timestamp=datetime.now(timezone.utc)
                ),
                connectionType="5G_NR",
                operatorId="310260",
                numberOfUsers=0,
                resourceURL=f"/location/v2/accessPoints/{cell['id']}"
            )
            self.access_points[cell["id"]] = ap

    async def get_ue_location_from_amf(self, ue_address: str) -> Optional[TerminalLocation]:
        """Query AMF for UE location

        Integration point with 5G Core AMF (amf_nas.py).
        """
        try:
            async with httpx.AsyncClient() as client:
                # Query AMF for UE context
                response = await client.get(
                    f"{self.amf_url}/namf-comm/v1/ue-contexts/{ue_address}/location"
                )
                if response.status_code == 200:
                    data = response.json()
                    # Convert AMF response to MEC Location format
                    location = TerminalLocation(
                        address=ue_address,
                        locationInfo=LocationInfo(
                            latitude=data.get("latitude", 37.7749),
                            longitude=data.get("longitude", -122.4194),
                            accuracy=data.get("accuracy", 100),
                            timestamp=datetime.now(timezone.utc)
                        ),
                        currentAccessPointId=data.get("cellId", "cell001"),
                        timestamp=datetime.now(timezone.utc)
                    )
                    self.ue_locations[ue_address] = location
                    return location
        except Exception as e:
            logger.warning(f"Could not query AMF for location: {e}")

        # Return cached or simulated location
        if ue_address in self.ue_locations:
            return self.ue_locations[ue_address]

        # Simulate location for demo purposes
        return self._simulate_ue_location(ue_address)

    def _simulate_ue_location(self, ue_address: str) -> TerminalLocation:
        """Simulate UE location for demo/testing"""
        import random

        # Pick a random cell
        cell_id = random.choice(list(self.access_points.keys()))
        cell = self.access_points[cell_id]

        # Add some random offset to cell location
        location = TerminalLocation(
            address=ue_address,
            locationInfo=LocationInfo(
                latitude=cell.locationInfo.latitude + random.uniform(-0.01, 0.01),
                longitude=cell.locationInfo.longitude + random.uniform(-0.01, 0.01),
                accuracy=random.randint(10, 100),
                timestamp=datetime.now(timezone.utc)
            ),
            currentAccessPointId=cell_id,
            timestamp=datetime.now(timezone.utc),
            resourceURL=f"/location/v2/users/{ue_address}"
        )

        self.ue_locations[ue_address] = location
        return location

    def get_zone_info(self, zone_id: str) -> Optional[ZoneInfo]:
        """Get information about a zone"""
        return self.zones.get(zone_id)

    def get_zones(self) -> List[ZoneInfo]:
        """Get all zones"""
        return list(self.zones.values())

    def get_access_points(self, zone_id: str = None) -> List[AccessPointInfo]:
        """Get access points, optionally filtered by zone"""
        return list(self.access_points.values())

    def get_access_point(self, ap_id: str) -> Optional[AccessPointInfo]:
        """Get specific access point"""
        return self.access_points.get(ap_id)

    def create_user_tracking_subscription(self, sub: UserTrackingSubscription) -> UserTrackingSubscription:
        """Create subscription for user location tracking"""
        sub_id = str(uuid.uuid4())
        sub.resourceURL = f"/location/v2/subscriptions/userTracking/{sub_id}"
        self.user_subscriptions[sub_id] = sub
        logger.info(f"Created user tracking subscription: {sub_id} for {sub.address}")
        return sub

    def create_zonal_traffic_subscription(self, sub: ZonalTrafficSubscription) -> ZonalTrafficSubscription:
        """Create subscription for zonal traffic monitoring"""
        sub_id = str(uuid.uuid4())
        sub.resourceURL = f"/location/v2/subscriptions/zonalTraffic/{sub_id}"
        self.zonal_subscriptions[sub_id] = sub
        logger.info(f"Created zonal traffic subscription: {sub_id} for zone {sub.zoneId}")
        return sub


# =============================================================================
# FastAPI Application - Location API
# =============================================================================

app = FastAPI(
    title="ETSI MEC Location API",
    description="Location Service implementing ETSI GS MEC 013",
    version="3.1.1",
    docs_url="/location/docs",
    openapi_url="/location/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Location Service instance
location_service = LocationService()


# -----------------------------------------------------------------------------
# ETSI GS MEC 013 - User Location Query
# -----------------------------------------------------------------------------

@app.get("/location/v2/queries/users",
         response_model=Dict[str, Any],
         tags=["Location Queries"])
async def get_users_location(
    address: List[str] = Query(..., description="User addresses (MSISDN or acr:)")
):
    """ETSI GS MEC 013 - Query User Location

    Spec: Section 7.3.2 - GET /queries/users
    Returns location information for specified users.
    """
    with tracer.start_as_current_span("mec_location_query") as span:
        span.set_attribute("etsi.spec", "GS MEC 013")
        span.set_attribute("etsi.operation", "QueryUserLocation")

        user_list = []
        for addr in address:
            location = await location_service.get_ue_location_from_amf(addr)
            if location:
                user_list.append(location.dict())

        span.set_attribute("users.count", len(user_list))

        return {
            "userList": {
                "user": user_list
            }
        }


@app.get("/location/v2/users/{userId}",
         response_model=TerminalLocation,
         tags=["Location Queries"])
async def get_user_location(
    userId: str = Path(..., description="User ID (MSISDN or acr:)")
):
    """ETSI GS MEC 013 - Get Specific User Location

    Spec: Section 7.4.2
    """
    location = await location_service.get_ue_location_from_amf(userId)
    if not location:
        raise HTTPException(status_code=404, detail="User not found")
    return location


# -----------------------------------------------------------------------------
# ETSI GS MEC 013 - Zone Information
# -----------------------------------------------------------------------------

@app.get("/location/v2/zones",
         response_model=Dict[str, Any],
         tags=["Zone Information"])
async def get_zones():
    """ETSI GS MEC 013 - Get All Zones

    Spec: Section 7.5.2 - GET /zones
    """
    zones = location_service.get_zones()
    return {
        "zoneList": {
            "zone": [z.dict() for z in zones]
        }
    }


@app.get("/location/v2/zones/{zoneId}",
         response_model=ZoneInfo,
         tags=["Zone Information"])
async def get_zone(
    zoneId: str = Path(..., description="Zone ID")
):
    """ETSI GS MEC 013 - Get Zone Information

    Spec: Section 7.5.3
    """
    zone = location_service.get_zone_info(zoneId)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return zone


# -----------------------------------------------------------------------------
# ETSI GS MEC 013 - Access Point Information
# -----------------------------------------------------------------------------

@app.get("/location/v2/zones/{zoneId}/accessPoints",
         response_model=Dict[str, Any],
         tags=["Access Point Information"])
async def get_zone_access_points(
    zoneId: str = Path(..., description="Zone ID")
):
    """ETSI GS MEC 013 - Get Access Points in Zone

    Spec: Section 7.6.2
    """
    aps = location_service.get_access_points(zoneId)
    return {
        "accessPointList": {
            "zoneId": zoneId,
            "accessPoint": [ap.dict() for ap in aps]
        }
    }


@app.get("/location/v2/zones/{zoneId}/accessPoints/{accessPointId}",
         response_model=AccessPointInfo,
         tags=["Access Point Information"])
async def get_access_point(
    zoneId: str = Path(..., description="Zone ID"),
    accessPointId: str = Path(..., description="Access Point ID")
):
    """ETSI GS MEC 013 - Get Access Point Information

    Spec: Section 7.6.3
    """
    ap = location_service.get_access_point(accessPointId)
    if not ap:
        raise HTTPException(status_code=404, detail="Access point not found")
    return ap


# -----------------------------------------------------------------------------
# ETSI GS MEC 013 - Subscriptions
# -----------------------------------------------------------------------------

@app.post("/location/v2/subscriptions/userTracking",
          response_model=UserTrackingSubscription,
          status_code=201,
          tags=["Subscriptions"])
async def create_user_tracking_subscription(
    subscription: UserTrackingSubscription
):
    """ETSI GS MEC 013 - Create User Tracking Subscription

    Spec: Section 7.7.2
    Subscribe to location updates for a specific user.
    """
    return location_service.create_user_tracking_subscription(subscription)


@app.post("/location/v2/subscriptions/zonalTraffic",
          response_model=ZonalTrafficSubscription,
          status_code=201,
          tags=["Subscriptions"])
async def create_zonal_traffic_subscription(
    subscription: ZonalTrafficSubscription
):
    """ETSI GS MEC 013 - Create Zonal Traffic Subscription

    Spec: Section 7.8.2
    Subscribe to traffic updates for a zone.
    """
    return location_service.create_zonal_traffic_subscription(subscription)


# -----------------------------------------------------------------------------
# Health and Metrics
# -----------------------------------------------------------------------------

@app.get("/health", tags=["Operations"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service_id": location_service.service_id,
        "zones": len(location_service.zones),
        "access_points": len(location_service.access_points),
        "cached_locations": len(location_service.ue_locations),
        "subscriptions": {
            "userTracking": len(location_service.user_subscriptions),
            "zonalTraffic": len(location_service.zonal_subscriptions)
        }
    }


@app.get("/metrics", tags=["Operations"])
async def get_metrics():
    """Prometheus-compatible metrics"""
    metrics = []
    metrics.append(f'mec_location_zones_total {len(location_service.zones)}')
    metrics.append(f'mec_location_access_points_total {len(location_service.access_points)}')
    metrics.append(f'mec_location_cached_ues_total {len(location_service.ue_locations)}')
    metrics.append(f'mec_location_user_subscriptions_total {len(location_service.user_subscriptions)}')
    metrics.append(f'mec_location_zonal_subscriptions_total {len(location_service.zonal_subscriptions)}')
    return "\n".join(metrics)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("MEC_LOCATION_PORT", 8091))
    logger.info(f"Starting MEC Location API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
