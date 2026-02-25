# File location: 5G_Emulator_API/etsi/zsm/closed_loop.py
# ETSI GS ZSM 002 - Zero-touch Service Management Closed Loop
# Implements automated observe-analyze-act loop for network management

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Callable
import uvicorn
import uuid
import logging
from datetime import datetime, timezone
import os
import asyncio
import httpx
from opentelemetry import trace
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

# =============================================================================
# ETSI GS ZSM 002 Data Models
# =============================================================================

class ClosedLoopState(str, Enum):
    """ETSI GS ZSM 002 - Closed Loop State"""
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"

class ClosedLoopPhase(str, Enum):
    """ETSI GS ZSM 002 - Closed Loop Phases"""
    OBSERVE = "OBSERVE"
    ORIENT = "ORIENT"
    DECIDE = "DECIDE"
    ACT = "ACT"

class IntentType(str, Enum):
    """ZSM Intent Types"""
    ASSURANCE = "ASSURANCE"
    OPTIMIZATION = "OPTIMIZATION"
    HEALING = "HEALING"
    SCALING = "SCALING"

class Intent(BaseModel):
    """ETSI GS ZSM 002 - Intent Object

    Represents a high-level goal for the closed loop to achieve.
    """
    intentId: str = Field(..., description="Intent identifier")
    intentType: IntentType = Field(..., description="Type of intent")
    target: str = Field(..., description="Target NF or service")
    objective: str = Field(..., description="Objective description")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Constraints")
    priority: int = Field(5, ge=1, le=10, description="Priority (1=highest)")
    state: str = Field("ACTIVE", description="Intent state")

class Observation(BaseModel):
    """ETSI GS ZSM 002 - Observation from monitoring"""
    observationId: str = Field(..., description="Observation ID")
    source: str = Field(..., description="Data source (e.g., Prometheus)")
    metric: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    unit: Optional[str] = Field(None, description="Unit")
    timestamp: datetime = Field(..., description="Observation time")
    tags: Optional[Dict[str, str]] = Field(None, description="Tags/labels")

class Analysis(BaseModel):
    """ETSI GS ZSM 002 - Analysis result"""
    analysisId: str = Field(..., description="Analysis ID")
    intentId: str = Field(..., description="Related intent")
    observations: List[str] = Field(..., description="Observation IDs used")
    assessment: str = Field(..., description="Assessment result")
    deviation: Optional[float] = Field(None, description="Deviation from target")
    recommendation: Optional[str] = Field(None, description="Recommended action")
    confidence: float = Field(1.0, ge=0, le=1, description="Confidence level")
    timestamp: datetime = Field(..., description="Analysis time")

class Action(BaseModel):
    """ETSI GS ZSM 002 - Action to execute"""
    actionId: str = Field(..., description="Action ID")
    analysisId: str = Field(..., description="Related analysis")
    actionType: str = Field(..., description="Action type (SCALE, HEAL, CONFIGURE)")
    target: str = Field(..., description="Target NF/resource")
    parameters: Dict[str, Any] = Field(..., description="Action parameters")
    state: str = Field("PENDING", description="Action state")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result")
    timestamp: datetime = Field(..., description="Action time")

class ClosedLoop(BaseModel):
    """ETSI GS ZSM 002 - Closed Loop Definition"""
    loopId: str = Field(..., description="Closed loop ID")
    name: str = Field(..., description="Loop name")
    description: Optional[str] = Field(None, description="Description")
    state: ClosedLoopState = Field(ClosedLoopState.INACTIVE, description="Loop state")
    intents: List[str] = Field(default_factory=list, description="Associated intent IDs")
    observeInterval: int = Field(30, description="Observe interval in seconds")
    analyzeInterval: int = Field(60, description="Analyze interval in seconds")
    createdAt: datetime = Field(..., description="Creation time")
    lastExecution: Optional[datetime] = Field(None, description="Last execution time")

# =============================================================================
# ZSM Closed Loop Engine
# =============================================================================

class ZSMEngine:
    """ETSI GS ZSM 002 - Zero-touch Service Management Engine

    Implements the closed-loop automation for network management.
    Integrates with OpenTelemetry for observability data.
    """

    def __init__(self, prometheus_url: str = None, vnfm_url: str = None):
        """Initialize ZSM Engine

        Args:
            prometheus_url: URL for Prometheus metrics
            vnfm_url: URL for VNFM (lifecycle operations)
        """
        self.engine_id = str(uuid.uuid4())
        self.prometheus_url = prometheus_url or os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:9090")
        self.vnfm_url = vnfm_url or os.environ.get("VNFM_URL", "http://127.0.0.1:8093")

        # Closed loops registry
        self.closed_loops: Dict[str, ClosedLoop] = {}

        # Intents registry
        self.intents: Dict[str, Intent] = {}

        # Observations cache
        self.observations: Dict[str, Observation] = {}

        # Analysis history
        self.analyses: Dict[str, Analysis] = {}

        # Actions history
        self.actions: Dict[str, Action] = {}

        # Background tasks
        self._running_loops: Dict[str, asyncio.Task] = {}

        logger.info(f"ZSM Engine initialized: {self.engine_id}")

    def create_intent(self, intent: Intent) -> Intent:
        """Create a new intent"""
        if not intent.intentId:
            intent.intentId = str(uuid.uuid4())
        self.intents[intent.intentId] = intent
        logger.info(f"Created intent: {intent.intentId} - {intent.objective}")
        return intent

    def create_closed_loop(self, loop: ClosedLoop) -> ClosedLoop:
        """Create a new closed loop"""
        if not loop.loopId:
            loop.loopId = str(uuid.uuid4())
        loop.createdAt = datetime.now(timezone.utc)
        self.closed_loops[loop.loopId] = loop
        logger.info(f"Created closed loop: {loop.loopId} - {loop.name}")
        return loop

    async def start_closed_loop(self, loop_id: str) -> bool:
        """Start a closed loop execution"""
        loop = self.closed_loops.get(loop_id)
        if not loop:
            return False

        if loop_id in self._running_loops:
            return True  # Already running

        loop.state = ClosedLoopState.ACTIVE
        task = asyncio.create_task(self._run_closed_loop(loop_id))
        self._running_loops[loop_id] = task
        logger.info(f"Started closed loop: {loop_id}")
        return True

    async def stop_closed_loop(self, loop_id: str) -> bool:
        """Stop a closed loop"""
        loop = self.closed_loops.get(loop_id)
        if not loop:
            return False

        if loop_id in self._running_loops:
            self._running_loops[loop_id].cancel()
            del self._running_loops[loop_id]

        loop.state = ClosedLoopState.INACTIVE
        logger.info(f"Stopped closed loop: {loop_id}")
        return True

    async def _run_closed_loop(self, loop_id: str):
        """Main closed loop execution"""
        loop = self.closed_loops[loop_id]

        while loop.state == ClosedLoopState.ACTIVE:
            try:
                with tracer.start_as_current_span(f"zsm_closed_loop_{loop_id}") as span:
                    span.set_attribute("etsi.spec", "GS ZSM 002")
                    span.set_attribute("loop.id", loop_id)

                    # OBSERVE phase
                    observations = await self._observe(loop)
                    span.set_attribute("observations.count", len(observations))

                    # ORIENT/ANALYZE phase
                    for intent_id in loop.intents:
                        intent = self.intents.get(intent_id)
                        if not intent:
                            continue

                        analysis = await self._analyze(intent, observations)
                        if analysis:
                            span.set_attribute(f"analysis.{intent_id}", analysis.assessment)

                            # DECIDE phase
                            if analysis.recommendation:
                                # ACT phase
                                action = await self._act(analysis, intent)
                                if action:
                                    span.set_attribute(f"action.{intent_id}", action.actionType)

                    loop.lastExecution = datetime.now(timezone.utc)

                await asyncio.sleep(loop.observeInterval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Closed loop error: {e}")
                await asyncio.sleep(5)

    async def _observe(self, loop: ClosedLoop) -> List[Observation]:
        """OBSERVE phase - Collect metrics from Prometheus"""
        observations = []

        try:
            async with httpx.AsyncClient() as client:
                # Query key metrics
                metrics = [
                    "amf_registered_ues_total",
                    "smf_active_sessions_total",
                    "upf_throughput_bytes",
                    "nf_response_time_seconds",
                ]

                for metric in metrics:
                    try:
                        response = await client.get(
                            f"{self.prometheus_url}/api/v1/query",
                            params={"query": metric}
                        )
                        if response.status_code == 200:
                            data = response.json()
                            for result in data.get("data", {}).get("result", []):
                                obs = Observation(
                                    observationId=str(uuid.uuid4()),
                                    source="prometheus",
                                    metric=metric,
                                    value=float(result.get("value", [0, 0])[1]),
                                    timestamp=datetime.now(timezone.utc),
                                    tags=result.get("metric", {})
                                )
                                observations.append(obs)
                                self.observations[obs.observationId] = obs
                    except Exception:
                        pass

        except Exception as e:
            logger.warning(f"Observe phase error: {e}")
            # Generate simulated observations for demo
            for metric in ["latency_ms", "throughput_mbps", "cpu_percent"]:
                import random
                obs = Observation(
                    observationId=str(uuid.uuid4()),
                    source="simulation",
                    metric=metric,
                    value=random.uniform(10, 100),
                    timestamp=datetime.now(timezone.utc)
                )
                observations.append(obs)
                self.observations[obs.observationId] = obs

        return observations

    async def _analyze(self, intent: Intent, observations: List[Observation]) -> Optional[Analysis]:
        """ORIENT/ANALYZE phase - Analyze observations against intent"""

        # Simple threshold-based analysis
        relevant_obs = [o for o in observations if intent.target.lower() in o.metric.lower() or intent.target.lower() in str(o.tags).lower()]

        if not relevant_obs:
            return None

        # Calculate deviation from constraints
        deviation = 0.0
        assessment = "NORMAL"
        recommendation = None

        for obs in relevant_obs:
            if intent.constraints:
                threshold = intent.constraints.get("max_value")
                if threshold and obs.value > threshold:
                    deviation = (obs.value - threshold) / threshold * 100
                    assessment = "THRESHOLD_EXCEEDED"
                    if intent.intentType == IntentType.SCALING:
                        recommendation = "SCALE_OUT"
                    elif intent.intentType == IntentType.HEALING:
                        recommendation = "RESTART"

        analysis = Analysis(
            analysisId=str(uuid.uuid4()),
            intentId=intent.intentId,
            observations=[o.observationId for o in relevant_obs],
            assessment=assessment,
            deviation=deviation,
            recommendation=recommendation,
            confidence=0.85,
            timestamp=datetime.now(timezone.utc)
        )

        self.analyses[analysis.analysisId] = analysis
        return analysis

    async def _act(self, analysis: Analysis, intent: Intent) -> Optional[Action]:
        """ACT phase - Execute remediation action"""

        if not analysis.recommendation:
            return None

        action = Action(
            actionId=str(uuid.uuid4()),
            analysisId=analysis.analysisId,
            actionType=analysis.recommendation,
            target=intent.target,
            parameters={"intent": intent.intentId, "deviation": analysis.deviation},
            state="EXECUTING",
            timestamp=datetime.now(timezone.utc)
        )
        self.actions[action.actionId] = action

        try:
            if analysis.recommendation == "SCALE_OUT":
                # Call VNFM to scale
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{self.vnfm_url}/vnflcm/v1/vnf_instances/{intent.target}/scale",
                        json={"type": "SCALE_OUT", "aspectId": "default", "numberOfSteps": 1}
                    )
                action.state = "COMPLETED"
                action.result = {"status": "scaled"}

            elif analysis.recommendation == "RESTART":
                # Restart action
                action.state = "COMPLETED"
                action.result = {"status": "restarted"}

            else:
                action.state = "COMPLETED"
                action.result = {"status": "unknown_action"}

            logger.info(f"Executed action: {action.actionType} on {action.target}")

        except Exception as e:
            action.state = "FAILED"
            action.result = {"error": str(e)}
            logger.error(f"Action failed: {e}")

        return action


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ETSI ZSM Closed Loop API",
    description="Zero-touch Service Management implementing ETSI GS ZSM 002",
    version="1.0.0",
    docs_url="/zsm/docs",
    openapi_url="/zsm/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

zsm_engine = ZSMEngine()


# -----------------------------------------------------------------------------
# Intent Management
# -----------------------------------------------------------------------------

@app.post("/zsm/v1/intents",
          response_model=Intent,
          status_code=201,
          tags=["Intents"])
async def create_intent(intent: Intent):
    """Create a new intent for closed-loop automation"""
    return zsm_engine.create_intent(intent)


@app.get("/zsm/v1/intents",
         response_model=List[Intent],
         tags=["Intents"])
async def get_intents():
    """Get all intents"""
    return list(zsm_engine.intents.values())


@app.get("/zsm/v1/intents/{intentId}",
         response_model=Intent,
         tags=["Intents"])
async def get_intent(intentId: str = Path(..., description="Intent ID")):
    """Get specific intent"""
    intent = zsm_engine.intents.get(intentId)
    if not intent:
        raise HTTPException(status_code=404, detail="Intent not found")
    return intent


# -----------------------------------------------------------------------------
# Closed Loop Management
# -----------------------------------------------------------------------------

@app.post("/zsm/v1/closed-loops",
          response_model=ClosedLoop,
          status_code=201,
          tags=["Closed Loops"])
async def create_closed_loop(loop: ClosedLoop):
    """Create a new closed loop"""
    return zsm_engine.create_closed_loop(loop)


@app.get("/zsm/v1/closed-loops",
         response_model=List[ClosedLoop],
         tags=["Closed Loops"])
async def get_closed_loops():
    """Get all closed loops"""
    return list(zsm_engine.closed_loops.values())


@app.post("/zsm/v1/closed-loops/{loopId}/start",
          status_code=200,
          tags=["Closed Loops"])
async def start_closed_loop(loopId: str = Path(..., description="Loop ID")):
    """Start a closed loop"""
    if not await zsm_engine.start_closed_loop(loopId):
        raise HTTPException(status_code=404, detail="Closed loop not found")
    return {"status": "started", "loopId": loopId}


@app.post("/zsm/v1/closed-loops/{loopId}/stop",
          status_code=200,
          tags=["Closed Loops"])
async def stop_closed_loop(loopId: str = Path(..., description="Loop ID")):
    """Stop a closed loop"""
    if not await zsm_engine.stop_closed_loop(loopId):
        raise HTTPException(status_code=404, detail="Closed loop not found")
    return {"status": "stopped", "loopId": loopId}


# -----------------------------------------------------------------------------
# Observations and Analysis
# -----------------------------------------------------------------------------

@app.get("/zsm/v1/observations",
         response_model=List[Observation],
         tags=["Observations"])
async def get_observations(limit: int = Query(100, le=1000)):
    """Get recent observations"""
    obs = list(zsm_engine.observations.values())
    return sorted(obs, key=lambda x: x.timestamp, reverse=True)[:limit]


@app.get("/zsm/v1/analyses",
         response_model=List[Analysis],
         tags=["Analysis"])
async def get_analyses(limit: int = Query(100, le=1000)):
    """Get recent analyses"""
    analyses = list(zsm_engine.analyses.values())
    return sorted(analyses, key=lambda x: x.timestamp, reverse=True)[:limit]


@app.get("/zsm/v1/actions",
         response_model=List[Action],
         tags=["Actions"])
async def get_actions(limit: int = Query(100, le=1000)):
    """Get recent actions"""
    actions = list(zsm_engine.actions.values())
    return sorted(actions, key=lambda x: x.timestamp, reverse=True)[:limit]


# -----------------------------------------------------------------------------
# Health and Metrics
# -----------------------------------------------------------------------------

@app.get("/health", tags=["Operations"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine_id": zsm_engine.engine_id,
        "intents": len(zsm_engine.intents),
        "closed_loops": len(zsm_engine.closed_loops),
        "active_loops": len(zsm_engine._running_loops),
        "observations": len(zsm_engine.observations),
        "analyses": len(zsm_engine.analyses),
        "actions": len(zsm_engine.actions)
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("ZSM_PORT", 8094))
    logger.info(f"Starting ZSM Engine on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
