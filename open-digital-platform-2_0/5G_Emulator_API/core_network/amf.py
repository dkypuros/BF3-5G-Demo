import logging
import os
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
import uvicorn
import requests
from contextlib import asynccontextmanager
import time
from typing import Dict
import json

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import start_http_server

# Create logs directory if it doesn't exist
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{logs_dir}/amf_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom JSON File Exporter
class JsonFileExporter(SpanExporter):
    def __init__(self, file_path):
        self.file_path = file_path

    def export(self, spans):
        with open(self.file_path, 'a') as f:
            for span in spans:
                json.dump(self._span_to_dict(span), f)
                f.write('\n')
        return None

    def shutdown(self):
        pass

    def _span_to_dict(self, span: Span):
        return {
            "name": span.name,
            "context": {
                "trace_id": hex(span.context.trace_id),
                "span_id": hex(span.context.span_id),
            },
            "kind": span.kind.name,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "status": {
                "status_code": span.status.status_code.name,
            },
            "attributes": dict(span.attributes),
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": dict(event.attributes),
                }
                for event in span.events
            ],
        }

# Set up tracing
resource = Resource.create({"service.name": "amf-service"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# File Exporter (Saves traces to a file)
trace_filename = f"{logs_dir}/trace_output_{timestamp}.json"
file_exporter = JsonFileExporter(trace_filename)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(file_exporter))

# Set up metrics
metric_reader = PrometheusMetricReader()
console_metric_exporter = ConsoleMetricExporter()
periodic_metric_reader = PeriodicExportingMetricReader(console_metric_exporter, export_interval_millis=5000)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader, periodic_metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)

# Create metrics
handover_request_counter = meter.create_counter("ngap_handover_requests_total", description="Total number of NGAP Handover Requests")
handover_duration_histogram = meter.create_histogram("ngap_handover_duration_seconds", description="Duration of NGAP Handover process")

nrf_url = "http://127.0.0.1:8000"
smf_url = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global smf_url
    nf_registration = {
        "nf_type": "AMF",
        "ip": "0.0.0.0",
        "port": 9000
    }
    try:
        response = requests.post(f"{nrf_url}/register", json=nf_registration)
        response.raise_for_status()

        smf_info = requests.get(f"{nrf_url}/discover/SMF").json()
        if 'message' in smf_info:
            logger.error(f"SMF discovery failed: {smf_info['message']}")
        else:
            smf_url = f"http://{smf_info.get('ip')}:{smf_info.get('port')}"
            logger.info(f"SMF discovered at {smf_url}")
    except requests.RequestException as e:
        logger.error(f"Failed to register with NRF or discover SMF: {str(e)}")

    start_http_server(9100)
    yield

app = FastAPI(lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)

# Simulated UE and gNB data
ue_contexts: Dict[str, Dict] = {}
gnb_data: Dict[str, Dict] = {
    "gnb001": {"ip": "192.168.1.1", "capacity": 100},
    "gnb002": {"ip": "192.168.1.2", "capacity": 100},
}

def handle_ngap_handover_request(request_data: Dict):
    with tracer.start_as_current_span("ngap_handover_request") as span:
        span.set_attribute("ngap.message_type", "NGAP_HANDOVER_REQUEST")
        span.set_attribute("ngap.source_gnb_id", request_data['source_gnb_id'])
        
        logger.info(f"Handling NGAP Handover Request for UE: {request_data['ue_id']} from source gNB: {request_data['source_gnb_id']}")
        
        # Simulate processing of handover request
        ue_id = request_data['ue_id']
        if ue_id not in ue_contexts:
            logger.error(f"UE context for {ue_id} not found!")
            raise HTTPException(status_code=404, detail="UE context not found")
        
        # Decide on target gNB (simplified logic)
        target_gnb_id = "gnb002" if request_data['source_gnb_id'] == "gnb001" else "gnb001"
        logger.info(f"Decided target gNB for UE {ue_id}: {target_gnb_id}")
        
        # Update UE context
        ue_contexts[ue_id]['target_gnb_id'] = target_gnb_id
        
        # Simulate sending NGAP Handover Request Acknowledge
        send_ngap_handover_request_ack(request_data['source_gnb_id'])
        
        return target_gnb_id

def send_ngap_handover_request_ack(source_gnb_id: str):
    with tracer.start_as_current_span("ngap_handover_request_ack") as span:
        span.set_attribute("ngap.message_type", "NGAP_HANDOVER_REQUEST_ACK")
        span.set_attribute("ngap.source_gnb_id", source_gnb_id)
        logger.info(f"Sending NGAP Handover Request Acknowledge to source gNB: {source_gnb_id}")
        # Simulate sending acknowledgment
        time.sleep(0.01)  # Simulate network delay

def initiate_ngap_resource_setup(target_gnb_id: str):
    with tracer.start_as_current_span("ngap_resource_setup") as span:
        span.set_attribute("ngap.message_type", "NGAP_RESOURCE_SETUP_REQUEST")
        span.set_attribute("ngap.target_gnb_id", target_gnb_id)
        
        logger.info(f"Initiating resource setup for target gNB: {target_gnb_id}")
        
        # Simulate sending resource setup request to target gNB
        time.sleep(0.02)  # Simulate network delay
        logger.info(f"Resource setup request sent to {target_gnb_id}")
        
        # Simulate receiving resource setup response
        time.sleep(0.02)  # Simulate network delay
        logger.info(f"Resource setup response received from {target_gnb_id}")
        
        span.add_event("Resource setup completed")

def send_ngap_handover_command(source_gnb_id: str, target_gnb_id: str):
    with tracer.start_as_current_span("ngap_handover_command") as span:
        span.set_attribute("ngap.message_type", "NGAP_HANDOVER_COMMAND")
        span.set_attribute("ngap.source_gnb_id", source_gnb_id)
        span.set_attribute("ngap.target_gnb_id", target_gnb_id)
        
        logger.info(f"Sending NGAP Handover Command from source gNB: {source_gnb_id} to target gNB: {target_gnb_id}")
        
        # Simulate sending handover command to source gNB
        time.sleep(0.01)  # Simulate network delay
        
        # Wait for handover complete message
        time.sleep(0.03)  # Simulate handover execution time
        span.add_event("Handover completed")
        logger.info(f"Handover completed for UE at target gNB: {target_gnb_id}")

@app.post("/amf/handover")
async def amf_handover(request_data: Dict):
    start_time = time.time()
    handover_request_counter.add(1)
    
    try:
        # Simulate receiving NGAP Handover Request
        target_gnb_id = handle_ngap_handover_request(request_data)
        
        # Simulate NGAP Resource Setup
        initiate_ngap_resource_setup(target_gnb_id)
        
        # Simulate sending NGAP Handover Command
        send_ngap_handover_command(request_data['source_gnb_id'], target_gnb_id)
        
        # Record handover duration
        duration = time.time() - start_time
        handover_duration_histogram.record(duration)
        
        logger.info(f"Handover process completed in {duration} seconds")
        return {"message": "Handover process completed", "duration": duration}
    except Exception as e:
        logger.error(f"Handover failed: {str(e)}")
        return {"message": f"Handover failed: {str(e)}"}

@app.get("/amf/ue/{ue_id}")
async def get_ue_context(ue_id: str):
    if ue_id not in ue_contexts:
        raise HTTPException(status_code=404, detail="UE context not found")
    return ue_contexts[ue_id]

@app.post("/amf/ue/{ue_id}")
async def create_ue_context(ue_id: str, context: Dict):
    ue_contexts[ue_id] = context
    logger.info(f"UE context created for {ue_id}")
    return {"message": "UE context created"}

def trigger_pdu_session_creation(ue_context: dict):
    """
    Models the AMF sending a Create SM Context Request to the SMF over N11.
    Implements 3GPP TS 23.502 Section 4.3.2.2.1 - PDU Session Establishment
    This replaces the old custom API call with 3GPP-aligned endpoints.
    """
    global smf_url
    if not smf_url:
        logger.error("SMF URL not available - service discovery failed")
        return None
        
    # 1. Define the 3GPP Service-Based Interface (SBI) endpoint on the SMF
    # This comes from TS 29.502 (Nsmf_PDUSession service)
    sm_context_endpoint = f"{smf_url}/nsmf-pdusession/v1/sm-contexts"

    # 2. Construct a 3GPP-aligned JSON payload
    # Parameter names like 'supi', 'dnn', 'sNssai' are defined in the specs.
    pdu_session_data = {
        "supi": ue_context.get("supi"), # Standardized identifier
        "pduSessionId": ue_context.get("pduSessionId"),
        "dnn": "internet",  # Data Network Name
        "sNssai": {       # Single Network Slice Selection Assistance Information
            "sst": 1,
            "sd": "010203"
        },
        "gpsi": f"msisdn-{ue_context.get('imsi')}", # Generic Public Subscription Identifier
        "anType": "3GPP_ACCESS",
        "ratType": "NR",
        "ueLocation": {
            "nrLocation": {
                "tai": {
                    "plmnId": {"mcc": "001", "mnc": "01"},
                    "tac": "000001"
                },
                "ncgi": {
                    "plmnId": {"mcc": "001", "mnc": "01"},
                    "nrCellId": "000000001"
                }
            }
        }
    }

    logger.info(f"AMF -> SMF: Sending Create SM Context Request for SUPI {ue_context['supi']}")
    
    try:
        with tracer.start_as_current_span("amf_pdu_session_create_request") as span:
            span.set_attribute("3gpp.procedure", "pdu_session_establishment")
            span.set_attribute("3gpp.interface", "N11")
            span.set_attribute("3gpp.service", "Nsmf_PDUSession")
            span.set_attribute("ue.supi", ue_context['supi'])
            span.set_attribute("pdu.session.id", str(ue_context['pduSessionId']))
            
            response = requests.post(sm_context_endpoint, json=pdu_session_data, timeout=5)
            response.raise_for_status()
            
            # The SMF's response will contain data needed for the RAN
            sm_context_response = response.json()
            logger.info(f"AMF <- SMF: Create SM Context Response received.")
            
            span.add_event("sm_context_created", {
                "response.status": sm_context_response.get("status"),
                "n2.sm.info.present": "n2SmInfo" in sm_context_response
            })
            
            return sm_context_response
            
    except requests.RequestException as e:
        logger.error(f"AMF -> SMF: Failed to create PDU session context: {e}")
        return None

@app.post("/amf/pdu-session/create")
async def create_pdu_session(request_data: dict):
    """
    3GPP-compliant PDU Session Establishment procedure.
    Reference: 3GPP TS 23.502 Section 4.3.2.2.1
    """
    logger.info(f"Starting PDU Session Establishment for UE: {request_data.get('ue_id')}")
    
    ue_id = request_data.get('ue_id')
    if not ue_id or ue_id not in ue_contexts:
        raise HTTPException(status_code=400, detail="Invalid or missing UE context")
    
    ue_context = ue_contexts[ue_id]
    
    # Ensure UE context has required 3GPP fields
    if not ue_context.get('supi'):
        ue_context['supi'] = f"imsi-00101{ue_id}"  # Generate SUPI if missing
    if not ue_context.get('pduSessionId'):
        ue_context['pduSessionId'] = len(ue_contexts) + 1  # Simple ID assignment
    if not ue_context.get('imsi'):
        ue_context['imsi'] = f"00101{ue_id}"
    
    # Trigger the 3GPP-compliant PDU session creation
    sm_response = trigger_pdu_session_creation(ue_context)
    
    if sm_response:
        logger.info(f"PDU Session Establishment successful for UE {ue_id}")
        return {
            "status": "SUCCESS",
            "pduSessionId": ue_context['pduSessionId'],
            "message": "PDU Session established successfully",
            "smContextResponse": sm_response
        }
    else:
        raise HTTPException(status_code=500, detail="PDU Session Establishment failed")

# NGAP Endpoints for gNB communication (TS 38.413)
@app.post("/ngap/ng-setup")
async def ng_setup(ngap_message: Dict):
    """
    Handle NG Setup Request from gNB per TS 38.413 Section 9.2.6.1
    """
    with tracer.start_as_current_span("amf_ng_setup") as span:
        span.set_attribute("3gpp.interface", "N2")
        span.set_attribute("3gpp.protocol", "NGAP")
        span.set_attribute("ngap.procedure", "NGSetup")

        try:
            # Extract gNB info from NG Setup Request
            initiating_msg = ngap_message.get("initiatingMessage", {})
            protocol_ies = initiating_msg.get("value", {}).get("protocolIEs", {})

            global_ran_node_id = protocol_ies.get("GlobalRANNodeID", {})
            ran_node_name = protocol_ies.get("RANNodeName", "Unknown")
            supported_ta_list = protocol_ies.get("SupportedTAList", [])

            logger.info(f"NG Setup Request received from {ran_node_name}")

            # Store gNB info
            gnb_id = global_ran_node_id.get("globalGNB-ID", {}).get("gNB-ID", {}).get("gNB-ID", "unknown")
            gnb_data[gnb_id] = {
                "ranNodeName": ran_node_name,
                "supportedTAs": supported_ta_list,
                "status": "connected",
                "connectedAt": datetime.now().isoformat()
            }

            span.set_attribute("ngap.gnb_id", gnb_id)
            span.set_attribute("status", "SUCCESS")

            # Return NG Setup Response per TS 38.413 Section 9.2.6.2
            return {
                "successfulOutcome": {
                    "procedureCode": 21,
                    "criticality": "reject",
                    "value": {
                        "protocolIEs": {
                            "AMFName": "AMF-001",
                            "ServedGUAMIList": [
                                {
                                    "gUAMI": {
                                        "pLMNIdentity": {"mcc": "001", "mnc": "01"},
                                        "aMFRegionID": "01",
                                        "aMFSetID": "001",
                                        "aMFPointer": "01"
                                    }
                                }
                            ],
                            "RelativeAMFCapacity": 255,
                            "PLMNSupportList": [
                                {
                                    "pLMNIdentity": {"mcc": "001", "mnc": "01"},
                                    "sliceSupportList": [
                                        {"sNSSAI": {"sST": 1, "sD": "010203"}}
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"NG Setup failed: {e}")
            raise HTTPException(status_code=500, detail=f"NG Setup failed: {e}")

@app.post("/ngap/initial-ue-message")
async def initial_ue_message(ngap_message: Dict):
    """
    Handle Initial UE Message from gNB per TS 38.413 Section 9.2.5.1
    """
    with tracer.start_as_current_span("amf_initial_ue_message") as span:
        span.set_attribute("3gpp.interface", "N2")
        span.set_attribute("3gpp.protocol", "NGAP")
        span.set_attribute("ngap.procedure", "InitialUEMessage")

        try:
            initiating_msg = ngap_message.get("initiatingMessage", {})
            protocol_ies = initiating_msg.get("value", {}).get("protocolIEs", {})

            ran_ue_ngap_id = protocol_ies.get("RAN-UE-NGAP-ID")
            nas_pdu = protocol_ies.get("NAS-PDU", "")
            user_location = protocol_ies.get("UserLocationInformation", {})

            # Generate AMF UE NGAP ID
            import uuid
            amf_ue_ngap_id = abs(hash(uuid.uuid4())) % (2**40)

            # Create UE context
            ue_id = f"ue-{ran_ue_ngap_id}"
            ue_contexts[ue_id] = {
                "ranUeNgapId": ran_ue_ngap_id,
                "amfUeNgapId": amf_ue_ngap_id,
                "state": "REGISTERING",
                "userLocation": user_location,
                "createdAt": datetime.now().isoformat()
            }

            span.set_attribute("ngap.ran_ue_ngap_id", ran_ue_ngap_id)
            span.set_attribute("ngap.amf_ue_ngap_id", amf_ue_ngap_id)
            span.set_attribute("status", "SUCCESS")

            logger.info(f"Initial UE Message: RAN-UE-NGAP-ID={ran_ue_ngap_id}, AMF-UE-NGAP-ID={amf_ue_ngap_id}")

            return {
                "status": "SUCCESS",
                "amfUeNgapId": amf_ue_ngap_id,
                "ranUeNgapId": ran_ue_ngap_id,
                "message": "Initial UE Message processed"
            }
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Initial UE Message failed: {e}")
            raise HTTPException(status_code=500, detail=f"Initial UE Message failed: {e}")

@app.post("/ngap/uplink-nas-transport")
async def uplink_nas_transport(ngap_message: Dict):
    """
    Handle Uplink NAS Transport from gNB per TS 38.413 Section 9.2.5.4
    """
    with tracer.start_as_current_span("amf_uplink_nas_transport") as span:
        span.set_attribute("3gpp.interface", "N2")
        span.set_attribute("3gpp.protocol", "NGAP")
        span.set_attribute("ngap.procedure", "UplinkNASTransport")

        try:
            initiating_msg = ngap_message.get("initiatingMessage", {})
            protocol_ies = initiating_msg.get("value", {}).get("protocolIEs", {})

            amf_ue_ngap_id = protocol_ies.get("AMF-UE-NGAP-ID")
            ran_ue_ngap_id = protocol_ies.get("RAN-UE-NGAP-ID")
            nas_pdu = protocol_ies.get("NAS-PDU", "")

            span.set_attribute("ngap.amf_ue_ngap_id", amf_ue_ngap_id)
            span.set_attribute("ngap.ran_ue_ngap_id", ran_ue_ngap_id)

            logger.info(f"Uplink NAS Transport: AMF-UE-NGAP-ID={amf_ue_ngap_id}, NAS-PDU length={len(nas_pdu)}")

            return {
                "status": "SUCCESS",
                "message": "Uplink NAS Transport processed"
            }
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Uplink NAS Transport failed: {e}")
            raise HTTPException(status_code=500, detail=f"Uplink NAS Transport failed: {e}")

@app.post("/ngap/ue-context-release")
async def ue_context_release(ngap_message: Dict):
    """
    Handle UE Context Release per TS 38.413 Section 9.2.2.4
    Procedure Code: 41
    """
    with tracer.start_as_current_span("amf_ue_context_release") as span:
        span.set_attribute("3gpp.interface", "N2")
        span.set_attribute("3gpp.protocol", "NGAP")
        span.set_attribute("ngap.procedure", "UEContextRelease")
        span.set_attribute("ngap.procedure_code", 41)

        try:
            initiating_msg = ngap_message.get("initiatingMessage", {})
            protocol_ies = initiating_msg.get("value", {}).get("protocolIEs", {})

            amf_ue_ngap_id = protocol_ies.get("AMF-UE-NGAP-ID")
            ran_ue_ngap_id = protocol_ies.get("RAN-UE-NGAP-ID")
            cause = protocol_ies.get("Cause", {"radioNetwork": "unspecified"})

            span.set_attribute("ngap.amf_ue_ngap_id", str(amf_ue_ngap_id))
            span.set_attribute("ngap.cause", str(cause))

            # Find and remove UE context
            ue_id_to_remove = None
            for ue_id, context in ue_contexts.items():
                if context.get("amfUeNgapId") == amf_ue_ngap_id:
                    ue_id_to_remove = ue_id
                    break

            if ue_id_to_remove:
                released_context = ue_contexts.pop(ue_id_to_remove)
                logger.info(f"UE Context Released: {ue_id_to_remove}, Cause: {cause}")
                span.set_attribute("status", "SUCCESS")

                # Return UE Context Release Complete per TS 38.413 Section 9.2.2.5
                return {
                    "successfulOutcome": {
                        "procedureCode": 41,
                        "criticality": "reject",
                        "value": {
                            "protocolIEs": {
                                "AMF-UE-NGAP-ID": amf_ue_ngap_id,
                                "RAN-UE-NGAP-ID": ran_ue_ngap_id,
                                "UserLocationInformation": released_context.get("userLocation", {}),
                                "CriticalityDiagnostics": None
                            }
                        }
                    }
                }
            else:
                logger.warning(f"UE Context not found for AMF-UE-NGAP-ID: {amf_ue_ngap_id}")
                span.set_attribute("status", "NOT_FOUND")
                return {
                    "unsuccessfulOutcome": {
                        "procedureCode": 41,
                        "criticality": "reject",
                        "value": {
                            "cause": {"misc": "unknown-local-UE-NGAP-ID"}
                        }
                    }
                }
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"UE Context Release failed: {e}")
            raise HTTPException(status_code=500, detail=f"UE Context Release failed: {e}")


@app.post("/ngap/paging")
async def paging(paging_request: Dict):
    """
    Initiate Paging per TS 38.413 Section 9.2.3.1
    Procedure Code: 20
    Used for mobile-terminated services when UE is in IDLE mode
    """
    with tracer.start_as_current_span("amf_paging") as span:
        span.set_attribute("3gpp.interface", "N2")
        span.set_attribute("3gpp.protocol", "NGAP")
        span.set_attribute("ngap.procedure", "Paging")
        span.set_attribute("ngap.procedure_code", 20)

        try:
            ue_paging_identity = paging_request.get("uePagingIdentity", {})
            tai_list_for_paging = paging_request.get("taiListForPaging", [])
            paging_drx = paging_request.get("pagingDRX", "v128")
            paging_priority = paging_request.get("pagingPriority", "priolevel1")

            # Get 5G-S-TMSI from paging identity
            five_g_s_tmsi = ue_paging_identity.get("fiveG-S-TMSI", {})
            amf_set_id = five_g_s_tmsi.get("aMFSetID", "")
            amf_pointer = five_g_s_tmsi.get("aMFPointer", "")
            five_g_tmsi = five_g_s_tmsi.get("fiveG-TMSI", "")

            span.set_attribute("ngap.five_g_tmsi", five_g_tmsi)
            span.set_attribute("ngap.paging_drx", paging_drx)

            logger.info(f"Initiating Paging for 5G-TMSI: {five_g_tmsi}, TAI List: {tai_list_for_paging}")

            # Build Paging message per TS 38.413 Section 9.2.3.1
            paging_message = {
                "initiatingMessage": {
                    "procedureCode": 20,
                    "criticality": "ignore",
                    "value": {
                        "protocolIEs": {
                            "UEPagingIdentity": {
                                "fiveG-S-TMSI": five_g_s_tmsi
                            },
                            "PagingDRX": paging_drx,
                            "TAIListForPaging": tai_list_for_paging,
                            "PagingPriority": paging_priority,
                            "UERadioCapabilityForPaging": None,
                            "PagingOrigin": "non-3gpp",
                            "AssistanceDataForPaging": None
                        }
                    }
                }
            }

            span.set_attribute("status", "SUCCESS")

            # In a real implementation, this would be sent to all gNBs in the TAI list
            return {
                "status": "PAGING_INITIATED",
                "message": f"Paging initiated for 5G-TMSI: {five_g_tmsi}",
                "taiListForPaging": tai_list_for_paging,
                "pagingMessage": paging_message
            }
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Paging failed: {e}")
            raise HTTPException(status_code=500, detail=f"Paging failed: {e}")


@app.post("/ngap/error-indication")
async def error_indication(ngap_message: Dict):
    """
    Handle Error Indication per TS 38.413 Section 9.2.1.3
    Procedure Code: 5
    Used to report errors in NGAP procedures
    """
    with tracer.start_as_current_span("amf_error_indication") as span:
        span.set_attribute("3gpp.interface", "N2")
        span.set_attribute("3gpp.protocol", "NGAP")
        span.set_attribute("ngap.procedure", "ErrorIndication")
        span.set_attribute("ngap.procedure_code", 5)

        try:
            initiating_msg = ngap_message.get("initiatingMessage", {})
            protocol_ies = initiating_msg.get("value", {}).get("protocolIEs", {})

            amf_ue_ngap_id = protocol_ies.get("AMF-UE-NGAP-ID")
            ran_ue_ngap_id = protocol_ies.get("RAN-UE-NGAP-ID")
            cause = protocol_ies.get("Cause", {})
            criticality_diagnostics = protocol_ies.get("CriticalityDiagnostics", {})

            span.set_attribute("ngap.amf_ue_ngap_id", str(amf_ue_ngap_id))
            span.set_attribute("ngap.ran_ue_ngap_id", str(ran_ue_ngap_id))
            span.set_attribute("ngap.cause", str(cause))

            # Log the error
            logger.warning(f"Error Indication received: AMF-UE-NGAP-ID={amf_ue_ngap_id}, "
                          f"RAN-UE-NGAP-ID={ran_ue_ngap_id}, Cause={cause}")

            # Process error based on cause
            error_category = None
            if "radioNetwork" in cause:
                error_category = "radioNetwork"
            elif "transport" in cause:
                error_category = "transport"
            elif "nas" in cause:
                error_category = "nas"
            elif "protocol" in cause:
                error_category = "protocol"
            elif "misc" in cause:
                error_category = "misc"

            span.set_attribute("ngap.error_category", error_category or "unknown")
            span.set_attribute("status", "LOGGED")

            return {
                "status": "ERROR_LOGGED",
                "amfUeNgapId": amf_ue_ngap_id,
                "ranUeNgapId": ran_ue_ngap_id,
                "cause": cause,
                "errorCategory": error_category,
                "message": "Error indication received and logged"
            }
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Error Indication processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Error Indication processing failed: {e}")


@app.post("/ngap/downlink-nas-transport")
async def downlink_nas_transport(ngap_message: Dict):
    """
    Send Downlink NAS Transport to gNB per TS 38.413 Section 9.2.5.3
    Procedure Code: 4
    """
    with tracer.start_as_current_span("amf_downlink_nas_transport") as span:
        span.set_attribute("3gpp.interface", "N2")
        span.set_attribute("3gpp.protocol", "NGAP")
        span.set_attribute("ngap.procedure", "DownlinkNASTransport")
        span.set_attribute("ngap.procedure_code", 4)

        try:
            amf_ue_ngap_id = ngap_message.get("amfUeNgapId")
            ran_ue_ngap_id = ngap_message.get("ranUeNgapId")
            nas_pdu = ngap_message.get("nasPdu", "")

            span.set_attribute("ngap.amf_ue_ngap_id", str(amf_ue_ngap_id))
            span.set_attribute("ngap.ran_ue_ngap_id", str(ran_ue_ngap_id))

            # Build Downlink NAS Transport message
            dl_nas_transport = {
                "initiatingMessage": {
                    "procedureCode": 4,
                    "criticality": "ignore",
                    "value": {
                        "protocolIEs": {
                            "AMF-UE-NGAP-ID": amf_ue_ngap_id,
                            "RAN-UE-NGAP-ID": ran_ue_ngap_id,
                            "NAS-PDU": nas_pdu,
                            "MobilityRestrictionList": None,
                            "UEAggregateMaximumBitRate": None,
                            "AllowedNSSAI": [{"sNSSAI": {"sST": 1}}]
                        }
                    }
                }
            }

            logger.info(f"Downlink NAS Transport: AMF-UE-NGAP-ID={amf_ue_ngap_id}, NAS-PDU length={len(nas_pdu)}")
            span.set_attribute("status", "SUCCESS")

            return {
                "status": "SUCCESS",
                "message": "Downlink NAS Transport sent",
                "ngapMessage": dl_nas_transport
            }
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"Downlink NAS Transport failed: {e}")
            raise HTTPException(status_code=500, detail=f"Downlink NAS Transport failed: {e}")


@app.post("/amf/ue/register")
async def register_ue(registration_request: Dict):
    """
    Handle UE Registration per TS 23.502 Section 4.2.2.2
    Creates a proper UE context with SUPI as the key
    """
    with tracer.start_as_current_span("amf_ue_registration") as span:
        span.set_attribute("3gpp.procedure", "Registration")

        try:
            supi = registration_request.get("supi")
            if not supi:
                raise HTTPException(status_code=400, detail="SUPI is required")

            ran_ue_ngap_id = registration_request.get("ranUeNgapId")
            amf_ue_ngap_id = registration_request.get("amfUeNgapId")
            registration_type = registration_request.get("registrationType", "initialRegistration")

            # Generate AMF UE NGAP ID if not provided
            if not amf_ue_ngap_id:
                import uuid
                amf_ue_ngap_id = abs(hash(uuid.uuid4())) % (2**40)

            # Create or update UE context
            ue_contexts[supi] = {
                "supi": supi,
                "ranUeNgapId": ran_ue_ngap_id,
                "amfUeNgapId": amf_ue_ngap_id,
                "registrationType": registration_type,
                "rmState": "RM-REGISTERED",  # Registration Management state
                "cmState": "CM-CONNECTED",    # Connection Management state
                "registeredAt": datetime.now(timezone.utc).isoformat(),
                "guami": {
                    "plmnId": {"mcc": "001", "mnc": "01"},
                    "amfId": {"amfRegionId": "01", "amfSetId": "001", "amfPointer": "01"}
                },
                "allowedNssai": [{"sst": 1, "sd": "010203"}],
                "pduSessions": []
            }

            span.set_attribute("ue.supi", supi)
            span.set_attribute("ue.rm_state", "RM-REGISTERED")
            span.set_attribute("status", "SUCCESS")

            logger.info(f"UE Registered: SUPI={supi}, AMF-UE-NGAP-ID={amf_ue_ngap_id}")

            return {
                "status": "SUCCESS",
                "supi": supi,
                "amfUeNgapId": amf_ue_ngap_id,
                "rmState": "RM-REGISTERED",
                "cmState": "CM-CONNECTED",
                "message": "UE registration successful"
            }
        except HTTPException:
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"UE Registration failed: {e}")
            raise HTTPException(status_code=500, detail=f"UE Registration failed: {e}")


@app.post("/amf/ue/{supi}/deregister")
async def deregister_ue(supi: str, deregistration_request: Dict = None):
    """
    Handle UE Deregistration per TS 23.502 Section 4.2.2.3
    """
    with tracer.start_as_current_span("amf_ue_deregistration") as span:
        span.set_attribute("3gpp.procedure", "Deregistration")
        span.set_attribute("ue.supi", supi)

        try:
            if supi not in ue_contexts:
                raise HTTPException(status_code=404, detail="UE context not found")

            ue_context = ue_contexts[supi]
            deregistration_type = "switchOff"
            if deregistration_request:
                deregistration_type = deregistration_request.get("deregistrationType", "switchOff")

            # Update UE state
            ue_context["rmState"] = "RM-DEREGISTERED"
            ue_context["cmState"] = "CM-IDLE"
            ue_context["deregisteredAt"] = datetime.now(timezone.utc).isoformat()

            # Release PDU sessions
            released_sessions = ue_context.get("pduSessions", [])
            ue_context["pduSessions"] = []

            span.set_attribute("ue.rm_state", "RM-DEREGISTERED")
            span.set_attribute("status", "SUCCESS")

            logger.info(f"UE Deregistered: SUPI={supi}, Type={deregistration_type}")

            # Remove context after deregistration
            del ue_contexts[supi]

            return {
                "status": "SUCCESS",
                "supi": supi,
                "rmState": "RM-DEREGISTERED",
                "deregistrationType": deregistration_type,
                "releasedPduSessions": len(released_sessions),
                "message": "UE deregistration successful"
            }
        except HTTPException:
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"UE Deregistration failed: {e}")
            raise HTTPException(status_code=500, detail=f"UE Deregistration failed: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AMF",
        "compliance": "3GPP TS 29.518, TS 38.413",
        "version": "1.0.0",
        "connected_gnbs": len([g for g in gnb_data.values() if g.get("status") == "connected"]),
        "active_ues": len(ue_contexts)
    }

@app.get("/metrics")
async def metrics():
    return {"message": "Metrics are exposed on port 9100"}

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="AMF - Access and Mobility Management Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("amf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)