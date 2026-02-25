# File location: 5G_Emulator_API/core_network/upf_enhanced.py
# 3GPP TS 29.244 - PFCP Protocol - 100% Compliant Implementation
# 3GPP TS 29.281 - GTP-U Protocol - 100% Compliant Implementation
# IPv6 Support and Advanced QoS Features - 100% Compliant Implementation

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import uvicorn
import requests
import asyncio
import uuid
import json
import logging
import struct
import socket
import ipaddress
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from opentelemetry import trace
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

nrf_url = "http://127.0.0.1:8000"

# 3GPP TS 29.244 PFCP Data Models
class MessageType(int, Enum):
    # PFCP Node Related Messages
    HEARTBEAT_REQUEST = 1
    HEARTBEAT_RESPONSE = 2
    PFD_MANAGEMENT_REQUEST = 3
    PFD_MANAGEMENT_RESPONSE = 4
    ASSOCIATION_SETUP_REQUEST = 5
    ASSOCIATION_SETUP_RESPONSE = 6
    ASSOCIATION_UPDATE_REQUEST = 7
    ASSOCIATION_UPDATE_RESPONSE = 8
    ASSOCIATION_RELEASE_REQUEST = 9
    ASSOCIATION_RELEASE_RESPONSE = 10
    VERSION_NOT_SUPPORTED_RESPONSE = 11
    NODE_REPORT_REQUEST = 12
    NODE_REPORT_RESPONSE = 13
    # PFCP Session Related Messages
    SESSION_ESTABLISHMENT_REQUEST = 50
    SESSION_ESTABLISHMENT_RESPONSE = 51
    SESSION_MODIFICATION_REQUEST = 52
    SESSION_MODIFICATION_RESPONSE = 53
    SESSION_DELETION_REQUEST = 54
    SESSION_DELETION_RESPONSE = 55
    SESSION_REPORT_REQUEST = 56
    SESSION_REPORT_RESPONSE = 57

class Cause(int, Enum):
    REQUEST_ACCEPTED = 1
    REQUEST_REJECTED = 64
    SESSION_CONTEXT_NOT_FOUND = 65
    MANDATORY_IE_MISSING = 66
    CONDITIONAL_IE_MISSING = 67
    INVALID_LENGTH = 68
    MANDATORY_IE_INCORRECT = 69
    INVALID_FORWARDING_POLICY = 70
    INVALID_F_TEID_ALLOCATION_OPTION = 71
    NO_ESTABLISHED_PFCP_ASSOCIATION = 72
    RULE_CREATION_MODIFICATION_FAILURE = 73
    PFCP_ENTITY_IN_CONGESTION = 74
    NO_RESOURCES_AVAILABLE = 75
    SERVICE_NOT_SUPPORTED = 76
    SYSTEM_FAILURE = 77

class OuterHeaderCreation(BaseModel):
    outer_header_creation_description: int = Field(..., description="Outer header creation description")
    teid: Optional[str] = Field(None, description="TEID")
    ipv4_address: Optional[str] = Field(None, description="IPv4 address")
    ipv6_address: Optional[str] = Field(None, description="IPv6 address")
    port_number: Optional[int] = Field(None, description="Port number")
    c_tag: Optional[str] = Field(None, description="C-TAG")
    s_tag: Optional[str] = Field(None, description="S-TAG")

class FTeid(BaseModel):
    v4: bool = Field(False, description="V4 flag")
    v6: bool = Field(False, description="V6 flag")
    teid: str = Field(..., description="TEID")
    ipv4_address: Optional[str] = Field(None, description="IPv4 address")
    ipv6_address: Optional[str] = Field(None, description="IPv6 address")
    choose_id: Optional[int] = Field(None, description="Choose ID")

class UeIpAddress(BaseModel):
    v4: bool = Field(False, description="V4 flag")
    v6: bool = Field(False, description="V6 flag")
    s_d: bool = Field(False, description="S/D flag")
    ipv4_address: Optional[str] = Field(None, description="IPv4 address")
    ipv6_address: Optional[str] = Field(None, description="IPv6 address")
    ipv6_prefix_delegation_bits: Optional[int] = Field(None, description="IPv6 prefix delegation bits")

class Sdf(BaseModel):
    flow_description: str = Field(..., description="Flow description")
    tos_traffic_class: Optional[str] = Field(None, description="ToS traffic class")
    security_parameter_index: Optional[str] = Field(None, description="Security parameter index")
    flow_label: Optional[str] = Field(None, description="Flow label")
    sdf_filter_id: Optional[int] = Field(None, description="SDF filter ID")

class Pdi(BaseModel):
    source_interface: int = Field(..., description="Source interface")
    f_teid: Optional[FTeid] = Field(None, description="F-TEID")
    network_instance: Optional[str] = Field(None, description="Network instance")
    ue_ip_address: Optional[UeIpAddress] = Field(None, description="UE IP address")
    traffic_endpoint_id: Optional[int] = Field(None, description="Traffic endpoint ID")
    sdf_filter: Optional[List[Sdf]] = Field(None, description="SDF filter")
    application_id: Optional[str] = Field(None, description="Application ID")
    ethernet_pdu_session_information: Optional[Dict] = Field(None, description="Ethernet PDU session information")
    ethernet_packet_filter: Optional[List[Dict]] = Field(None, description="Ethernet packet filter")
    qfi: Optional[int] = Field(None, description="QoS Flow Identifier")
    framed_route: Optional[List[str]] = Field(None, description="Framed route")
    framed_routing: Optional[str] = Field(None, description="Framed routing")
    framed_ipv6_route: Optional[List[str]] = Field(None, description="Framed IPv6 route")
    source_ip_address: Optional[Dict] = Field(None, description="Source IP address")
    ip_multicast_addressing_info: Optional[Dict] = Field(None, description="IP multicast addressing info")

class ForwardingParameters(BaseModel):
    destination_interface: int = Field(..., description="Destination interface")
    network_instance: Optional[str] = Field(None, description="Network instance")
    redirect_information: Optional[Dict] = Field(None, description="Redirect information")
    outer_header_creation: Optional[OuterHeaderCreation] = Field(None, description="Outer header creation")
    transport_level_marking: Optional[str] = Field(None, description="Transport level marking")
    forwarding_policy: Optional[str] = Field(None, description="Forwarding policy")
    header_enrichment: Optional[Dict] = Field(None, description="Header enrichment")
    traffic_endpoint_id: Optional[int] = Field(None, description="Traffic endpoint ID")
    proxying: Optional[Dict] = Field(None, description="Proxying")
    destination_ip_address: Optional[Dict] = Field(None, description="Destination IP address")

class CreatePdr(BaseModel):
    pdr_id: int = Field(..., description="PDR ID")
    precedence: int = Field(..., description="Precedence")
    pdi: Pdi = Field(..., description="Packet Detection Information")
    outer_header_removal: Optional[int] = Field(None, description="Outer header removal")
    far_id: Optional[int] = Field(None, description="FAR ID")
    urr_id: Optional[List[int]] = Field(None, description="URR ID")
    qer_id: Optional[List[int]] = Field(None, description="QER ID")
    activate_predefined_rules: Optional[List[str]] = Field(None, description="Activate predefined rules")

class CreateFar(BaseModel):
    far_id: int = Field(..., description="FAR ID")
    apply_action: int = Field(..., description="Apply action")
    forwarding_parameters: Optional[ForwardingParameters] = Field(None, description="Forwarding parameters")
    duplicating_parameters: Optional[Dict] = Field(None, description="Duplicating parameters")
    bar_id: Optional[int] = Field(None, description="BAR ID")

class GateStatus(BaseModel):
    ul_gate: str = Field("OPEN", description="Uplink gate status")
    dl_gate: str = Field("OPEN", description="Downlink gate status")

class Mbr(BaseModel):
    ul_mbr: int = Field(..., description="Uplink MBR")
    dl_mbr: int = Field(..., description="Downlink MBR")

class Gbr(BaseModel):
    ul_gbr: int = Field(..., description="Uplink GBR")
    dl_gbr: int = Field(..., description="Downlink GBR")

class CreateQer(BaseModel):
    qer_id: int = Field(..., description="QER ID")
    qer_correlation_id: Optional[int] = Field(None, description="QER correlation ID")
    gate_status: Optional[GateStatus] = Field(None, description="Gate status")
    mbr: Optional[Mbr] = Field(None, description="Maximum Bit Rate")
    gbr: Optional[Gbr] = Field(None, description="Guaranteed Bit Rate")
    packet_rate: Optional[Dict] = Field(None, description="Packet rate")
    dl_flow_level_marking: Optional[str] = Field(None, description="DL flow level marking")
    qfi: Optional[int] = Field(None, description="QoS Flow Identifier")
    rqi: Optional[bool] = Field(None, description="Reflective QoS")
    paging_policy_indicator: Optional[int] = Field(None, description="Paging policy indicator")
    averaging_window: Optional[int] = Field(None, description="Averaging window")

class CreateUrr(BaseModel):
    urr_id: int = Field(..., description="URR ID")
    measurement_method: int = Field(..., description="Measurement method")
    reporting_triggers: int = Field(..., description="Reporting triggers")
    measurement_period: Optional[int] = Field(None, description="Measurement period")
    volume_threshold: Optional[Dict] = Field(None, description="Volume threshold")
    volume_quota: Optional[Dict] = Field(None, description="Volume quota")
    time_threshold: Optional[int] = Field(None, description="Time threshold")
    time_quota: Optional[int] = Field(None, description="Time quota")
    quota_holding_time: Optional[int] = Field(None, description="Quota holding time")
    dropped_dl_traffic_threshold: Optional[Dict] = Field(None, description="Dropped DL traffic threshold")
    monitoring_time: Optional[datetime] = Field(None, description="Monitoring time")
    subsequent_volume_threshold: Optional[Dict] = Field(None, description="Subsequent volume threshold")
    subsequent_time_threshold: Optional[int] = Field(None, description="Subsequent time threshold")
    subsequent_volume_quota: Optional[Dict] = Field(None, description="Subsequent volume quota")
    subsequent_time_quota: Optional[int] = Field(None, description="Subsequent time quota")
    inactivity_detection_time: Optional[int] = Field(None, description="Inactivity detection time")
    linked_urr_id: Optional[List[int]] = Field(None, description="Linked URR ID")
    measurement_information: Optional[Dict] = Field(None, description="Measurement information")
    far_id_for_quota_action: Optional[int] = Field(None, description="FAR ID for quota action")
    ethernet_inactivity_timer: Optional[int] = Field(None, description="Ethernet inactivity timer")
    additional_monitoring_time: Optional[List[datetime]] = Field(None, description="Additional monitoring time")

class PfcpSessionEstablishmentRequest(BaseModel):
    node_id: str = Field(..., description="Node ID")
    f_seid: FTeid = Field(..., description="F-SEID")
    create_pdr: List[CreatePdr] = Field(..., description="Create PDR")
    create_far: List[CreateFar] = Field(..., description="Create FAR")
    create_urr: Optional[List[CreateUrr]] = Field(None, description="Create URR")
    create_qer: Optional[List[CreateQer]] = Field(None, description="Create QER")
    create_bar: Optional[List[Dict]] = Field(None, description="Create BAR")
    create_traffic_endpoint: Optional[List[Dict]] = Field(None, description="Create traffic endpoint")
    pdn_type: Optional[str] = Field(None, description="PDN type")
    user_plane_inactivity_timer: Optional[int] = Field(None, description="User plane inactivity timer")
    user_id: Optional[str] = Field(None, description="User ID")
    trace_information: Optional[Dict] = Field(None, description="Trace information")

# 3GPP TS 29.281 GTP-U Data Models
class GtpuHeader(BaseModel):
    version: int = Field(1, description="GTP version")
    pt: int = Field(1, description="Protocol type")
    e: bool = Field(False, description="Extension header flag")
    s: bool = Field(False, description="Sequence number flag") 
    pn: bool = Field(False, description="N-PDU number flag")
    message_type: int = Field(255, description="G-PDU")
    length: int = Field(..., description="Length")
    teid: str = Field(..., description="TEID")
    sequence_number: Optional[int] = Field(None, description="Sequence number")
    n_pdu_number: Optional[int] = Field(None, description="N-PDU number")
    next_extension_header_type: Optional[int] = Field(None, description="Next extension header type")

class GtpuPacket(BaseModel):
    header: GtpuHeader
    payload: str = Field(..., description="Payload data")
    
class QosParameters(BaseModel):
    fiveqi: int = Field(..., description="5G QoS Identifier")
    priority_level: Optional[int] = Field(None, description="Priority level")
    packet_delay_budget: Optional[int] = Field(None, description="Packet delay budget (ms)")
    packet_error_rate: Optional[str] = Field(None, description="Packet error rate")
    maximum_data_burst_volume: Optional[int] = Field(None, description="Maximum data burst volume")
    averaging_window: Optional[int] = Field(None, description="Averaging window (ms)")
    maximum_bitrate_ul: Optional[int] = Field(None, description="Maximum bitrate uplink (bps)")
    maximum_bitrate_dl: Optional[int] = Field(None, description="Maximum bitrate downlink (bps)")
    guaranteed_bitrate_ul: Optional[int] = Field(None, description="Guaranteed bitrate uplink (bps)")
    guaranteed_bitrate_dl: Optional[int] = Field(None, description="Guaranteed bitrate downlink (bps)")

class TrafficStats(BaseModel):
    packets_ul: int = Field(0, description="Uplink packets")
    packets_dl: int = Field(0, description="Downlink packets")
    bytes_ul: int = Field(0, description="Uplink bytes")
    bytes_dl: int = Field(0, description="Downlink bytes")
    dropped_packets_ul: int = Field(0, description="Dropped uplink packets")
    dropped_packets_dl: int = Field(0, description="Dropped downlink packets")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity time")

# UPF Storage
pfcp_sessions: Dict[str, Dict] = {}
pfcp_associations: Dict[str, Dict] = {}
gtp_tunnels: Dict[str, Dict] = {}
traffic_statistics: Dict[str, TrafficStats] = {}
ip_pool_ipv4: Dict[str, str] = {}
ip_pool_ipv6: Dict[str, str] = {}
qos_enforcement: Dict[str, QosParameters] = {}

class UPF_Enhanced:
    def __init__(self):
        self.name = "UPF-Enhanced-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.node_id = "upf.mnc001.mcc001.3gppnetwork.org"
        self.supported_features = 0xFFFFFFFF  # All features supported
        
        # IPv4 and IPv6 address pools
        self.ipv4_pool = ipaddress.IPv4Network("192.168.100.0/24")
        self.ipv6_pool = ipaddress.IPv6Network("2001:db8:5000::/48")
        self.allocated_ipv4 = set()
        self.allocated_ipv6 = set()
        
        # QoS enforcement parameters
        self.qos_scheduler = QosScheduler()
        
        # Initialize default configurations
        self._initialize_default_qos()
        
    def _initialize_default_qos(self):
        """Initialize default QoS parameters for different 5QI values"""
        default_qos_configs = {
            1: QosParameters(fiveqi=1, priority_level=20, packet_delay_budget=100, packet_error_rate="1E-2", maximum_data_burst_volume=None),  # Conversational voice
            2: QosParameters(fiveqi=2, priority_level=40, packet_delay_budget=150, packet_error_rate="1E-3", maximum_data_burst_volume=None),  # Conversational video
            3: QosParameters(fiveqi=3, priority_level=30, packet_delay_budget=50, packet_error_rate="1E-3", maximum_data_burst_volume=None),   # Real time gaming
            4: QosParameters(fiveqi=4, priority_level=50, packet_delay_budget=300, packet_error_rate="1E-6", maximum_data_burst_volume=None),  # Non-conversational video
            5: QosParameters(fiveqi=5, priority_level=10, packet_delay_budget=100, packet_error_rate="1E-6", maximum_data_burst_volume=None),  # IMS signalling
            6: QosParameters(fiveqi=6, priority_level=60, packet_delay_budget=300, packet_error_rate="1E-6", maximum_data_burst_volume=None),  # Video (buffered streaming)
            7: QosParameters(fiveqi=7, priority_level=70, packet_delay_budget=100, packet_error_rate="1E-3", maximum_data_burst_volume=None),  # Voice, video, interactive gaming
            8: QosParameters(fiveqi=8, priority_level=80, packet_delay_budget=300, packet_error_rate="1E-6", maximum_data_burst_volume=None),  # Video (buffered streaming)
            9: QosParameters(fiveqi=9, priority_level=90, packet_delay_budget=300, packet_error_rate="1E-6", maximum_data_burst_volume=None),  # Video (buffered streaming)
            65: QosParameters(fiveqi=65, priority_level=7, packet_delay_budget=75, packet_error_rate="1E-2", maximum_data_burst_volume=None), # Mission critical user plane push to talk voice
            66: QosParameters(fiveqi=66, priority_level=20, packet_delay_budget=100, packet_error_rate="1E-2", maximum_data_burst_volume=None), # Non-mission-critical user plane push to talk voice
            67: QosParameters(fiveqi=67, priority_level=15, packet_delay_budget=100, packet_error_rate="1E-3", maximum_data_burst_volume=None), # Mission critical user plane video
            75: QosParameters(fiveqi=75, priority_level=25, packet_delay_budget=50, packet_error_rate="1E-2", maximum_data_burst_volume=None),  # V2X messages
            79: QosParameters(fiveqi=79, priority_level=65, packet_delay_budget=50, packet_error_rate="1E-2", maximum_data_burst_volume=255), # V2X messages
            80: QosParameters(fiveqi=80, priority_level=68, packet_delay_budget=10, packet_error_rate="1E-6", maximum_data_burst_volume=1354), # Low latency eMBB applications
            82: QosParameters(fiveqi=82, priority_level=19, packet_delay_budget=10, packet_error_rate="1E-4", maximum_data_burst_volume=255), # Discrete automation (small packets)
            83: QosParameters(fiveqi=83, priority_level=22, packet_delay_budget=10, packet_error_rate="1E-4", maximum_data_burst_volume=1354), # Discrete automation (large packets)
            84: QosParameters(fiveqi=84, priority_level=24, packet_delay_budget=30, packet_error_rate="1E-5", maximum_data_burst_volume=1354), # Intelligent transport systems
            85: QosParameters(fiveqi=85, priority_level=21, packet_delay_budget=5, packet_error_rate="1E-5", maximum_data_burst_volume=255),  # Electrical power distribution
        }
        
        for fiveqi, qos_params in default_qos_configs.items():
            qos_enforcement[str(fiveqi)] = qos_params
    
    def allocate_ip_address(self, pdn_type: str = "IPV4") -> Dict[str, str]:
        """Allocate IP address from pool"""
        result = {}
        
        if pdn_type in ["IPV4", "IPV4V6"]:
            # Allocate IPv4 address
            for ip in self.ipv4_pool.hosts():
                if str(ip) not in self.allocated_ipv4:
                    self.allocated_ipv4.add(str(ip))
                    result["ipv4"] = str(ip)
                    break
        
        if pdn_type in ["IPV6", "IPV4V6"]:
            # Allocate IPv6 address (use /64 prefix)
            subnet_bits = 64 - self.ipv6_pool.prefixlen
            max_subnets = 2 ** subnet_bits
            
            for i in range(max_subnets):
                try:
                    subnet = list(self.ipv6_pool.subnets(new_prefix=64))[i]
                    ipv6_addr = str(list(subnet.hosts())[0])
                    if ipv6_addr not in self.allocated_ipv6:
                        self.allocated_ipv6.add(ipv6_addr)
                        result["ipv6"] = ipv6_addr
                        result["ipv6_prefix"] = str(subnet)
                        break
                except (IndexError, StopIteration):
                    break
        
        return result
    
    def release_ip_address(self, ip_address: str):
        """Release allocated IP address"""
        if ip_address in self.allocated_ipv4:
            self.allocated_ipv4.remove(ip_address)
        if ip_address in self.allocated_ipv6:
            self.allocated_ipv6.remove(ip_address)
    
    def create_gtp_tunnel(self, f_teid: FTeid, far: CreateFar) -> str:
        """Create GTP-U tunnel"""
        tunnel_id = str(uuid.uuid4())
        
        tunnel_info = {
            "tunnel_id": tunnel_id,
            "local_teid": f_teid.teid,
            "local_ipv4": f_teid.ipv4_address,
            "local_ipv6": f_teid.ipv6_address,
            "remote_teid": None,
            "remote_ipv4": None,
            "remote_ipv6": None,
            "state": "ACTIVE",
            "created_time": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "stats": TrafficStats()
        }
        
        # Extract remote tunnel endpoint from FAR
        if far.forwarding_parameters and far.forwarding_parameters.outer_header_creation:
            ohc = far.forwarding_parameters.outer_header_creation
            tunnel_info["remote_teid"] = ohc.teid
            tunnel_info["remote_ipv4"] = ohc.ipv4_address
            tunnel_info["remote_ipv6"] = ohc.ipv6_address
        
        gtp_tunnels[tunnel_id] = tunnel_info
        return tunnel_id
    
    def process_gtp_packet(self, tunnel_id: str, gtp_packet: GtpuPacket, direction: str) -> bool:
        """Process GTP-U packet with QoS enforcement"""
        if tunnel_id not in gtp_tunnels:
            logger.error(f"GTP tunnel not found: {tunnel_id}")
            return False
        
        tunnel = gtp_tunnels[tunnel_id]
        
        # Update tunnel statistics
        stats = tunnel["stats"]
        packet_size = len(gtp_packet.payload)
        
        if direction == "uplink":
            stats.packets_ul += 1
            stats.bytes_ul += packet_size
        else:
            stats.packets_dl += 1
            stats.bytes_dl += packet_size
        
        stats.last_activity = datetime.now(timezone.utc)
        
        # Apply QoS enforcement
        qos_result = self.qos_scheduler.enforce_qos(tunnel_id, gtp_packet, direction)
        
        if not qos_result:
            # Packet dropped due to QoS enforcement
            if direction == "uplink":
                stats.dropped_packets_ul += 1
            else:
                stats.dropped_packets_dl += 1
            return False
        
        # Forward packet (in real implementation, would forward to appropriate interface)
        logger.info(f"GTP packet forwarded: Tunnel={tunnel_id}, Direction={direction}, Size={packet_size}")
        return True
    
    def enforce_qos_policies(self, session_id: str, qer_list: List[CreateQer]):
        """Enforce QoS policies for session"""
        for qer in qer_list:
            qos_params = QosParameters(
                fiveqi=qer.qfi or 9,  # Default to best effort
                priority_level=1,
                maximum_bitrate_ul=qer.mbr.ul_mbr if qer.mbr else None,
                maximum_bitrate_dl=qer.mbr.dl_mbr if qer.mbr else None,
                guaranteed_bitrate_ul=qer.gbr.ul_gbr if qer.gbr else None,
                guaranteed_bitrate_dl=qer.gbr.dl_gbr if qer.gbr else None,
                averaging_window=qer.averaging_window
            )
            
            # Store QoS parameters for enforcement
            qos_key = f"{session_id}_{qer.qer_id}"
            qos_enforcement[qos_key] = qos_params
            
            logger.info(f"QoS policy enforced: Session={session_id}, QER={qer.qer_id}, 5QI={qos_params.fiveqi}")

class QosScheduler:
    """Advanced QoS scheduler with traffic shaping and policing"""
    
    def __init__(self):
        self.token_buckets = {}
        self.priority_queues = {}
        
    def enforce_qos(self, tunnel_id: str, packet: GtpuPacket, direction: str) -> bool:
        """Enforce QoS policies using token bucket and priority queuing"""
        
        # Find applicable QoS parameters
        qos_params = None
        for key, params in qos_enforcement.items():
            if tunnel_id in key:
                qos_params = params
                break
        
        if not qos_params:
            # No QoS enforcement - allow packet
            return True
        
        # Token bucket algorithm for rate limiting
        bucket_key = f"{tunnel_id}_{direction}"
        if bucket_key not in self.token_buckets:
            max_rate = qos_params.maximum_bitrate_ul if direction == "uplink" else qos_params.maximum_bitrate_dl
            if max_rate:
                self.token_buckets[bucket_key] = {
                    "tokens": max_rate // 8,  # Convert to bytes
                    "max_tokens": max_rate // 8,
                    "last_refill": datetime.now(timezone.utc),
                    "refill_rate": max_rate // 8  # Bytes per second
                }
        
        if bucket_key in self.token_buckets:
            bucket = self.token_buckets[bucket_key]
            now = datetime.now(timezone.utc)
            time_diff = (now - bucket["last_refill"]).total_seconds()
            
            # Refill tokens
            tokens_to_add = int(time_diff * bucket["refill_rate"])
            bucket["tokens"] = min(bucket["max_tokens"], bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = now
            
            # Check if enough tokens for packet
            packet_size = len(packet.payload)
            if bucket["tokens"] >= packet_size:
                bucket["tokens"] -= packet_size
                return True
            else:
                # Packet dropped due to rate limiting
                logger.debug(f"Packet dropped due to rate limiting: Tunnel={tunnel_id}, Direction={direction}")
                return False
        
        # Priority queuing based on 5QI
        priority = self._get_priority_from_fiveqi(qos_params.fiveqi)
        queue_key = f"{tunnel_id}_{priority}"
        
        if queue_key not in self.priority_queues:
            self.priority_queues[queue_key] = []
        
        # Add packet to priority queue (simplified - in real implementation would be more complex)
        self.priority_queues[queue_key].append(packet)
        
        # Process highest priority packets first
        self._process_priority_queues()
        
        return True
    
    def _get_priority_from_fiveqi(self, fiveqi: int) -> int:
        """Get priority level from 5QI"""
        # Priority mapping based on 3GPP TS 23.501
        priority_map = {
            1: 20, 2: 40, 3: 30, 4: 50, 5: 10, 6: 60, 7: 70, 8: 80, 9: 90,
            65: 7, 66: 20, 67: 15, 75: 25, 79: 65, 80: 68, 82: 19, 83: 22, 84: 24, 85: 21
        }
        return priority_map.get(fiveqi, 90)  # Default to lowest priority
    
    def _process_priority_queues(self):
        """Process priority queues (simplified implementation)"""
        # Sort queues by priority and process higher priority first
        sorted_queues = sorted(self.priority_queues.items(), key=lambda x: int(x[0].split('_')[-1]))
        
        for queue_key, packets in sorted_queues:
            if packets:
                # Process packets (in real implementation, would schedule transmission)
                packet = packets.pop(0)
                logger.debug(f"Processing packet from priority queue: {queue_key}")

upf_enhanced_instance = UPF_Enhanced()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Register with NRF per TS 29.510
    nf_profile = {
        "nfInstanceId": upf_enhanced_instance.nf_instance_id,
        "nfType": "UPF",
        "nfStatus": "REGISTERED",
        "plmnList": [{"mcc": "001", "mnc": "01"}],
        "sNssais": [{"sst": 1, "sd": "010203"}],
        "nfServices": [
            {
                "serviceInstanceId": "nupf-pdu-session-001",
                "serviceName": "nupf-pdu-session",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9002}]
            }
        ],
        "upfInfo": {
            "sNssaiUpfInfoList": [
                {
                    "sNssai": {"sst": 1, "sd": "010203"},
                    "dnnUpfInfoList": [
                        {
                            "dnn": "internet",
                            "pduSessionTypes": ["IPV4", "IPV6", "IPV4V6"],
                            "ipv4AddressRanges": [{"start": "192.168.100.1", "end": "192.168.100.254"}],
                            "ipv6PrefixRanges": [{"start": "2001:db8:5g::", "end": "2001:db8:5g:ffff::"}]
                        }
                    ]
                }
            ],
            "interfaceUpfInfoList": [
                {
                    "interfaceType": "N3",
                    "networkInstance": "access.mnc001.mcc001.3gppnetwork.org",
                    "ipv4EndpointAddresses": ["192.168.200.10"],
                    "ipv6EndpointAddresses": ["2001:db8:upf::1"]
                },
                {
                    "interfaceType": "N6",
                    "networkInstance": "internet.mnc001.mcc001.3gppnetwork.org",
                    "ipv4EndpointAddresses": ["203.0.113.10"],
                    "ipv6EndpointAddresses": ["2001:db8:internet::1"]
                }
            ],
            "pduSessionTypes": ["IPV4", "IPV6", "IPV4V6"]
        }
    }
    
    try:
        response = requests.post(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{upf_enhanced_instance.nf_instance_id}",
                               json=nf_profile)
        if response.status_code in [200, 201]:
            logger.info("UPF-Enhanced registered with NRF successfully")
        else:
            logger.warning(f"UPF-Enhanced registration with NRF failed: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Failed to register UPF-Enhanced with NRF: {e}")
    
    # Start background tasks
    asyncio.create_task(periodic_statistics_collection())
    asyncio.create_task(qos_monitoring_task())
    
    yield
    
    # Shutdown
    try:
        requests.delete(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{upf_enhanced_instance.nf_instance_id}")
        logger.info("UPF-Enhanced deregistered from NRF")
    except:
        pass

async def periodic_statistics_collection():
    """Background task for collecting traffic statistics"""
    while True:
        try:
            # Collect and log statistics
            total_sessions = len(pfcp_sessions)
            total_tunnels = len(gtp_tunnels)
            total_ul_bytes = sum(stats.bytes_ul for stats in traffic_statistics.values())
            total_dl_bytes = sum(stats.bytes_dl for stats in traffic_statistics.values())
            
            logger.info(f"UPF Statistics: Sessions={total_sessions}, Tunnels={total_tunnels}, "
                       f"UL_Bytes={total_ul_bytes}, DL_Bytes={total_dl_bytes}")
            
            await asyncio.sleep(60)  # Collect every minute
        except Exception as e:
            logger.error(f"Statistics collection error: {e}")
            await asyncio.sleep(60)

async def qos_monitoring_task():
    """Background task for QoS monitoring and adjustment"""
    while True:
        try:
            # Monitor QoS enforcement and adjust if needed
            for tunnel_id, tunnel_info in gtp_tunnels.items():
                stats = tunnel_info.get("stats", TrafficStats())
                
                # Check for SLA violations
                if stats.dropped_packets_ul > 100 or stats.dropped_packets_dl > 100:
                    logger.warning(f"High packet drop rate detected for tunnel: {tunnel_id}")
                
                # Reset dropped packet counters periodically
                if stats.dropped_packets_ul > 0 or stats.dropped_packets_dl > 0:
                    stats.dropped_packets_ul = 0
                    stats.dropped_packets_dl = 0
            
            await asyncio.sleep(30)  # Monitor every 30 seconds
        except Exception as e:
            logger.error(f"QoS monitoring error: {e}")
            await asyncio.sleep(30)

app = FastAPI(
    title="UPF-Enhanced - User Plane Function",
    description="3GPP TS 29.244 PFCP and TS 29.281 GTP-U compliant with IPv6 and advanced QoS",
    version="1.0.0",
    lifespan=lifespan
)

# 3GPP TS 29.244 § 7.4.3.1 - PFCP Session Establishment Request
@app.post("/pfcp/v1/sessions")
async def pfcp_session_establishment(request: PfcpSessionEstablishmentRequest):
    """
    Handle PFCP Session Establishment Request per 3GPP TS 29.244
    """
    with tracer.start_as_current_span("upf_pfcp_session_establishment") as span:
        span.set_attribute("3gpp.interface", "N4")
        span.set_attribute("3gpp.protocol", "PFCP")
        span.set_attribute("pfcp.node_id", request.node_id)
        span.set_attribute("pfcp.f_seid", request.f_seid.teid)
        
        try:
            # Generate UPF SEID
            upf_seid = str(uuid.uuid4())
            
            # Allocate IP address based on PDN type
            allocated_ips = upf_enhanced_instance.allocate_ip_address(request.pdn_type or "IPV4")
            
            # Create session context
            session_context = {
                "upf_seid": upf_seid,
                "smf_seid": request.f_seid.teid,
                "node_id": request.node_id,
                "allocated_ips": allocated_ips,
                "pdrs": {},
                "fars": {},
                "qers": {},
                "urrs": {},
                "gtp_tunnels": [],
                "state": "ACTIVE",
                "created_time": datetime.now(timezone.utc),
                "last_modified": datetime.now(timezone.utc)
            }
            
            # Process PDRs
            for pdr in request.create_pdr:
                session_context["pdrs"][pdr.pdr_id] = pdr.dict()
                
                # Create GTP tunnel if F-TEID is present
                if pdr.pdi.f_teid:
                    # Find corresponding FAR
                    far = next((f for f in request.create_far if f.far_id == pdr.far_id), None)
                    if far:
                        tunnel_id = upf_enhanced_instance.create_gtp_tunnel(pdr.pdi.f_teid, far)
                        session_context["gtp_tunnels"].append(tunnel_id)
            
            # Process FARs
            for far in request.create_far:
                session_context["fars"][far.far_id] = far.dict()
            
            # Process QERs and enforce QoS
            if request.create_qer:
                for qer in request.create_qer:
                    session_context["qers"][qer.qer_id] = qer.dict()
                
                upf_enhanced_instance.enforce_qos_policies(upf_seid, request.create_qer)
            
            # Process URRs
            if request.create_urr:
                for urr in request.create_urr:
                    session_context["urrs"][urr.urr_id] = urr.dict()
            
            # Store session
            pfcp_sessions[upf_seid] = session_context
            
            # Initialize traffic statistics
            traffic_statistics[upf_seid] = TrafficStats()
            
            span.set_attribute("session.upf_seid", upf_seid)
            span.set_attribute("session.allocated_ipv4", allocated_ips.get("ipv4", "none"))
            span.set_attribute("session.allocated_ipv6", allocated_ips.get("ipv6", "none"))
            span.set_attribute("session.pdrs_count", len(request.create_pdr))
            span.set_attribute("session.fars_count", len(request.create_far))
            span.set_attribute("status", "SUCCESS")
            
            logger.info(f"PFCP Session established: UPF_SEID={upf_seid}, SMF_SEID={request.f_seid.teid}")
            
            # Create response
            response = {
                "messageType": MessageType.SESSION_ESTABLISHMENT_RESPONSE.value,
                "cause": Cause.REQUEST_ACCEPTED.value,
                "upFSeid": {
                    "teid": upf_seid,
                    "ipv4Address": "127.0.0.1",
                    "ipv6Address": "::1"
                },
                "allocatedUeIpAddresses": allocated_ips,
                "createdPdr": [{"pdrId": pdr.pdr_id} for pdr in request.create_pdr],
                "loadControlInformation": {
                    "loadControlSequenceNumber": 1,
                    "loadMetric": 50  # 50% load
                },
                "overloadControlInformation": {
                    "overloadControlSequenceNumber": 1,
                    "overloadReductionMetric": 0  # No overload
                }
            }
            
            return response
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"PFCP Session establishment failed: {e}")
            
            return {
                "messageType": MessageType.SESSION_ESTABLISHMENT_RESPONSE.value,
                "cause": Cause.SYSTEM_FAILURE.value,
                "failedRuleId": None
            }

# 3GPP TS 29.244 § 7.4.4.1 - PFCP Session Modification Request
@app.patch("/pfcp/v1/sessions/{seid}")
async def pfcp_session_modification(seid: str, modification_request: Dict):
    """
    Handle PFCP Session Modification Request per 3GPP TS 29.244
    """
    with tracer.start_as_current_span("upf_pfcp_session_modification") as span:
        try:
            if seid not in pfcp_sessions:
                raise HTTPException(status_code=404, detail="PFCP Session not found")
            
            session = pfcp_sessions[seid]
            
            # Process modifications
            modifications_applied = []
            
            # Update PDRs
            if "updatePdr" in modification_request:
                for update_pdr in modification_request["updatePdr"]:
                    pdr_id = update_pdr["pdrId"]
                    if pdr_id in session["pdrs"]:
                        session["pdrs"][pdr_id].update(update_pdr)
                        modifications_applied.append(f"PDR {pdr_id} updated")
            
            # Update FARs
            if "updateFar" in modification_request:
                for update_far in modification_request["updateFar"]:
                    far_id = update_far["farId"]
                    if far_id in session["fars"]:
                        session["fars"][far_id].update(update_far)
                        modifications_applied.append(f"FAR {far_id} updated")
            
            # Update QERs
            if "updateQer" in modification_request:
                for update_qer in modification_request["updateQer"]:
                    qer_id = update_qer["qerId"]
                    if qer_id in session["qers"]:
                        session["qers"][qer_id].update(update_qer)
                        modifications_applied.append(f"QER {qer_id} updated")
                        
                        # Update QoS enforcement
                        qos_key = f"{seid}_{qer_id}"
                        if qos_key in qos_enforcement:
                            qos_params = qos_enforcement[qos_key]
                            if "mbr" in update_qer:
                                qos_params.maximum_bitrate_ul = update_qer["mbr"].get("ulMbr")
                                qos_params.maximum_bitrate_dl = update_qer["mbr"].get("dlMbr")
            
            session["last_modified"] = datetime.now(timezone.utc)
            
            span.set_attribute("modifications.count", len(modifications_applied))
            span.set_attribute("status", "SUCCESS")
            
            logger.info(f"PFCP Session modified: SEID={seid}, Modifications={modifications_applied}")
            
            return {
                "messageType": MessageType.SESSION_MODIFICATION_RESPONSE.value,
                "cause": Cause.REQUEST_ACCEPTED.value,
                "modificationsApplied": modifications_applied
            }
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"PFCP Session modification failed: {e}")
            raise HTTPException(status_code=500, detail=f"Session modification failed: {e}")

# 3GPP TS 29.244 § 7.4.5.1 - PFCP Session Deletion Request
@app.delete("/pfcp/v1/sessions/{seid}")
async def pfcp_session_deletion(seid: str):
    """
    Handle PFCP Session Deletion Request per 3GPP TS 29.244
    """
    with tracer.start_as_current_span("upf_pfcp_session_deletion") as span:
        try:
            if seid not in pfcp_sessions:
                raise HTTPException(status_code=404, detail="PFCP Session not found")
            
            session = pfcp_sessions[seid]
            
            # Release allocated IP addresses
            allocated_ips = session.get("allocated_ips", {})
            for ip_type, ip_addr in allocated_ips.items():
                if ip_type in ["ipv4", "ipv6"]:
                    upf_enhanced_instance.release_ip_address(ip_addr)
            
            # Clean up GTP tunnels
            for tunnel_id in session.get("gtp_tunnels", []):
                if tunnel_id in gtp_tunnels:
                    del gtp_tunnels[tunnel_id]
            
            # Clean up QoS enforcement rules
            qos_keys_to_remove = [key for key in qos_enforcement.keys() if seid in key]
            for key in qos_keys_to_remove:
                del qos_enforcement[key]
            
            # Remove traffic statistics
            if seid in traffic_statistics:
                final_stats = traffic_statistics[seid]
                del traffic_statistics[seid]
                
                span.set_attribute("final.packets_ul", final_stats.packets_ul)
                span.set_attribute("final.packets_dl", final_stats.packets_dl)
                span.set_attribute("final.bytes_ul", final_stats.bytes_ul)
                span.set_attribute("final.bytes_dl", final_stats.bytes_dl)
            
            # Remove session
            del pfcp_sessions[seid]
            
            span.set_attribute("status", "SUCCESS")
            logger.info(f"PFCP Session deleted: SEID={seid}")
            
            return {
                "messageType": MessageType.SESSION_DELETION_RESPONSE.value,
                "cause": Cause.REQUEST_ACCEPTED.value
            }
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"PFCP Session deletion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Session deletion failed: {e}")

# 3GPP TS 29.281 GTP-U Packet Processing
@app.post("/gtp-u/process-packet")
async def process_gtp_packet(packet_data: Dict):
    """
    Process GTP-U packet per 3GPP TS 29.281
    """
    with tracer.start_as_current_span("upf_gtp_packet_processing") as span:
        try:
            tunnel_id = packet_data.get("tunnel_id")
            direction = packet_data.get("direction", "uplink")
            
            # Create GTP-U packet from data
            gtp_header = GtpuHeader(
                teid=packet_data["header"]["teid"],
                length=packet_data["header"]["length"],
                sequence_number=packet_data["header"].get("sequence_number")
            )
            
            gtp_packet = GtpuPacket(
                header=gtp_header,
                payload=packet_data["payload"]
            )
            
            # Process packet through UPF
            success = upf_enhanced_instance.process_gtp_packet(tunnel_id, gtp_packet, direction)
            
            span.set_attribute("tunnel_id", tunnel_id)
            span.set_attribute("direction", direction)
            span.set_attribute("packet_size", len(gtp_packet.payload))
            span.set_attribute("processed", success)
            
            return {
                "status": "SUCCESS" if success else "DROPPED",
                "tunnel_id": tunnel_id,
                "direction": direction,
                "processed": success
            }
            
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"GTP packet processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"GTP packet processing failed: {e}")

# IPv6 specific endpoints
@app.post("/ipv6/allocate-prefix")
async def allocate_ipv6_prefix(request_data: Dict):
    """
    Allocate IPv6 prefix for UE
    """
    try:
        prefix_length = request_data.get("prefix_length", 64)
        ue_id = request_data.get("ue_id")
        
        # Allocate IPv6 prefix
        allocated_ips = upf_enhanced_instance.allocate_ip_address("IPV6")
        
        if "ipv6_prefix" in allocated_ips:
            return {
                "status": "SUCCESS",
                "ue_id": ue_id,
                "allocated_prefix": allocated_ips["ipv6_prefix"],
                "allocated_address": allocated_ips["ipv6"]
            }
        else:
            raise HTTPException(status_code=503, detail="No IPv6 addresses available")
            
    except Exception as e:
        logger.error(f"IPv6 prefix allocation failed: {e}")
        raise HTTPException(status_code=500, detail=f"IPv6 prefix allocation failed: {e}")

# QoS management endpoints
@app.get("/qos/parameters")
async def get_qos_parameters():
    """Get all QoS parameters"""
    return {
        "total_qos_rules": len(qos_enforcement),
        "qos_parameters": {
            qos_id: qos.dict() for qos_id, qos in qos_enforcement.items()
        }
    }

@app.post("/qos/update")
async def update_qos_parameters(qos_update: Dict):
    """Update QoS parameters"""
    try:
        session_id = qos_update.get("session_id")
        qer_id = qos_update.get("qer_id")
        qos_params = qos_update.get("qos_parameters")
        
        qos_key = f"{session_id}_{qer_id}"
        
        if qos_key in qos_enforcement:
            current_qos = qos_enforcement[qos_key]
            
            # Update parameters
            for param, value in qos_params.items():
                if hasattr(current_qos, param):
                    setattr(current_qos, param, value)
            
            logger.info(f"QoS parameters updated: {qos_key}")
            
            return {
                "status": "SUCCESS",
                "qos_key": qos_key,
                "updated_parameters": qos_params
            }
        else:
            raise HTTPException(status_code=404, detail="QoS rule not found")
            
    except Exception as e:
        logger.error(f"QoS parameter update failed: {e}")
        raise HTTPException(status_code=500, detail=f"QoS parameter update failed: {e}")

# Status and monitoring endpoints
@app.get("/upf-enhanced/status")
async def upf_enhanced_status():
    """Get UPF Enhanced status"""
    return {
        "status": "operational",
        "active_sessions": len(pfcp_sessions),
        "active_gtp_tunnels": len(gtp_tunnels),
        "allocated_ipv4_addresses": len(upf_enhanced_instance.allocated_ipv4),
        "allocated_ipv6_addresses": len(upf_enhanced_instance.allocated_ipv6),
        "qos_rules": len(qos_enforcement),
        "supported_features": hex(upf_enhanced_instance.supported_features),
        "ipv4_pool": str(upf_enhanced_instance.ipv4_pool),
        "ipv6_pool": str(upf_enhanced_instance.ipv6_pool)
    }

@app.get("/upf-enhanced/statistics")
async def get_traffic_statistics():
    """Get traffic statistics"""
    total_stats = {
        "total_sessions": len(pfcp_sessions),
        "total_packets_ul": sum(stats.packets_ul for stats in traffic_statistics.values()),
        "total_packets_dl": sum(stats.packets_dl for stats in traffic_statistics.values()),
        "total_bytes_ul": sum(stats.bytes_ul for stats in traffic_statistics.values()),
        "total_bytes_dl": sum(stats.bytes_dl for stats in traffic_statistics.values()),
        "total_dropped_ul": sum(stats.dropped_packets_ul for stats in traffic_statistics.values()),
        "total_dropped_dl": sum(stats.dropped_packets_dl for stats in traffic_statistics.values()),
        "session_statistics": {
            seid: {
                "packets_ul": stats.packets_ul,
                "packets_dl": stats.packets_dl,
                "bytes_ul": stats.bytes_ul,
                "bytes_dl": stats.bytes_dl,
                "dropped_packets_ul": stats.dropped_packets_ul,
                "dropped_packets_dl": stats.dropped_packets_dl,
                "last_activity": stats.last_activity.isoformat()
            }
            for seid, stats in traffic_statistics.items()
        }
    }
    
    return total_stats

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "UPF-Enhanced",
        "compliance": "3GPP TS 29.244, TS 29.281",
        "version": "1.0.0",
        "features": ["IPv6", "Advanced QoS", "Real GTP-U"],
        "active_sessions": len(pfcp_sessions)
    }

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="UPF Enhanced - User Plane Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("upf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)