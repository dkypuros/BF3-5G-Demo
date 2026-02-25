"""
PFCP Protocol Implementation (TS 29.244)

This module implements the Packet Forwarding Control Protocol (PFCP) for
communication between SMF and UPF (N4 interface) as specified in 3GPP TS 29.244.

PFCP is a binary protocol running over UDP (port 8805).

Key Features:
- Binary message encoding/decoding
- All PFCP message types
- Information Element (IE) handling
- Session management
- Node management

Reference: 3GPP TS 29.244 V17.7.0 (2023-03)
"""

import struct
import socket
import asyncio
from enum import IntEnum
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol Constants (TS 29.244 Section 7)
# =============================================================================

PFCP_PORT = 8805
PFCP_VERSION = 1


class MessageType(IntEnum):
    """PFCP Message Types (TS 29.244 Section 7.2.3.1)"""
    # Node Related Messages
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

    # Session Related Messages
    SESSION_ESTABLISHMENT_REQUEST = 50
    SESSION_ESTABLISHMENT_RESPONSE = 51
    SESSION_MODIFICATION_REQUEST = 52
    SESSION_MODIFICATION_RESPONSE = 53
    SESSION_DELETION_REQUEST = 54
    SESSION_DELETION_RESPONSE = 55
    SESSION_REPORT_REQUEST = 56
    SESSION_REPORT_RESPONSE = 57


class IEType(IntEnum):
    """Information Element Types (TS 29.244 Section 8.1)"""
    # Basic IEs
    CREATE_PDR = 1
    PDI = 2
    CREATE_FAR = 3
    FORWARDING_PARAMETERS = 4
    DUPLICATING_PARAMETERS = 5
    CREATE_URR = 6
    CREATE_QER = 7
    CREATED_PDR = 8
    UPDATE_PDR = 9
    UPDATE_FAR = 10
    UPDATE_FORWARDING_PARAMETERS = 11
    UPDATE_BAR_PFCP_SESSION_REPORT_RESPONSE = 12
    UPDATE_URR = 13
    UPDATE_QER = 14
    REMOVE_PDR = 15
    REMOVE_FAR = 16
    REMOVE_URR = 17
    REMOVE_QER = 18

    # Cause and Other IEs
    CAUSE = 19
    SOURCE_INTERFACE = 20
    F_TEID = 21
    NETWORK_INSTANCE = 22
    SDF_FILTER = 23
    APPLICATION_ID = 24
    GATE_STATUS = 25
    MBR = 26
    GBR = 27
    QER_CORRELATION_ID = 28
    PRECEDENCE = 29
    TRANSPORT_LEVEL_MARKING = 30
    VOLUME_THRESHOLD = 31
    TIME_THRESHOLD = 32
    MONITORING_TIME = 33
    SUBSEQUENT_VOLUME_THRESHOLD = 34
    SUBSEQUENT_TIME_THRESHOLD = 35
    INACTIVITY_DETECTION_TIME = 36
    REPORTING_TRIGGERS = 37
    REDIRECT_INFORMATION = 38
    REPORT_TYPE = 39
    OFFENDING_IE = 40
    FORWARDING_POLICY = 41
    DESTINATION_INTERFACE = 42
    UP_FUNCTION_FEATURES = 43
    APPLY_ACTION = 44
    DOWNLINK_DATA_SERVICE_INFORMATION = 45
    DOWNLINK_DATA_NOTIFICATION_DELAY = 46
    DL_BUFFERING_DURATION = 47
    DL_BUFFERING_SUGGESTED_PACKET_COUNT = 48
    PFCPSMREQ_FLAGS = 49
    PFCPSRRSP_FLAGS = 50
    LOAD_CONTROL_INFORMATION = 51
    SEQUENCE_NUMBER = 52
    METRIC = 53
    OVERLOAD_CONTROL_INFORMATION = 54
    TIMER = 55
    PDR_ID = 56
    F_SEID = 57
    APPLICATION_ID_PFDS = 58
    PFD_CONTEXT = 59
    NODE_ID = 60
    PFD_CONTENTS = 61
    MEASUREMENT_METHOD = 62
    USAGE_REPORT_TRIGGER = 63
    MEASUREMENT_PERIOD = 64
    FQ_CSID = 65
    VOLUME_MEASUREMENT = 66
    DURATION_MEASUREMENT = 67
    APPLICATION_DETECTION_INFORMATION = 68
    TIME_OF_FIRST_PACKET = 69
    TIME_OF_LAST_PACKET = 70
    QUOTA_HOLDING_TIME = 71
    DROPPED_DL_TRAFFIC_THRESHOLD = 72
    VOLUME_QUOTA = 73
    TIME_QUOTA = 74
    START_TIME = 75
    END_TIME = 76
    QUERY_URR = 77
    USAGE_REPORT_SMR = 78
    USAGE_REPORT_SDR = 79
    USAGE_REPORT_SRR = 80
    URR_ID = 81
    LINKED_URR_ID = 82
    DOWNLINK_DATA_REPORT = 83
    OUTER_HEADER_CREATION = 84
    CREATE_BAR = 85
    UPDATE_BAR_SESSION_MODIFICATION_REQUEST = 86
    REMOVE_BAR = 87
    BAR_ID = 88
    CP_FUNCTION_FEATURES = 89
    USAGE_INFORMATION = 90
    APPLICATION_INSTANCE_ID = 91
    FLOW_INFORMATION = 92
    UE_IP_ADDRESS = 93
    PACKET_RATE = 94
    OUTER_HEADER_REMOVAL = 95
    RECOVERY_TIME_STAMP = 96
    DL_FLOW_LEVEL_MARKING = 97
    HEADER_ENRICHMENT = 98
    ERROR_INDICATION_REPORT = 99
    MEASUREMENT_INFORMATION = 100
    NODE_REPORT_TYPE = 101
    USER_PLANE_PATH_FAILURE_REPORT = 102
    REMOTE_GTP_U_PEER = 103
    UR_SEQN = 104
    UPDATE_DUPLICATING_PARAMETERS = 105
    ACTIVATE_PREDEFINED_RULES = 106
    DEACTIVATE_PREDEFINED_RULES = 107
    FAR_ID = 108
    QER_ID = 109
    OCI_FLAGS = 110
    PFCP_ASSOCIATION_RELEASE_REQUEST = 111
    GRACEFUL_RELEASE_PERIOD = 112
    PDN_TYPE = 113
    FAILED_RULE_ID = 114
    TIME_QUOTA_MECHANISM = 115
    USER_PLANE_IP_RESOURCE_INFORMATION = 116
    USER_PLANE_INACTIVITY_TIMER = 117
    AGGREGATED_URRS = 118
    MULTIPLIER = 119
    AGGREGATED_URR_ID = 120
    SUBSEQUENT_VOLUME_QUOTA = 121
    SUBSEQUENT_TIME_QUOTA = 122
    RQI = 123
    QFI = 124
    QUERY_URR_REFERENCE = 125
    ADDITIONAL_USAGE_REPORTS_INFORMATION = 126
    CREATE_TRAFFIC_ENDPOINT = 127
    CREATED_TRAFFIC_ENDPOINT = 128
    UPDATE_TRAFFIC_ENDPOINT = 129
    REMOVE_TRAFFIC_ENDPOINT = 130
    TRAFFIC_ENDPOINT_ID = 131
    ETHERNET_PACKET_FILTER = 132
    MAC_ADDRESS = 133
    C_TAG = 134
    S_TAG = 135
    ETHERTYPE = 136
    PROXYING = 137
    ETHERNET_FILTER_ID = 138
    ETHERNET_FILTER_PROPERTIES = 139
    SUGGESTED_BUFFERING_PACKETS_COUNT = 140
    USER_ID = 141
    ETHERNET_PDU_SESSION_INFORMATION = 142
    ETHERNET_TRAFFIC_INFORMATION = 143
    MAC_ADDRESSES_DETECTED = 144
    MAC_ADDRESSES_REMOVED = 145
    ETHERNET_INACTIVITY_TIMER = 146
    ADDITIONAL_MONITORING_TIME = 147
    EVENT_QUOTA = 148
    EVENT_THRESHOLD = 149
    SUBSEQUENT_EVENT_QUOTA = 150
    SUBSEQUENT_EVENT_THRESHOLD = 151
    TRACE_INFORMATION = 152
    FRAMED_ROUTE = 153
    FRAMED_ROUTING = 154
    FRAMED_IPV6_ROUTE = 155
    TIME_STAMP = 156
    AVERAGING_WINDOW = 157
    PAGING_POLICY_INDICATOR = 158
    APN_DNN = 159
    TGPP_INTERFACE_TYPE = 160
    PFCPSRREQ_FLAGS = 161


class CauseValue(IntEnum):
    """Cause Values (TS 29.244 Section 8.2.1)"""
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


class SourceInterface(IntEnum):
    """Source Interface Values (TS 29.244 Section 8.2.2)"""
    ACCESS = 0
    CORE = 1
    SGI_LAN_N6_LAN = 2
    CP_FUNCTION = 3
    LI_FUNCTION = 4


class DestinationInterface(IntEnum):
    """Destination Interface Values (TS 29.244 Section 8.2.24)"""
    ACCESS = 0
    CORE = 1
    SGI_LAN_N6_LAN = 2
    CP_FUNCTION = 3
    LI_FUNCTION = 4


class ApplyAction(IntEnum):
    """Apply Action Flags (TS 29.244 Section 8.2.26)"""
    DROP = 0x01
    FORW = 0x02  # Forward
    BUFF = 0x04  # Buffer
    NOCP = 0x08  # Notify CP function
    DUPL = 0x10  # Duplicate
    IPMA = 0x20  # IP Multicast Accept
    IPMD = 0x40  # IP Multicast Deny


class GateStatus(IntEnum):
    """Gate Status Values (TS 29.244 Section 8.2.9)"""
    OPEN = 0
    CLOSED = 1


# =============================================================================
# Information Element Classes
# =============================================================================

@dataclass
class InformationElement:
    """Base class for PFCP Information Elements"""
    ie_type: int
    length: int
    value: bytes

    def encode(self) -> bytes:
        """Encode IE to bytes"""
        return struct.pack('>HH', self.ie_type, self.length) + self.value

    @classmethod
    def decode(cls, data: bytes) -> Tuple['InformationElement', int]:
        """Decode IE from bytes"""
        if len(data) < 4:
            raise ValueError("Insufficient data for IE header")
        ie_type, length = struct.unpack('>HH', data[:4])
        value = data[4:4+length]
        return cls(ie_type, length, value), 4 + length


@dataclass
class NodeId:
    """Node ID IE (TS 29.244 Section 8.2.38)"""
    node_id_type: int  # 0=IPv4, 1=IPv6, 2=FQDN
    node_id: Union[str, bytes]

    def encode(self) -> bytes:
        """Encode Node ID IE"""
        if self.node_id_type == 0:  # IPv4
            ip_bytes = socket.inet_aton(self.node_id)
            value = bytes([self.node_id_type]) + ip_bytes
        elif self.node_id_type == 1:  # IPv6
            ip_bytes = socket.inet_pton(socket.AF_INET6, self.node_id)
            value = bytes([self.node_id_type]) + ip_bytes
        else:  # FQDN
            fqdn_bytes = self.node_id.encode('utf-8')
            value = bytes([self.node_id_type]) + fqdn_bytes

        return struct.pack('>HH', IEType.NODE_ID, len(value)) + value

    @classmethod
    def decode(cls, data: bytes) -> 'NodeId':
        """Decode Node ID from IE value"""
        node_id_type = data[0]
        if node_id_type == 0:
            node_id = socket.inet_ntoa(data[1:5])
        elif node_id_type == 1:
            node_id = socket.inet_ntop(socket.AF_INET6, data[1:17])
        else:
            node_id = data[1:].decode('utf-8')
        return cls(node_id_type, node_id)


@dataclass
class FSEID:
    """F-SEID IE (TS 29.244 Section 8.2.37)"""
    seid: int           # 8 bytes
    ipv4_address: Optional[str] = None
    ipv6_address: Optional[str] = None

    def encode(self) -> bytes:
        """Encode F-SEID IE"""
        flags = 0
        value = b''

        if self.ipv4_address:
            flags |= 0x02
        if self.ipv6_address:
            flags |= 0x01

        value = struct.pack('>BQ', flags, self.seid)

        if self.ipv4_address:
            value += socket.inet_aton(self.ipv4_address)
        if self.ipv6_address:
            value += socket.inet_pton(socket.AF_INET6, self.ipv6_address)

        return struct.pack('>HH', IEType.F_SEID, len(value)) + value

    @classmethod
    def decode(cls, data: bytes) -> 'FSEID':
        """Decode F-SEID from IE value"""
        flags = data[0]
        seid = struct.unpack('>Q', data[1:9])[0]
        offset = 9

        ipv4 = None
        ipv6 = None

        if flags & 0x02:
            ipv4 = socket.inet_ntoa(data[offset:offset+4])
            offset += 4
        if flags & 0x01:
            ipv6 = socket.inet_ntop(socket.AF_INET6, data[offset:offset+16])

        return cls(seid, ipv4, ipv6)


@dataclass
class FTEID:
    """F-TEID IE (TS 29.244 Section 8.2.3)"""
    teid: int           # 4 bytes
    ipv4_address: Optional[str] = None
    ipv6_address: Optional[str] = None
    choose_id: Optional[int] = None

    def encode(self) -> bytes:
        """Encode F-TEID IE"""
        flags = 0

        if self.choose_id is not None:
            flags |= 0x04  # CH bit
        if self.ipv4_address:
            flags |= 0x02  # V4 bit
        if self.ipv6_address:
            flags |= 0x01  # V6 bit

        value = bytes([flags])
        value += struct.pack('>I', self.teid)

        if self.ipv4_address:
            value += socket.inet_aton(self.ipv4_address)
        if self.ipv6_address:
            value += socket.inet_pton(socket.AF_INET6, self.ipv6_address)
        if self.choose_id is not None:
            value += bytes([self.choose_id])

        return struct.pack('>HH', IEType.F_TEID, len(value)) + value

    @classmethod
    def decode(cls, data: bytes) -> 'FTEID':
        """Decode F-TEID from IE value"""
        flags = data[0]
        teid = struct.unpack('>I', data[1:5])[0]
        offset = 5

        ipv4 = None
        ipv6 = None
        choose_id = None

        if flags & 0x02:
            ipv4 = socket.inet_ntoa(data[offset:offset+4])
            offset += 4
        if flags & 0x01:
            ipv6 = socket.inet_ntop(socket.AF_INET6, data[offset:offset+16])
            offset += 16
        if flags & 0x04:
            choose_id = data[offset]

        return cls(teid, ipv4, ipv6, choose_id)


@dataclass
class UEIPAddress:
    """UE IP Address IE (TS 29.244 Section 8.2.62)"""
    source_destination: int  # 0=source, 1=destination
    ipv4_address: Optional[str] = None
    ipv6_address: Optional[str] = None
    ipv6_prefix_length: Optional[int] = None

    def encode(self) -> bytes:
        """Encode UE IP Address IE"""
        flags = 0

        if self.source_destination:
            flags |= 0x04  # S/D bit
        if self.ipv4_address:
            flags |= 0x02  # V4 bit
        if self.ipv6_address:
            flags |= 0x01  # V6 bit

        value = bytes([flags])

        if self.ipv4_address:
            value += socket.inet_aton(self.ipv4_address)
        if self.ipv6_address:
            value += socket.inet_pton(socket.AF_INET6, self.ipv6_address)
            if self.ipv6_prefix_length:
                value += bytes([self.ipv6_prefix_length])

        return struct.pack('>HH', IEType.UE_IP_ADDRESS, len(value)) + value


@dataclass
class PDRId:
    """PDR ID IE (TS 29.244 Section 8.2.36)"""
    rule_id: int  # 2 bytes

    def encode(self) -> bytes:
        value = struct.pack('>H', self.rule_id)
        return struct.pack('>HH', IEType.PDR_ID, 2) + value


@dataclass
class FARId:
    """FAR ID IE (TS 29.244 Section 8.2.74)"""
    far_id: int  # 4 bytes

    def encode(self) -> bytes:
        value = struct.pack('>I', self.far_id)
        return struct.pack('>HH', IEType.FAR_ID, 4) + value


@dataclass
class QERId:
    """QER ID IE (TS 29.244 Section 8.2.75)"""
    qer_id: int  # 4 bytes

    def encode(self) -> bytes:
        value = struct.pack('>I', self.qer_id)
        return struct.pack('>HH', IEType.QER_ID, 4) + value


@dataclass
class URRId:
    """URR ID IE (TS 29.244 Section 8.2.54)"""
    urr_id: int  # 4 bytes

    def encode(self) -> bytes:
        value = struct.pack('>I', self.urr_id)
        return struct.pack('>HH', IEType.URR_ID, 4) + value


@dataclass
class Precedence:
    """Precedence IE (TS 29.244 Section 8.2.11)"""
    precedence: int  # 4 bytes, lower = higher priority

    def encode(self) -> bytes:
        value = struct.pack('>I', self.precedence)
        return struct.pack('>HH', IEType.PRECEDENCE, 4) + value


@dataclass
class ApplyActionIE:
    """Apply Action IE (TS 29.244 Section 8.2.26)"""
    flags: int  # Combination of ApplyAction enum values

    def encode(self) -> bytes:
        # Apply Action is 2 bytes in later releases
        value = struct.pack('>H', self.flags)
        return struct.pack('>HH', IEType.APPLY_ACTION, 2) + value


@dataclass
class OuterHeaderCreation:
    """Outer Header Creation IE (TS 29.244 Section 8.2.56)"""
    description: int      # 2 bytes (flags)
    teid: Optional[int] = None
    ipv4_address: Optional[str] = None
    ipv6_address: Optional[str] = None
    port_number: Optional[int] = None

    def encode(self) -> bytes:
        value = struct.pack('>H', self.description)

        if self.teid is not None:
            value += struct.pack('>I', self.teid)
        if self.ipv4_address:
            value += socket.inet_aton(self.ipv4_address)
        if self.ipv6_address:
            value += socket.inet_pton(socket.AF_INET6, self.ipv6_address)
        if self.port_number:
            value += struct.pack('>H', self.port_number)

        return struct.pack('>HH', IEType.OUTER_HEADER_CREATION, len(value)) + value


@dataclass
class MBR:
    """MBR (Maximum Bit Rate) IE (TS 29.244 Section 8.2.8)"""
    ul_mbr: int  # kbps, 5 bytes
    dl_mbr: int  # kbps, 5 bytes

    def encode(self) -> bytes:
        # Each rate is 5 bytes (40 bits)
        ul_bytes = struct.pack('>Q', self.ul_mbr)[3:]  # Take lower 5 bytes
        dl_bytes = struct.pack('>Q', self.dl_mbr)[3:]
        value = ul_bytes + dl_bytes
        return struct.pack('>HH', IEType.MBR, 10) + value


@dataclass
class GBR:
    """GBR (Guaranteed Bit Rate) IE (TS 29.244 Section 8.2.10)"""
    ul_gbr: int  # kbps, 5 bytes
    dl_gbr: int  # kbps, 5 bytes

    def encode(self) -> bytes:
        ul_bytes = struct.pack('>Q', self.ul_gbr)[3:]
        dl_bytes = struct.pack('>Q', self.dl_gbr)[3:]
        value = ul_bytes + dl_bytes
        return struct.pack('>HH', IEType.GBR, 10) + value


@dataclass
class GateStatusIE:
    """Gate Status IE (TS 29.244 Section 8.2.25)"""
    ul_gate: GateStatus
    dl_gate: GateStatus

    def encode(self) -> bytes:
        value = bytes([(self.dl_gate << 2) | self.ul_gate])
        return struct.pack('>HH', IEType.GATE_STATUS, 1) + value


@dataclass
class QFI_IE:
    """QFI IE (TS 29.244 Section 8.2.89)"""
    qfi: int  # 6 bits

    def encode(self) -> bytes:
        value = bytes([self.qfi & 0x3F])
        return struct.pack('>HH', IEType.QFI, 1) + value


@dataclass
class Cause:
    """Cause IE (TS 29.244 Section 8.2.1)"""
    cause: CauseValue

    def encode(self) -> bytes:
        value = bytes([self.cause])
        return struct.pack('>HH', IEType.CAUSE, 1) + value

    @classmethod
    def decode(cls, data: bytes) -> 'Cause':
        return cls(CauseValue(data[0]))


@dataclass
class RecoveryTimeStamp:
    """Recovery Time Stamp IE (TS 29.244 Section 8.2.65)"""
    timestamp: int  # NTP timestamp (seconds since 1900)

    def encode(self) -> bytes:
        value = struct.pack('>I', self.timestamp)
        return struct.pack('>HH', IEType.RECOVERY_TIME_STAMP, 4) + value

    @classmethod
    def decode(cls, data: bytes) -> 'RecoveryTimeStamp':
        timestamp = struct.unpack('>I', data[:4])[0]
        return cls(timestamp)


# =============================================================================
# Grouped IEs (Create PDR, Create FAR, etc.)
# =============================================================================

@dataclass
class PDI:
    """Packet Detection Information grouped IE (TS 29.244 Section 8.2.137)"""
    source_interface: SourceInterface
    local_fteid: Optional[FTEID] = None
    ue_ip_address: Optional[UEIPAddress] = None
    network_instance: Optional[str] = None
    sdf_filter: Optional[bytes] = None
    application_id: Optional[str] = None
    qfi: Optional[int] = None

    def encode(self) -> bytes:
        """Encode PDI grouped IE"""
        content = b''

        # Source Interface (mandatory)
        src_value = struct.pack('>HHB', IEType.SOURCE_INTERFACE, 1, self.source_interface)
        content += src_value

        # Local F-TEID (optional)
        if self.local_fteid:
            content += self.local_fteid.encode()

        # UE IP Address (optional)
        if self.ue_ip_address:
            content += self.ue_ip_address.encode()

        # Network Instance (optional)
        if self.network_instance:
            ni_bytes = self.network_instance.encode('utf-8')
            content += struct.pack('>HH', IEType.NETWORK_INSTANCE, len(ni_bytes)) + ni_bytes

        # QFI (optional)
        if self.qfi is not None:
            content += QFI_IE(self.qfi).encode()

        return struct.pack('>HH', IEType.PDI, len(content)) + content


@dataclass
class ForwardingParameters:
    """Forwarding Parameters grouped IE (TS 29.244 Section 8.2.138)"""
    destination_interface: DestinationInterface
    network_instance: Optional[str] = None
    outer_header_creation: Optional[OuterHeaderCreation] = None

    def encode(self) -> bytes:
        """Encode Forwarding Parameters grouped IE"""
        content = b''

        # Destination Interface (mandatory)
        dst_value = struct.pack('>HHB', IEType.DESTINATION_INTERFACE, 1, self.destination_interface)
        content += dst_value

        # Network Instance (optional)
        if self.network_instance:
            ni_bytes = self.network_instance.encode('utf-8')
            content += struct.pack('>HH', IEType.NETWORK_INSTANCE, len(ni_bytes)) + ni_bytes

        # Outer Header Creation (optional)
        if self.outer_header_creation:
            content += self.outer_header_creation.encode()

        return struct.pack('>HH', IEType.FORWARDING_PARAMETERS, len(content)) + content


@dataclass
class CreatePDR:
    """Create PDR grouped IE (TS 29.244 Section 7.5.2.2)"""
    pdr_id: int
    precedence: int
    pdi: PDI
    outer_header_removal: Optional[int] = None
    far_id: Optional[int] = None
    urr_ids: List[int] = field(default_factory=list)
    qer_id: Optional[int] = None

    def encode(self) -> bytes:
        """Encode Create PDR grouped IE"""
        content = b''

        # PDR ID (mandatory)
        content += PDRId(self.pdr_id).encode()

        # Precedence (mandatory)
        content += Precedence(self.precedence).encode()

        # PDI (mandatory)
        content += self.pdi.encode()

        # Outer Header Removal (optional)
        if self.outer_header_removal is not None:
            content += struct.pack('>HHB', IEType.OUTER_HEADER_REMOVAL, 1, self.outer_header_removal)

        # FAR ID (optional)
        if self.far_id is not None:
            content += FARId(self.far_id).encode()

        # URR IDs (optional)
        for urr_id in self.urr_ids:
            content += URRId(urr_id).encode()

        # QER ID (optional)
        if self.qer_id is not None:
            content += QERId(self.qer_id).encode()

        return struct.pack('>HH', IEType.CREATE_PDR, len(content)) + content


@dataclass
class CreateFAR:
    """Create FAR grouped IE (TS 29.244 Section 7.5.2.3)"""
    far_id: int
    apply_action: int
    forwarding_parameters: Optional[ForwardingParameters] = None
    bar_id: Optional[int] = None

    def encode(self) -> bytes:
        """Encode Create FAR grouped IE"""
        content = b''

        # FAR ID (mandatory)
        content += FARId(self.far_id).encode()

        # Apply Action (mandatory)
        content += ApplyActionIE(self.apply_action).encode()

        # Forwarding Parameters (conditional)
        if self.forwarding_parameters:
            content += self.forwarding_parameters.encode()

        # BAR ID (optional)
        if self.bar_id is not None:
            content += struct.pack('>HHB', IEType.BAR_ID, 1, self.bar_id)

        return struct.pack('>HH', IEType.CREATE_FAR, len(content)) + content


@dataclass
class CreateQER:
    """Create QER grouped IE (TS 29.244 Section 7.5.2.5)"""
    qer_id: int
    gate_status_ul: GateStatus = GateStatus.OPEN
    gate_status_dl: GateStatus = GateStatus.OPEN
    mbr: Optional[MBR] = None
    gbr: Optional[GBR] = None
    qfi: Optional[int] = None

    def encode(self) -> bytes:
        """Encode Create QER grouped IE"""
        content = b''

        # QER ID (mandatory)
        content += QERId(self.qer_id).encode()

        # Gate Status (mandatory)
        content += GateStatusIE(self.gate_status_ul, self.gate_status_dl).encode()

        # MBR (optional)
        if self.mbr:
            content += self.mbr.encode()

        # GBR (optional)
        if self.gbr:
            content += self.gbr.encode()

        # QFI (optional)
        if self.qfi is not None:
            content += QFI_IE(self.qfi).encode()

        return struct.pack('>HH', IEType.CREATE_QER, len(content)) + content


# =============================================================================
# PFCP Message Classes
# =============================================================================

@dataclass
class PFCPHeader:
    """PFCP Message Header (TS 29.244 Section 7.2.2)"""
    version: int = PFCP_VERSION
    mp: int = 0           # Message Priority
    s: int = 0            # SEID flag
    message_type: int = 0
    length: int = 0
    seid: Optional[int] = None
    sequence_number: int = 0
    spare: int = 0

    def encode(self) -> bytes:
        """Encode PFCP header"""
        # First byte: Version (3 bits) + Spare (2 bits) + MP (1 bit) + S (1 bit) + Spare (1 bit)
        first_byte = ((self.version & 0x07) << 5) | ((self.mp & 0x01) << 1) | (self.s & 0x01)

        if self.s and self.seid is not None:
            # Header with SEID (16 bytes)
            return struct.pack('>BBHQIH',
                first_byte,
                self.message_type,
                self.length,
                self.seid,
                self.sequence_number,
                self.spare
            )
        else:
            # Header without SEID (8 bytes)
            return struct.pack('>BBHIH',
                first_byte,
                self.message_type,
                self.length,
                self.sequence_number,
                self.spare
            )

    @classmethod
    def decode(cls, data: bytes) -> Tuple['PFCPHeader', int]:
        """Decode PFCP header"""
        if len(data) < 8:
            raise ValueError("Insufficient data for PFCP header")

        first_byte = data[0]
        version = (first_byte >> 5) & 0x07
        mp = (first_byte >> 1) & 0x01
        s = first_byte & 0x01

        message_type = data[1]
        length = struct.unpack('>H', data[2:4])[0]

        if s:
            # Header with SEID
            if len(data) < 16:
                raise ValueError("Insufficient data for PFCP header with SEID")
            seid = struct.unpack('>Q', data[4:12])[0]
            seq_num = struct.unpack('>I', data[12:16])[0] >> 8
            return cls(version, mp, s, message_type, length, seid, seq_num), 16
        else:
            # Header without SEID
            seq_num = struct.unpack('>I', data[4:8])[0] >> 8
            return cls(version, mp, s, message_type, length, None, seq_num), 8


@dataclass
class HeartbeatRequest:
    """Heartbeat Request Message (TS 29.244 Section 7.4.2.1)"""
    recovery_time_stamp: int

    def encode(self, sequence_number: int) -> bytes:
        """Encode Heartbeat Request"""
        # IEs
        ie_content = RecoveryTimeStamp(self.recovery_time_stamp).encode()

        # Header
        header = PFCPHeader(
            message_type=MessageType.HEARTBEAT_REQUEST,
            length=len(ie_content),
            sequence_number=sequence_number
        )

        return header.encode() + ie_content


@dataclass
class HeartbeatResponse:
    """Heartbeat Response Message (TS 29.244 Section 7.4.2.2)"""
    recovery_time_stamp: int

    def encode(self, sequence_number: int) -> bytes:
        """Encode Heartbeat Response"""
        ie_content = RecoveryTimeStamp(self.recovery_time_stamp).encode()

        header = PFCPHeader(
            message_type=MessageType.HEARTBEAT_RESPONSE,
            length=len(ie_content),
            sequence_number=sequence_number
        )

        return header.encode() + ie_content


@dataclass
class AssociationSetupRequest:
    """Association Setup Request Message (TS 29.244 Section 7.4.3.1)"""
    node_id: NodeId
    recovery_time_stamp: int
    cp_function_features: Optional[int] = None

    def encode(self, sequence_number: int) -> bytes:
        """Encode Association Setup Request"""
        ie_content = b''
        ie_content += self.node_id.encode()
        ie_content += RecoveryTimeStamp(self.recovery_time_stamp).encode()

        if self.cp_function_features is not None:
            ie_content += struct.pack('>HHH', IEType.CP_FUNCTION_FEATURES, 2, self.cp_function_features)

        header = PFCPHeader(
            message_type=MessageType.ASSOCIATION_SETUP_REQUEST,
            length=len(ie_content),
            sequence_number=sequence_number
        )

        return header.encode() + ie_content


@dataclass
class AssociationSetupResponse:
    """Association Setup Response Message (TS 29.244 Section 7.4.3.2)"""
    node_id: NodeId
    cause: CauseValue
    recovery_time_stamp: int
    up_function_features: Optional[int] = None

    def encode(self, sequence_number: int) -> bytes:
        """Encode Association Setup Response"""
        ie_content = b''
        ie_content += self.node_id.encode()
        ie_content += Cause(self.cause).encode()
        ie_content += RecoveryTimeStamp(self.recovery_time_stamp).encode()

        if self.up_function_features is not None:
            ie_content += struct.pack('>HHH', IEType.UP_FUNCTION_FEATURES, 2, self.up_function_features)

        header = PFCPHeader(
            message_type=MessageType.ASSOCIATION_SETUP_RESPONSE,
            length=len(ie_content),
            sequence_number=sequence_number
        )

        return header.encode() + ie_content


@dataclass
class SessionEstablishmentRequest:
    """Session Establishment Request Message (TS 29.244 Section 7.5.2.1)"""
    node_id: NodeId
    cp_fseid: FSEID
    create_pdrs: List[CreatePDR] = field(default_factory=list)
    create_fars: List[CreateFAR] = field(default_factory=list)
    create_qers: List[CreateQER] = field(default_factory=list)
    pdn_type: Optional[int] = None

    def encode(self, sequence_number: int, seid: int) -> bytes:
        """Encode Session Establishment Request"""
        ie_content = b''
        ie_content += self.node_id.encode()
        ie_content += self.cp_fseid.encode()

        for pdr in self.create_pdrs:
            ie_content += pdr.encode()

        for far in self.create_fars:
            ie_content += far.encode()

        for qer in self.create_qers:
            ie_content += qer.encode()

        if self.pdn_type is not None:
            ie_content += struct.pack('>HHB', IEType.PDN_TYPE, 1, self.pdn_type)

        header = PFCPHeader(
            s=1,
            message_type=MessageType.SESSION_ESTABLISHMENT_REQUEST,
            length=len(ie_content),
            seid=seid,
            sequence_number=sequence_number
        )

        return header.encode() + ie_content


@dataclass
class SessionEstablishmentResponse:
    """Session Establishment Response Message (TS 29.244 Section 7.5.3.1)"""
    node_id: NodeId
    cause: CauseValue
    up_fseid: Optional[FSEID] = None
    created_pdrs: List[Dict[str, Any]] = field(default_factory=list)

    def encode(self, sequence_number: int, seid: int) -> bytes:
        """Encode Session Establishment Response"""
        ie_content = b''
        ie_content += self.node_id.encode()
        ie_content += Cause(self.cause).encode()

        if self.up_fseid:
            ie_content += self.up_fseid.encode()

        # Created PDR IEs would go here

        header = PFCPHeader(
            s=1,
            message_type=MessageType.SESSION_ESTABLISHMENT_RESPONSE,
            length=len(ie_content),
            seid=seid,
            sequence_number=sequence_number
        )

        return header.encode() + ie_content


@dataclass
class SessionDeletionRequest:
    """Session Deletion Request Message (TS 29.244 Section 7.5.4.1)"""
    # No mandatory IEs besides header

    def encode(self, sequence_number: int, seid: int) -> bytes:
        """Encode Session Deletion Request"""
        header = PFCPHeader(
            s=1,
            message_type=MessageType.SESSION_DELETION_REQUEST,
            length=0,
            seid=seid,
            sequence_number=sequence_number
        )

        return header.encode()


@dataclass
class SessionDeletionResponse:
    """Session Deletion Response Message (TS 29.244 Section 7.5.5.1)"""
    cause: CauseValue

    def encode(self, sequence_number: int, seid: int) -> bytes:
        """Encode Session Deletion Response"""
        ie_content = Cause(self.cause).encode()

        header = PFCPHeader(
            s=1,
            message_type=MessageType.SESSION_DELETION_RESPONSE,
            length=len(ie_content),
            seid=seid,
            sequence_number=sequence_number
        )

        return header.encode() + ie_content


# =============================================================================
# PFCP Codec and Protocol Handler
# =============================================================================

class PFCPCodec:
    """
    PFCP Protocol Codec for encoding/decoding messages.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sequence_number = 0

    def get_next_sequence(self) -> int:
        """Get next sequence number"""
        self.sequence_number = (self.sequence_number + 1) & 0xFFFFFF
        return self.sequence_number

    def decode_message(self, data: bytes) -> Dict[str, Any]:
        """
        Decode a PFCP message from bytes.

        Args:
            data: Raw PFCP message bytes

        Returns:
            Dictionary containing decoded message fields
        """
        header, offset = PFCPHeader.decode(data)

        result = {
            'version': header.version,
            'message_type': header.message_type,
            'message_type_name': MessageType(header.message_type).name,
            'length': header.length,
            'sequence_number': header.sequence_number,
            'seid': header.seid,
            'ies': []
        }

        # Parse IEs
        ie_data = data[offset:]
        ie_offset = 0

        while ie_offset < len(ie_data):
            if len(ie_data) - ie_offset < 4:
                break
            ie, consumed = InformationElement.decode(ie_data[ie_offset:])
            result['ies'].append({
                'type': ie.ie_type,
                'type_name': IEType(ie.ie_type).name if ie.ie_type in IEType._value2member_map_ else 'UNKNOWN',
                'length': ie.length,
                'value': ie.value.hex()
            })
            ie_offset += consumed

        return result


class PFCPNode:
    """
    PFCP Node (SMF or UPF) for handling PFCP protocol.
    """

    def __init__(self, node_type: str, node_id: str, port: int = PFCP_PORT):
        """
        Initialize PFCP node.

        Args:
            node_type: "SMF" or "UPF"
            node_id: Node IP address
            port: PFCP port (default 8805)
        """
        self.node_type = node_type
        self.node_id = NodeId(0, node_id)
        self.port = port
        self.codec = PFCPCodec()
        self.recovery_timestamp = self._get_ntp_timestamp()
        self.associations: Dict[str, Dict] = {}
        self.sessions: Dict[int, Dict] = {}
        self.logger = logging.getLogger(__name__)

    def _get_ntp_timestamp(self) -> int:
        """Get current time as NTP timestamp"""
        import time
        # NTP epoch is 1900, Unix epoch is 1970
        # Difference is 2208988800 seconds
        return int(time.time()) + 2208988800

    def create_association_setup_request(self) -> bytes:
        """Create Association Setup Request message"""
        msg = AssociationSetupRequest(
            node_id=self.node_id,
            recovery_time_stamp=self.recovery_timestamp,
            cp_function_features=0x0001 if self.node_type == "SMF" else None
        )
        return msg.encode(self.codec.get_next_sequence())

    def create_heartbeat_request(self) -> bytes:
        """Create Heartbeat Request message"""
        msg = HeartbeatRequest(recovery_time_stamp=self.recovery_timestamp)
        return msg.encode(self.codec.get_next_sequence())

    def create_session_establishment_request(
        self,
        seid: int,
        ue_ip: str,
        gnb_teid: int,
        gnb_ip: str,
        upf_teid: int,
        qfi: int = 1
    ) -> bytes:
        """
        Create Session Establishment Request for a PDU session.

        Args:
            seid: Session Endpoint Identifier
            ue_ip: UE IP address
            gnb_teid: gNB TEID for downlink
            gnb_ip: gNB IP address
            upf_teid: UPF TEID for uplink
            qfi: QoS Flow Identifier

        Returns:
            Encoded PFCP message bytes
        """
        # Create uplink PDR (N3 -> N6)
        ul_pdr = CreatePDR(
            pdr_id=1,
            precedence=100,
            pdi=PDI(
                source_interface=SourceInterface.ACCESS,
                local_fteid=FTEID(teid=upf_teid, ipv4_address=self.node_id.node_id),
                qfi=qfi
            ),
            outer_header_removal=0,  # GTP-U/UDP/IPv4
            far_id=1
        )

        # Create downlink PDR (N6 -> N3)
        dl_pdr = CreatePDR(
            pdr_id=2,
            precedence=100,
            pdi=PDI(
                source_interface=SourceInterface.CORE,
                ue_ip_address=UEIPAddress(1, ue_ip),
                qfi=qfi
            ),
            far_id=2
        )

        # Create uplink FAR (forward to N6/DN)
        ul_far = CreateFAR(
            far_id=1,
            apply_action=ApplyAction.FORW,
            forwarding_parameters=ForwardingParameters(
                destination_interface=DestinationInterface.CORE,
                network_instance="internet"
            )
        )

        # Create downlink FAR (forward to gNB)
        dl_far = CreateFAR(
            far_id=2,
            apply_action=ApplyAction.FORW,
            forwarding_parameters=ForwardingParameters(
                destination_interface=DestinationInterface.ACCESS,
                outer_header_creation=OuterHeaderCreation(
                    description=0x0100,  # GTP-U/UDP/IPv4
                    teid=gnb_teid,
                    ipv4_address=gnb_ip
                )
            )
        )

        # Create QER for rate limiting
        qer = CreateQER(
            qer_id=1,
            gate_status_ul=GateStatus.OPEN,
            gate_status_dl=GateStatus.OPEN,
            mbr=MBR(ul_mbr=100000, dl_mbr=100000),  # 100 Mbps
            qfi=qfi
        )

        msg = SessionEstablishmentRequest(
            node_id=self.node_id,
            cp_fseid=FSEID(seid=seid, ipv4_address=self.node_id.node_id),
            create_pdrs=[ul_pdr, dl_pdr],
            create_fars=[ul_far, dl_far],
            create_qers=[qer]
        )

        return msg.encode(self.codec.get_next_sequence(), seid)

    def create_session_deletion_request(self, seid: int) -> bytes:
        """Create Session Deletion Request message"""
        msg = SessionDeletionRequest()
        return msg.encode(self.codec.get_next_sequence(), seid)


# =============================================================================
# Async UDP Transport
# =============================================================================

class PFCPTransport:
    """
    Async UDP transport for PFCP messages.
    """

    def __init__(self, local_addr: str, local_port: int = PFCP_PORT):
        self.local_addr = local_addr
        self.local_port = local_port
        self.transport = None
        self.protocol = None
        self.logger = logging.getLogger(__name__)
        self.message_handlers: Dict[int, callable] = {}
        self.pending_responses: Dict[int, asyncio.Future] = {}

    async def start(self):
        """Start the UDP transport"""
        loop = asyncio.get_event_loop()

        class PFCPProtocol(asyncio.DatagramProtocol):
            def __init__(self, handler):
                self.handler = handler

            def datagram_received(self, data, addr):
                asyncio.create_task(self.handler._handle_message(data, addr))

        self.transport, self.protocol = await loop.create_datagram_endpoint(
            lambda: PFCPProtocol(self),
            local_addr=(self.local_addr, self.local_port)
        )
        self.logger.info(f"PFCP transport started on {self.local_addr}:{self.local_port}")

    async def stop(self):
        """Stop the UDP transport"""
        if self.transport:
            self.transport.close()
            self.logger.info("PFCP transport stopped")

    async def send(self, data: bytes, addr: Tuple[str, int]):
        """Send PFCP message to remote address"""
        if self.transport:
            self.transport.sendto(data, addr)
            self.logger.debug(f"Sent {len(data)} bytes to {addr}")

    async def send_and_wait(self, data: bytes, addr: Tuple[str, int], timeout: float = 5.0) -> Optional[bytes]:
        """Send message and wait for response"""
        # Extract sequence number from message
        seq = struct.unpack('>I', data[4:8])[0] >> 8 if data[0] & 0x01 == 0 else struct.unpack('>I', data[12:16])[0] >> 8

        future = asyncio.get_event_loop().create_future()
        self.pending_responses[seq] = future

        await self.send(data, addr)

        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for response to sequence {seq}")
            return None
        finally:
            self.pending_responses.pop(seq, None)

    async def _handle_message(self, data: bytes, addr: Tuple[str, int]):
        """Handle received PFCP message"""
        try:
            codec = PFCPCodec()
            msg = codec.decode_message(data)
            self.logger.debug(f"Received {msg['message_type_name']} from {addr}")

            # Check if this is a response to a pending request
            seq = msg['sequence_number']
            if seq in self.pending_responses:
                self.pending_responses[seq].set_result(data)
                return

            # Otherwise, dispatch to message handler
            msg_type = msg['message_type']
            if msg_type in self.message_handlers:
                await self.message_handlers[msg_type](msg, addr)
            else:
                self.logger.warning(f"No handler for message type {msg_type}")

        except Exception as e:
            self.logger.error(f"Error handling PFCP message: {e}")


# =============================================================================
# Demo/Test Functions
# =============================================================================

def demo_pfcp_encoding():
    """Demonstrate PFCP message encoding/decoding"""
    print("=" * 60)
    print("PFCP Protocol Demo")
    print("=" * 60)

    # Create a PFCP node (SMF)
    smf = PFCPNode("SMF", "10.0.0.1")

    # Create Association Setup Request
    print("\n=== Association Setup Request ===")
    assoc_req = smf.create_association_setup_request()
    print(f"Encoded ({len(assoc_req)} bytes): {assoc_req.hex()}")

    # Decode it
    codec = PFCPCodec()
    decoded = codec.decode_message(assoc_req)
    print(f"Decoded: {decoded['message_type_name']}")
    for ie in decoded['ies']:
        print(f"  IE: {ie['type_name']} ({ie['length']} bytes)")

    # Create Heartbeat Request
    print("\n=== Heartbeat Request ===")
    hb_req = smf.create_heartbeat_request()
    print(f"Encoded ({len(hb_req)} bytes): {hb_req.hex()}")

    # Create Session Establishment Request
    print("\n=== Session Establishment Request ===")
    session_req = smf.create_session_establishment_request(
        seid=0x123456789ABCDEF0,
        ue_ip="10.45.0.1",
        gnb_teid=0x12345678,
        gnb_ip="192.168.1.100",
        upf_teid=0x87654321,
        qfi=1
    )
    print(f"Encoded ({len(session_req)} bytes): {session_req[:64].hex()}...")

    decoded = codec.decode_message(session_req)
    print(f"Decoded: {decoded['message_type_name']}")
    print(f"  SEID: {decoded['seid']:016x}")
    print(f"  IEs: {len(decoded['ies'])}")
    for ie in decoded['ies'][:5]:
        print(f"    - {ie['type_name']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    demo_pfcp_encoding()
