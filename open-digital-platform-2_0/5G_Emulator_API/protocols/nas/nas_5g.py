"""
5G NAS Protocol Implementation (TS 24.501)

This module implements the 5G Non-Access Stratum (NAS) protocol for communication
between UE and AMF. It provides binary encoding/decoding of NAS messages as
specified in 3GPP TS 24.501.

Key Features:
- Binary NAS message encoding/decoding
- All 5GMM (5G Mobility Management) message types
- All 5GSM (5G Session Management) message types
- Security header processing
- Information Element (IE) encoding/decoding

Reference: 3GPP TS 24.501 V17.7.0 (2023-03)
"""

import struct
from enum import IntEnum
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol Constants (TS 24.501 Section 9)
# =============================================================================

class ExtendedProtocolDiscriminator(IntEnum):
    """Extended Protocol Discriminator (TS 24.501 Section 9.2)"""
    MOBILITY_MANAGEMENT_5G = 0x7E  # 5GMM
    SESSION_MANAGEMENT_5G = 0x2E   # 5GSM


class SecurityHeaderType(IntEnum):
    """Security Header Type (TS 24.501 Section 9.3)"""
    PLAIN_NAS = 0x00
    INTEGRITY_PROTECTED = 0x01
    INTEGRITY_PROTECTED_CIPHERED = 0x02
    INTEGRITY_PROTECTED_NEW_CONTEXT = 0x03
    INTEGRITY_PROTECTED_CIPHERED_NEW_CONTEXT = 0x04


class MessageType5GMM(IntEnum):
    """5GMM Message Types (TS 24.501 Section 9.7)"""
    REGISTRATION_REQUEST = 0x41
    REGISTRATION_ACCEPT = 0x42
    REGISTRATION_COMPLETE = 0x43
    REGISTRATION_REJECT = 0x44
    DEREGISTRATION_REQUEST_UE = 0x45
    DEREGISTRATION_ACCEPT_UE = 0x46
    DEREGISTRATION_REQUEST_NW = 0x47
    DEREGISTRATION_ACCEPT_NW = 0x48
    SERVICE_REQUEST = 0x4C
    SERVICE_REJECT = 0x4D
    SERVICE_ACCEPT = 0x4E
    CONFIGURATION_UPDATE_COMMAND = 0x54
    CONFIGURATION_UPDATE_COMPLETE = 0x55
    AUTHENTICATION_REQUEST = 0x56
    AUTHENTICATION_RESPONSE = 0x57
    AUTHENTICATION_REJECT = 0x58
    AUTHENTICATION_FAILURE = 0x59
    AUTHENTICATION_RESULT = 0x5A
    IDENTITY_REQUEST = 0x5B
    IDENTITY_RESPONSE = 0x5C
    SECURITY_MODE_COMMAND = 0x5D
    SECURITY_MODE_COMPLETE = 0x5E
    SECURITY_MODE_REJECT = 0x5F
    STATUS_5GMM = 0x64
    NOTIFICATION = 0x65
    NOTIFICATION_RESPONSE = 0x66
    UL_NAS_TRANSPORT = 0x67
    DL_NAS_TRANSPORT = 0x68


class MessageType5GSM(IntEnum):
    """5GSM Message Types (TS 24.501 Section 9.7)"""
    PDU_SESSION_ESTABLISHMENT_REQUEST = 0xC1
    PDU_SESSION_ESTABLISHMENT_ACCEPT = 0xC2
    PDU_SESSION_ESTABLISHMENT_REJECT = 0xC3
    PDU_SESSION_AUTHENTICATION_COMMAND = 0xC5
    PDU_SESSION_AUTHENTICATION_COMPLETE = 0xC6
    PDU_SESSION_AUTHENTICATION_RESULT = 0xC7
    PDU_SESSION_MODIFICATION_REQUEST = 0xC9
    PDU_SESSION_MODIFICATION_REJECT = 0xCA
    PDU_SESSION_MODIFICATION_COMMAND = 0xCB
    PDU_SESSION_MODIFICATION_COMPLETE = 0xCC
    PDU_SESSION_MODIFICATION_COMMAND_REJECT = 0xCD
    PDU_SESSION_RELEASE_REQUEST = 0xD1
    PDU_SESSION_RELEASE_REJECT = 0xD2
    PDU_SESSION_RELEASE_COMMAND = 0xD3
    PDU_SESSION_RELEASE_COMPLETE = 0xD4
    STATUS_5GSM = 0xD6


class RegistrationType(IntEnum):
    """5GS Registration Type (TS 24.501 Section 9.11.3.7)"""
    INITIAL = 0x01
    MOBILITY_UPDATING = 0x02
    PERIODIC_UPDATING = 0x03
    EMERGENCY = 0x04


class MobileIdentityType(IntEnum):
    """5GS Mobile Identity Type (TS 24.501 Section 9.11.3.4)"""
    NO_IDENTITY = 0x00
    SUCI = 0x01
    GUTI_5G = 0x02
    IMEI = 0x03
    TMSI_5G = 0x04
    IMEISV = 0x05
    MAC_ADDRESS = 0x06
    EUI_64 = 0x07


class PDUSessionType(IntEnum):
    """PDU Session Type (TS 24.501 Section 9.11.4.11)"""
    IPV4 = 0x01
    IPV6 = 0x02
    IPV4V6 = 0x03
    UNSTRUCTURED = 0x04
    ETHERNET = 0x05


class CauseValue5GMM(IntEnum):
    """5GMM Cause Values (TS 24.501 Section 9.11.3.2)"""
    ILLEGAL_UE = 0x03
    PEI_NOT_ACCEPTED = 0x05
    ILLEGAL_ME = 0x06
    SERVICES_NOT_ALLOWED = 0x07
    UE_IDENTITY_NOT_DERIVED = 0x09
    IMPLICITLY_DEREGISTERED = 0x0A
    PLMN_NOT_ALLOWED = 0x0B
    TA_NOT_ALLOWED = 0x0C
    ROAMING_NOT_ALLOWED = 0x0D
    NO_SUITABLE_CELLS = 0x0F
    MAC_FAILURE = 0x14
    SYNCH_FAILURE = 0x15
    CONGESTION = 0x16
    UE_SECURITY_CAP_MISMATCH = 0x17
    SECURITY_MODE_REJECTED = 0x18
    NON_5G_AUTH_UNACCEPTABLE = 0x1A
    N1_MODE_NOT_ALLOWED = 0x1B
    RESTRICTED_SERVICE_AREA = 0x1C
    LADN_NOT_AVAILABLE = 0x2B
    NO_NETWORK_SLICES_AVAILABLE = 0x3E
    MAX_PDU_SESSIONS_REACHED = 0x41
    INSUFFICIENT_RESOURCES_SLICE = 0x43
    MISSING_OR_UNKNOWN_DNN_SLICE = 0x44
    PAYLOAD_NOT_FORWARDED = 0x5A
    DNN_NOT_SUPPORTED_IN_SLICE = 0x5B
    INVALID_MANDATORY_INFO = 0x60
    MESSAGE_TYPE_NONEXISTENT = 0x61
    MESSAGE_TYPE_NOT_COMPATIBLE = 0x62
    IE_NONEXISTENT = 0x63
    CONDITIONAL_IE_ERROR = 0x64
    MESSAGE_NOT_COMPATIBLE = 0x65
    UNSPECIFIED = 0x6F


# =============================================================================
# Information Element Tags (TS 24.501 Section 9.11)
# =============================================================================

class IEI:
    """Information Element Identifiers"""
    # 5GMM IEIs
    NON_CURRENT_NATIVE_NAS_KSI = 0x0C
    GUTI_5G = 0x77
    ALLOWED_NSSAI = 0x15
    REJECTED_NSSAI = 0x11
    CONFIGURED_NSSAI = 0x31
    NETWORK_FEATURE_SUPPORT = 0x21
    PDU_SESSION_STATUS = 0x50
    PDU_SESSION_REACTIVATION_RESULT = 0x26
    PDU_SESSION_REACTIVATION_ERROR = 0x72
    LADN_INFORMATION = 0x79
    MICO_INDICATION = 0x0B
    NETWORK_SLICING_INDICATION = 0x09
    SERVICE_AREA_LIST = 0x27
    GPRS_TIMER_3 = 0x5E
    GPRS_TIMER_2 = 0x5D
    EMERGENCY_NUMBER_LIST = 0x34
    EXTENDED_EMERGENCY_NUMBER_LIST = 0x7A
    SOR_TRANSPARENT_CONTAINER = 0x73
    EAP_MESSAGE = 0x78
    NSSAI_INCLUSION_MODE = 0x0A
    OPERATOR_DEFINED_ACCESS_CAT = 0x76
    NEGOTIATED_DRX = 0x51
    NON_3GPP_NW_POLICIES = 0x0D
    EPS_BEARER_CONTEXT_STATUS = 0x60
    EXTENDED_DRX = 0x6E
    GPRS_TIMER_T3447 = 0x6C
    GPRS_TIMER_T3448 = 0x6B
    GPRS_TIMER_T3324 = 0x6A
    UE_RADIO_CAP_ID = 0x67
    UE_RADIO_CAP_ID_DELETION = 0x68
    PENDING_NSSAI = 0x39
    CIPHERING_KEY_DATA = 0x74
    CAG_INFO_LIST = 0x75
    TRUNCATED_5G_S_TMSI = 0x6D
    WUS_ASSISTANCE = 0x1B
    NB_N1_MODE_DRX = 0x29

    # 5GSM IEIs
    PDU_SESSION_TYPE = 0x09
    SSC_MODE = 0x0A
    QUALITY_OF_SERVICE_RULES = 0x7A
    SESSION_AMBR = 0x2A
    PDU_ADDRESS = 0x29
    RQ_TIMER = 0x56
    S_NSSAI = 0x22
    ALWAYS_ON_PDU_SESSION = 0x08
    MAPPED_EPS_BEARER_CONTEXTS = 0x75
    EAP_MESSAGE_SM = 0x78
    AUTHORIZED_QOS_FLOW = 0x79
    EXTENDED_PCO = 0x7B
    DNN = 0x25


# =============================================================================
# Data Classes for NAS Messages
# =============================================================================

@dataclass
class NASHeader:
    """NAS Message Header (TS 24.501 Section 9.2/9.3)"""
    extended_protocol_discriminator: int
    security_header_type: int
    message_type: int
    spare_half_octet: int = 0

    def encode(self) -> bytes:
        """Encode header to bytes"""
        octet1 = self.extended_protocol_discriminator
        octet2 = (self.spare_half_octet << 4) | (self.security_header_type & 0x0F)
        octet3 = self.message_type
        return struct.pack('BBB', octet1, octet2, octet3)

    @classmethod
    def decode(cls, data: bytes) -> Tuple['NASHeader', int]:
        """Decode header from bytes, returns (header, bytes_consumed)"""
        if len(data) < 3:
            raise ValueError("Insufficient data for NAS header")
        epd, octet2, msg_type = struct.unpack('BBB', data[:3])
        security_header = octet2 & 0x0F
        spare = (octet2 >> 4) & 0x0F
        return cls(epd, security_header, msg_type, spare), 3


@dataclass
class SecurityHeader:
    """Security Protected NAS Header (TS 24.501 Section 9.1)"""
    extended_protocol_discriminator: int
    security_header_type: int
    message_auth_code: bytes  # 4 bytes
    sequence_number: int      # 1 byte

    def encode(self) -> bytes:
        """Encode security header"""
        return struct.pack('BB',
            self.extended_protocol_discriminator,
            self.security_header_type
        ) + self.message_auth_code + struct.pack('B', self.sequence_number)

    @classmethod
    def decode(cls, data: bytes) -> Tuple['SecurityHeader', int]:
        """Decode security header"""
        if len(data) < 7:
            raise ValueError("Insufficient data for security header")
        epd, sht = struct.unpack('BB', data[:2])
        mac = data[2:6]
        seq = data[6]
        return cls(epd, sht, mac, seq), 7


@dataclass
class PLMN:
    """Public Land Mobile Network Identity (TS 24.501 Section 9.11.3.4)"""
    mcc: str  # 3 digits
    mnc: str  # 2-3 digits

    def encode(self) -> bytes:
        """Encode PLMN to 3 bytes per TS 24.008 Section 10.5.1.13"""
        mcc1, mcc2, mcc3 = int(self.mcc[0]), int(self.mcc[1]), int(self.mcc[2])
        if len(self.mnc) == 2:
            mnc1, mnc2, mnc3 = int(self.mnc[0]), int(self.mnc[1]), 0x0F
        else:
            mnc1, mnc2, mnc3 = int(self.mnc[0]), int(self.mnc[1]), int(self.mnc[2])

        byte1 = (mcc2 << 4) | mcc1
        byte2 = (mnc3 << 4) | mcc3
        byte3 = (mnc2 << 4) | mnc1
        return bytes([byte1, byte2, byte3])

    @classmethod
    def decode(cls, data: bytes) -> 'PLMN':
        """Decode PLMN from 3 bytes"""
        if len(data) < 3:
            raise ValueError("Insufficient data for PLMN")
        mcc1 = data[0] & 0x0F
        mcc2 = (data[0] >> 4) & 0x0F
        mcc3 = data[1] & 0x0F
        mnc3 = (data[1] >> 4) & 0x0F
        mnc1 = data[2] & 0x0F
        mnc2 = (data[2] >> 4) & 0x0F

        mcc = f"{mcc1}{mcc2}{mcc3}"
        if mnc3 == 0x0F:
            mnc = f"{mnc1}{mnc2}"
        else:
            mnc = f"{mnc1}{mnc2}{mnc3}"
        return cls(mcc, mnc)


@dataclass
class SNSSAI:
    """Single Network Slice Selection Assistance Information (TS 24.501 Section 9.11.2.8)"""
    sst: int          # Slice/Service Type (1 byte)
    sd: Optional[bytes] = None  # Slice Differentiator (3 bytes, optional)

    def encode(self) -> bytes:
        """Encode S-NSSAI"""
        if self.sd:
            length = 4
            return struct.pack('BB', length, self.sst) + self.sd
        else:
            length = 1
            return struct.pack('BB', length, self.sst)

    @classmethod
    def decode(cls, data: bytes) -> Tuple['SNSSAI', int]:
        """Decode S-NSSAI"""
        if len(data) < 2:
            raise ValueError("Insufficient data for S-NSSAI")
        length = data[0]
        sst = data[1]
        sd = data[2:5] if length >= 4 else None
        return cls(sst, sd), length + 1


@dataclass
class GUTI5G:
    """5G Globally Unique Temporary Identifier (TS 24.501 Section 9.11.3.4)"""
    plmn: PLMN
    amf_region_id: int    # 8 bits
    amf_set_id: int       # 10 bits
    amf_pointer: int      # 6 bits
    tmsi_5g: bytes        # 4 bytes

    def encode(self) -> bytes:
        """Encode 5G-GUTI"""
        # Type of identity (3 bits) = 010 (5G-GUTI)
        # Odd/even indication (1 bit) = 0 (even)
        type_id = (MobileIdentityType.GUTI_5G << 4) | 0x0F

        plmn_bytes = self.plmn.encode()

        # AMF Identifier encoding (3 bytes)
        amf_id = (self.amf_region_id << 16) | (self.amf_set_id << 6) | self.amf_pointer
        amf_bytes = struct.pack('>I', amf_id)[1:]  # 3 bytes big-endian

        content = bytes([type_id]) + plmn_bytes + amf_bytes + self.tmsi_5g
        length = len(content)
        return struct.pack('BB', IEI.GUTI_5G, length) + content

    @classmethod
    def decode(cls, data: bytes) -> Tuple['GUTI5G', int]:
        """Decode 5G-GUTI from bytes"""
        if len(data) < 2:
            raise ValueError("Insufficient data for 5G-GUTI")

        iei = data[0]
        length = data[1]
        content = data[2:2+length]

        # Parse identity type
        type_id = (content[0] >> 4) & 0x07

        # Parse PLMN (bytes 1-3)
        plmn = PLMN.decode(content[1:4])

        # Parse AMF ID (bytes 4-6)
        amf_bytes = b'\x00' + content[4:7]
        amf_id = struct.unpack('>I', amf_bytes)[0]
        amf_region_id = (amf_id >> 16) & 0xFF
        amf_set_id = (amf_id >> 6) & 0x3FF
        amf_pointer = amf_id & 0x3F

        # Parse 5G-TMSI (bytes 7-10)
        tmsi_5g = content[7:11]

        return cls(plmn, amf_region_id, amf_set_id, amf_pointer, tmsi_5g), length + 2


@dataclass
class SUCI:
    """Subscription Concealed Identifier (TS 24.501 Section 9.11.3.4)"""
    supi_format: int      # IMSI=0, NAI=1
    plmn: PLMN
    routing_indicator: str  # 1-4 digits
    protection_scheme: int  # null=0, ECIES-A=1, ECIES-B=2
    home_network_pki: int   # Public Key Identifier
    scheme_output: bytes    # Encrypted MSIN or clear MSIN

    def encode(self) -> bytes:
        """Encode SUCI"""
        # First byte: type=001 (SUCI), supi_format
        type_byte = (MobileIdentityType.SUCI << 4) | (self.supi_format & 0x07)

        plmn_bytes = self.plmn.encode()

        # Routing indicator (BCD encoded, 2 bytes)
        ri_padded = self.routing_indicator.ljust(4, 'F')
        ri_bytes = bytes([
            (int(ri_padded[1], 16) << 4) | int(ri_padded[0], 16),
            (int(ri_padded[3], 16) << 4) | int(ri_padded[2], 16)
        ])

        # Protection scheme and HN PKI
        prot_pki = struct.pack('BB', self.protection_scheme, self.home_network_pki)

        content = bytes([type_byte]) + plmn_bytes + ri_bytes + prot_pki + self.scheme_output
        length = len(content)

        return struct.pack('B', length) + content


@dataclass
class UESecurityCapability:
    """UE Security Capability (TS 24.501 Section 9.11.3.54)"""
    ea: int   # 5G encryption algorithms (1 byte, bit field)
    ia: int   # 5G integrity algorithms (1 byte, bit field)
    eea: Optional[int] = None  # EPS encryption (optional)
    eia: Optional[int] = None  # EPS integrity (optional)

    def encode(self) -> bytes:
        """Encode UE Security Capability"""
        if self.eea is not None and self.eia is not None:
            length = 4
            return struct.pack('BBBBB', length, self.ea, self.ia, self.eea, self.eia)
        else:
            length = 2
            return struct.pack('BBB', length, self.ea, self.ia)

    @classmethod
    def decode(cls, data: bytes) -> Tuple['UESecurityCapability', int]:
        """Decode UE Security Capability"""
        length = data[0]
        ea = data[1]
        ia = data[2]
        eea = data[3] if length >= 4 else None
        eia = data[4] if length >= 4 else None
        return cls(ea, ia, eea, eia), length + 1


# =============================================================================
# NAS Message Classes
# =============================================================================

@dataclass
class RegistrationRequest:
    """5GMM Registration Request (TS 24.501 Section 8.2.6)"""
    ngksi: int                          # NAS Key Set Identifier (4 bits)
    registration_type: int              # 5GS Registration Type (4 bits)
    mobile_identity: Union[SUCI, GUTI5G, bytes]  # 5GS Mobile Identity
    ue_security_capability: Optional[UESecurityCapability] = None
    requested_nssai: Optional[List[SNSSAI]] = None
    last_visited_tai: Optional[bytes] = None
    s1_ue_network_capability: Optional[bytes] = None
    uplink_data_status: Optional[bytes] = None
    pdu_session_status: Optional[bytes] = None
    mico_indication: Optional[bool] = None
    ue_status: Optional[bytes] = None
    additional_guti: Optional[GUTI5G] = None
    allowed_pdu_session_status: Optional[bytes] = None
    ue_usage_setting: Optional[int] = None
    drx_parameter: Optional[bytes] = None
    eps_nas_message: Optional[bytes] = None
    ladn_indication: Optional[bytes] = None
    payload_container_type: Optional[int] = None
    payload_container: Optional[bytes] = None
    network_slicing_indication: Optional[int] = None
    update_type: Optional[int] = None

    def encode(self) -> bytes:
        """Encode Registration Request to bytes"""
        header = NASHeader(
            ExtendedProtocolDiscriminator.MOBILITY_MANAGEMENT_5G,
            SecurityHeaderType.PLAIN_NAS,
            MessageType5GMM.REGISTRATION_REQUEST
        )

        # ngKSI and registration type (1 byte)
        ngksi_regtype = ((self.ngksi & 0x0F) << 4) | (self.registration_type & 0x0F)

        # Mobile identity
        if isinstance(self.mobile_identity, SUCI):
            mi_bytes = self.mobile_identity.encode()
        elif isinstance(self.mobile_identity, GUTI5G):
            mi_bytes = self.mobile_identity.encode()
        else:
            mi_bytes = self.mobile_identity

        result = header.encode() + struct.pack('B', ngksi_regtype) + mi_bytes

        # Optional IEs
        if self.ue_security_capability:
            result += struct.pack('B', 0x2E) + self.ue_security_capability.encode()

        if self.requested_nssai:
            nssai_data = b''.join(s.encode() for s in self.requested_nssai)
            result += struct.pack('BB', 0x2F, len(nssai_data)) + nssai_data

        return result

    @classmethod
    def decode(cls, data: bytes) -> 'RegistrationRequest':
        """Decode Registration Request from bytes"""
        header, offset = NASHeader.decode(data)

        ngksi_regtype = data[offset]
        ngksi = (ngksi_regtype >> 4) & 0x0F
        registration_type = ngksi_regtype & 0x0F
        offset += 1

        # Parse mobile identity length and type
        mi_length = data[offset]
        mi_type = (data[offset + 1] >> 4) & 0x07
        mi_data = data[offset:offset + mi_length + 1]
        offset += mi_length + 1

        # Parse based on identity type
        if mi_type == MobileIdentityType.SUCI:
            mobile_identity = mi_data  # TODO: Full SUCI parsing
        elif mi_type == MobileIdentityType.GUTI_5G:
            mobile_identity, _ = GUTI5G.decode(mi_data)
        else:
            mobile_identity = mi_data

        # Parse optional IEs
        ue_security_capability = None
        requested_nssai = None

        while offset < len(data):
            iei = data[offset]
            offset += 1

            if iei == 0x2E:  # UE Security Capability
                ue_security_capability, consumed = UESecurityCapability.decode(data[offset:])
                offset += consumed
            elif iei == 0x2F:  # Requested NSSAI
                nssai_len = data[offset]
                offset += 1
                nssai_end = offset + nssai_len
                requested_nssai = []
                while offset < nssai_end:
                    snssai, consumed = SNSSAI.decode(data[offset:])
                    requested_nssai.append(snssai)
                    offset += consumed
            else:
                # Skip unknown IE
                if iei & 0x80:  # Type 1 TV or T format
                    pass
                else:  # Type 4 TLV format
                    ie_len = data[offset]
                    offset += 1 + ie_len

        return cls(
            ngksi=ngksi,
            registration_type=registration_type,
            mobile_identity=mobile_identity,
            ue_security_capability=ue_security_capability,
            requested_nssai=requested_nssai
        )


@dataclass
class RegistrationAccept:
    """5GMM Registration Accept (TS 24.501 Section 8.2.7)"""
    registration_result: int       # 5GS Registration Result
    guti_5g: Optional[GUTI5G] = None
    tai_list: Optional[bytes] = None
    allowed_nssai: Optional[List[SNSSAI]] = None
    rejected_nssai: Optional[List[SNSSAI]] = None
    configured_nssai: Optional[List[SNSSAI]] = None
    network_feature_support: Optional[bytes] = None
    pdu_session_status: Optional[bytes] = None
    pdu_session_reactivation_result: Optional[bytes] = None
    gprs_timer_t3512: Optional[int] = None
    gprs_timer_t3502: Optional[int] = None
    emergency_number_list: Optional[bytes] = None
    extended_emergency_number_list: Optional[bytes] = None
    sor_transparent_container: Optional[bytes] = None
    eap_message: Optional[bytes] = None
    nssai_inclusion_mode: Optional[int] = None
    negotiated_drx: Optional[int] = None
    non_3gpp_nw_policies: Optional[int] = None
    eps_bearer_context_status: Optional[bytes] = None
    extended_drx: Optional[bytes] = None
    gprs_timer_t3447: Optional[int] = None
    gprs_timer_t3448: Optional[int] = None
    gprs_timer_t3324: Optional[int] = None
    ue_radio_capability_id: Optional[bytes] = None
    pending_nssai: Optional[List[SNSSAI]] = None

    def encode(self) -> bytes:
        """Encode Registration Accept to bytes"""
        header = NASHeader(
            ExtendedProtocolDiscriminator.MOBILITY_MANAGEMENT_5G,
            SecurityHeaderType.PLAIN_NAS,
            MessageType5GMM.REGISTRATION_ACCEPT
        )

        # Registration result (1 byte)
        reg_result = struct.pack('B', self.registration_result & 0x07)

        result = header.encode() + reg_result

        # Optional IEs
        if self.guti_5g:
            result += self.guti_5g.encode()

        if self.allowed_nssai:
            nssai_data = b''.join(s.encode() for s in self.allowed_nssai)
            result += struct.pack('BB', IEI.ALLOWED_NSSAI, len(nssai_data)) + nssai_data

        if self.tai_list:
            result += struct.pack('BB', 0x54, len(self.tai_list)) + self.tai_list

        return result


@dataclass
class AuthenticationRequest:
    """5GMM Authentication Request (TS 24.501 Section 8.2.1)"""
    ngksi: int               # NAS Key Set Identifier
    abba: bytes              # Anti-Bidding down Between Architectures
    rand: bytes              # RAND (16 bytes)
    autn: bytes              # AUTN (16 bytes)
    eap_message: Optional[bytes] = None

    def encode(self) -> bytes:
        """Encode Authentication Request"""
        header = NASHeader(
            ExtendedProtocolDiscriminator.MOBILITY_MANAGEMENT_5G,
            SecurityHeaderType.PLAIN_NAS,
            MessageType5GMM.AUTHENTICATION_REQUEST
        )

        # Spare + ngKSI
        ngksi_byte = self.ngksi & 0x0F

        # ABBA
        abba_encoded = struct.pack('B', len(self.abba)) + self.abba

        # RAND (IEI 0x21)
        rand_encoded = struct.pack('B', 0x21) + self.rand

        # AUTN (IEI 0x20)
        autn_encoded = struct.pack('BB', 0x20, len(self.autn)) + self.autn

        return header.encode() + struct.pack('B', ngksi_byte) + abba_encoded + rand_encoded + autn_encoded


@dataclass
class AuthenticationResponse:
    """5GMM Authentication Response (TS 24.501 Section 8.2.2)"""
    res_star: bytes          # RES* (16 bytes)
    eap_message: Optional[bytes] = None

    def encode(self) -> bytes:
        """Encode Authentication Response"""
        header = NASHeader(
            ExtendedProtocolDiscriminator.MOBILITY_MANAGEMENT_5G,
            SecurityHeaderType.PLAIN_NAS,
            MessageType5GMM.AUTHENTICATION_RESPONSE
        )

        # RES* (IEI 0x2D)
        res_encoded = struct.pack('BB', 0x2D, len(self.res_star)) + self.res_star

        return header.encode() + res_encoded

    @classmethod
    def decode(cls, data: bytes) -> 'AuthenticationResponse':
        """Decode Authentication Response"""
        header, offset = NASHeader.decode(data)

        res_star = None
        eap_message = None

        while offset < len(data):
            iei = data[offset]
            offset += 1

            if iei == 0x2D:  # Authentication response parameter (RES*)
                length = data[offset]
                offset += 1
                res_star = data[offset:offset + length]
                offset += length
            elif iei == 0x78:  # EAP message
                length = struct.unpack('>H', data[offset:offset+2])[0]
                offset += 2
                eap_message = data[offset:offset + length]
                offset += length

        return cls(res_star=res_star, eap_message=eap_message)


@dataclass
class SecurityModeCommand:
    """5GMM Security Mode Command (TS 24.501 Section 8.2.25)"""
    selected_nas_security_algorithms: int  # NAS security algorithms
    ngksi: int                             # NAS Key Set Identifier
    replayed_ue_security_capability: UESecurityCapability
    imeisv_request: Optional[int] = None
    selected_eps_nas_security_algorithms: Optional[int] = None
    additional_security_info: Optional[int] = None
    eap_message: Optional[bytes] = None
    abba: Optional[bytes] = None
    replayed_s1_ue_security_capability: Optional[bytes] = None

    def encode(self) -> bytes:
        """Encode Security Mode Command"""
        header = NASHeader(
            ExtendedProtocolDiscriminator.MOBILITY_MANAGEMENT_5G,
            SecurityHeaderType.PLAIN_NAS,
            MessageType5GMM.SECURITY_MODE_COMMAND
        )

        # Selected NAS security algorithms (1 byte)
        sec_alg = struct.pack('B', self.selected_nas_security_algorithms)

        # Spare + ngKSI (1 byte)
        ngksi_byte = struct.pack('B', self.ngksi & 0x0F)

        # Replayed UE security capability
        ue_sec_cap = self.replayed_ue_security_capability.encode()

        result = header.encode() + sec_alg + ngksi_byte + ue_sec_cap

        # Optional: IMEISV request
        if self.imeisv_request is not None:
            result += struct.pack('B', 0xE0 | (self.imeisv_request & 0x07))

        # Optional: ABBA
        if self.abba:
            result += struct.pack('BB', 0x38, len(self.abba)) + self.abba

        return result


@dataclass
class SecurityModeComplete:
    """5GMM Security Mode Complete (TS 24.501 Section 8.2.26)"""
    imeisv: Optional[bytes] = None
    nas_message_container: Optional[bytes] = None
    non_imeisv_pei: Optional[bytes] = None

    def encode(self) -> bytes:
        """Encode Security Mode Complete"""
        header = NASHeader(
            ExtendedProtocolDiscriminator.MOBILITY_MANAGEMENT_5G,
            SecurityHeaderType.PLAIN_NAS,
            MessageType5GMM.SECURITY_MODE_COMPLETE
        )

        result = header.encode()

        if self.imeisv:
            result += struct.pack('BB', 0x77, len(self.imeisv)) + self.imeisv

        if self.nas_message_container:
            length = len(self.nas_message_container)
            result += struct.pack('>BH', 0x71, length) + self.nas_message_container

        return result


@dataclass
class PDUSessionEstablishmentRequest:
    """5GSM PDU Session Establishment Request (TS 24.501 Section 8.3.1)"""
    pdu_session_id: int
    pti: int                 # Procedure Transaction Identity
    integrity_protection_max_data_rate: bytes  # 2 bytes
    pdu_session_type: Optional[int] = None
    ssc_mode: Optional[int] = None
    capability_5gsm: Optional[bytes] = None
    max_packet_filters: Optional[int] = None
    always_on_pdu_session: Optional[bool] = None
    sm_pdu_dn_request_container: Optional[bytes] = None
    extended_pco: Optional[bytes] = None
    header_compression_config: Optional[bytes] = None
    ds_tt_ethernet_port_mac: Optional[bytes] = None
    ue_ds_tt_residence_time: Optional[bytes] = None
    port_management_info_container: Optional[bytes] = None

    def encode(self) -> bytes:
        """Encode PDU Session Establishment Request"""
        header = NASHeader(
            ExtendedProtocolDiscriminator.SESSION_MANAGEMENT_5G,
            self.pdu_session_id,  # PDU session ID in security header field for 5GSM
            MessageType5GSM.PDU_SESSION_ESTABLISHMENT_REQUEST
        )

        # PTI
        pti_byte = struct.pack('B', self.pti)

        # Integrity protection maximum data rate (mandatory)
        ip_max_rate = self.integrity_protection_max_data_rate

        result = header.encode() + pti_byte + ip_max_rate

        # Optional IEs
        if self.pdu_session_type is not None:
            result += struct.pack('B', 0x90 | (self.pdu_session_type & 0x0F))

        if self.ssc_mode is not None:
            result += struct.pack('B', 0xA0 | (self.ssc_mode & 0x0F))

        if self.extended_pco:
            length = len(self.extended_pco)
            result += struct.pack('>BH', IEI.EXTENDED_PCO, length) + self.extended_pco

        return result


@dataclass
class PDUSessionEstablishmentAccept:
    """5GSM PDU Session Establishment Accept (TS 24.501 Section 8.3.2)"""
    pdu_session_id: int
    pti: int
    selected_pdu_session_type: int
    selected_ssc_mode: int
    authorized_qos_rules: bytes
    session_ambr: bytes       # 6 bytes
    pdu_address: Optional[bytes] = None
    rq_timer: Optional[int] = None
    s_nssai: Optional[SNSSAI] = None
    always_on_pdu_session: Optional[bool] = None
    mapped_eps_bearer_contexts: Optional[bytes] = None
    eap_message: Optional[bytes] = None
    authorized_qos_flow_descriptions: Optional[bytes] = None
    extended_pco: Optional[bytes] = None
    dnn: Optional[str] = None

    def encode(self) -> bytes:
        """Encode PDU Session Establishment Accept"""
        header = NASHeader(
            ExtendedProtocolDiscriminator.SESSION_MANAGEMENT_5G,
            self.pdu_session_id,
            MessageType5GSM.PDU_SESSION_ESTABLISHMENT_ACCEPT
        )

        # PTI
        pti_byte = struct.pack('B', self.pti)

        # Selected PDU session type and SSC mode (1 byte)
        type_ssc = ((self.selected_ssc_mode & 0x07) << 4) | (self.selected_pdu_session_type & 0x07)

        # Authorized QoS rules (TLV-E)
        qos_rules_len = len(self.authorized_qos_rules)
        qos_rules = struct.pack('>H', qos_rules_len) + self.authorized_qos_rules

        # Session AMBR
        session_ambr = struct.pack('B', 6) + self.session_ambr

        result = header.encode() + pti_byte + struct.pack('B', type_ssc) + qos_rules + session_ambr

        # Optional: PDU Address
        if self.pdu_address:
            result += struct.pack('BB', IEI.PDU_ADDRESS, len(self.pdu_address)) + self.pdu_address

        # Optional: DNN
        if self.dnn:
            dnn_bytes = self.dnn.encode('utf-8')
            result += struct.pack('BB', IEI.DNN, len(dnn_bytes)) + dnn_bytes

        # Optional: S-NSSAI
        if self.s_nssai:
            result += struct.pack('B', IEI.S_NSSAI) + self.s_nssai.encode()

        return result


# =============================================================================
# NAS Codec Class
# =============================================================================

class NASCodec:
    """
    NAS Protocol Codec for encoding/decoding 5G NAS messages.

    This class provides the main interface for working with NAS messages,
    including security protection/verification.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def decode_message(self, data: bytes) -> Dict[str, Any]:
        """
        Decode a NAS message from bytes.

        Args:
            data: Raw NAS message bytes

        Returns:
            Dictionary containing decoded message fields
        """
        if len(data) < 3:
            raise ValueError("NAS message too short")

        epd = data[0]
        security_header = data[1] & 0x0F

        # Check if security protected
        if security_header != SecurityHeaderType.PLAIN_NAS:
            # Parse security header
            sec_header, offset = SecurityHeader.decode(data)
            # The actual NAS message follows
            inner_data = data[offset:]
            return self._decode_plain_message(inner_data, sec_header)
        else:
            return self._decode_plain_message(data)

    def _decode_plain_message(self, data: bytes, security_header: Optional[SecurityHeader] = None) -> Dict[str, Any]:
        """Decode a plain (unprotected) NAS message"""
        header, offset = NASHeader.decode(data)

        result = {
            'protocol': 'NAS-5G',
            'epd': header.extended_protocol_discriminator,
            'security_header_type': header.security_header_type,
            'message_type': header.message_type,
            'security_header': security_header,
        }

        # Decode based on EPD
        if header.extended_protocol_discriminator == ExtendedProtocolDiscriminator.MOBILITY_MANAGEMENT_5G:
            result['message_type_name'] = MessageType5GMM(header.message_type).name
            result.update(self._decode_5gmm_message(header.message_type, data[offset:]))
        elif header.extended_protocol_discriminator == ExtendedProtocolDiscriminator.SESSION_MANAGEMENT_5G:
            result['message_type_name'] = MessageType5GSM(header.message_type).name
            result.update(self._decode_5gsm_message(header.message_type, data[offset:]))

        return result

    def _decode_5gmm_message(self, message_type: int, data: bytes) -> Dict[str, Any]:
        """Decode 5GMM specific message content"""
        if message_type == MessageType5GMM.REGISTRATION_REQUEST:
            msg = RegistrationRequest.decode(b'\x7e\x00' + bytes([message_type]) + data)
            return {
                'ngksi': msg.ngksi,
                'registration_type': msg.registration_type,
                'registration_type_name': RegistrationType(msg.registration_type).name,
            }
        elif message_type == MessageType5GMM.AUTHENTICATION_RESPONSE:
            msg = AuthenticationResponse.decode(b'\x7e\x00' + bytes([message_type]) + data)
            return {
                'res_star': msg.res_star.hex() if msg.res_star else None,
            }
        # Add more message types as needed
        return {}

    def _decode_5gsm_message(self, message_type: int, data: bytes) -> Dict[str, Any]:
        """Decode 5GSM specific message content"""
        # PDU session ID is in the header for 5GSM
        return {}

    def encode_registration_request(
        self,
        registration_type: RegistrationType,
        supi: str,
        plmn: PLMN,
        ngksi: int = 7,  # No key available
        requested_nssai: Optional[List[SNSSAI]] = None
    ) -> bytes:
        """
        Encode a Registration Request message.

        Args:
            registration_type: Type of registration
            supi: SUPI (e.g., "imsi-001010000000001")
            plmn: Serving PLMN
            ngksi: NAS Key Set Identifier (default 7 = no key)
            requested_nssai: List of requested S-NSSAIs

        Returns:
            Encoded NAS message bytes
        """
        # Create SUCI from SUPI (null protection for simplicity)
        msin = supi.split('-')[1][5:]  # Extract MSIN from IMSI
        suci = SUCI(
            supi_format=0,  # IMSI
            plmn=plmn,
            routing_indicator="0000",
            protection_scheme=0,  # Null scheme
            home_network_pki=0,
            scheme_output=msin.encode()
        )

        msg = RegistrationRequest(
            ngksi=ngksi,
            registration_type=registration_type,
            mobile_identity=suci,
            requested_nssai=requested_nssai
        )

        return msg.encode()

    def encode_registration_accept(
        self,
        registration_result: int,
        guti: Optional[GUTI5G] = None,
        allowed_nssai: Optional[List[SNSSAI]] = None
    ) -> bytes:
        """
        Encode a Registration Accept message.

        Args:
            registration_result: 5GS registration result
            guti: Allocated 5G-GUTI
            allowed_nssai: List of allowed S-NSSAIs

        Returns:
            Encoded NAS message bytes
        """
        msg = RegistrationAccept(
            registration_result=registration_result,
            guti_5g=guti,
            allowed_nssai=allowed_nssai
        )
        return msg.encode()

    def encode_authentication_request(
        self,
        ngksi: int,
        abba: bytes,
        rand: bytes,
        autn: bytes
    ) -> bytes:
        """
        Encode an Authentication Request message.

        Args:
            ngksi: NAS Key Set Identifier
            abba: ABBA parameter
            rand: RAND (16 bytes)
            autn: AUTN (16 bytes)

        Returns:
            Encoded NAS message bytes
        """
        msg = AuthenticationRequest(
            ngksi=ngksi,
            abba=abba,
            rand=rand,
            autn=autn
        )
        return msg.encode()

    def encode_authentication_response(self, res_star: bytes) -> bytes:
        """
        Encode an Authentication Response message.

        Args:
            res_star: RES* value (16 bytes)

        Returns:
            Encoded NAS message bytes
        """
        msg = AuthenticationResponse(res_star=res_star)
        return msg.encode()

    def encode_security_mode_command(
        self,
        selected_algorithms: int,
        ngksi: int,
        ue_security_capability: UESecurityCapability,
        abba: Optional[bytes] = None
    ) -> bytes:
        """
        Encode a Security Mode Command message.

        Args:
            selected_algorithms: Selected NAS security algorithms
            ngksi: NAS Key Set Identifier
            ue_security_capability: Replayed UE security capability
            abba: ABBA parameter

        Returns:
            Encoded NAS message bytes
        """
        msg = SecurityModeCommand(
            selected_nas_security_algorithms=selected_algorithms,
            ngksi=ngksi,
            replayed_ue_security_capability=ue_security_capability,
            abba=abba
        )
        return msg.encode()

    def encode_pdu_session_establishment_request(
        self,
        pdu_session_id: int,
        pti: int,
        pdu_session_type: PDUSessionType = PDUSessionType.IPV4,
        ssc_mode: int = 1
    ) -> bytes:
        """
        Encode a PDU Session Establishment Request message.

        Args:
            pdu_session_id: PDU Session ID
            pti: Procedure Transaction Identity
            pdu_session_type: Requested PDU session type
            ssc_mode: SSC mode

        Returns:
            Encoded NAS message bytes
        """
        msg = PDUSessionEstablishmentRequest(
            pdu_session_id=pdu_session_id,
            pti=pti,
            integrity_protection_max_data_rate=b'\xff\xff',  # Full rate
            pdu_session_type=pdu_session_type,
            ssc_mode=ssc_mode
        )
        return msg.encode()

    def encode_pdu_session_establishment_accept(
        self,
        pdu_session_id: int,
        pti: int,
        pdu_session_type: PDUSessionType,
        ssc_mode: int,
        qos_rules: bytes,
        session_ambr: bytes,
        pdu_address: Optional[bytes] = None,
        dnn: Optional[str] = None,
        s_nssai: Optional[SNSSAI] = None
    ) -> bytes:
        """
        Encode a PDU Session Establishment Accept message.

        Args:
            pdu_session_id: PDU Session ID
            pti: Procedure Transaction Identity
            pdu_session_type: Selected PDU session type
            ssc_mode: Selected SSC mode
            qos_rules: Authorized QoS rules
            session_ambr: Session AMBR (6 bytes)
            pdu_address: PDU address (IP address)
            dnn: Data Network Name
            s_nssai: S-NSSAI

        Returns:
            Encoded NAS message bytes
        """
        msg = PDUSessionEstablishmentAccept(
            pdu_session_id=pdu_session_id,
            pti=pti,
            selected_pdu_session_type=pdu_session_type,
            selected_ssc_mode=ssc_mode,
            authorized_qos_rules=qos_rules,
            session_ambr=session_ambr,
            pdu_address=pdu_address,
            dnn=dnn,
            s_nssai=s_nssai
        )
        return msg.encode()


# =============================================================================
# Utility Functions
# =============================================================================

def create_default_qos_rules(qfi: int = 1) -> bytes:
    """
    Create default QoS rules for a PDU session.

    Args:
        qfi: QoS Flow Identifier

    Returns:
        Encoded QoS rules bytes
    """
    # QoS Rule (TS 24.501 Section 9.11.4.13)
    # Simplified: Single rule allowing all traffic
    rule_id = 0x01
    rule_length = 0x06
    rule_operation = 0x21  # Create new QoS rule, DQR=1
    dqr_and_num_pf = 0x11  # DQR=1, num packet filters = 1
    pf_id = 0x10           # Packet filter ID with direction
    pf_length = 0x01
    pf_component = 0x01    # Match all
    rule_precedence = 0xFF
    qfi_byte = qfi & 0x3F

    return bytes([
        rule_id,
        0x00, rule_length,  # Length (2 bytes)
        rule_operation,
        dqr_and_num_pf,
        pf_id, pf_length, pf_component,
        rule_precedence,
        qfi_byte
    ])


def create_session_ambr(dl_mbps: int = 1000, ul_mbps: int = 1000) -> bytes:
    """
    Create Session AMBR IE.

    Args:
        dl_mbps: Downlink AMBR in Mbps
        ul_mbps: Uplink AMBR in Mbps

    Returns:
        Encoded Session AMBR bytes (6 bytes)
    """
    # Unit: 1 = 1 Kbps, 2 = 4 Kbps, ... (TS 24.501 Section 9.11.4.14)
    # For simplicity, use unit = 0x06 (1 Mbps)
    dl_unit = 0x06
    ul_unit = 0x06
    return struct.pack('>BHBH', dl_unit, dl_mbps, ul_unit, ul_mbps)


def create_pdu_address_ipv4(ip_address: str) -> bytes:
    """
    Create PDU Address IE for IPv4.

    Args:
        ip_address: IPv4 address string (e.g., "10.0.0.1")

    Returns:
        Encoded PDU address bytes
    """
    import socket
    pdu_session_type = PDUSessionType.IPV4
    ip_bytes = socket.inet_aton(ip_address)
    return bytes([pdu_session_type]) + ip_bytes


# =============================================================================
# Test/Demo Functions
# =============================================================================

def demo_nas_encoding():
    """Demonstrate NAS message encoding/decoding"""
    codec = NASCodec()
    plmn = PLMN("001", "01")

    # Create Registration Request
    print("=== Registration Request ===")
    reg_req = codec.encode_registration_request(
        registration_type=RegistrationType.INITIAL,
        supi="imsi-001010000000001",
        plmn=plmn,
        requested_nssai=[SNSSAI(sst=1, sd=bytes.fromhex("010203"))]
    )
    print(f"Encoded: {reg_req.hex()}")

    # Create Authentication Request
    print("\n=== Authentication Request ===")
    auth_req = codec.encode_authentication_request(
        ngksi=0,
        abba=bytes([0x00, 0x00]),
        rand=bytes(16),  # Random value
        autn=bytes(16)   # AUTN value
    )
    print(f"Encoded: {auth_req.hex()}")

    # Create PDU Session Establishment Accept
    print("\n=== PDU Session Establishment Accept ===")
    pdu_accept = codec.encode_pdu_session_establishment_accept(
        pdu_session_id=1,
        pti=1,
        pdu_session_type=PDUSessionType.IPV4,
        ssc_mode=1,
        qos_rules=create_default_qos_rules(),
        session_ambr=create_session_ambr(),
        pdu_address=create_pdu_address_ipv4("10.0.0.1"),
        dnn="internet"
    )
    print(f"Encoded: {pdu_accept.hex()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    demo_nas_encoding()
