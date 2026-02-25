"""
5G Protocol Implementations

This package provides real 3GPP protocol implementations:
- nas: 5G NAS protocol (TS 24.501)
- crypto: 5G-AKA authentication (TS 33.501)
- pfcp: PFCP protocol for N4 interface (TS 29.244)
- sctp: SCTP transport for NGAP (TS 38.412)
"""

from .nas.nas_5g import (
    NASCodec,
    RegistrationRequest,
    RegistrationAccept,
    AuthenticationRequest,
    AuthenticationResponse,
    SecurityModeCommand,
    PDUSessionEstablishmentRequest,
    PDUSessionEstablishmentAccept,
    PLMN,
    SNSSAI,
    GUTI5G,
    MessageType5GMM,
    MessageType5GSM,
    RegistrationType,
    PDUSessionType,
)

from .crypto.aka_5g import (
    AKA5GHandler,
    Milenage,
    KeyDerivation5G,
    AuthVector5G,
    SubscriberData,
    SUCIHandler,
)

from .pfcp.pfcp import (
    PFCPCodec,
    PFCPNode,
    PFCPTransport,
    SessionEstablishmentRequest,
    SessionEstablishmentResponse,
    CreatePDR,
    CreateFAR,
    CreateQER,
    MessageType as PFCPMessageType,
    CauseValue as PFCPCauseValue,
)

from .sctp.sctp_transport import (
    NGAPSCTPHandler,
    SCTPMessage,
    SCTPAssociation,
    create_sctp_transport,
    NGAP_SCTP_PORT,
    NGAP_PPID,
)

__all__ = [
    # NAS
    'NASCodec',
    'RegistrationRequest',
    'RegistrationAccept',
    'AuthenticationRequest',
    'AuthenticationResponse',
    'SecurityModeCommand',
    'PDUSessionEstablishmentRequest',
    'PDUSessionEstablishmentAccept',
    'PLMN',
    'SNSSAI',
    'GUTI5G',
    'MessageType5GMM',
    'MessageType5GSM',
    'RegistrationType',
    'PDUSessionType',
    # Crypto
    'AKA5GHandler',
    'Milenage',
    'KeyDerivation5G',
    'AuthVector5G',
    'SubscriberData',
    'SUCIHandler',
    # PFCP
    'PFCPCodec',
    'PFCPNode',
    'PFCPTransport',
    'SessionEstablishmentRequest',
    'SessionEstablishmentResponse',
    'CreatePDR',
    'CreateFAR',
    'CreateQER',
    'PFCPMessageType',
    'PFCPCauseValue',
    # SCTP
    'NGAPSCTPHandler',
    'SCTPMessage',
    'SCTPAssociation',
    'create_sctp_transport',
    'NGAP_SCTP_PORT',
    'NGAP_PPID',
]
