#!/usr/bin/env python3
"""
E2 Application Protocol (E2AP) Implementation
ETSI TS 104039 Compliant

This module provides E2AP message creation and handling for:
- E2 Setup procedures
- RIC Subscription procedures
- RIC Indication handling
- RIC Control procedures
- RIC Query procedures
- Error handling

The E2AP is the application protocol used over the E2 interface between
Near-RT RIC and E2 Nodes (O-CU, O-DU, O-eNB).
"""

import uuid
from datetime import datetime
from enum import IntEnum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# =============================================================================
# E2AP Procedure Codes (per ETSI TS 104039 Section 9.3.3)
# =============================================================================

class E2apProcedureCode(IntEnum):
    """E2AP Elementary Procedure codes per ETSI TS 104039"""
    # RIC-initiated procedures
    E2_SETUP = 1
    RIC_SUBSCRIPTION = 8
    RIC_SUBSCRIPTION_DELETE = 9
    RIC_SUBSCRIPTION_MODIFICATION = 10
    RIC_CONTROL = 4
    RIC_QUERY = 6
    RIC_SERVICE_UPDATE = 7

    # E2 Node-initiated procedures
    RIC_INDICATION = 5
    RIC_SUBSCRIPTION_DELETE_REQUIRED = 11
    E2_NODE_CONFIGURATION_UPDATE = 3
    E2_REMOVAL = 12
    E2_CONNECTION_UPDATE = 13

    # Common procedures
    RESET = 2
    ERROR_INDICATION = 14


class E2apCriticality(IntEnum):
    """Criticality values per ETSI TS 104039"""
    REJECT = 0
    IGNORE = 1
    NOTIFY = 2


class E2apMessageType(IntEnum):
    """E2AP Message types"""
    INITIATING_MESSAGE = 0
    SUCCESSFUL_OUTCOME = 1
    UNSUCCESSFUL_OUTCOME = 2


# =============================================================================
# E2AP Information Elements (IEs)
# =============================================================================

class RicRequestId(BaseModel):
    """RIC Request ID per ETSI TS 104039 Section 9.2.6"""
    ricRequestorId: int = Field(..., ge=0, le=65535, description="RIC Requestor ID")
    ricInstanceId: int = Field(..., ge=0, le=65535, description="RIC Instance ID")


class GlobalE2NodeId(BaseModel):
    """Global E2 Node ID per ETSI TS 104039 Section 9.2.3"""
    plmnId: str = Field(..., description="PLMN Identity (3 bytes hex)")
    nodeIdType: str = Field(..., description="Node ID type (gNB, gNB-CU, gNB-DU, etc.)")
    nodeId: str = Field(..., description="Node ID value")


class GlobalRicId(BaseModel):
    """Global RIC ID per ETSI TS 104039 Section 9.2.4"""
    plmnId: str = Field(..., description="PLMN Identity")
    ricId: str = Field(..., description="RIC ID (20 bits)")


class RanFunctionDefinition(BaseModel):
    """RAN Function definition per ETSI TS 104039"""
    ranFunctionId: int = Field(..., ge=0, le=4095)
    ranFunctionDefinition: bytes = Field(..., description="E2SM-encoded definition")
    ranFunctionRevision: int = Field(default=1)
    ranFunctionOid: Optional[str] = Field(default=None)


class CauseRic(BaseModel):
    """RIC-related cause values"""
    causeType: str = Field(..., description="Cause type")
    causeValue: int = Field(..., description="Cause value")


class TimeToWait(IntEnum):
    """Time to wait values (in seconds)"""
    V1S = 1
    V2S = 2
    V5S = 5
    V10S = 10
    V20S = 20
    V60S = 60


# =============================================================================
# E2AP Message Base Classes
# =============================================================================

class E2apMessage(BaseModel):
    """Base E2AP message structure"""
    procedureCode: E2apProcedureCode
    criticality: E2apCriticality = E2apCriticality.REJECT
    messageType: E2apMessageType
    transactionId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class E2apProcedure:
    """E2AP Procedure handler base class"""

    @staticmethod
    def create_transaction_id() -> str:
        return str(uuid.uuid4())[:8]


# =============================================================================
# E2 Setup Procedure (ETSI TS 104039 Section 8.2.1)
# =============================================================================

class E2SetupRequestIEs(BaseModel):
    """E2 Setup Request IEs"""
    globalE2NodeId: GlobalE2NodeId
    ranFunctionsAdded: List[RanFunctionDefinition] = Field(default=[])
    e2NodeComponentConfigAddition: Optional[List[Dict]] = Field(default=None)
    transactionId: str = Field(default_factory=E2apProcedure.create_transaction_id)


class E2SetupResponseIEs(BaseModel):
    """E2 Setup Response IEs"""
    globalRicId: GlobalRicId
    ranFunctionsAccepted: List[Dict] = Field(default=[])
    ranFunctionsRejected: List[Dict] = Field(default=[])
    e2NodeComponentConfigAdditionAck: Optional[List[Dict]] = Field(default=None)
    transactionId: str


class E2SetupFailureIEs(BaseModel):
    """E2 Setup Failure IEs"""
    cause: CauseRic
    timeToWait: Optional[TimeToWait] = Field(default=None)
    criticalityDiagnostics: Optional[Dict] = Field(default=None)
    transactionId: str


def create_e2_setup_request(
    global_e2_node_id: GlobalE2NodeId,
    ran_functions: List[RanFunctionDefinition],
    e2_node_config: Optional[List[Dict]] = None
) -> Dict:
    """
    Create E2 Setup Request message

    Per ETSI TS 104039 Section 8.2.1.2:
    This message is sent by E2 Node to Near-RT RIC to establish E2 interface.
    """
    transaction_id = E2apProcedure.create_transaction_id()

    return {
        "procedureCode": E2apProcedureCode.E2_SETUP,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.INITIATING_MESSAGE,
        "transactionId": transaction_id,
        "protocolIEs": {
            "globalE2NodeId": global_e2_node_id.model_dump(),
            "ranFunctionsAdded": [rf.model_dump() for rf in ran_functions],
            "e2NodeComponentConfigAddition": e2_node_config
        }
    }


def create_e2_setup_response(
    global_ric_id: GlobalRicId,
    ran_functions_accepted: List[Dict],
    ran_functions_rejected: List[Dict],
    transaction_id: str
) -> Dict:
    """
    Create E2 Setup Response message

    Per ETSI TS 104039 Section 8.2.1.3
    """
    return {
        "procedureCode": E2apProcedureCode.E2_SETUP,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.SUCCESSFUL_OUTCOME,
        "transactionId": transaction_id,
        "protocolIEs": {
            "globalRicId": global_ric_id.model_dump(),
            "ranFunctionsAccepted": ran_functions_accepted,
            "ranFunctionsRejected": ran_functions_rejected
        }
    }


def create_e2_setup_failure(
    cause: CauseRic,
    transaction_id: str,
    time_to_wait: Optional[TimeToWait] = None
) -> Dict:
    """Create E2 Setup Failure message"""
    return {
        "procedureCode": E2apProcedureCode.E2_SETUP,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.UNSUCCESSFUL_OUTCOME,
        "transactionId": transaction_id,
        "protocolIEs": {
            "cause": cause.model_dump(),
            "timeToWait": time_to_wait
        }
    }


# =============================================================================
# RIC Subscription Procedure (ETSI TS 104039 Section 8.2.3)
# =============================================================================

class RicEventTriggerDefinition(BaseModel):
    """RIC Event Trigger Definition"""
    eventTriggerType: str = Field(..., description="Trigger type")
    eventTriggerDefinition: Dict = Field(..., description="E2SM-encoded trigger")


class RicActionDefinition(BaseModel):
    """RIC Action Definition"""
    ricActionId: int = Field(..., ge=0, le=255)
    ricActionType: str = Field(..., description="REPORT, INSERT, POLICY")
    ricActionDefinition: Optional[Dict] = Field(default=None)
    ricSubsequentAction: Optional[Dict] = Field(default=None)


class RicSubscriptionRequestIEs(BaseModel):
    """RIC Subscription Request IEs"""
    ricRequestId: RicRequestId
    ranFunctionId: int
    ricEventTriggerDefinition: RicEventTriggerDefinition
    ricActionToBeSetupList: List[RicActionDefinition]


class RicSubscriptionResponseIEs(BaseModel):
    """RIC Subscription Response IEs"""
    ricRequestId: RicRequestId
    ranFunctionId: int
    ricActionAdmittedList: List[Dict]
    ricActionNotAdmittedList: Optional[List[Dict]] = Field(default=None)


def create_ric_subscription_request(
    ric_request_id: RicRequestId,
    ran_function_id: int,
    event_trigger: RicEventTriggerDefinition,
    action_list: List[RicActionDefinition]
) -> Dict:
    """
    Create RIC Subscription Request

    Per ETSI TS 104039 Section 8.2.3.2:
    Sent by Near-RT RIC to E2 Node to subscribe for reporting/control.
    """
    return {
        "procedureCode": E2apProcedureCode.RIC_SUBSCRIPTION,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.INITIATING_MESSAGE,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id,
            "ricEventTriggerDefinition": event_trigger.model_dump(),
            "ricActionToBeSetupList": [a.model_dump() for a in action_list]
        }
    }


def create_ric_subscription_response(
    ric_request_id: RicRequestId,
    ran_function_id: int,
    admitted_actions: List[int],
    not_admitted_actions: Optional[List[Dict]] = None
) -> Dict:
    """Create RIC Subscription Response"""
    return {
        "procedureCode": E2apProcedureCode.RIC_SUBSCRIPTION,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.SUCCESSFUL_OUTCOME,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id,
            "ricActionAdmittedList": [{"ricActionId": a} for a in admitted_actions],
            "ricActionNotAdmittedList": not_admitted_actions
        }
    }


def create_ric_subscription_failure(
    ric_request_id: RicRequestId,
    ran_function_id: int,
    cause: CauseRic,
    not_admitted_actions: List[Dict]
) -> Dict:
    """Create RIC Subscription Failure"""
    return {
        "procedureCode": E2apProcedureCode.RIC_SUBSCRIPTION,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.UNSUCCESSFUL_OUTCOME,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id,
            "cause": cause.model_dump(),
            "ricActionNotAdmittedList": not_admitted_actions
        }
    }


# =============================================================================
# RIC Subscription Delete Procedure (ETSI TS 104039 Section 8.2.4)
# =============================================================================

def create_ric_subscription_delete_request(
    ric_request_id: RicRequestId,
    ran_function_id: int
) -> Dict:
    """Create RIC Subscription Delete Request"""
    return {
        "procedureCode": E2apProcedureCode.RIC_SUBSCRIPTION_DELETE,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.INITIATING_MESSAGE,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id
        }
    }


def create_ric_subscription_delete_response(
    ric_request_id: RicRequestId,
    ran_function_id: int
) -> Dict:
    """Create RIC Subscription Delete Response"""
    return {
        "procedureCode": E2apProcedureCode.RIC_SUBSCRIPTION_DELETE,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.SUCCESSFUL_OUTCOME,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id
        }
    }


# =============================================================================
# RIC Indication Procedure (ETSI TS 104039 Section 8.2.5)
# =============================================================================

class RicIndicationType(IntEnum):
    """RIC Indication types"""
    REPORT = 0
    INSERT = 1


class RicIndicationIEs(BaseModel):
    """RIC Indication IEs"""
    ricRequestId: RicRequestId
    ranFunctionId: int
    ricActionId: int
    ricIndicationType: RicIndicationType
    ricIndicationHeader: bytes
    ricIndicationMessage: bytes
    ricCallProcessId: Optional[bytes] = Field(default=None)


def create_ric_indication(
    ric_request_id: RicRequestId,
    ran_function_id: int,
    action_id: int,
    indication_type: RicIndicationType,
    indication_header: Dict,
    indication_message: Dict,
    call_process_id: Optional[str] = None
) -> Dict:
    """
    Create RIC Indication message

    Per ETSI TS 104039 Section 8.2.5.2:
    Sent by E2 Node to Near-RT RIC for REPORT or INSERT service.
    """
    return {
        "procedureCode": E2apProcedureCode.RIC_INDICATION,
        "criticality": E2apCriticality.IGNORE,
        "messageType": E2apMessageType.INITIATING_MESSAGE,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id,
            "ricActionId": action_id,
            "ricIndicationType": indication_type,
            "ricIndicationHeader": indication_header,
            "ricIndicationMessage": indication_message,
            "ricCallProcessId": call_process_id
        }
    }


# =============================================================================
# RIC Control Procedure (ETSI TS 104039 Section 8.2.6)
# =============================================================================

class RicControlAckRequest(IntEnum):
    """Control acknowledgment request"""
    NO_ACK = 0
    ACK = 1


class RicControlIEs(BaseModel):
    """RIC Control Request IEs"""
    ricRequestId: RicRequestId
    ranFunctionId: int
    ricCallProcessId: Optional[bytes] = Field(default=None)
    ricControlHeader: bytes
    ricControlMessage: bytes
    ricControlAckRequest: RicControlAckRequest = RicControlAckRequest.ACK


def create_ric_control_request(
    ric_request_id: RicRequestId,
    ran_function_id: int,
    control_header: Dict,
    control_message: Dict,
    ack_request: RicControlAckRequest = RicControlAckRequest.ACK,
    call_process_id: Optional[str] = None
) -> Dict:
    """
    Create RIC Control Request

    Per ETSI TS 104039 Section 8.2.6.2:
    Sent by Near-RT RIC to E2 Node to execute control action.
    """
    return {
        "procedureCode": E2apProcedureCode.RIC_CONTROL,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.INITIATING_MESSAGE,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id,
            "ricControlHeader": control_header,
            "ricControlMessage": control_message,
            "ricControlAckRequest": ack_request,
            "ricCallProcessId": call_process_id
        }
    }


def create_ric_control_acknowledge(
    ric_request_id: RicRequestId,
    ran_function_id: int,
    control_outcome: Optional[Dict] = None,
    call_process_id: Optional[str] = None
) -> Dict:
    """Create RIC Control Acknowledge"""
    return {
        "procedureCode": E2apProcedureCode.RIC_CONTROL,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.SUCCESSFUL_OUTCOME,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id,
            "ricControlOutcome": control_outcome,
            "ricCallProcessId": call_process_id
        }
    }


def create_ric_control_failure(
    ric_request_id: RicRequestId,
    ran_function_id: int,
    cause: CauseRic,
    control_outcome: Optional[Dict] = None
) -> Dict:
    """Create RIC Control Failure"""
    return {
        "procedureCode": E2apProcedureCode.RIC_CONTROL,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.UNSUCCESSFUL_OUTCOME,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id,
            "cause": cause.model_dump(),
            "ricControlOutcome": control_outcome
        }
    }


# =============================================================================
# RIC Query Procedure (ETSI TS 104039 Section 8.2.7)
# =============================================================================

def create_ric_query_request(
    ric_request_id: RicRequestId,
    ran_function_id: int,
    query_header: Dict,
    query_definition: Dict
) -> Dict:
    """
    Create RIC Query Request

    Per ETSI TS 104039 Section 8.2.7:
    Sent by Near-RT RIC to query E2 Node for specific information.
    """
    return {
        "procedureCode": E2apProcedureCode.RIC_QUERY,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.INITIATING_MESSAGE,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id,
            "ricQueryHeader": query_header,
            "ricQueryDefinition": query_definition
        }
    }


def create_ric_query_response(
    ric_request_id: RicRequestId,
    ran_function_id: int,
    query_outcome: Dict
) -> Dict:
    """Create RIC Query Response"""
    return {
        "procedureCode": E2apProcedureCode.RIC_QUERY,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.SUCCESSFUL_OUTCOME,
        "protocolIEs": {
            "ricRequestId": ric_request_id.model_dump(),
            "ranFunctionId": ran_function_id,
            "ricQueryOutcome": query_outcome
        }
    }


# =============================================================================
# Error Indication Procedure (ETSI TS 104039 Section 8.2.8)
# =============================================================================

class CauseType(str):
    """Cause types for error indication"""
    RIC_REQUEST = "ricRequest"
    RIC_SERVICE = "ricService"
    E2_NODE = "e2Node"
    TRANSPORT = "transport"
    PROTOCOL = "protocol"
    MISC = "misc"


def create_error_indication(
    cause_type: str,
    cause_value: int,
    ric_request_id: Optional[RicRequestId] = None,
    ran_function_id: Optional[int] = None,
    criticality_diagnostics: Optional[Dict] = None
) -> Dict:
    """
    Create Error Indication

    Per ETSI TS 104039 Section 8.2.8:
    Sent by either Near-RT RIC or E2 Node to report error conditions.
    """
    protocol_ies = {
        "cause": {
            "causeType": cause_type,
            "causeValue": cause_value
        }
    }

    if ric_request_id:
        protocol_ies["ricRequestId"] = ric_request_id.model_dump()
    if ran_function_id is not None:
        protocol_ies["ranFunctionId"] = ran_function_id
    if criticality_diagnostics:
        protocol_ies["criticalityDiagnostics"] = criticality_diagnostics

    return {
        "procedureCode": E2apProcedureCode.ERROR_INDICATION,
        "criticality": E2apCriticality.IGNORE,
        "messageType": E2apMessageType.INITIATING_MESSAGE,
        "protocolIEs": protocol_ies
    }


# =============================================================================
# Reset Procedure (ETSI TS 104039 Section 8.2.9)
# =============================================================================

class ResetType(str):
    """Reset types"""
    E2_INTERFACE = "e2Interface"
    PART_OF_E2_INTERFACE = "partOfE2Interface"


def create_reset_request(
    reset_type: str,
    cause: CauseRic
) -> Dict:
    """Create Reset Request"""
    return {
        "procedureCode": E2apProcedureCode.RESET,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.INITIATING_MESSAGE,
        "transactionId": E2apProcedure.create_transaction_id(),
        "protocolIEs": {
            "cause": cause.model_dump(),
            "resetType": reset_type
        }
    }


def create_reset_response(
    transaction_id: str
) -> Dict:
    """Create Reset Response"""
    return {
        "procedureCode": E2apProcedureCode.RESET,
        "criticality": E2apCriticality.REJECT,
        "messageType": E2apMessageType.SUCCESSFUL_OUTCOME,
        "transactionId": transaction_id,
        "protocolIEs": {}
    }


# =============================================================================
# E2AP Message Parser
# =============================================================================

class E2apMessageParser:
    """Parse and validate E2AP messages"""

    @staticmethod
    def parse_message(message: Dict) -> Dict:
        """Parse raw E2AP message and validate structure"""
        required_fields = ["procedureCode", "messageType", "protocolIEs"]

        for field in required_fields:
            if field not in message:
                raise ValueError(f"Missing required field: {field}")

        procedure_code = message["procedureCode"]
        message_type = message["messageType"]

        # Validate procedure code
        try:
            E2apProcedureCode(procedure_code)
        except ValueError:
            raise ValueError(f"Invalid procedure code: {procedure_code}")

        # Validate message type
        try:
            E2apMessageType(message_type)
        except ValueError:
            raise ValueError(f"Invalid message type: {message_type}")

        return {
            "procedureCode": E2apProcedureCode(procedure_code),
            "messageType": E2apMessageType(message_type),
            "protocolIEs": message["protocolIEs"],
            "transactionId": message.get("transactionId")
        }

    @staticmethod
    def get_procedure_name(procedure_code: E2apProcedureCode) -> str:
        """Get human-readable procedure name"""
        names = {
            E2apProcedureCode.E2_SETUP: "E2 Setup",
            E2apProcedureCode.RIC_SUBSCRIPTION: "RIC Subscription",
            E2apProcedureCode.RIC_SUBSCRIPTION_DELETE: "RIC Subscription Delete",
            E2apProcedureCode.RIC_INDICATION: "RIC Indication",
            E2apProcedureCode.RIC_CONTROL: "RIC Control",
            E2apProcedureCode.RIC_QUERY: "RIC Query",
            E2apProcedureCode.ERROR_INDICATION: "Error Indication",
            E2apProcedureCode.RESET: "Reset"
        }
        return names.get(procedure_code, f"Unknown ({procedure_code})")


# =============================================================================
# E2SM Service Model Registry
# =============================================================================

class E2smRegistry:
    """
    E2SM (E2 Service Model) Registry

    Tracks supported E2 Service Models per ETSI TS 104040.
    Common service models:
    - E2SM-KPM: Key Performance Measurement
    - E2SM-RC: RAN Control
    - E2SM-NI: Network Interface
    """

    # Standard E2SM OIDs
    E2SM_KPM_OID = "1.3.6.1.4.1.53148.1.2.2.2"  # KPM v2.0
    E2SM_RC_OID = "1.3.6.1.4.1.53148.1.2.2.3"   # RC v1.0
    E2SM_NI_OID = "1.3.6.1.4.1.53148.1.2.2.1"   # NI v1.0

    def __init__(self):
        self.service_models: Dict[str, Dict] = {
            self.E2SM_KPM_OID: {
                "name": "E2SM-KPM",
                "version": "2.0",
                "description": "Key Performance Measurement Service Model",
                "supportedActions": ["REPORT"]
            },
            self.E2SM_RC_OID: {
                "name": "E2SM-RC",
                "version": "1.0",
                "description": "RAN Control Service Model",
                "supportedActions": ["REPORT", "INSERT", "CONTROL", "POLICY"]
            }
        }

    def get_service_model(self, oid: str) -> Optional[Dict]:
        """Get service model by OID"""
        return self.service_models.get(oid)

    def is_supported(self, oid: str) -> bool:
        """Check if service model is supported"""
        return oid in self.service_models

    def list_service_models(self) -> List[Dict]:
        """List all supported service models"""
        return [
            {"oid": oid, **info}
            for oid, info in self.service_models.items()
        ]


# Global E2SM registry instance
e2sm_registry = E2smRegistry()
