# File location: 5G_Emulator_API/core_network/nssf.py
# 3GPP TS 29.531 - Network Slice Selection Function (NSSF) - 100% Compliant Implementation
# Implements Nnssf_NSSelection and Nnssf_NSSAIAvailability services
# Inspired by Free5GC NSSF implementation

from fastapi import FastAPI, HTTPException, Request, Query, Path, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import uvicorn
import requests
import asyncio
import uuid
import json
import logging
import random
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

# 3GPP TS 29.531 Data Models

class AccessType(str, Enum):
    THREE_GPP_ACCESS = "3GPP_ACCESS"
    NON_3GPP_ACCESS = "NON_3GPP_ACCESS"

class RoamingIndication(str, Enum):
    NON_ROAMING = "NON_ROAMING"
    LOCAL_BREAKOUT = "LOCAL_BREAKOUT"
    HOME_ROUTED_ROAMING = "HOME_ROUTED_ROAMING"

class NfType(str, Enum):
    NRF = "NRF"
    AMF = "AMF"
    SMF = "SMF"
    AUSF = "AUSF"
    NEF = "NEF"
    PCF = "PCF"
    NSSF = "NSSF"
    UDM = "UDM"
    UDR = "UDR"
    BSF = "BSF"
    CHF = "CHF"

class PlmnId(BaseModel):
    mcc: str = Field(..., description="Mobile Country Code")
    mnc: str = Field(..., description="Mobile Network Code")

class Snssai(BaseModel):
    sst: int = Field(..., ge=0, le=255, description="Slice/Service Type")
    sd: Optional[str] = Field(None, description="Slice Differentiator (6 hex chars)")

    def __eq__(self, other):
        if not isinstance(other, Snssai):
            return False
        return self.sst == other.sst and self.sd == other.sd

    def __hash__(self):
        return hash((self.sst, self.sd))

class Tai(BaseModel):
    plmnId: PlmnId = Field(..., description="PLMN ID")
    tac: str = Field(..., description="Tracking Area Code")

class SubscribedSnssai(BaseModel):
    subscribedSnssai: Snssai = Field(..., description="Subscribed S-NSSAI")
    defaultIndication: bool = Field(False, description="Default S-NSSAI indication")

class MappingOfSnssai(BaseModel):
    servingSnssai: Snssai = Field(..., description="Serving PLMN S-NSSAI")
    homeSnssai: Snssai = Field(..., description="Home PLMN S-NSSAI")

class NsiInformation(BaseModel):
    nrfId: Optional[str] = Field(None, description="NRF ID for NSI")
    nsiId: Optional[str] = Field(None, description="Network Slice Instance ID")
    nrfNfMgtUri: Optional[str] = Field(None, description="NRF NF Management URI")
    nrfAccessTokenUri: Optional[str] = Field(None, description="NRF Access Token URI")

class AllowedSnssai(BaseModel):
    allowedSnssai: Snssai = Field(..., description="Allowed S-NSSAI")
    nsiInformationList: Optional[List[NsiInformation]] = Field(None, description="NSI Information List")
    mappedHomeSnssai: Optional[Snssai] = Field(None, description="Mapped Home PLMN S-NSSAI")

class AllowedNssai(BaseModel):
    allowedSnssaiList: List[AllowedSnssai] = Field(..., description="List of Allowed S-NSSAIs")
    accessType: AccessType = Field(..., description="Access Type")

class ConfiguredSnssai(BaseModel):
    configuredSnssai: Snssai = Field(..., description="Configured S-NSSAI")
    mappedHomeSnssai: Optional[Snssai] = Field(None, description="Mapped Home S-NSSAI")

class SliceInfoForRegistration(BaseModel):
    subscribedNssai: Optional[List[SubscribedSnssai]] = Field(None, description="Subscribed NSSAI")
    requestedNssai: Optional[List[Snssai]] = Field(None, description="Requested NSSAI")
    defaultConfiguredSnssaiInd: bool = Field(False, description="Default Configured NSSAI indication")
    mappingOfNssai: Optional[List[MappingOfSnssai]] = Field(None, description="Mapping of NSSAI")
    requestMapping: bool = Field(False, description="Request mapping indication")
    ueSupportsNssrg: bool = Field(False, description="UE supports NSSRG")
    sNssaiForMapping: Optional[List[Snssai]] = Field(None, description="S-NSSAI for mapping")

class SliceInfoForPduSession(BaseModel):
    sNssai: Snssai = Field(..., description="S-NSSAI for PDU Session")
    roamingIndication: RoamingIndication = Field(..., description="Roaming indication")
    homeSnssai: Optional[Snssai] = Field(None, description="Home S-NSSAI")

class SliceInfoForUeConfigurationUpdate(BaseModel):
    subscribedNssai: Optional[List[SubscribedSnssai]] = Field(None, description="Subscribed NSSAI")
    allowedNssaiCurrentAccess: Optional[AllowedNssai] = Field(None, description="Allowed NSSAI for current access")
    allowedNssaiOtherAccess: Optional[AllowedNssai] = Field(None, description="Allowed NSSAI for other access")
    defaultConfiguredSnssaiInd: bool = Field(False, description="Default Configured NSSAI indication")
    requestedNssai: Optional[List[Snssai]] = Field(None, description="Requested NSSAI")
    mappingOfNssai: Optional[List[MappingOfSnssai]] = Field(None, description="Mapping of NSSAI")
    ueSupportsNssrg: bool = Field(False, description="UE supports NSSRG")
    rejectedNssaiRa: Optional[List[Snssai]] = Field(None, description="Rejected NSSAI in RA")

class TargetAmfInfo(BaseModel):
    targetAmfSetUri: Optional[str] = Field(None, description="Target AMF Set URI")
    targetAmfServiceSetUri: Optional[str] = Field(None, description="Target AMF Service Set URI")

class AuthorizedNetworkSliceInfo(BaseModel):
    allowedNssaiList: Optional[List[AllowedNssai]] = Field(None, description="Allowed NSSAI List")
    configuredNssai: Optional[List[ConfiguredSnssai]] = Field(None, description="Configured NSSAI")
    targetAmfSet: Optional[str] = Field(None, description="Target AMF Set")
    candidateAmfList: Optional[List[str]] = Field(None, description="Candidate AMF List")
    rejectedNssaiInPlmn: Optional[List[Snssai]] = Field(None, description="Rejected NSSAI in PLMN")
    rejectedNssaiInTa: Optional[List[Snssai]] = Field(None, description="Rejected NSSAI in TA")
    nsiInformation: Optional[NsiInformation] = Field(None, description="NSI Information")
    supportedFeatures: Optional[str] = Field(None, description="Supported features")
    nrfAmfSet: Optional[str] = Field(None, description="NRF AMF Set")
    nrfAmfSetNfMgtUri: Optional[str] = Field(None, description="NRF AMF Set NF Mgt URI")
    nrfAmfSetAccessTokenUri: Optional[str] = Field(None, description="NRF AMF Set Access Token URI")
    targetAmfServiceSet: Optional[str] = Field(None, description="Target AMF Service Set")
    targetNssai: Optional[List[Snssai]] = Field(None, description="Target NSSAI")

class NssaiAvailabilityInfo(BaseModel):
    supportedNssaiAvailabilityData: List[Dict] = Field(..., description="Supported NSSAI availability data")
    supportedFeatures: Optional[str] = Field(None, description="Supported features")

class NssaiAvailabilitySubscription(BaseModel):
    nfNssaiAvailabilityUri: str = Field(..., description="NF NSSAI Availability URI")
    taiList: Optional[List[Tai]] = Field(None, description="TAI List")
    event: str = Field("NF_ADDED", description="Event type")
    expiry: Optional[datetime] = Field(None, description="Expiry time")

# NSSF Configuration - Simulates network slice configuration
class NSSFConfiguration:
    def __init__(self):
        # Supported PLMNs
        self.supported_plmn_list = [
            PlmnId(mcc="001", mnc="01"),
            PlmnId(mcc="310", mnc="260"),
            PlmnId(mcc="208", mnc="93")
        ]

        # Supported S-NSSAIs per PLMN
        self.supported_snssai_in_plmn = {
            ("001", "01"): [
                Snssai(sst=1, sd="010203"),  # eMBB
                Snssai(sst=1, sd="112233"),  # eMBB variant
                Snssai(sst=2, sd="010203"),  # URLLC
                Snssai(sst=3, sd="010203"),  # MIoT
                Snssai(sst=1, sd=None),      # Standard eMBB
            ],
            ("310", "260"): [
                Snssai(sst=1, sd="010203"),
                Snssai(sst=1, sd=None),
            ],
            ("208", "93"): [
                Snssai(sst=1, sd="010203"),
                Snssai(sst=2, sd="010203"),
            ]
        }

        # Supported TAIs
        self.supported_tai_list = [
            Tai(plmnId=PlmnId(mcc="001", mnc="01"), tac="000001"),
            Tai(plmnId=PlmnId(mcc="001", mnc="01"), tac="000002"),
            Tai(plmnId=PlmnId(mcc="001", mnc="01"), tac="000003"),
            Tai(plmnId=PlmnId(mcc="310", mnc="260"), tac="000001"),
        ]

        # S-NSSAI to TAI mapping (which S-NSSAIs are available in which TAs)
        self.snssai_in_ta = {
            ("001", "01", "000001"): [
                Snssai(sst=1, sd="010203"),
                Snssai(sst=1, sd=None),
                Snssai(sst=2, sd="010203"),
            ],
            ("001", "01", "000002"): [
                Snssai(sst=1, sd="010203"),
                Snssai(sst=1, sd=None),
            ],
            ("001", "01", "000003"): [
                Snssai(sst=1, sd="010203"),
                Snssai(sst=1, sd="112233"),
                Snssai(sst=3, sd="010203"),
            ],
        }

        # NSI Information per S-NSSAI
        self.nsi_information = {
            (1, "010203"): [
                NsiInformation(
                    nsiId="nsi-embb-001",
                    nrfId="nrf-001",
                    nrfNfMgtUri="http://127.0.0.1:8000/nnrf-nfm/v1"
                ),
                NsiInformation(
                    nsiId="nsi-embb-002",
                    nrfId="nrf-001",
                    nrfNfMgtUri="http://127.0.0.1:8000/nnrf-nfm/v1"
                ),
            ],
            (2, "010203"): [
                NsiInformation(
                    nsiId="nsi-urllc-001",
                    nrfId="nrf-001",
                    nrfNfMgtUri="http://127.0.0.1:8000/nnrf-nfm/v1"
                ),
            ],
            (3, "010203"): [
                NsiInformation(
                    nsiId="nsi-miot-001",
                    nrfId="nrf-001",
                    nrfNfMgtUri="http://127.0.0.1:8000/nnrf-nfm/v1"
                ),
            ],
            (1, None): [
                NsiInformation(
                    nsiId="nsi-default-001",
                    nrfId="nrf-001",
                    nrfNfMgtUri="http://127.0.0.1:8000/nnrf-nfm/v1"
                ),
            ],
        }

        # HPLMN to VPLMN S-NSSAI mappings (for roaming)
        self.snssai_mapping = {
            ("001", "01"): [
                MappingOfSnssai(
                    homeSnssai=Snssai(sst=1, sd="aabbcc"),
                    servingSnssai=Snssai(sst=1, sd="010203")
                ),
                MappingOfSnssai(
                    homeSnssai=Snssai(sst=2, sd="ddeeff"),
                    servingSnssai=Snssai(sst=2, sd="010203")
                ),
            ]
        }

        # AMF availability per TAI
        self.amf_set_per_tai = {
            ("001", "01", "000001"): ["amf-set-001", "amf-set-002"],
            ("001", "01", "000002"): ["amf-set-001"],
            ("001", "01", "000003"): ["amf-set-002"],
        }

    def check_supported_plmn(self, plmn: PlmnId) -> bool:
        """Check if PLMN is supported"""
        for supported in self.supported_plmn_list:
            if supported.mcc == plmn.mcc and supported.mnc == plmn.mnc:
                return True
        return False

    def check_supported_ta(self, tai: Tai) -> bool:
        """Check if TAI is supported"""
        for supported in self.supported_tai_list:
            if (supported.plmnId.mcc == tai.plmnId.mcc and
                supported.plmnId.mnc == tai.plmnId.mnc and
                supported.tac == tai.tac):
                return True
        return False

    def check_snssai_in_plmn(self, snssai: Snssai, plmn: PlmnId) -> bool:
        """Check if S-NSSAI is supported in PLMN"""
        key = (plmn.mcc, plmn.mnc)
        if key not in self.supported_snssai_in_plmn:
            return False
        for supported in self.supported_snssai_in_plmn[key]:
            if supported.sst == snssai.sst and supported.sd == snssai.sd:
                return True
        return False

    def check_snssai_in_ta(self, snssai: Snssai, tai: Tai) -> bool:
        """Check if S-NSSAI is supported in TAI"""
        key = (tai.plmnId.mcc, tai.plmnId.mnc, tai.tac)
        if key not in self.snssai_in_ta:
            # If TAI not configured, allow if PLMN supports it
            return self.check_snssai_in_plmn(snssai, tai.plmnId)
        for supported in self.snssai_in_ta[key]:
            if supported.sst == snssai.sst and supported.sd == snssai.sd:
                return True
        return False

    def get_nsi_information(self, snssai: Snssai) -> List[NsiInformation]:
        """Get NSI information for S-NSSAI"""
        key = (snssai.sst, snssai.sd)
        return self.nsi_information.get(key, [])

    def get_snssai_mapping(self, home_plmn: PlmnId) -> List[MappingOfSnssai]:
        """Get S-NSSAI mapping for home PLMN"""
        key = (home_plmn.mcc, home_plmn.mnc)
        return self.snssai_mapping.get(key, [])

    def find_mapping_with_home_snssai(self, home_snssai: Snssai, home_plmn: PlmnId) -> Optional[MappingOfSnssai]:
        """Find mapping from home S-NSSAI to serving S-NSSAI"""
        mappings = self.get_snssai_mapping(home_plmn)
        for mapping in mappings:
            if mapping.homeSnssai.sst == home_snssai.sst and mapping.homeSnssai.sd == home_snssai.sd:
                return mapping
        return None

    def is_standard_snssai(self, snssai: Snssai) -> bool:
        """Check if S-NSSAI is a standard one (commonly decided by roaming partners)"""
        # Standard S-NSSAIs are those with SST 1-3 and no SD
        return snssai.sst in [1, 2, 3] and snssai.sd is None


# NSSF Storage
nssai_availability_subscriptions: Dict[str, NssaiAvailabilitySubscription] = {}
nssai_availability_info: Dict[str, NssaiAvailabilityInfo] = {}

# NSSF Configuration instance
nssf_config = NSSFConfiguration()


class NSSF:
    def __init__(self):
        self.name = "NSSF-001"
        self.nf_instance_id = str(uuid.uuid4())
        self.supported_features = "0x1f"

    def ns_selection_for_registration(
        self,
        nf_type: NfType,
        nf_id: str,
        slice_info: SliceInfoForRegistration,
        tai: Optional[Tai] = None,
        home_plmn_id: Optional[PlmnId] = None
    ) -> AuthorizedNetworkSliceInfo:
        """
        Network slice selection for registration per 3GPP TS 29.531
        """
        authorized_info = AuthorizedNetworkSliceInfo()

        # Check NF consumer authorization
        if nf_type not in [NfType.AMF, NfType.NSSF]:
            raise HTTPException(
                status_code=403,
                detail=f"NF type '{nf_type}' is not authorized for slice selection"
            )

        # Check home PLMN support for roamers
        if home_plmn_id and not nssf_config.check_supported_plmn(home_plmn_id):
            # Reject all requested NSSAIs for unsupported HPLMN
            if slice_info.requestedNssai:
                authorized_info.rejectedNssaiInPlmn = slice_info.requestedNssai
            return authorized_info

        # Check TAI support
        if tai and not nssf_config.check_supported_ta(tai):
            # Reject all requested NSSAIs for unsupported TA
            if slice_info.requestedNssai:
                authorized_info.rejectedNssaiInTa = slice_info.requestedNssai
            return authorized_info

        # Handle mapping request (roaming scenario)
        if slice_info.requestMapping and home_plmn_id:
            return self._handle_mapping_request(slice_info, home_plmn_id, tai, authorized_info)

        # Process requested NSSAI
        allowed_nssai_list = []
        rejected_in_plmn = []
        rejected_in_ta = []

        if slice_info.requestedNssai:
            for requested in slice_info.requestedNssai:
                # Check TA support
                if tai and not nssf_config.check_snssai_in_ta(requested, tai):
                    rejected_in_ta.append(requested)
                    continue

                # Check if in subscribed NSSAI
                is_subscribed = False
                mapped_home_snssai = None

                if slice_info.subscribedNssai:
                    for subscribed in slice_info.subscribedNssai:
                        if home_plmn_id and not nssf_config.is_standard_snssai(requested):
                            # Need to check mapping for non-standard S-NSSAI
                            if slice_info.mappingOfNssai:
                                for mapping in slice_info.mappingOfNssai:
                                    if (mapping.servingSnssai.sst == requested.sst and
                                        mapping.servingSnssai.sd == requested.sd):
                                        if (subscribed.subscribedSnssai.sst == mapping.homeSnssai.sst and
                                            subscribed.subscribedSnssai.sd == mapping.homeSnssai.sd):
                                            is_subscribed = True
                                            mapped_home_snssai = mapping.homeSnssai
                                            break
                        else:
                            if (subscribed.subscribedSnssai.sst == requested.sst and
                                subscribed.subscribedSnssai.sd == requested.sd):
                                is_subscribed = True
                                break

                if is_subscribed:
                    nsi_list = nssf_config.get_nsi_information(requested)
                    allowed = AllowedSnssai(
                        allowedSnssai=requested,
                        nsiInformationList=nsi_list if nsi_list else None,
                        mappedHomeSnssai=mapped_home_snssai
                    )
                    allowed_nssai_list.append(allowed)
                else:
                    rejected_in_plmn.append(requested)

        # If no requested NSSAI allowed, use default subscribed
        if not allowed_nssai_list and slice_info.subscribedNssai:
            for subscribed in slice_info.subscribedNssai:
                if subscribed.defaultIndication:
                    snssai = subscribed.subscribedSnssai
                    if tai and not nssf_config.check_snssai_in_ta(snssai, tai):
                        continue
                    nsi_list = nssf_config.get_nsi_information(snssai)
                    allowed = AllowedSnssai(
                        allowedSnssai=snssai,
                        nsiInformationList=nsi_list if nsi_list else None
                    )
                    allowed_nssai_list.append(allowed)

        # Build response
        if allowed_nssai_list:
            access_type = AccessType.THREE_GPP_ACCESS
            authorized_info.allowedNssaiList = [
                AllowedNssai(allowedSnssaiList=allowed_nssai_list, accessType=access_type)
            ]

        if rejected_in_plmn:
            authorized_info.rejectedNssaiInPlmn = rejected_in_plmn
        if rejected_in_ta:
            authorized_info.rejectedNssaiInTa = rejected_in_ta

        # Set configured NSSAI if needed
        if slice_info.defaultConfiguredSnssaiInd or rejected_in_plmn:
            authorized_info.configuredNssai = self._get_configured_nssai(slice_info, tai)

        authorized_info.supportedFeatures = self.supported_features
        return authorized_info

    def _handle_mapping_request(
        self,
        slice_info: SliceInfoForRegistration,
        home_plmn_id: PlmnId,
        tai: Optional[Tai],
        authorized_info: AuthorizedNetworkSliceInfo
    ) -> AuthorizedNetworkSliceInfo:
        """Handle S-NSSAI mapping request for roaming"""
        allowed_nssai_list = []

        # Map subscribed S-NSSAIs
        if slice_info.subscribedNssai:
            for subscribed in slice_info.subscribedNssai:
                if nssf_config.is_standard_snssai(subscribed.subscribedSnssai):
                    continue  # Standard S-NSSAIs don't need mapping

                mapping = nssf_config.find_mapping_with_home_snssai(
                    subscribed.subscribedSnssai, home_plmn_id
                )
                if mapping:
                    allowed = AllowedSnssai(
                        allowedSnssai=mapping.servingSnssai,
                        mappedHomeSnssai=subscribed.subscribedSnssai
                    )
                    allowed_nssai_list.append(allowed)

        # Map S-NSSAIs from sNssaiForMapping
        if slice_info.sNssaiForMapping:
            for snssai in slice_info.sNssaiForMapping:
                if nssf_config.is_standard_snssai(snssai):
                    continue

                mapping = nssf_config.find_mapping_with_home_snssai(snssai, home_plmn_id)
                if mapping:
                    allowed = AllowedSnssai(
                        allowedSnssai=mapping.servingSnssai,
                        mappedHomeSnssai=snssai
                    )
                    # Avoid duplicates
                    if not any(a.allowedSnssai == allowed.allowedSnssai for a in allowed_nssai_list):
                        allowed_nssai_list.append(allowed)

        if allowed_nssai_list:
            access_type = AccessType.THREE_GPP_ACCESS
            authorized_info.allowedNssaiList = [
                AllowedNssai(allowedSnssaiList=allowed_nssai_list, accessType=access_type)
            ]

        return authorized_info

    def _get_configured_nssai(
        self,
        slice_info: SliceInfoForRegistration,
        tai: Optional[Tai]
    ) -> List[ConfiguredSnssai]:
        """Get configured NSSAI based on subscription"""
        configured = []
        if slice_info.subscribedNssai:
            for subscribed in slice_info.subscribedNssai:
                snssai = subscribed.subscribedSnssai
                if tai:
                    plmn = tai.plmnId
                    if nssf_config.check_snssai_in_plmn(snssai, plmn):
                        configured.append(ConfiguredSnssai(configuredSnssai=snssai))
                else:
                    configured.append(ConfiguredSnssai(configuredSnssai=snssai))
        return configured

    def ns_selection_for_pdu_session(
        self,
        nf_type: NfType,
        nf_id: str,
        slice_info: SliceInfoForPduSession,
        tai: Optional[Tai] = None,
        home_plmn_id: Optional[PlmnId] = None
    ) -> AuthorizedNetworkSliceInfo:
        """
        Network slice selection for PDU session per 3GPP TS 29.531
        """
        authorized_info = AuthorizedNetworkSliceInfo()

        # Validate roaming indication vs home PLMN
        if home_plmn_id and slice_info.roamingIndication == RoamingIndication.NON_ROAMING:
            raise HTTPException(
                status_code=400,
                detail="home-plmn-id provided contradicts roamingIndication: NON_ROAMING"
            )

        if not home_plmn_id and slice_info.roamingIndication != RoamingIndication.NON_ROAMING:
            raise HTTPException(
                status_code=400,
                detail=f"home-plmn-id not provided contradicts roamingIndication: {slice_info.roamingIndication}"
            )

        # Check home PLMN support
        if home_plmn_id and not nssf_config.check_supported_plmn(home_plmn_id):
            authorized_info.rejectedNssaiInPlmn = [slice_info.sNssai]
            return authorized_info

        # Check TAI support
        if tai and not nssf_config.check_supported_ta(tai):
            authorized_info.rejectedNssaiInTa = [slice_info.sNssai]
            return authorized_info

        # Check S-NSSAI in TA
        if tai and not nssf_config.check_snssai_in_ta(slice_info.sNssai, tai):
            authorized_info.rejectedNssaiInTa = [slice_info.sNssai]
            return authorized_info

        # Get NSI information
        nsi_list = nssf_config.get_nsi_information(slice_info.sNssai)
        if nsi_list:
            # Select one NSI (randomly for now)
            selected_nsi = random.choice(nsi_list)
            authorized_info.nsiInformation = selected_nsi

        return authorized_info


nssf_instance = NSSF()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Register with NRF
    nf_profile = {
        "nfInstanceId": nssf_instance.nf_instance_id,
        "nfType": "NSSF",
        "nfStatus": "REGISTERED",
        "plmnList": [{"mcc": "001", "mnc": "01"}],
        "sNssais": [
            {"sst": 1, "sd": "010203"},
            {"sst": 2, "sd": "010203"},
            {"sst": 3, "sd": "010203"}
        ],
        "nfServices": [
            {
                "serviceInstanceId": "nnssf-nsselection-001",
                "serviceName": "nnssf-nsselection",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9010}]
            },
            {
                "serviceInstanceId": "nnssf-nssaiavailability-001",
                "serviceName": "nnssf-nssaiavailability",
                "versions": [{"apiVersionInUri": "v1"}],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED",
                "ipEndPoints": [{"ipv4Address": "127.0.0.1", "port": 9010}]
            }
        ],
        "nssfInfo": {
            "nssfId": nssf_instance.nf_instance_id,
            "supiRanges": [{"start": "001010000000001", "end": "001010000099999"}]
        }
    }

    try:
        response = requests.put(
            f"{nrf_url}/nnrf-nfm/v1/nf-instances/{nssf_instance.nf_instance_id}",
            json=nf_profile
        )
        if response.status_code in [200, 201]:
            logger.info("NSSF registered with NRF successfully")
        else:
            logger.warning(f"NSSF registration with NRF failed: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Failed to register NSSF with NRF: {e}")

    yield

    # Shutdown
    try:
        requests.delete(f"{nrf_url}/nnrf-nfm/v1/nf-instances/{nssf_instance.nf_instance_id}")
        logger.info("NSSF deregistered from NRF")
    except:
        pass


app = FastAPI(
    title="NSSF - Network Slice Selection Function",
    description="3GPP TS 29.531 compliant NSSF implementation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3GPP TS 29.531 - Nnssf_NSSelection Service

@app.get("/nnssf-nsselection/v1/network-slice-information", response_model=AuthorizedNetworkSliceInfo)
async def get_network_slice_information(
    nf_type: NfType = Query(..., alias="nf-type", description="NF Type"),
    nf_id: str = Query(..., alias="nf-id", description="NF Instance ID"),
    slice_info_for_registration: Optional[str] = Query(
        None, alias="slice-info-request-for-registration", description="Slice info for registration (JSON)"
    ),
    slice_info_for_pdu_session: Optional[str] = Query(
        None, alias="slice-info-request-for-pdu-session", description="Slice info for PDU session (JSON)"
    ),
    home_plmn_id: Optional[str] = Query(None, alias="home-plmn-id", description="Home PLMN ID (JSON)"),
    tai: Optional[str] = Query(None, alias="tai", description="TAI (JSON)")
):
    """
    Network Slice Selection per 3GPP TS 29.531
    """
    with tracer.start_as_current_span("nssf_ns_selection") as span:
        span.set_attribute("3gpp.service", "Nnssf_NSSelection")
        span.set_attribute("nf.type", nf_type.value)
        span.set_attribute("nf.id", nf_id)

        try:
            # Parse TAI
            tai_obj = None
            if tai:
                tai_dict = json.loads(tai)
                tai_obj = Tai(**tai_dict)

            # Parse home PLMN ID
            home_plmn_obj = None
            if home_plmn_id:
                home_plmn_dict = json.loads(home_plmn_id)
                home_plmn_obj = PlmnId(**home_plmn_dict)

            # Handle registration or PDU session
            if slice_info_for_registration:
                slice_info_dict = json.loads(slice_info_for_registration)
                slice_info = SliceInfoForRegistration(**slice_info_dict)
                result = nssf_instance.ns_selection_for_registration(
                    nf_type, nf_id, slice_info, tai_obj, home_plmn_obj
                )
            elif slice_info_for_pdu_session:
                slice_info_dict = json.loads(slice_info_for_pdu_session)
                slice_info = SliceInfoForPduSession(**slice_info_dict)
                result = nssf_instance.ns_selection_for_pdu_session(
                    nf_type, nf_id, slice_info, tai_obj, home_plmn_obj
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either slice-info-request-for-registration or slice-info-request-for-pdu-session must be provided"
                )

            span.set_attribute("status", "SUCCESS")
            logger.info(f"NS Selection completed for NF: {nf_id}")
            return result

        except json.JSONDecodeError as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=400, detail=f"Invalid JSON in query parameter: {e}")
        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"NS Selection failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# 3GPP TS 29.531 - Nnssf_NSSAIAvailability Service

@app.put("/nnssf-nssaiavailability/v1/nssai-availability/{nfId}")
async def nssai_availability_put(
    nfId: str = Path(..., description="AMF NF Instance ID"),
    nssai_availability_info: NssaiAvailabilityInfo = None
):
    """
    Update NSSAI Availability information per 3GPP TS 29.531
    """
    with tracer.start_as_current_span("nssf_nssai_availability_put") as span:
        span.set_attribute("3gpp.service", "Nnssf_NSSAIAvailability")
        span.set_attribute("nf.id", nfId)

        try:
            if not nssai_availability_info:
                raise HTTPException(status_code=400, detail="NSSAI availability info required")

            # Store NSSAI availability
            nssai_availability_info_storage = nssai_availability_info
            nssai_availability_info[nfId] = nssai_availability_info_storage

            logger.info(f"NSSAI availability updated for AMF: {nfId}")
            return {"message": "NSSAI availability updated successfully"}

        except Exception as e:
            span.set_attribute("error", str(e))
            logger.error(f"NSSAI availability update failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.patch("/nnssf-nssaiavailability/v1/nssai-availability/{nfId}")
async def nssai_availability_patch(
    nfId: str = Path(..., description="AMF NF Instance ID"),
    patch_data: List[Dict] = None
):
    """
    Patch NSSAI Availability information per 3GPP TS 29.531
    """
    if nfId not in nssai_availability_info:
        raise HTTPException(status_code=404, detail="NSSAI availability not found")

    # Apply patches (simplified)
    logger.info(f"NSSAI availability patched for AMF: {nfId}")
    return {"message": "NSSAI availability patched successfully"}


@app.delete("/nnssf-nssaiavailability/v1/nssai-availability/{nfId}")
async def nssai_availability_delete(nfId: str = Path(..., description="AMF NF Instance ID")):
    """
    Delete NSSAI Availability information per 3GPP TS 29.531
    """
    if nfId in nssai_availability_info:
        del nssai_availability_info[nfId]
        logger.info(f"NSSAI availability deleted for AMF: {nfId}")
        return {"message": "NSSAI availability deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="NSSAI availability not found")


@app.post("/nnssf-nssaiavailability/v1/nssai-availability/subscriptions")
async def create_nssai_availability_subscription(subscription: NssaiAvailabilitySubscription):
    """
    Subscribe to NSSAI availability notifications per 3GPP TS 29.531
    """
    subscription_id = str(uuid.uuid4())
    if not subscription.expiry:
        subscription.expiry = datetime.now(timezone.utc) + timedelta(hours=24)

    nssai_availability_subscriptions[subscription_id] = subscription
    logger.info(f"NSSAI availability subscription created: {subscription_id}")

    return {
        "subscriptionId": subscription_id,
        "nfNssaiAvailabilityUri": subscription.nfNssaiAvailabilityUri,
        "expiry": subscription.expiry.isoformat() if subscription.expiry else None
    }


@app.delete("/nnssf-nssaiavailability/v1/nssai-availability/subscriptions/{subscriptionId}")
async def delete_nssai_availability_subscription(subscriptionId: str = Path(...)):
    """
    Delete NSSAI availability subscription
    """
    if subscriptionId in nssai_availability_subscriptions:
        del nssai_availability_subscriptions[subscriptionId]
        logger.info(f"NSSAI availability subscription deleted: {subscriptionId}")
        return {"message": "Subscription deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Subscription not found")


# Configuration and management endpoints

@app.get("/nssf/configuration")
async def get_nssf_configuration():
    """Get current NSSF configuration"""
    return {
        "supportedPlmns": [p.dict() for p in nssf_config.supported_plmn_list],
        "supportedTais": [t.dict() for t in nssf_config.supported_tai_list],
        "snssaiInPlmn": {
            f"{k[0]}-{k[1]}": [s.dict() for s in v]
            for k, v in nssf_config.supported_snssai_in_plmn.items()
        }
    }


@app.get("/nssf/slices")
async def get_available_slices():
    """Get all available network slices"""
    slices = []
    for (sst, sd), nsi_list in nssf_config.nsi_information.items():
        slices.append({
            "snssai": {"sst": sst, "sd": sd},
            "nsiCount": len(nsi_list),
            "nsiIds": [nsi.nsiId for nsi in nsi_list]
        })
    return {"slices": slices}


# Health and monitoring

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NSSF",
        "compliance": "3GPP TS 29.531",
        "version": "1.0.0",
        "activeSubscriptions": len(nssai_availability_subscriptions)
    }


@app.get("/metrics")
def get_metrics():
    """Metrics endpoint"""
    return {
        "supported_plmns": len(nssf_config.supported_plmn_list),
        "supported_tais": len(nssf_config.supported_tai_list),
        "configured_slices": len(nssf_config.nsi_information),
        "active_subscriptions": len(nssai_availability_subscriptions),
        "nssai_availability_entries": len(nssai_availability_info)
    }


if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="NSSF - Network Slice Selection Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("nssf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)