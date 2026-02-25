# File: core_network/ipsec.py
# IPSec/XFRM Kernel Integration Module
# Inspired by Free5GC n3iwf/internal/ike/xfrm/xfrm.go
# Provides kernel-level IPSec tunnel management using Linux XFRM

import os
import sys
import logging
import subprocess
import struct
import secrets
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# Constants - Based on Linux XFRM and IKEv2 definitions
# ============================================================================

class XfrmProto(Enum):
    """XFRM Protocol Types"""
    ESP = 50  # Encapsulating Security Payload
    AH = 51   # Authentication Header
    COMP = 108  # IP Compression

class XfrmMode(Enum):
    """XFRM Mode"""
    TRANSPORT = 0
    TUNNEL = 1
    ROUTE_OPTIMIZATION = 2
    IN_TRIGGER = 3
    BEET = 4

class XfrmDir(Enum):
    """XFRM Policy Direction"""
    IN = 0
    OUT = 1
    FWD = 2

class EncryptionAlgorithm(Enum):
    """IKEv2 Encryption Algorithms - RFC 7296"""
    ENCR_DES = 2
    ENCR_3DES = 3
    ENCR_CAST = 6
    ENCR_BLOWFISH = 7
    ENCR_NULL = 11
    ENCR_AES_CBC = 12
    ENCR_AES_CTR = 13
    ENCR_AES_GCM_8 = 18
    ENCR_AES_GCM_12 = 19
    ENCR_AES_GCM_16 = 20

class IntegrityAlgorithm(Enum):
    """IKEv2 Integrity Algorithms - RFC 7296"""
    AUTH_NONE = 0
    AUTH_HMAC_MD5_96 = 1
    AUTH_HMAC_SHA1_96 = 2
    AUTH_DES_MAC = 3
    AUTH_KPDK_MD5 = 4
    AUTH_AES_XCBC_96 = 5
    AUTH_HMAC_SHA2_256_128 = 12
    AUTH_HMAC_SHA2_384_192 = 13
    AUTH_HMAC_SHA2_512_256 = 14

class DiffieHellmanGroup(Enum):
    """IKEv2 DH Groups - RFC 3526, RFC 5903"""
    DH_NONE = 0
    DH_MODP_768 = 1
    DH_MODP_1024 = 2
    DH_MODP_1536 = 5
    DH_MODP_2048 = 14
    DH_MODP_3072 = 15
    DH_MODP_4096 = 16
    DH_ECP_256 = 19
    DH_ECP_384 = 20
    DH_ECP_521 = 21

class PRFAlgorithm(Enum):
    """IKEv2 PRF Algorithms"""
    PRF_HMAC_MD5 = 1
    PRF_HMAC_SHA1 = 2
    PRF_HMAC_TIGER = 3
    PRF_AES128_XCBC = 4
    PRF_HMAC_SHA2_256 = 5
    PRF_HMAC_SHA2_384 = 6
    PRF_HMAC_SHA2_512 = 7

# XFRM Algorithm Name Mappings
ENCR_TO_XFRM = {
    EncryptionAlgorithm.ENCR_DES: "cbc(des)",
    EncryptionAlgorithm.ENCR_3DES: "cbc(des3_ede)",
    EncryptionAlgorithm.ENCR_CAST: "cbc(cast5)",
    EncryptionAlgorithm.ENCR_BLOWFISH: "cbc(blowfish)",
    EncryptionAlgorithm.ENCR_NULL: "ecb(cipher_null)",
    EncryptionAlgorithm.ENCR_AES_CBC: "cbc(aes)",
    EncryptionAlgorithm.ENCR_AES_CTR: "rfc3686(ctr(aes))",
    EncryptionAlgorithm.ENCR_AES_GCM_16: "rfc4106(gcm(aes))",
}

INTEG_TO_XFRM = {
    IntegrityAlgorithm.AUTH_HMAC_MD5_96: "hmac(md5)",
    IntegrityAlgorithm.AUTH_HMAC_SHA1_96: "hmac(sha1)",
    IntegrityAlgorithm.AUTH_AES_XCBC_96: "xcbc(aes)",
    IntegrityAlgorithm.AUTH_HMAC_SHA2_256_128: "hmac(sha256)",
    IntegrityAlgorithm.AUTH_HMAC_SHA2_384_192: "hmac(sha384)",
    IntegrityAlgorithm.AUTH_HMAC_SHA2_512_256: "hmac(sha512)",
}

INTEG_TRUNCATE_LEN = {
    IntegrityAlgorithm.AUTH_HMAC_MD5_96: 96,
    IntegrityAlgorithm.AUTH_HMAC_SHA1_96: 96,
    IntegrityAlgorithm.AUTH_AES_XCBC_96: 96,
    IntegrityAlgorithm.AUTH_HMAC_SHA2_256_128: 128,
    IntegrityAlgorithm.AUTH_HMAC_SHA2_384_192: 192,
    IntegrityAlgorithm.AUTH_HMAC_SHA2_512_256: 256,
}

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class XfrmState:
    """XFRM State - represents an SA"""
    src: str
    dst: str
    proto: XfrmProto
    mode: XfrmMode
    spi: int
    reqid: int = 0
    encr_algo: Optional[str] = None
    encr_key: Optional[bytes] = None
    auth_algo: Optional[str] = None
    auth_key: Optional[bytes] = None
    auth_trunc_len: int = 96
    ifid: int = 0
    esn: bool = False
    encap_type: Optional[str] = None  # "espinudp"
    encap_sport: int = 0
    encap_dport: int = 0

@dataclass
class XfrmPolicy:
    """XFRM Policy - traffic selector and action"""
    src_net: str
    dst_net: str
    src_port: int = 0
    dst_port: int = 0
    proto: int = 0  # IP protocol (0 = any)
    direction: XfrmDir = XfrmDir.IN
    priority: int = 0
    ifid: int = 0
    tmpl_src: Optional[str] = None
    tmpl_dst: Optional[str] = None
    tmpl_spi: int = 0
    tmpl_proto: XfrmProto = XfrmProto.ESP
    tmpl_mode: XfrmMode = XfrmMode.TUNNEL

@dataclass
class ChildSecurityAssociation:
    """Child SA - IPSec tunnel parameters"""
    child_sa_id: str = field(default_factory=lambda: secrets.token_hex(8))
    inbound_spi: int = 0
    outbound_spi: int = 0

    # Encryption
    encr_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.ENCR_AES_CBC
    initiator_to_responder_encr_key: bytes = field(default_factory=bytes)
    responder_to_initiator_encr_key: bytes = field(default_factory=bytes)

    # Integrity
    integ_algorithm: Optional[IntegrityAlgorithm] = IntegrityAlgorithm.AUTH_HMAC_SHA2_256_128
    initiator_to_responder_integ_key: bytes = field(default_factory=bytes)
    responder_to_initiator_integ_key: bytes = field(default_factory=bytes)

    # Traffic selectors
    local_public_ip: str = ""
    peer_public_ip: str = ""
    traffic_selector_local: str = "0.0.0.0/0"  # CIDR
    traffic_selector_remote: str = "0.0.0.0/0"
    selected_ip_protocol: int = 0  # 0 = any

    # NAT-T
    enable_encapsulate: bool = False
    nat_port: int = 4500
    n3iwf_port: int = 4500

    # ESN
    esn_enabled: bool = False

    # Kernel state tracking
    xfrm_states: List[XfrmState] = field(default_factory=list)
    xfrm_policies: List[XfrmPolicy] = field(default_factory=list)

@dataclass
class IKESecurityAssociation:
    """IKE SA - IKEv2 session parameters"""
    local_spi: int = field(default_factory=lambda: int.from_bytes(secrets.token_bytes(8), 'big'))
    remote_spi: int = 0

    # Negotiated algorithms
    encr_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.ENCR_AES_CBC
    integ_algorithm: IntegrityAlgorithm = IntegrityAlgorithm.AUTH_HMAC_SHA2_256_128
    prf_algorithm: PRFAlgorithm = PRFAlgorithm.PRF_HMAC_SHA2_256
    dh_group: DiffieHellmanGroup = DiffieHellmanGroup.DH_MODP_2048

    # Derived keys
    sk_d: bytes = field(default_factory=bytes)  # For deriving child keys
    sk_ai: bytes = field(default_factory=bytes)  # IKE auth initiator
    sk_ar: bytes = field(default_factory=bytes)  # IKE auth responder
    sk_ei: bytes = field(default_factory=bytes)  # IKE encr initiator
    sk_er: bytes = field(default_factory=bytes)  # IKE encr responder
    sk_pi: bytes = field(default_factory=bytes)  # PRF initiator
    sk_pr: bytes = field(default_factory=bytes)  # PRF responder

    # NAT detection
    ue_behind_nat: bool = False
    n3iwf_behind_nat: bool = False

    # Message IDs
    initiator_message_id: int = 0
    responder_message_id: int = 0

    # Auth data
    initiator_signed_octets: bytes = field(default_factory=bytes)
    responder_signed_octets: bytes = field(default_factory=bytes)

    # Child SAs
    child_security_associations: List[ChildSecurityAssociation] = field(default_factory=list)

# ============================================================================
# XFRM Operations - Kernel Interface
# ============================================================================

class XfrmManager:
    """
    XFRM Manager - Manages kernel-level IPSec using Linux XFRM
    Based on Free5GC n3iwf/internal/ike/xfrm/xfrm.go
    """

    def __init__(self, use_kernel: bool = False):
        """
        Initialize XFRM Manager

        Args:
            use_kernel: If True, actually configure kernel XFRM (requires root)
                       If False, simulate operations (for testing)
        """
        self.use_kernel = use_kernel and os.geteuid() == 0
        self.xfrm_interface_counter = 0
        self.xfrm_interfaces: Dict[str, Dict] = {}
        self.active_states: List[XfrmState] = []
        self.active_policies: List[XfrmPolicy] = []

        if self.use_kernel:
            logger.info("XFRM Manager initialized with kernel mode")
        else:
            logger.info("XFRM Manager initialized in simulation mode")

    def setup_ipsec_xfrmi(
        self,
        xfrm_iface_name: str,
        parent_iface_name: str,
        xfrm_iface_id: int,
        xfrm_iface_addr: str
    ) -> bool:
        """
        Setup XFRM interface for IPSec tunnel

        Equivalent to:
            ip link add <name> type xfrm dev <parent> if_id <id>
            ip addr add <addr> dev <name>
            ip link set <name> up
        """
        if self.use_kernel:
            try:
                # Create XFRM interface
                cmd = [
                    "ip", "link", "add", xfrm_iface_name,
                    "type", "xfrm",
                    "dev", parent_iface_name,
                    "if_id", str(xfrm_iface_id)
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                # Add IP address
                cmd = ["ip", "addr", "add", xfrm_iface_addr, "dev", xfrm_iface_name]
                subprocess.run(cmd, check=True, capture_output=True)

                # Bring interface up
                cmd = ["ip", "link", "set", xfrm_iface_name, "up"]
                subprocess.run(cmd, check=True, capture_output=True)

                logger.info(f"Created XFRM interface {xfrm_iface_name} with id {xfrm_iface_id}")
                return True

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create XFRM interface: {e.stderr.decode()}")
                return False
        else:
            # Simulation mode
            self.xfrm_interfaces[xfrm_iface_name] = {
                "parent": parent_iface_name,
                "if_id": xfrm_iface_id,
                "addr": xfrm_iface_addr,
                "state": "up"
            }
            logger.info(f"[SIM] Created XFRM interface {xfrm_iface_name}")
            return True

    def apply_xfrm_rule(
        self,
        n3iwf_is_initiator: bool,
        xfrm_iface_id: int,
        child_sa: ChildSecurityAssociation
    ) -> bool:
        """
        Apply XFRM state and policy rules for Child SA

        Based on Free5GC ApplyXFRMRule()
        Creates bidirectional ESP tunnel with states and policies
        """
        # Get algorithm names for XFRM
        encr_name = ENCR_TO_XFRM.get(child_sa.encr_algorithm, "cbc(aes)")
        integ_name = INTEG_TO_XFRM.get(child_sa.integ_algorithm, "hmac(sha256)") if child_sa.integ_algorithm else None
        trunc_len = INTEG_TRUNCATE_LEN.get(child_sa.integ_algorithm, 128) if child_sa.integ_algorithm else 0

        # Determine key direction
        if n3iwf_is_initiator:
            inbound_encr_key = child_sa.responder_to_initiator_encr_key
            inbound_integ_key = child_sa.responder_to_initiator_integ_key
            outbound_encr_key = child_sa.initiator_to_responder_encr_key
            outbound_integ_key = child_sa.initiator_to_responder_integ_key
        else:
            inbound_encr_key = child_sa.initiator_to_responder_encr_key
            inbound_integ_key = child_sa.initiator_to_responder_integ_key
            outbound_encr_key = child_sa.responder_to_initiator_encr_key
            outbound_integ_key = child_sa.responder_to_initiator_integ_key

        # === INBOUND (peer -> local) ===
        inbound_state = XfrmState(
            src=child_sa.peer_public_ip,
            dst=child_sa.local_public_ip,
            proto=XfrmProto.ESP,
            mode=XfrmMode.TUNNEL,
            spi=child_sa.inbound_spi,
            ifid=xfrm_iface_id,
            encr_algo=encr_name,
            encr_key=inbound_encr_key,
            auth_algo=integ_name,
            auth_key=inbound_integ_key,
            auth_trunc_len=trunc_len,
            esn=child_sa.esn_enabled
        )

        inbound_policy = XfrmPolicy(
            src_net=child_sa.traffic_selector_remote,
            dst_net=child_sa.traffic_selector_local,
            proto=child_sa.selected_ip_protocol,
            direction=XfrmDir.IN,
            ifid=xfrm_iface_id,
            tmpl_src=child_sa.peer_public_ip,
            tmpl_dst=child_sa.local_public_ip,
            tmpl_spi=child_sa.inbound_spi,
            tmpl_proto=XfrmProto.ESP,
            tmpl_mode=XfrmMode.TUNNEL
        )

        # === OUTBOUND (local -> peer) ===
        outbound_state = XfrmState(
            src=child_sa.local_public_ip,
            dst=child_sa.peer_public_ip,
            proto=XfrmProto.ESP,
            mode=XfrmMode.TUNNEL,
            spi=child_sa.outbound_spi,
            ifid=xfrm_iface_id,
            encr_algo=encr_name,
            encr_key=outbound_encr_key,
            auth_algo=integ_name,
            auth_key=outbound_integ_key,
            auth_trunc_len=trunc_len,
            esn=child_sa.esn_enabled
        )

        # NAT-T encapsulation for outbound
        if child_sa.enable_encapsulate:
            outbound_state.encap_type = "espinudp"
            outbound_state.encap_sport = child_sa.n3iwf_port
            outbound_state.encap_dport = child_sa.nat_port

        outbound_policy = XfrmPolicy(
            src_net=child_sa.traffic_selector_local,
            dst_net=child_sa.traffic_selector_remote,
            proto=child_sa.selected_ip_protocol,
            direction=XfrmDir.OUT,
            ifid=xfrm_iface_id,
            tmpl_src=child_sa.local_public_ip,
            tmpl_dst=child_sa.peer_public_ip,
            tmpl_spi=child_sa.outbound_spi,
            tmpl_proto=XfrmProto.ESP,
            tmpl_mode=XfrmMode.TUNNEL
        )

        # Apply rules
        success = True
        success &= self._add_xfrm_state(inbound_state)
        success &= self._add_xfrm_policy(inbound_policy)
        success &= self._add_xfrm_state(outbound_state)
        success &= self._add_xfrm_policy(outbound_policy)

        if success:
            child_sa.xfrm_states.extend([inbound_state, outbound_state])
            child_sa.xfrm_policies.extend([inbound_policy, outbound_policy])
            logger.info(f"Applied XFRM rules for Child SA {child_sa.child_sa_id}")

        return success

    def _add_xfrm_state(self, state: XfrmState) -> bool:
        """Add XFRM state to kernel"""
        if self.use_kernel:
            try:
                # Build ip xfrm state add command
                cmd = [
                    "ip", "xfrm", "state", "add",
                    "src", state.src,
                    "dst", state.dst,
                    "proto", "esp",
                    "spi", str(state.spi),
                    "mode", "tunnel",
                    "if_id", str(state.ifid)
                ]

                if state.encr_algo and state.encr_key:
                    cmd.extend(["enc", state.encr_algo, f"0x{state.encr_key.hex()}"])

                if state.auth_algo and state.auth_key:
                    cmd.extend([
                        "auth-trunc", state.auth_algo,
                        f"0x{state.auth_key.hex()}", str(state.auth_trunc_len)
                    ])

                if state.encap_type:
                    cmd.extend([
                        "encap", state.encap_type,
                        str(state.encap_sport), str(state.encap_dport), "0.0.0.0"
                    ])

                subprocess.run(cmd, check=True, capture_output=True)
                self.active_states.append(state)
                return True

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to add XFRM state: {e.stderr.decode()}")
                return False
        else:
            # Simulation
            self.active_states.append(state)
            logger.info(f"[SIM] Added XFRM state: {state.src} -> {state.dst} SPI={state.spi:#x}")
            return True

    def _add_xfrm_policy(self, policy: XfrmPolicy) -> bool:
        """Add XFRM policy to kernel"""
        if self.use_kernel:
            try:
                dir_str = {XfrmDir.IN: "in", XfrmDir.OUT: "out", XfrmDir.FWD: "fwd"}[policy.direction]

                cmd = [
                    "ip", "xfrm", "policy", "add",
                    "src", policy.src_net,
                    "dst", policy.dst_net,
                    "dir", dir_str,
                    "if_id", str(policy.ifid),
                    "tmpl",
                    "src", policy.tmpl_src,
                    "dst", policy.tmpl_dst,
                    "proto", "esp",
                    "mode", "tunnel",
                    "spi", str(policy.tmpl_spi)
                ]

                subprocess.run(cmd, check=True, capture_output=True)
                self.active_policies.append(policy)
                return True

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to add XFRM policy: {e.stderr.decode()}")
                return False
        else:
            # Simulation
            self.active_policies.append(policy)
            logger.info(f"[SIM] Added XFRM policy: {policy.src_net} -> {policy.dst_net} dir={policy.direction.name}")
            return True

    def delete_xfrm_state(self, src: str, dst: str, spi: int) -> bool:
        """Delete XFRM state from kernel"""
        if self.use_kernel:
            try:
                cmd = [
                    "ip", "xfrm", "state", "delete",
                    "src", src,
                    "dst", dst,
                    "proto", "esp",
                    "spi", str(spi)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to delete XFRM state: {e.stderr.decode()}")
                return False
        else:
            self.active_states = [s for s in self.active_states
                                  if not (s.src == src and s.dst == dst and s.spi == spi)]
            logger.info(f"[SIM] Deleted XFRM state: {src} -> {dst} SPI={spi:#x}")
            return True

    def delete_xfrm_policy(self, src_net: str, dst_net: str, direction: XfrmDir) -> bool:
        """Delete XFRM policy from kernel"""
        if self.use_kernel:
            try:
                dir_str = {XfrmDir.IN: "in", XfrmDir.OUT: "out", XfrmDir.FWD: "fwd"}[direction]
                cmd = [
                    "ip", "xfrm", "policy", "delete",
                    "src", src_net,
                    "dst", dst_net,
                    "dir", dir_str
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to delete XFRM policy: {e.stderr.decode()}")
                return False
        else:
            self.active_policies = [p for p in self.active_policies
                                    if not (p.src_net == src_net and p.dst_net == dst_net
                                            and p.direction == direction)]
            logger.info(f"[SIM] Deleted XFRM policy: {src_net} -> {dst_net}")
            return True

    def cleanup_child_sa(self, child_sa: ChildSecurityAssociation) -> None:
        """Clean up all XFRM resources for a Child SA"""
        for state in child_sa.xfrm_states:
            self.delete_xfrm_state(state.src, state.dst, state.spi)

        for policy in child_sa.xfrm_policies:
            self.delete_xfrm_policy(policy.src_net, policy.dst_net, policy.direction)

        child_sa.xfrm_states.clear()
        child_sa.xfrm_policies.clear()
        logger.info(f"Cleaned up Child SA {child_sa.child_sa_id}")

    def delete_xfrm_interface(self, iface_name: str) -> bool:
        """Delete XFRM interface"""
        if self.use_kernel:
            try:
                cmd = ["ip", "link", "delete", iface_name]
                subprocess.run(cmd, check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError:
                return False
        else:
            if iface_name in self.xfrm_interfaces:
                del self.xfrm_interfaces[iface_name]
            logger.info(f"[SIM] Deleted XFRM interface {iface_name}")
            return True

    def get_status(self) -> Dict[str, Any]:
        """Get XFRM manager status"""
        return {
            "kernel_mode": self.use_kernel,
            "interfaces": list(self.xfrm_interfaces.keys()),
            "active_states": len(self.active_states),
            "active_policies": len(self.active_policies),
            "states": [
                {
                    "src": s.src,
                    "dst": s.dst,
                    "spi": f"{s.spi:#x}",
                    "proto": s.proto.name,
                    "mode": s.mode.name
                }
                for s in self.active_states
            ],
            "policies": [
                {
                    "src": p.src_net,
                    "dst": p.dst_net,
                    "direction": p.direction.name
                }
                for p in self.active_policies
            ]
        }


# ============================================================================
# Helper Functions
# ============================================================================

def generate_child_sa_keys(
    ike_sa: IKESecurityAssociation,
    nonce_i: bytes,
    nonce_r: bytes,
    key_length: int = 32
) -> ChildSecurityAssociation:
    """
    Generate keys for Child SA using IKE SA's SK_d

    Per RFC 7296 Section 2.17:
    KEYMAT = prf+(SK_d, Ni | Nr)
    """
    import hmac
    import hashlib

    # Generate KEYMAT using prf+
    seed = nonce_i + nonce_r
    keymat = b""

    # prf+ (RFC 7296 Section 2.13)
    t = b""
    counter = 1
    while len(keymat) < key_length * 4:  # 4 keys needed
        t = hmac.new(ike_sa.sk_d, t + seed + bytes([counter]), hashlib.sha256).digest()
        keymat += t
        counter += 1

    # Split KEYMAT into keys
    child_sa = ChildSecurityAssociation()
    offset = 0
    child_sa.initiator_to_responder_encr_key = keymat[offset:offset + key_length]
    offset += key_length
    child_sa.responder_to_initiator_encr_key = keymat[offset:offset + key_length]
    offset += key_length
    child_sa.initiator_to_responder_integ_key = keymat[offset:offset + key_length]
    offset += key_length
    child_sa.responder_to_initiator_integ_key = keymat[offset:offset + key_length]

    # Generate random SPIs
    child_sa.inbound_spi = int.from_bytes(secrets.token_bytes(4), 'big')
    child_sa.outbound_spi = int.from_bytes(secrets.token_bytes(4), 'big')

    return child_sa


# Global XFRM manager instance
xfrm_manager = XfrmManager(use_kernel=False)  # Default to simulation mode
