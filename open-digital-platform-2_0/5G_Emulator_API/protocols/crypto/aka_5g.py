"""
5G-AKA Authentication Implementation (TS 33.501)

This module implements the 5G Authentication and Key Agreement (5G-AKA) protocol
as specified in 3GPP TS 33.501. It includes:
- Milenage algorithm (TS 35.206)
- TUAK algorithm (TS 35.231) - placeholder
- Key derivation functions per TS 33.501
- SUCI/SUPI conversion

Reference:
- 3GPP TS 33.501 V17.7.0 (Security architecture and procedures for 5G)
- 3GPP TS 35.206 (Milenage algorithm)
"""

import os
import hmac
import hashlib
import struct
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

class AuthMethod(IntEnum):
    """Authentication methods (TS 33.501)"""
    AKA_5G = 0
    EAP_AKA_PRIME = 1
    EAP_TLS = 2


class ProtectionScheme(IntEnum):
    """SUCI Protection Schemes (TS 33.501 Section 6.12)"""
    NULL_SCHEME = 0
    PROFILE_A = 1  # ECIES scheme profile A
    PROFILE_B = 2  # ECIES scheme profile B


# AKA Algorithm Parameters
AMF_SEPARATION_BIT = 0x80  # Bit position for AMF separation
AUTN_LENGTH = 16
RAND_LENGTH = 16
RES_LENGTH = 8
XRES_STAR_LENGTH = 16
KAUSF_LENGTH = 32
KSEAF_LENGTH = 32
KAMF_LENGTH = 32


# =============================================================================
# Milenage Algorithm Implementation (TS 35.206)
# =============================================================================

class Milenage:
    """
    Milenage Algorithm Implementation per 3GPP TS 35.206.

    The Milenage algorithm is a set of cryptographic functions used in
    3GPP authentication. It uses AES-128 as the core encryption function.
    """

    # Milenage constants (default OP configuration)
    C1 = bytes(16)  # All zeros
    C2 = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01])
    C3 = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02])
    C4 = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04])
    C5 = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08])

    # Rotation amounts
    R1 = 64
    R2 = 0
    R3 = 32
    R4 = 64
    R5 = 96

    def __init__(self, k: bytes, op: bytes, opc: Optional[bytes] = None):
        """
        Initialize Milenage with key and operator parameters.

        Args:
            k: Subscriber key K (16 bytes)
            op: Operator variant algorithm configuration field OP (16 bytes)
            opc: Pre-computed OPc (16 bytes), if available
        """
        if len(k) != 16:
            raise ValueError("K must be 16 bytes")
        if len(op) != 16 and opc is None:
            raise ValueError("OP must be 16 bytes")

        self.k = k
        self.op = op
        self.opc = opc if opc else self._compute_opc(k, op)

    def _aes_encrypt(self, key: bytes, data: bytes) -> bytes:
        """AES-128 encryption in ECB mode"""
        cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(data) + encryptor.finalize()

    def _compute_opc(self, k: bytes, op: bytes) -> bytes:
        """Compute OPc from K and OP per TS 35.206 Section 8.2"""
        return self._xor(self._aes_encrypt(k, op), op)

    def _xor(self, a: bytes, b: bytes) -> bytes:
        """XOR two byte strings"""
        return bytes(x ^ y for x, y in zip(a, b))

    def _rotate(self, data: bytes, bits: int) -> bytes:
        """Rotate bytes left by specified bits"""
        if bits == 0:
            return data
        byte_shift = bits // 8
        result = data[byte_shift:] + data[:byte_shift]
        return result

    def f1(self, rand: bytes, sqn: bytes, amf: bytes) -> bytes:
        """
        f1: Network authentication function (MAC-A)

        Args:
            rand: Random challenge (16 bytes)
            sqn: Sequence number (6 bytes)
            amf: Authentication Management Field (2 bytes)

        Returns:
            MAC-A (8 bytes)
        """
        # Compute temp = E_K(RAND XOR OPc)
        temp = self._aes_encrypt(self.k, self._xor(rand, self.opc))

        # Construct input: SQN || AMF || SQN || AMF
        inp = sqn + amf + sqn + amf

        # Out1 = E_K(temp XOR rotate(inp XOR OPc, r1) XOR c1) XOR OPc
        rotated = self._rotate(self._xor(inp, self.opc), self.R1)
        out1 = self._xor(
            self._aes_encrypt(self.k, self._xor(self._xor(temp, rotated), self.C1)),
            self.opc
        )

        # MAC-A = out1[0:8]
        return out1[0:8]

    def f1_star(self, rand: bytes, sqn: bytes, amf: bytes) -> bytes:
        """
        f1*: Re-synchronization function (MAC-S)

        Args:
            rand: Random challenge (16 bytes)
            sqn: Sequence number (6 bytes)
            amf: Authentication Management Field (2 bytes)

        Returns:
            MAC-S (8 bytes)
        """
        # Compute temp = E_K(RAND XOR OPc)
        temp = self._aes_encrypt(self.k, self._xor(rand, self.opc))

        # Construct input: SQN || AMF || SQN || AMF
        inp = sqn + amf + sqn + amf

        # Out1 = E_K(temp XOR rotate(inp XOR OPc, r1) XOR c1) XOR OPc
        rotated = self._rotate(self._xor(inp, self.opc), self.R1)
        out1 = self._xor(
            self._aes_encrypt(self.k, self._xor(self._xor(temp, rotated), self.C1)),
            self.opc
        )

        # MAC-S = out1[8:16]
        return out1[8:16]

    def f2345(self, rand: bytes) -> Tuple[bytes, bytes, bytes, bytes]:
        """
        f2, f3, f4, f5: User authentication and key generation functions

        Args:
            rand: Random challenge (16 bytes)

        Returns:
            Tuple of (RES, CK, IK, AK) - (8, 16, 16, 6 bytes)
        """
        # Compute temp = E_K(RAND XOR OPc)
        temp = self._aes_encrypt(self.k, self._xor(rand, self.opc))

        # Out2 = E_K(rotate(temp XOR OPc, r2) XOR c2) XOR OPc
        out2 = self._xor(
            self._aes_encrypt(self.k, self._xor(self._rotate(self._xor(temp, self.opc), self.R2), self.C2)),
            self.opc
        )

        # Out3 = E_K(rotate(temp XOR OPc, r3) XOR c3) XOR OPc
        out3 = self._xor(
            self._aes_encrypt(self.k, self._xor(self._rotate(self._xor(temp, self.opc), self.R3), self.C3)),
            self.opc
        )

        # Out4 = E_K(rotate(temp XOR OPc, r4) XOR c4) XOR OPc
        out4 = self._xor(
            self._aes_encrypt(self.k, self._xor(self._rotate(self._xor(temp, self.opc), self.R4), self.C4)),
            self.opc
        )

        # Out5 = E_K(rotate(temp XOR OPc, r5) XOR c5) XOR OPc
        out5 = self._xor(
            self._aes_encrypt(self.k, self._xor(self._rotate(self._xor(temp, self.opc), self.R5), self.C5)),
            self.opc
        )

        # Extract outputs
        res = out2[8:16]   # f2: RES (8 bytes)
        ck = out3          # f3: CK (16 bytes)
        ik = out4          # f4: IK (16 bytes)
        ak = out2[0:6]     # f5: AK (6 bytes)

        return res, ck, ik, ak

    def f5_star(self, rand: bytes) -> bytes:
        """
        f5*: Re-synchronization function for AK

        Args:
            rand: Random challenge (16 bytes)

        Returns:
            AK (6 bytes)
        """
        # Compute temp = E_K(RAND XOR OPc)
        temp = self._aes_encrypt(self.k, self._xor(rand, self.opc))

        # Out5 = E_K(rotate(temp XOR OPc, r5) XOR c5) XOR OPc
        out5 = self._xor(
            self._aes_encrypt(self.k, self._xor(self._rotate(self._xor(temp, self.opc), self.R5), self.C5)),
            self.opc
        )

        return out5[0:6]

    def generate_auth_vector(self, rand: bytes, sqn: bytes, amf: bytes) -> Tuple[bytes, bytes, bytes, bytes, bytes]:
        """
        Generate complete authentication vector.

        Args:
            rand: Random challenge (16 bytes)
            sqn: Sequence number (6 bytes)
            amf: Authentication Management Field (2 bytes)

        Returns:
            Tuple of (XRES, AUTN, CK, IK, AK)
        """
        # Generate f1-f5 outputs
        mac_a = self.f1(rand, sqn, amf)
        res, ck, ik, ak = self.f2345(rand)

        # Compute AUTN = SQN XOR AK || AMF || MAC-A
        sqn_xor_ak = self._xor(sqn, ak)
        autn = sqn_xor_ak + amf + mac_a

        return res, autn, ck, ik, ak


# =============================================================================
# 5G-AKA Key Derivation Functions (TS 33.501)
# =============================================================================

class KeyDerivation5G:
    """
    5G Key Derivation Functions per 3GPP TS 33.501.

    Implements the key hierarchy:
    K -> CK/IK -> CK'/IK' -> KAUSF -> KSEAF -> KAMF -> KgNB/KN3IWF
    """

    @staticmethod
    def _kdf(key: bytes, fc: int, *params: Tuple[bytes, int]) -> bytes:
        """
        Generic Key Derivation Function per TS 33.220 Annex B.

        Args:
            key: Input key
            fc: Function code
            params: List of (parameter, length) tuples

        Returns:
            Derived key (32 bytes)
        """
        # S = FC || P0 || L0 || P1 || L1 || ...
        s = bytes([fc])
        for param, length in params:
            s += param + struct.pack('>H', length)

        return hmac.new(key, s, hashlib.sha256).digest()

    @staticmethod
    def derive_ck_ik_prime(ck: bytes, ik: bytes, serving_network_name: str, sqn: bytes, ak: bytes) -> Tuple[bytes, bytes]:
        """
        Derive CK' and IK' from CK, IK per TS 33.501 Annex A.2.

        Args:
            ck: Cipher Key (16 bytes)
            ik: Integrity Key (16 bytes)
            serving_network_name: SN name string
            sqn: Sequence number (6 bytes)
            ak: Anonymity Key (6 bytes)

        Returns:
            Tuple of (CK', IK') - (16, 16 bytes)
        """
        # Key = CK || IK
        key = ck + ik

        # FC = 0x20
        fc = 0x20

        # P0 = Serving Network Name
        sn_bytes = serving_network_name.encode('utf-8')

        # P1 = SQN XOR AK
        sqn_xor_ak = bytes(a ^ b for a, b in zip(sqn, ak))

        # Derive using KDF
        derived = KeyDerivation5G._kdf(
            key, fc,
            (sn_bytes, len(sn_bytes)),
            (sqn_xor_ak, len(sqn_xor_ak))
        )

        # CK' = derived[0:16], IK' = derived[16:32]
        return derived[0:16], derived[16:32]

    @staticmethod
    def derive_kausf(ck_prime: bytes, ik_prime: bytes, serving_network_name: str, sqn: bytes, ak: bytes) -> bytes:
        """
        Derive KAUSF from CK', IK' per TS 33.501 Annex A.2.

        Args:
            ck_prime: CK' (16 bytes)
            ik_prime: IK' (16 bytes)
            serving_network_name: SN name string
            sqn: Sequence number (6 bytes)
            ak: Anonymity Key (6 bytes)

        Returns:
            KAUSF (32 bytes)
        """
        # Key = CK' || IK'
        key = ck_prime + ik_prime

        # FC = 0x6A
        fc = 0x6A

        # P0 = Serving Network Name
        sn_bytes = serving_network_name.encode('utf-8')

        # P1 = SQN XOR AK
        sqn_xor_ak = bytes(a ^ b for a, b in zip(sqn, ak))

        return KeyDerivation5G._kdf(
            key, fc,
            (sn_bytes, len(sn_bytes)),
            (sqn_xor_ak, len(sqn_xor_ak))
        )

    @staticmethod
    def derive_kseaf(kausf: bytes, serving_network_name: str) -> bytes:
        """
        Derive KSEAF from KAUSF per TS 33.501 Annex A.6.

        Args:
            kausf: KAUSF (32 bytes)
            serving_network_name: SN name string

        Returns:
            KSEAF (32 bytes)
        """
        # FC = 0x6C
        fc = 0x6C

        # P0 = Serving Network Name
        sn_bytes = serving_network_name.encode('utf-8')

        return KeyDerivation5G._kdf(kausf, fc, (sn_bytes, len(sn_bytes)))

    @staticmethod
    def derive_kamf(kseaf: bytes, supi: str, abba: bytes) -> bytes:
        """
        Derive KAMF from KSEAF per TS 33.501 Annex A.7.

        Args:
            kseaf: KSEAF (32 bytes)
            supi: SUPI string (e.g., "imsi-001010000000001")
            abba: ABBA parameter

        Returns:
            KAMF (32 bytes)
        """
        # FC = 0x6D
        fc = 0x6D

        # P0 = SUPI
        supi_bytes = supi.encode('utf-8')

        # P1 = ABBA
        return KeyDerivation5G._kdf(
            kseaf, fc,
            (supi_bytes, len(supi_bytes)),
            (abba, len(abba))
        )

    @staticmethod
    def derive_kgnb(kamf: bytes, nas_uplink_count: int, access_type: int = 1) -> bytes:
        """
        Derive KgNB from KAMF per TS 33.501 Annex A.9.

        Args:
            kamf: KAMF (32 bytes)
            nas_uplink_count: NAS uplink COUNT (4 bytes as int)
            access_type: Access type (1=3GPP, 2=non-3GPP)

        Returns:
            KgNB (32 bytes)
        """
        # FC = 0x6E
        fc = 0x6E

        # P0 = Uplink NAS COUNT
        count_bytes = struct.pack('>I', nas_uplink_count)

        # P1 = Access type distinguisher
        access_bytes = bytes([access_type])

        return KeyDerivation5G._kdf(
            kamf, fc,
            (count_bytes, 4),
            (access_bytes, 1)
        )

    @staticmethod
    def derive_nas_keys(kamf: bytes, algorithm_type: int, algorithm_id: int) -> bytes:
        """
        Derive NAS encryption/integrity keys per TS 33.501 Annex A.8.

        Args:
            kamf: KAMF (32 bytes)
            algorithm_type: 0x01=NEA, 0x02=NIA
            algorithm_id: Algorithm identifier (0-7)

        Returns:
            KNASint or KNASenc (32 bytes, use lower 16 bytes for 128-bit keys)
        """
        # FC = 0x69
        fc = 0x69

        # P0 = Algorithm type distinguisher
        type_bytes = bytes([algorithm_type])

        # P1 = Algorithm identity
        id_bytes = bytes([algorithm_id])

        return KeyDerivation5G._kdf(
            kamf, fc,
            (type_bytes, 1),
            (id_bytes, 1)
        )

    @staticmethod
    def compute_res_star(ck: bytes, ik: bytes, serving_network_name: str, rand: bytes, res: bytes) -> bytes:
        """
        Compute RES* per TS 33.501 Annex A.4.

        Args:
            ck: Cipher Key (16 bytes)
            ik: Integrity Key (16 bytes)
            serving_network_name: SN name string
            rand: RAND (16 bytes)
            res: RES from Milenage (8 bytes)

        Returns:
            RES* (16 bytes)
        """
        # Key = CK || IK
        key = ck + ik

        # FC = 0x6B
        fc = 0x6B

        # P0 = Serving Network Name
        sn_bytes = serving_network_name.encode('utf-8')

        # P1 = RAND
        # P2 = RES
        derived = KeyDerivation5G._kdf(
            key, fc,
            (sn_bytes, len(sn_bytes)),
            (rand, 16),
            (res, len(res))
        )

        # RES* = derived[16:32] (lower 128 bits)
        return derived[16:32]

    @staticmethod
    def compute_xres_star(ck: bytes, ik: bytes, serving_network_name: str, rand: bytes, xres: bytes) -> bytes:
        """
        Compute XRES* (network side) per TS 33.501 Annex A.4.

        Same computation as RES* but with XRES instead of RES.
        """
        return KeyDerivation5G.compute_res_star(ck, ik, serving_network_name, rand, xres)

    @staticmethod
    def compute_hres_star(rand: bytes, res_star: bytes) -> bytes:
        """
        Compute HRES* per TS 33.501 Annex A.5.

        Used by AUSF to verify authentication without knowing RES*.

        Args:
            rand: RAND (16 bytes)
            res_star: RES* (16 bytes)

        Returns:
            HRES* (16 bytes)
        """
        # HRES* = SHA-256(RAND || RES*)[0:16]
        h = hashlib.sha256(rand + res_star).digest()
        return h[0:16]

    @staticmethod
    def compute_hxres_star(rand: bytes, xres_star: bytes) -> bytes:
        """
        Compute HXRES* (network side) per TS 33.501 Annex A.5.
        """
        return KeyDerivation5G.compute_hres_star(rand, xres_star)


# =============================================================================
# 5G-AKA Protocol Handler
# =============================================================================

@dataclass
class AuthVector5G:
    """5G Authentication Vector"""
    rand: bytes           # 16 bytes
    autn: bytes           # 16 bytes
    xres_star: bytes      # 16 bytes
    hxres_star: bytes     # 16 bytes
    kausf: bytes          # 32 bytes
    kseaf: bytes          # 32 bytes


@dataclass
class SubscriberData:
    """Subscriber authentication data (stored in UDM/ARPF)"""
    supi: str
    k: bytes              # Subscriber key (16 bytes)
    opc: bytes            # OPc (16 bytes)
    sqn: bytes            # Sequence number (6 bytes)
    amf: bytes            # AMF (2 bytes)


class AKA5GHandler:
    """
    5G-AKA Protocol Handler.

    Manages authentication flows for both network and UE sides.
    """

    def __init__(self, serving_network_name: str = "5G:mnc001.mcc001.3gppnetwork.org"):
        """
        Initialize 5G-AKA handler.

        Args:
            serving_network_name: Serving Network Name (SN-name) per TS 24.501
        """
        self.serving_network_name = serving_network_name
        self.logger = logging.getLogger(__name__)

    def generate_auth_vector(self, subscriber: SubscriberData) -> AuthVector5G:
        """
        Generate 5G Authentication Vector (network side).

        Called by UDM/ARPF to create authentication vector for AUSF.

        Args:
            subscriber: Subscriber data including K, OPc, SQN

        Returns:
            5G Authentication Vector
        """
        # Generate random challenge
        rand = os.urandom(16)

        # Initialize Milenage
        milenage = Milenage(subscriber.k, bytes(16), subscriber.opc)

        # Generate authentication vector components
        xres, autn, ck, ik, ak = milenage.generate_auth_vector(
            rand, subscriber.sqn, subscriber.amf
        )

        self.logger.debug(f"Generated XRES: {xres.hex()}")
        self.logger.debug(f"Generated AUTN: {autn.hex()}")
        self.logger.debug(f"Generated CK: {ck.hex()}")
        self.logger.debug(f"Generated IK: {ik.hex()}")

        # Compute XRES*
        xres_star = KeyDerivation5G.compute_xres_star(
            ck, ik, self.serving_network_name, rand, xres
        )

        # Compute HXRES*
        hxres_star = KeyDerivation5G.compute_hxres_star(rand, xres_star)

        # Derive CK'/IK'
        ck_prime, ik_prime = KeyDerivation5G.derive_ck_ik_prime(
            ck, ik, self.serving_network_name, subscriber.sqn, ak
        )

        # Derive KAUSF
        kausf = KeyDerivation5G.derive_kausf(
            ck_prime, ik_prime, self.serving_network_name, subscriber.sqn, ak
        )

        # Derive KSEAF
        kseaf = KeyDerivation5G.derive_kseaf(kausf, self.serving_network_name)

        self.logger.info(f"Generated 5G-AV for {subscriber.supi}")
        self.logger.debug(f"RAND: {rand.hex()}")
        self.logger.debug(f"XRES*: {xres_star.hex()}")
        self.logger.debug(f"KAUSF: {kausf.hex()}")

        return AuthVector5G(
            rand=rand,
            autn=autn,
            xres_star=xres_star,
            hxres_star=hxres_star,
            kausf=kausf,
            kseaf=kseaf
        )

    def verify_auth_response(self, av: AuthVector5G, res_star: bytes) -> bool:
        """
        Verify authentication response from UE.

        Called by AUSF to verify RES* from UE.

        Args:
            av: Authentication vector used
            res_star: RES* received from UE

        Returns:
            True if authentication successful
        """
        # Compute HRES* from received RES*
        hres_star = KeyDerivation5G.compute_hres_star(av.rand, res_star)

        # Compare with expected HXRES*
        if hmac.compare_digest(hres_star, av.hxres_star):
            self.logger.info("Authentication successful")
            return True
        else:
            self.logger.warning("Authentication failed: RES* mismatch")
            return False

    def ue_compute_response(self, k: bytes, opc: bytes, rand: bytes, autn: bytes) -> Optional[Tuple[bytes, bytes, bytes]]:
        """
        Compute authentication response (UE side).

        Args:
            k: Subscriber key (16 bytes)
            opc: OPc (16 bytes)
            rand: RAND from network (16 bytes)
            autn: AUTN from network (16 bytes)

        Returns:
            Tuple of (RES*, CK, IK) if successful, None if AUTN verification fails
        """
        # Parse AUTN
        sqn_xor_ak = autn[0:6]
        amf = autn[6:8]
        mac_a = autn[8:16]

        # Initialize Milenage
        milenage = Milenage(k, bytes(16), opc)

        # Compute f2-f5
        res, ck, ik, ak = milenage.f2345(rand)

        # Recover SQN
        sqn = bytes(a ^ b for a, b in zip(sqn_xor_ak, ak))

        # Verify MAC-A
        expected_mac = milenage.f1(rand, sqn, amf)
        if not hmac.compare_digest(mac_a, expected_mac):
            self.logger.error("AUTN verification failed: MAC mismatch")
            return None

        # Verify AMF separation bit (for 5G)
        if not (amf[0] & AMF_SEPARATION_BIT):
            self.logger.warning("AMF separation bit not set - may be 4G vector")

        # Compute RES*
        res_star = KeyDerivation5G.compute_res_star(
            ck, ik, self.serving_network_name, rand, res
        )

        self.logger.info("UE computed authentication response")
        return res_star, ck, ik

    def derive_ue_keys(self, ck: bytes, ik: bytes, sqn: bytes, ak: bytes, supi: str, abba: bytes) -> Dict[str, bytes]:
        """
        Derive UE-side keys after successful authentication.

        Args:
            ck: Cipher Key
            ik: Integrity Key
            sqn: Sequence number (recovered from AUTN)
            ak: Anonymity Key
            supi: SUPI
            abba: ABBA parameter

        Returns:
            Dictionary of derived keys
        """
        # Derive CK'/IK'
        ck_prime, ik_prime = KeyDerivation5G.derive_ck_ik_prime(
            ck, ik, self.serving_network_name, sqn, ak
        )

        # Derive KAUSF
        kausf = KeyDerivation5G.derive_kausf(
            ck_prime, ik_prime, self.serving_network_name, sqn, ak
        )

        # Derive KSEAF
        kseaf = KeyDerivation5G.derive_kseaf(kausf, self.serving_network_name)

        # Derive KAMF
        kamf = KeyDerivation5G.derive_kamf(kseaf, supi, abba)

        # Derive NAS keys
        knas_enc = KeyDerivation5G.derive_nas_keys(kamf, 0x01, 0x01)  # NEA1
        knas_int = KeyDerivation5G.derive_nas_keys(kamf, 0x02, 0x02)  # NIA2

        return {
            'KAUSF': kausf,
            'KSEAF': kseaf,
            'KAMF': kamf,
            'KNASenc': knas_enc[16:32],  # Use lower 128 bits
            'KNASint': knas_int[16:32],
        }


# =============================================================================
# SUCI/SUPI Conversion
# =============================================================================

class SUCIHandler:
    """
    Handler for SUCI (Subscription Concealed Identifier) operations.

    SUCI is used to conceal SUPI during initial registration.
    """

    def __init__(self, home_network_public_key: Optional[bytes] = None):
        """
        Initialize SUCI handler.

        Args:
            home_network_public_key: Home network public key for ECIES encryption
        """
        self.hn_public_key = home_network_public_key

    def conceal_supi(self, supi: str, protection_scheme: ProtectionScheme = ProtectionScheme.NULL_SCHEME) -> Dict[str, Any]:
        """
        Conceal SUPI to create SUCI.

        Args:
            supi: SUPI string (e.g., "imsi-001010000000001")
            protection_scheme: Protection scheme to use

        Returns:
            Dictionary containing SUCI components
        """
        # Parse SUPI
        if not supi.startswith("imsi-"):
            raise ValueError("Only IMSI-based SUPI supported")

        imsi = supi[5:]  # Remove "imsi-" prefix
        mcc = imsi[0:3]
        mnc = imsi[3:5] if len(imsi) == 15 else imsi[3:6]
        msin = imsi[len(mcc) + len(mnc):]

        if protection_scheme == ProtectionScheme.NULL_SCHEME:
            # Null scheme: MSIN is sent in clear
            scheme_output = msin
        else:
            # ECIES encryption would go here
            # For now, use null scheme as fallback
            self.logger.warning("ECIES not implemented, using null scheme")
            scheme_output = msin

        return {
            'supi_format': 0,  # IMSI
            'mcc': mcc,
            'mnc': mnc,
            'routing_indicator': '0000',
            'protection_scheme': protection_scheme,
            'home_network_pki': 0,
            'scheme_output': scheme_output,
        }

    def reveal_supi(self, suci: Dict[str, Any], private_key: Optional[bytes] = None) -> str:
        """
        Reveal SUPI from SUCI.

        Args:
            suci: SUCI components dictionary
            private_key: Home network private key for ECIES decryption

        Returns:
            SUPI string
        """
        protection_scheme = suci.get('protection_scheme', 0)

        if protection_scheme == ProtectionScheme.NULL_SCHEME:
            # Null scheme: MSIN is in clear
            msin = suci['scheme_output']
        else:
            # ECIES decryption would go here
            raise NotImplementedError("ECIES decryption not implemented")

        # Reconstruct SUPI
        mcc = suci['mcc']
        mnc = suci['mnc']
        return f"imsi-{mcc}{mnc}{msin}"


# =============================================================================
# Demo/Test Functions
# =============================================================================

def demo_5g_aka():
    """Demonstrate 5G-AKA authentication flow"""
    print("=" * 60)
    print("5G-AKA Authentication Demo")
    print("=" * 60)

    # Subscriber data (would be stored in UDM/ARPF)
    subscriber = SubscriberData(
        supi="imsi-001010000000001",
        k=bytes.fromhex("465B5CE8B199B49FAA5F0A2EE238A6BC"),
        opc=bytes.fromhex("E8ED289DEBA952E4283B54E88E6183CA"),
        sqn=bytes.fromhex("000000000001"),
        amf=bytes.fromhex("8000")  # Bit 7 set for 5G
    )

    # Initialize handler
    handler = AKA5GHandler()

    # Network side: Generate authentication vector
    print("\n[Network] Generating authentication vector...")
    av = handler.generate_auth_vector(subscriber)
    print(f"  RAND: {av.rand.hex()}")
    print(f"  AUTN: {av.autn.hex()}")
    print(f"  XRES*: {av.xres_star.hex()}")
    print(f"  HXRES*: {av.hxres_star.hex()}")

    # UE side: Compute response
    print("\n[UE] Computing authentication response...")
    result = handler.ue_compute_response(
        subscriber.k, subscriber.opc, av.rand, av.autn
    )

    if result:
        res_star, ck, ik = result
        print(f"  RES*: {res_star.hex()}")

        # Network side: Verify response
        print("\n[Network] Verifying authentication response...")
        if handler.verify_auth_response(av, res_star):
            print("  Authentication SUCCESSFUL!")

            # Derive keys
            print("\n[UE] Deriving session keys...")
            sqn = subscriber.sqn
            ak = Milenage(subscriber.k, bytes(16), subscriber.opc).f2345(av.rand)[3]
            keys = handler.derive_ue_keys(
                ck, ik, sqn, ak, subscriber.supi, bytes([0x00, 0x00])
            )
            for name, key in keys.items():
                print(f"  {name}: {key.hex()}")
        else:
            print("  Authentication FAILED!")
    else:
        print("  AUTN verification failed!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    demo_5g_aka()
