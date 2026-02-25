"""
SCTP Transport Implementation for NGAP (TS 38.412)

This module provides SCTP transport for NGAP signaling between gNB and AMF
as specified in 3GPP TS 38.412.

SCTP Features Required for NGAP:
- Multi-homing (multiple IP addresses per endpoint)
- Multi-streaming (parallel message streams)
- Message boundaries preserved
- Ordered/unordered delivery
- Partial reliability (PR-SCTP) for some messages

Note: This implementation provides both a real SCTP transport (when pysctp is available)
and a fallback TCP transport for environments without SCTP support.

Reference: 3GPP TS 38.412 V17.2.0 (NGAP signaling transport)
"""

import asyncio
import socket
import struct
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import IntEnum
from abc import ABC, abstractmethod
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

NGAP_SCTP_PORT = 38412
NGAP_PPID = 60  # NGAP Payload Protocol Identifier


class SCTPEvent(IntEnum):
    """SCTP notification events"""
    COMM_UP = 1
    COMM_LOST = 2
    RESTART = 3
    SHUTDOWN_COMP = 4
    CANT_STR_ASSOC = 5
    ADDR_AVAILABLE = 6
    ADDR_UNAVAILABLE = 7
    SEND_FAILED = 8
    PEER_ERROR = 9
    PARTIAL_DELIVERY = 10


# =============================================================================
# SCTP Abstraction Layer
# =============================================================================

@dataclass
class SCTPMessage:
    """SCTP message with metadata"""
    data: bytes
    stream_id: int = 0
    ppid: int = NGAP_PPID
    flags: int = 0
    tsn: Optional[int] = None
    ssn: Optional[int] = None
    peer_addr: Optional[Tuple[str, int]] = None


@dataclass
class SCTPAssociation:
    """SCTP Association state"""
    assoc_id: int
    local_addrs: List[str]
    remote_addrs: List[str]
    local_port: int
    remote_port: int
    state: str = "CLOSED"
    inbound_streams: int = 0
    outbound_streams: int = 0
    primary_addr: Optional[str] = None


class SCTPTransportBase(ABC):
    """Abstract base class for SCTP transport"""

    @abstractmethod
    async def connect(self, remote_addrs: List[str], remote_port: int) -> bool:
        """Connect to remote endpoint"""
        pass

    @abstractmethod
    async def listen(self, local_addrs: List[str], local_port: int):
        """Start listening for connections"""
        pass

    @abstractmethod
    async def send(self, message: SCTPMessage) -> bool:
        """Send message"""
        pass

    @abstractmethod
    async def recv(self, timeout: Optional[float] = None) -> Optional[SCTPMessage]:
        """Receive message"""
        pass

    @abstractmethod
    async def close(self):
        """Close transport"""
        pass

    @abstractmethod
    def get_association(self) -> Optional[SCTPAssociation]:
        """Get association information"""
        pass


# =============================================================================
# Real SCTP Implementation (using pysctp if available)
# =============================================================================

class RealSCTPTransport(SCTPTransportBase):
    """
    Real SCTP transport using pysctp library.

    This implementation uses the pysctp library for native SCTP support.
    Falls back to TCP if pysctp is not available.
    """

    def __init__(self, max_streams: int = 10):
        self.max_streams = max_streams
        self.socket = None
        self.association = None
        self.logger = logging.getLogger(__name__)
        self._sctp_available = self._check_sctp_available()

    def _check_sctp_available(self) -> bool:
        """Check if SCTP is available on this system"""
        try:
            # Try to import sctp module
            import sctp
            self._sctp_module = sctp
            return True
        except ImportError:
            self.logger.warning("pysctp not available, will use fallback")
            return False

    async def connect(self, remote_addrs: List[str], remote_port: int) -> bool:
        """Connect to remote SCTP endpoint"""
        if not self._sctp_available:
            return False

        try:
            # Create SCTP socket
            self.socket = self._sctp_module.sctpsocket_tcp(socket.AF_INET)

            # Set socket options
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Enable SCTP events
            self._enable_events()

            # Connect to primary address
            primary_addr = remote_addrs[0]
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.socket.connect((primary_addr, remote_port))
            )

            # Add additional addresses for multi-homing
            for addr in remote_addrs[1:]:
                try:
                    self._sctp_module.bindx(self.socket, [(addr, 0)], self._sctp_module.SCTP_BINDX_ADD_ADDR)
                except Exception as e:
                    self.logger.warning(f"Could not add address {addr}: {e}")

            # Get association info
            self._update_association_info()

            self.logger.info(f"SCTP connected to {primary_addr}:{remote_port}")
            return True

        except Exception as e:
            self.logger.error(f"SCTP connect failed: {e}")
            return False

    async def listen(self, local_addrs: List[str], local_port: int):
        """Start listening for SCTP connections"""
        if not self._sctp_available:
            return

        try:
            # Create SCTP socket
            self.socket = self._sctp_module.sctpsocket_tcp(socket.AF_INET)

            # Set socket options
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Bind to primary address
            primary_addr = local_addrs[0]
            self.socket.bind((primary_addr, local_port))

            # Add additional addresses for multi-homing
            for addr in local_addrs[1:]:
                try:
                    self._sctp_module.bindx(self.socket, [(addr, local_port)], self._sctp_module.SCTP_BINDX_ADD_ADDR)
                except Exception as e:
                    self.logger.warning(f"Could not bind address {addr}: {e}")

            # Enable events
            self._enable_events()

            # Listen
            self.socket.listen(5)

            self.logger.info(f"SCTP listening on {primary_addr}:{local_port}")

        except Exception as e:
            self.logger.error(f"SCTP listen failed: {e}")
            raise

    async def accept(self) -> Optional['RealSCTPTransport']:
        """Accept incoming connection"""
        if not self.socket:
            return None

        try:
            conn, addr = await asyncio.get_event_loop().run_in_executor(
                None, self.socket.accept
            )

            new_transport = RealSCTPTransport(self.max_streams)
            new_transport.socket = conn
            new_transport._sctp_available = self._sctp_available
            new_transport._sctp_module = self._sctp_module
            new_transport._update_association_info()

            self.logger.info(f"Accepted SCTP connection from {addr}")
            return new_transport

        except Exception as e:
            self.logger.error(f"SCTP accept failed: {e}")
            return None

    def _enable_events(self):
        """Enable SCTP event notifications"""
        if self._sctp_available and self.socket:
            try:
                # Enable various SCTP events
                self.socket.sctp_set_events({
                    'data_io': True,
                    'association': True,
                    'address': True,
                    'send_failure': True,
                    'peer_error': True,
                    'shutdown': True,
                    'partial_delivery': True,
                })
            except Exception as e:
                self.logger.warning(f"Could not enable SCTP events: {e}")

    def _update_association_info(self):
        """Update association information from socket"""
        if self._sctp_available and self.socket:
            try:
                # Get local addresses
                local_addrs = [addr[0] for addr in self.socket.getsockname() if isinstance(addr, tuple)]
                if not local_addrs:
                    local_addr, local_port = self.socket.getsockname()
                    local_addrs = [local_addr]

                # Get peer addresses
                remote_addrs = []
                remote_port = 0
                try:
                    peer_addr, peer_port = self.socket.getpeername()
                    remote_addrs = [peer_addr]
                    remote_port = peer_port
                except:
                    pass

                self.association = SCTPAssociation(
                    assoc_id=0,
                    local_addrs=local_addrs,
                    remote_addrs=remote_addrs,
                    local_port=local_addrs[0] if local_addrs else 0,
                    remote_port=remote_port,
                    state="ESTABLISHED",
                    inbound_streams=self.max_streams,
                    outbound_streams=self.max_streams
                )
            except Exception as e:
                self.logger.warning(f"Could not get association info: {e}")

    async def send(self, message: SCTPMessage) -> bool:
        """Send SCTP message"""
        if not self._sctp_available or not self.socket:
            return False

        try:
            # Use sctp_send for message-oriented send
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.socket.sctp_send(
                    message.data,
                    ppid=message.ppid,
                    stream=message.stream_id,
                    flags=message.flags
                )
            )
            self.logger.debug(f"Sent {len(message.data)} bytes on stream {message.stream_id}")
            return True

        except Exception as e:
            self.logger.error(f"SCTP send failed: {e}")
            return False

    async def recv(self, timeout: Optional[float] = None) -> Optional[SCTPMessage]:
        """Receive SCTP message"""
        if not self._sctp_available or not self.socket:
            return None

        try:
            # Set timeout if specified
            if timeout:
                self.socket.settimeout(timeout)
            else:
                self.socket.setblocking(True)

            # Receive with metadata
            data, notif, sender, flags = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.socket.sctp_recv(65535)
            )

            if data:
                return SCTPMessage(
                    data=data,
                    stream_id=notif.get('stream', 0) if notif else 0,
                    ppid=notif.get('ppid', 0) if notif else 0,
                    flags=flags,
                    peer_addr=sender
                )

            return None

        except socket.timeout:
            return None
        except Exception as e:
            self.logger.error(f"SCTP recv failed: {e}")
            return None

    async def close(self):
        """Close SCTP transport"""
        if self.socket:
            try:
                self.socket.close()
                self.logger.info("SCTP transport closed")
            except Exception as e:
                self.logger.error(f"Error closing SCTP socket: {e}")
            self.socket = None

    def get_association(self) -> Optional[SCTPAssociation]:
        """Get association information"""
        return self.association


# =============================================================================
# TCP Fallback Implementation
# =============================================================================

class TCPFallbackTransport(SCTPTransportBase):
    """
    TCP fallback transport for environments without SCTP support.

    This implementation emulates SCTP semantics over TCP:
    - Message framing (length-prefixed messages)
    - Stream ID in message header
    - No multi-homing (single address only)
    """

    # Message format: [4 bytes length][2 bytes stream_id][4 bytes ppid][data]
    HEADER_SIZE = 10

    def __init__(self):
        self.reader = None
        self.writer = None
        self.server = None
        self.association = None
        self.logger = logging.getLogger(__name__)

    async def connect(self, remote_addrs: List[str], remote_port: int) -> bool:
        """Connect using TCP"""
        try:
            primary_addr = remote_addrs[0]
            self.reader, self.writer = await asyncio.open_connection(
                primary_addr, remote_port
            )

            self.association = SCTPAssociation(
                assoc_id=0,
                local_addrs=[self.writer.get_extra_info('sockname')[0]],
                remote_addrs=[primary_addr],
                local_port=self.writer.get_extra_info('sockname')[1],
                remote_port=remote_port,
                state="ESTABLISHED",
                inbound_streams=1,
                outbound_streams=1
            )

            self.logger.info(f"TCP fallback connected to {primary_addr}:{remote_port}")
            return True

        except Exception as e:
            self.logger.error(f"TCP connect failed: {e}")
            return False

    async def listen(self, local_addrs: List[str], local_port: int):
        """Start listening using TCP"""
        try:
            primary_addr = local_addrs[0]
            self.server = await asyncio.start_server(
                self._handle_connection,
                primary_addr, local_port
            )

            self.logger.info(f"TCP fallback listening on {primary_addr}:{local_port}")

        except Exception as e:
            self.logger.error(f"TCP listen failed: {e}")
            raise

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming TCP connection"""
        self.reader = reader
        self.writer = writer

        peer_addr = writer.get_extra_info('peername')
        self.association = SCTPAssociation(
            assoc_id=0,
            local_addrs=[writer.get_extra_info('sockname')[0]],
            remote_addrs=[peer_addr[0]],
            local_port=writer.get_extra_info('sockname')[1],
            remote_port=peer_addr[1],
            state="ESTABLISHED",
            inbound_streams=1,
            outbound_streams=1
        )

        self.logger.info(f"TCP fallback accepted connection from {peer_addr}")

    async def accept(self) -> Optional['TCPFallbackTransport']:
        """Accept is handled by the server callback"""
        # Wait for connection
        await asyncio.sleep(0.1)
        if self.reader and self.writer:
            return self
        return None

    async def send(self, message: SCTPMessage) -> bool:
        """Send message with framing"""
        if not self.writer:
            return False

        try:
            # Create header
            header = struct.pack('>IHI',
                len(message.data),
                message.stream_id,
                message.ppid
            )

            # Send header + data
            self.writer.write(header + message.data)
            await self.writer.drain()

            self.logger.debug(f"Sent {len(message.data)} bytes (stream {message.stream_id})")
            return True

        except Exception as e:
            self.logger.error(f"TCP send failed: {e}")
            return False

    async def recv(self, timeout: Optional[float] = None) -> Optional[SCTPMessage]:
        """Receive message with framing"""
        if not self.reader:
            return None

        try:
            # Read header
            if timeout:
                header = await asyncio.wait_for(
                    self.reader.readexactly(self.HEADER_SIZE),
                    timeout
                )
            else:
                header = await self.reader.readexactly(self.HEADER_SIZE)

            # Parse header
            length, stream_id, ppid = struct.unpack('>IHI', header)

            # Read data
            data = await self.reader.readexactly(length)

            return SCTPMessage(
                data=data,
                stream_id=stream_id,
                ppid=ppid
            )

        except asyncio.TimeoutError:
            return None
        except asyncio.IncompleteReadError:
            self.logger.warning("Connection closed by peer")
            return None
        except Exception as e:
            self.logger.error(f"TCP recv failed: {e}")
            return None

    async def close(self):
        """Close TCP transport"""
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
                self.logger.info("TCP fallback transport closed")
            except Exception as e:
                self.logger.error(f"Error closing TCP socket: {e}")

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        self.reader = None
        self.writer = None
        self.server = None

    def get_association(self) -> Optional[SCTPAssociation]:
        """Get association information"""
        return self.association


# =============================================================================
# SCTP Transport Factory
# =============================================================================

def create_sctp_transport(use_fallback: bool = False) -> SCTPTransportBase:
    """
    Create SCTP transport, with automatic fallback to TCP if SCTP unavailable.

    Args:
        use_fallback: Force use of TCP fallback

    Returns:
        SCTP transport instance
    """
    if use_fallback:
        logger.info("Using TCP fallback transport (forced)")
        return TCPFallbackTransport()

    # Try to create real SCTP transport
    real_transport = RealSCTPTransport()
    if real_transport._sctp_available:
        logger.info("Using real SCTP transport")
        return real_transport
    else:
        logger.info("Using TCP fallback transport (SCTP unavailable)")
        return TCPFallbackTransport()


# =============================================================================
# NGAP SCTP Handler
# =============================================================================

class NGAPSCTPHandler:
    """
    NGAP-specific SCTP handler for gNB-AMF communication.

    Implements SCTP usage per TS 38.412:
    - Stream 0 for non-UE-associated signaling
    - Stream 1+ for UE-associated signaling
    - PPID = 60 for NGAP
    """

    # NGAP stream allocation
    NON_UE_STREAM = 0
    UE_STREAM_BASE = 1

    def __init__(self, role: str, local_addrs: List[str], local_port: int = NGAP_SCTP_PORT):
        """
        Initialize NGAP SCTP handler.

        Args:
            role: "gNB" or "AMF"
            local_addrs: List of local IP addresses for multi-homing
            local_port: Local port (default 38412)
        """
        self.role = role
        self.local_addrs = local_addrs
        self.local_port = local_port
        self.transport: Optional[SCTPTransportBase] = None
        self.message_handler: Optional[Callable] = None
        self.running = False
        self.ue_stream_map: Dict[int, int] = {}  # UE ID -> stream ID
        self.next_stream_id = self.UE_STREAM_BASE
        self.logger = logging.getLogger(__name__)

    def set_message_handler(self, handler: Callable[[bytes, int], None]):
        """Set handler for incoming NGAP messages"""
        self.message_handler = handler

    async def start(self, use_tcp_fallback: bool = False):
        """Start SCTP transport"""
        self.transport = create_sctp_transport(use_tcp_fallback)

        if self.role == "AMF":
            await self.transport.listen(self.local_addrs, self.local_port)
            self.logger.info(f"AMF NGAP listening on port {self.local_port}")
        else:
            self.logger.info(f"gNB NGAP handler started")

        self.running = True
        asyncio.create_task(self._receive_loop())

    async def connect(self, remote_addrs: List[str], remote_port: int = NGAP_SCTP_PORT) -> bool:
        """Connect to remote endpoint (gNB -> AMF)"""
        if self.role != "gNB":
            raise ValueError("Only gNB can initiate connection")

        return await self.transport.connect(remote_addrs, remote_port)

    async def accept(self) -> bool:
        """Accept incoming connection (AMF)"""
        if self.role != "AMF":
            raise ValueError("Only AMF can accept connections")

        accepted = await self.transport.accept()
        return accepted is not None

    async def send_non_ue_message(self, data: bytes) -> bool:
        """Send non-UE-associated NGAP message (stream 0)"""
        msg = SCTPMessage(
            data=data,
            stream_id=self.NON_UE_STREAM,
            ppid=NGAP_PPID
        )
        return await self.transport.send(msg)

    async def send_ue_message(self, ue_id: int, data: bytes) -> bool:
        """Send UE-associated NGAP message"""
        # Get or allocate stream for this UE
        if ue_id not in self.ue_stream_map:
            self.ue_stream_map[ue_id] = self.next_stream_id
            self.next_stream_id += 1
            # Wrap around if needed
            if self.next_stream_id > 10:  # Max streams
                self.next_stream_id = self.UE_STREAM_BASE

        stream_id = self.ue_stream_map[ue_id]

        msg = SCTPMessage(
            data=data,
            stream_id=stream_id,
            ppid=NGAP_PPID
        )
        return await self.transport.send(msg)

    async def _receive_loop(self):
        """Background receive loop"""
        while self.running:
            try:
                msg = await self.transport.recv(timeout=1.0)
                if msg and self.message_handler:
                    # Dispatch to handler
                    self.message_handler(msg.data, msg.stream_id)
            except Exception as e:
                if self.running:
                    self.logger.error(f"Receive error: {e}")
                    await asyncio.sleep(0.1)

    async def stop(self):
        """Stop SCTP transport"""
        self.running = False
        if self.transport:
            await self.transport.close()
        self.logger.info(f"{self.role} NGAP handler stopped")

    def get_association_info(self) -> Optional[Dict[str, Any]]:
        """Get SCTP association information"""
        if self.transport:
            assoc = self.transport.get_association()
            if assoc:
                return {
                    'state': assoc.state,
                    'local_addrs': assoc.local_addrs,
                    'remote_addrs': assoc.remote_addrs,
                    'inbound_streams': assoc.inbound_streams,
                    'outbound_streams': assoc.outbound_streams
                }
        return None


# =============================================================================
# Demo/Test Functions
# =============================================================================

async def demo_sctp_transport():
    """Demonstrate SCTP transport usage"""
    print("=" * 60)
    print("SCTP Transport Demo")
    print("=" * 60)

    # Check SCTP availability
    transport = create_sctp_transport()
    print(f"\nTransport type: {type(transport).__name__}")

    # Create AMF handler
    amf = NGAPSCTPHandler("AMF", ["127.0.0.1"], 38412)

    # Create gNB handler
    gnb = NGAPSCTPHandler("gNB", ["127.0.0.1"], 0)

    # Set message handlers
    received_messages = []

    def amf_handler(data: bytes, stream_id: int):
        print(f"  AMF received {len(data)} bytes on stream {stream_id}")
        received_messages.append(('amf', data, stream_id))

    def gnb_handler(data: bytes, stream_id: int):
        print(f"  gNB received {len(data)} bytes on stream {stream_id}")
        received_messages.append(('gnb', data, stream_id))

    amf.set_message_handler(amf_handler)
    gnb.set_message_handler(gnb_handler)

    print("\n1. Starting AMF...")
    await amf.start(use_tcp_fallback=True)  # Use TCP for demo

    print("\n2. Starting gNB and connecting to AMF...")
    await gnb.start(use_tcp_fallback=True)
    connected = await gnb.connect(["127.0.0.1"], 38412)
    print(f"   Connected: {connected}")

    if connected:
        # Give time for connection to establish
        await asyncio.sleep(0.5)

        print("\n3. Sending NG Setup Request (non-UE message)...")
        ng_setup_data = b'\x00\x15\x00\x2c\x00\x00\x04\x00\x1b\x00\x08\x02\xf8\x39\x00\x00\x00\x00\x01'
        await gnb.send_non_ue_message(ng_setup_data)

        print("\n4. Sending Initial UE Message (UE-associated)...")
        initial_ue_data = b'\x00\x0f\x40\x55\x00\x00\x05\x00\x55\x00\x02\x00\x01'
        await gnb.send_ue_message(1, initial_ue_data)

        # Wait for messages
        await asyncio.sleep(0.5)

        print(f"\n5. Messages received: {len(received_messages)}")

        # Show association info
        print("\n6. Association info:")
        gnb_info = gnb.get_association_info()
        if gnb_info:
            print(f"   gNB: state={gnb_info['state']}, streams={gnb_info['outbound_streams']}")

    print("\n7. Stopping...")
    await gnb.stop()
    await amf.stop()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(demo_sctp_transport())
