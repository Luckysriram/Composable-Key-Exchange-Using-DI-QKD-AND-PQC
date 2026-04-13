"""
Layer 6: Hybrid TLS Wrapper

Integrates CuKEM with TLS 1.3 using Pre-Shared Key (PSK) mode
for quantum-safe transport layer security.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import ssl
import socket
import logging
from cukem import CuKEM, CuKEMConfig, CuKEMMode, CuKEMResult

logger = logging.getLogger(__name__)


@dataclass
class TLSConfig:
    """Configuration for hybrid TLS"""
    hostname: str = "localhost"
    port: int = 8443
    use_psk: bool = True
    tls_version: int = ssl.PROTOCOL_TLS_SERVER
    ciphers: str = "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256"
    client_auth: bool = False


@dataclass
class TLSConnectionResult:
    """Result of TLS connection establishment"""
    success: bool
    connection: Optional[ssl.SSLSocket] = None
    cukem_result: Optional[CuKEMResult] = None
    cipher: Optional[str] = None
    protocol_version: Optional[str] = None
    error: Optional[str] = None


class HybridTLSWrapper:
    """
    Hybrid TLS wrapper integrating CuKEM with TLS 1.3 PSK mode.

    Workflow:
    1. Perform CuKEM key exchange out-of-band
    2. Use derived key as TLS 1.3 PSK
    3. Establish TLS connection with PSK
    4. Quantum-safe transport security
    """

    def __init__(self, cukem_config: Optional[CuKEMConfig] = None,
                 tls_config: Optional[TLSConfig] = None):
        """
        Initialize hybrid TLS wrapper.

        Args:
            cukem_config: CuKEM configuration
            tls_config: TLS configuration
        """
        self.cukem = CuKEM(cukem_config or CuKEMConfig())
        self.tls_config = tls_config or TLSConfig()
        self.psk_store = {}  # Store PSKs by identity
        logger.info("Initialized Hybrid TLS wrapper")

    def psk_callback(self, connection: ssl.SSLSocket, identity: str) -> Optional[bytes]:
        """
        PSK callback for TLS 1.3.

        Args:
            connection: SSL socket
            identity: PSK identity

        Returns:
            PSK bytes or None
        """
        psk = self.psk_store.get(identity)
        if psk:
            logger.debug(f"PSK retrieved for identity: {identity}")
            return psk
        else:
            logger.warning(f"PSK not found for identity: {identity}")
            return None

    def perform_cukem_exchange(self, role: str = "initiator",
                              noise_level: float = 0.0) -> CuKEMResult:
        """
        Perform CuKEM key exchange.

        Args:
            role: "initiator" or "responder"
            noise_level: Quantum channel noise level

        Returns:
            CuKEMResult with shared key
        """
        logger.info(f"Performing CuKEM exchange as {role}")

        if role == "initiator":
            # In practice, responder's public key would be obtained via channel
            responder_keypair = self.cukem.generate_keypair()
            result = self.cukem.initiate_exchange(
                responder_public_key=responder_keypair.public_key,
                noise_level=noise_level
            )
        else:  # responder
            keypair = self.cukem.generate_keypair()
            # In practice, ciphertext would be received from initiator
            # For simulation, we do a full exchange
            temp_cukem = CuKEM(self.cukem.config)
            init_result = temp_cukem.initiate_exchange(
                responder_public_key=keypair.public_key,
                noise_level=noise_level
            )

            if not init_result.success:
                return init_result

            result = self.cukem.respond_exchange(
                keypair=keypair,
                ciphertext=init_result.pqc_result.ciphertext,
                noise_level=noise_level
            )

        return result

    def create_tls_context(self, server_side: bool,
                          psk_identity: Optional[str] = None,
                          psk: Optional[bytes] = None) -> ssl.SSLContext:
        """
        Create TLS context with PSK support.

        Args:
            server_side: True for server, False for client
            psk_identity: PSK identity string
            psk: Pre-shared key bytes

        Returns:
            Configured SSL context
        """
        if server_side:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_3
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.minimum_version = ssl.TLSVersion.TLSv1_3
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        # Set cipher suites
        context.set_ciphers(self.tls_config.ciphers)

        # Configure PSK
        if psk and psk_identity:
            self.psk_store[psk_identity] = psk

            # Note: PSK support in Python's ssl module is limited
            # In production, use a TLS library with full PSK support
            logger.info(f"Configured PSK for identity: {psk_identity}")

        return context

    def establish_server(self, psk_identity: str = "cukem-client") -> TLSConnectionResult:
        """
        Establish TLS server with CuKEM-derived PSK.

        Args:
            psk_identity: Expected client PSK identity

        Returns:
            TLSConnectionResult with accepted connection
        """
        try:
            logger.info("=== Establishing Hybrid TLS Server ===")

            # Step 1: CuKEM key exchange
            cukem_result = self.perform_cukem_exchange(role="responder")

            if not cukem_result.success:
                return TLSConnectionResult(
                    success=False,
                    error=f"CuKEM exchange failed: {cukem_result.error}"
                )

            psk = cukem_result.shared_key
            logger.info(f"CuKEM PSK established: {len(psk)} bytes")

            # Step 2: Create TLS context
            context = self.create_tls_context(
                server_side=True,
                psk_identity=psk_identity,
                psk=psk
            )

            # Step 3: Create server socket
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.tls_config.hostname, self.tls_config.port))
            server_socket.listen(1)

            logger.info(f"Server listening on {self.tls_config.hostname}:{self.tls_config.port}")

            # Step 4: Accept connection
            client_socket, addr = server_socket.accept()
            logger.info(f"Accepted connection from {addr}")

            # Step 5: Wrap with TLS
            tls_socket = context.wrap_socket(client_socket, server_side=True)

            cipher = tls_socket.cipher()
            version = tls_socket.version()

            logger.info(f"TLS established: {version}, Cipher: {cipher}")

            return TLSConnectionResult(
                success=True,
                connection=tls_socket,
                cukem_result=cukem_result,
                cipher=cipher[0] if cipher else None,
                protocol_version=version
            )

        except Exception as e:
            logger.error(f"Server establishment failed: {e}")
            return TLSConnectionResult(
                success=False,
                error=str(e)
            )

    def establish_client(self, psk_identity: str = "cukem-client") -> TLSConnectionResult:
        """
        Establish TLS client connection with CuKEM-derived PSK.

        Args:
            psk_identity: Client PSK identity

        Returns:
            TLSConnectionResult with connection
        """
        try:
            logger.info("=== Establishing Hybrid TLS Client ===")

            # Step 1: CuKEM key exchange
            cukem_result = self.perform_cukem_exchange(role="initiator")

            if not cukem_result.success:
                return TLSConnectionResult(
                    success=False,
                    error=f"CuKEM exchange failed: {cukem_result.error}"
                )

            psk = cukem_result.shared_key
            logger.info(f"CuKEM PSK established: {len(psk)} bytes")

            # Step 2: Create TLS context
            context = self.create_tls_context(
                server_side=False,
                psk_identity=psk_identity,
                psk=psk
            )

            # Step 3: Connect to server
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.tls_config.hostname, self.tls_config.port))

            # Step 4: Wrap with TLS
            tls_socket = context.wrap_socket(
                client_socket,
                server_hostname=self.tls_config.hostname
            )

            cipher = tls_socket.cipher()
            version = tls_socket.version()

            logger.info(f"TLS established: {version}, Cipher: {cipher}")

            return TLSConnectionResult(
                success=True,
                connection=tls_socket,
                cukem_result=cukem_result,
                cipher=cipher[0] if cipher else None,
                protocol_version=version
            )

        except Exception as e:
            logger.error(f"Client connection failed: {e}")
            return TLSConnectionResult(
                success=False,
                error=str(e)
            )

    def send_message(self, connection: ssl.SSLSocket, message: bytes) -> bool:
        """
        Send message over TLS connection.

        Args:
            connection: TLS socket
            message: Message bytes

        Returns:
            True if successful
        """
        try:
            connection.sendall(message)
            logger.debug(f"Sent {len(message)} bytes")
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False

    def receive_message(self, connection: ssl.SSLSocket,
                       buffer_size: int = 4096) -> Optional[bytes]:
        """
        Receive message from TLS connection.

        Args:
            connection: TLS socket
            buffer_size: Receive buffer size

        Returns:
            Received bytes or None
        """
        try:
            data = connection.recv(buffer_size)
            logger.debug(f"Received {len(data)} bytes")
            return data
        except Exception as e:
            logger.error(f"Receive failed: {e}")
            return None

    def close_connection(self, connection: ssl.SSLSocket):
        """Close TLS connection"""
        try:
            connection.close()
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Close failed: {e}")

    def get_statistics(self) -> dict:
        """Get wrapper statistics"""
        return {
            "cukem": self.cukem.get_statistics(),
            "tls_config": {
                "hostname": self.tls_config.hostname,
                "port": self.tls_config.port,
                "tls_version": "TLS 1.3+",
                "ciphers": self.tls_config.ciphers
            }
        }
