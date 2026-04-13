"""
Layer 1: Post-Quantum Cryptography - MLKEM768 KEM Wrapper

This module provides a wrapper around the liboqs MLKEM768 (Kyber768) key
encapsulation mechanism for post-quantum key exchange.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import oqs
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class PQCKeyPair:
    """Container for post-quantum key pairs"""
    public_key: bytes
    secret_key: bytes
    algorithm: str = "MLKEM768"

    def __post_init__(self):
        """Validate key pair"""
        if not self.public_key or not self.secret_key:
            raise ValueError("Keys cannot be empty")
        if len(self.public_key) == 0 or len(self.secret_key) == 0:
            raise ValueError("Invalid key length")


@dataclass
class PQCResult:
    """Result container for PQC operations"""
    success: bool
    shared_secret: Optional[bytes] = None
    ciphertext: Optional[bytes] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None

    def get_key_fingerprint(self) -> Optional[str]:
        """Get SHA-256 fingerprint of shared secret"""
        if self.shared_secret:
            return hashlib.sha256(self.shared_secret).hexdigest()[:16]
        return None


class MLKEM768:
    """
    MLKEM768 (formerly Kyber768) post-quantum KEM implementation.

    NIST Level 3 security (~192-bit classical security).
    """

    ALGORITHM_NAME = "Kyber768"

    def __init__(self):
        """Initialize MLKEM768 KEM"""
        try:
            self.kem = oqs.KeyEncapsulation(self.ALGORITHM_NAME)
            logger.info(f"Initialized {self.ALGORITHM_NAME} KEM")
            logger.debug(f"Public key size: {self.kem.details['length_public_key']} bytes")
            logger.debug(f"Secret key size: {self.kem.details['length_secret_key']} bytes")
            logger.debug(f"Ciphertext size: {self.kem.details['length_ciphertext']} bytes")
            logger.debug(f"Shared secret size: {self.kem.details['length_shared_secret']} bytes")
        except Exception as e:
            logger.error(f"Failed to initialize {self.ALGORITHM_NAME}: {e}")
            raise

    def generate_keypair(self) -> PQCKeyPair:
        """
        Generate a new MLKEM768 key pair.

        Returns:
            PQCKeyPair containing public and secret keys

        Raises:
            RuntimeError: If key generation fails
        """
        try:
            public_key = self.kem.generate_keypair()
            secret_key = self.kem.export_secret_key()

            keypair = PQCKeyPair(
                public_key=public_key,
                secret_key=secret_key,
                algorithm=self.ALGORITHM_NAME
            )

            logger.info(f"Generated {self.ALGORITHM_NAME} keypair")
            return keypair

        except Exception as e:
            logger.error(f"Keypair generation failed: {e}")
            raise RuntimeError(f"Failed to generate keypair: {e}")

    def encapsulate(self, public_key: bytes) -> PQCResult:
        """
        Encapsulate a shared secret using recipient's public key.

        Args:
            public_key: Recipient's MLKEM768 public key

        Returns:
            PQCResult containing ciphertext and shared secret
        """
        try:
            ciphertext, shared_secret = self.kem.encap_secret(public_key)

            result = PQCResult(
                success=True,
                shared_secret=shared_secret,
                ciphertext=ciphertext,
                metadata={
                    "algorithm": self.ALGORITHM_NAME,
                    "ciphertext_size": len(ciphertext),
                    "shared_secret_size": len(shared_secret)
                }
            )

            logger.info(f"Encapsulation successful, fingerprint: {result.get_key_fingerprint()}")
            return result

        except Exception as e:
            logger.error(f"Encapsulation failed: {e}")
            return PQCResult(success=False, error=str(e))

    def decapsulate(self, ciphertext: bytes, secret_key: bytes) -> PQCResult:
        """
        Decapsulate shared secret using secret key.

        Args:
            ciphertext: Ciphertext from encapsulation
            secret_key: Recipient's MLKEM768 secret key

        Returns:
            PQCResult containing recovered shared secret
        """
        try:
            # Import the secret key
            self.kem = oqs.KeyEncapsulation(self.ALGORITHM_NAME, secret_key)
            shared_secret = self.kem.decap_secret(ciphertext)

            result = PQCResult(
                success=True,
                shared_secret=shared_secret,
                metadata={
                    "algorithm": self.ALGORITHM_NAME,
                    "shared_secret_size": len(shared_secret)
                }
            )

            logger.info(f"Decapsulation successful, fingerprint: {result.get_key_fingerprint()}")
            return result

        except Exception as e:
            logger.error(f"Decapsulation failed: {e}")
            return PQCResult(success=False, error=str(e))

    def get_details(self) -> dict:
        """Get KEM algorithm details"""
        return {
            "algorithm": self.ALGORITHM_NAME,
            "claimed_nist_level": self.kem.details.get("claimed_nist_level", 3),
            "public_key_bytes": self.kem.details["length_public_key"],
            "secret_key_bytes": self.kem.details["length_secret_key"],
            "ciphertext_bytes": self.kem.details["length_ciphertext"],
            "shared_secret_bytes": self.kem.details["length_shared_secret"]
        }
