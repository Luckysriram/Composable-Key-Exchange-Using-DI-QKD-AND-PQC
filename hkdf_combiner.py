"""
Layer 4: HKDF Key Combiner

Implements HKDF (HMAC-based Key Derivation Function) for combining
post-quantum and quantum-derived keys into a hybrid key.
"""

from dataclasses import dataclass
from typing import Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import logging

logger = logging.getLogger(__name__)


@dataclass
class CombinerResult:
    """Result container for key combination"""
    success: bool
    combined_key: Optional[bytes] = None
    key_length: int = 0
    error: Optional[str] = None


class HKDFCombiner:
    """
    HKDF-based key combiner for hybrid cryptography.

    Combines multiple key sources (PQC + Quantum) using HKDF
    to derive a single hybrid key with properties of both.

    HKDF(salt, IKM, info, L) = HKDF-Expand(HKDF-Extract(salt, IKM), info, L)
    """

    def __init__(self, hash_algorithm=hashes.SHA256()):
        """
        Initialize HKDF combiner.

        Args:
            hash_algorithm: Hash algorithm for HKDF (default: SHA-256)
        """
        self.hash_algorithm = hash_algorithm
        logger.info(f"Initialized HKDF combiner with {hash_algorithm.name}")

    def combine_keys(self, pqc_key: bytes, quantum_key: bytes,
                    output_length: int = 32,
                    salt: Optional[bytes] = None,
                    info: bytes = b"CuKEM-hybrid-key") -> CombinerResult:
        """
        Combine PQC and quantum keys using HKDF.

        Args:
            pqc_key: Post-quantum key material
            quantum_key: Quantum-derived key material
            output_length: Desired output key length in bytes
            salt: Optional salt value
            info: Context/application-specific info

        Returns:
            CombinerResult with combined key
        """
        try:
            if not pqc_key or not quantum_key:
                return CombinerResult(
                    success=False,
                    error="Both keys must be non-empty"
                )

            # Concatenate keys as input key material
            input_key_material = pqc_key + quantum_key

            # Derive combined key using HKDF
            hkdf = HKDF(
                algorithm=self.hash_algorithm,
                length=output_length,
                salt=salt,
                info=info
            )

            combined_key = hkdf.derive(input_key_material)

            logger.info(f"Combined keys: PQC({len(pqc_key)}B) + QKD({len(quantum_key)}B) "
                       f"→ {output_length}B")

            return CombinerResult(
                success=True,
                combined_key=combined_key,
                key_length=output_length
            )

        except Exception as e:
            logger.error(f"Key combination failed: {e}")
            return CombinerResult(
                success=False,
                error=str(e)
            )

    def derive_multiple_keys(self, master_key: bytes,
                           key_purposes: list,
                           key_length: int = 32,
                           salt: Optional[bytes] = None) -> dict:
        """
        Derive multiple purpose-specific keys from master key.

        Args:
            master_key: Master key material
            key_purposes: List of purpose strings (e.g., ["encryption", "authentication"])
            key_length: Length of each derived key
            salt: Optional salt

        Returns:
            Dictionary mapping purpose to derived key
        """
        derived_keys = {}

        for purpose in key_purposes:
            info = f"CuKEM-{purpose}".encode()
            result = self.combine_keys(
                pqc_key=master_key,
                quantum_key=b"",  # Single key derivation
                output_length=key_length,
                salt=salt,
                info=info
            )

            if result.success:
                derived_keys[purpose] = result.combined_key
                logger.debug(f"Derived {purpose} key: {key_length} bytes")

        return derived_keys

    def get_statistics(self) -> dict:
        """Get combiner configuration"""
        return {
            "kdf": "HKDF",
            "hash_algorithm": self.hash_algorithm.name,
            "mode": "hybrid_combination"
        }
