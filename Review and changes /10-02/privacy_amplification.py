"""
Layer 3: Privacy Amplification

Implements privacy amplification using SHAKE-256 extendable-output function
to reduce adversary knowledge of quantum keys.
"""

from dataclasses import dataclass
from typing import Optional
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)


@dataclass
class AmplificationResult:
    """Result container for privacy amplification"""
    success: bool
    amplified_key: Optional[bytes] = None
    output_length: int = 0
    compression_ratio: float = 0.0
    error: Optional[str] = None


class PrivacyAmplifier:
    """
    Privacy amplification using SHAKE-256.

    Privacy amplification reduces an adversary's information about a key
    by applying a universal hash function (SHAKE-256) to compress the key.

    For a key with min-entropy H_min and adversary information I:
    - Output length: L = H_min - I - security_parameter
    """

    def __init__(self, security_parameter: int = 256):
        """
        Initialize privacy amplifier.

        Args:
            security_parameter: Security parameter in bits (e.g., 256 for 256-bit security)
        """
        self.security_param = security_parameter
        logger.info(f"Initialized privacy amplifier (security: {security_parameter} bits)")

    def calculate_output_length(self, input_length: int, min_entropy: float,
                                adversary_info_bits: int = 0) -> int:
        """
        Calculate optimal output length for privacy amplification.

        Args:
            input_length: Input key length in bytes
            min_entropy: Min-entropy per bit (0.0 to 1.0)
            adversary_info_bits: Estimated adversary information in bits

        Returns:
            Output length in bytes
        """
        input_bits = input_length * 8
        total_entropy = input_bits * min_entropy

        # Output length = min_entropy - adversary_info - security_parameter
        available_entropy = total_entropy - adversary_info_bits - self.security_param

        if available_entropy <= 0:
            logger.warning("Insufficient entropy for privacy amplification")
            return 0

        output_bits = int(available_entropy)
        output_bytes = output_bits // 8

        logger.debug(f"Calculated output length: {output_bytes} bytes from {input_length} bytes")
        return output_bytes

    def amplify(self, key: bytes, output_length: Optional[int] = None,
               seed: Optional[bytes] = None) -> AmplificationResult:
        """
        Perform privacy amplification on input key.

        Args:
            key: Input key material
            output_length: Desired output length in bytes (if None, use input length)
            seed: Optional random seed for deterministic testing

        Returns:
            AmplificationResult with amplified key
        """
        try:
            if len(key) == 0:
                return AmplificationResult(
                    success=False,
                    error="Input key is empty"
                )

            # Default output length is same as input
            if output_length is None:
                output_length = len(key)

            if output_length <= 0:
                return AmplificationResult(
                    success=False,
                    error="Output length must be positive"
                )

            # Generate random seed if not provided
            if seed is None:
                seed = secrets.token_bytes(32)

            # Apply SHAKE-256 for privacy amplification
            # SHAKE-256(key || seed) -> output of desired length
            shake = hashlib.shake_256()
            shake.update(key)
            shake.update(seed)
            amplified_key = shake.digest(output_length)

            compression_ratio = len(key) / output_length if output_length > 0 else 0

            result = AmplificationResult(
                success=True,
                amplified_key=amplified_key,
                output_length=output_length,
                compression_ratio=compression_ratio
            )

            logger.info(f"Privacy amplification: {len(key)} → {output_length} bytes "
                       f"(ratio: {compression_ratio:.2f})")
            return result

        except Exception as e:
            logger.error(f"Privacy amplification failed: {e}")
            return AmplificationResult(
                success=False,
                error=str(e)
            )

    def amplify_with_entropy(self, key: bytes, min_entropy: float,
                            adversary_info_bits: int = 0) -> AmplificationResult:
        """
        Perform privacy amplification with automatic length calculation.

        Args:
            key: Input key material
            min_entropy: Min-entropy per bit (0.0 to 1.0)
            adversary_info_bits: Estimated adversary information

        Returns:
            AmplificationResult with amplified key
        """
        output_length = self.calculate_output_length(
            len(key), min_entropy, adversary_info_bits
        )

        if output_length == 0:
            return AmplificationResult(
                success=False,
                error="Insufficient entropy for secure amplification"
            )

        return self.amplify(key, output_length)

    def two_universal_hash(self, key: bytes, output_length: int,
                          hash_seed: bytes) -> bytes:
        """
        Apply 2-universal hash function for privacy amplification.

        Uses SHAKE-256 as a 2-universal hash function family.

        Args:
            key: Input key
            output_length: Output length in bytes
            hash_seed: Seed selecting hash function from family

        Returns:
            Hashed output
        """
        shake = hashlib.shake_256()
        shake.update(hash_seed)
        shake.update(key)
        return shake.digest(output_length)

    def get_statistics(self) -> dict:
        """Get amplifier configuration"""
        return {
            "security_parameter": self.security_param,
            "hash_function": "SHAKE-256",
            "theoretical_security": f"{self.security_param}-bit"
        }
