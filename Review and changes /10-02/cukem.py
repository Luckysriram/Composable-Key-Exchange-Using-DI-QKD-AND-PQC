"""
Layer 5: CuKEM (Custom Unified KEM) Abstraction Layer

Orchestrates the complete hybrid quantum-classical key exchange:
- Layer 1: Post-quantum KEM (MLKEM768)
- Layer 2: Quantum key distribution (BB84 + CHSH)
- Layer 3: Entropy validation and privacy amplification
- Layer 4: Key combination (HKDF)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import logging

# Import all layer components
from pqc_kem import MLKEM768, PQCKeyPair, PQCResult
from bb84_simulator import BB84Simulator, BB84Result
from chsh_bell_test import CHSHBellTest, CHSHResult
from entropy_estimator import EntropyEstimator, EntropyResult
from privacy_amplification import PrivacyAmplifier, AmplificationResult
from hkdf_combiner import HKDFCombiner, CombinerResult

logger = logging.getLogger(__name__)


class CuKEMMode(Enum):
    """CuKEM operational modes"""
    HYBRID = "hybrid"           # PQC + QKD
    PQC_ONLY = "pqc_only"      # Post-quantum only
    QKD_ONLY = "qkd_only"      # Quantum only
    FALLBACK = "fallback"       # Automatic fallback


@dataclass
class CuKEMConfig:
    """Configuration for CuKEM"""
    mode: CuKEMMode = CuKEMMode.HYBRID
    n_qubits: int = 256
    min_entropy: float = 0.8
    qber_threshold: float = 0.11
    chsh_verification: bool = True
    output_key_length: int = 32


@dataclass
class CuKEMResult:
    """Result of CuKEM key exchange"""
    success: bool
    mode: CuKEMMode
    shared_key: Optional[bytes] = None
    key_length: int = 0

    # Layer-specific results
    pqc_result: Optional[PQCResult] = None
    bb84_result: Optional[BB84Result] = None
    chsh_result: Optional[CHSHResult] = None
    entropy_result: Optional[EntropyResult] = None
    amplification_result: Optional[AmplificationResult] = None
    combiner_result: Optional[CombinerResult] = None

    # Metadata
    fallback_used: bool = False
    error: Optional[str] = None
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class CuKEM:
    """
    CuKEM: Custom Unified Key Encapsulation Mechanism

    Provides hybrid quantum-classical key exchange with:
    - Post-quantum security (MLKEM768)
    - Quantum key distribution (BB84)
    - Entanglement verification (CHSH)
    - Entropy validation (NIST 800-90B)
    - Privacy amplification (SHAKE-256)
    - Key combination (HKDF)
    """

    def __init__(self, config: Optional[CuKEMConfig] = None):
        """
        Initialize CuKEM.

        Args:
            config: CuKEM configuration (uses defaults if None)
        """
        self.config = config or CuKEMConfig()

        # Initialize all layers
        self.pqc_kem = MLKEM768()
        self.bb84 = BB84Simulator(n_qubits=self.config.n_qubits,
                                  error_correction_threshold=self.config.qber_threshold)
        self.chsh = CHSHBellTest()
        self.entropy = EntropyEstimator(min_entropy_per_bit=self.config.min_entropy)
        self.amplifier = PrivacyAmplifier()
        self.combiner = HKDFCombiner()

        logger.info(f"Initialized CuKEM in {self.config.mode.value} mode")

    def generate_keypair(self) -> PQCKeyPair:
        """
        Generate PQC keypair for the responder.

        Returns:
            PQCKeyPair for key exchange
        """
        return self.pqc_kem.generate_keypair()

    def establish_quantum_channel(self, noise_level: float = 0.0) -> Tuple[bool, Optional[bytes]]:
        """
        Establish quantum channel and generate quantum key.

        Args:
            noise_level: Simulated channel noise

        Returns:
            Tuple of (success, quantum_key)
        """
        warnings = []

        # Step 1: CHSH verification (if enabled)
        if self.config.chsh_verification:
            logger.info("Performing CHSH entanglement verification")
            chsh_result = self.chsh.execute_chsh_test()

            if not chsh_result.success or not chsh_result.violation:
                logger.warning("CHSH verification failed - quantum channel may be compromised")
                warnings.append("CHSH verification failed")
                if self.config.mode == CuKEMMode.QKD_ONLY:
                    return False, None

        # Step 2: BB84 quantum key distribution
        logger.info("Executing BB84 protocol")
        bb84_result = self.bb84.execute_protocol(noise_level=noise_level)

        if not bb84_result.success:
            logger.error(f"BB84 failed: {bb84_result.error}")
            return False, None

        quantum_key = bb84_result.raw_key

        # Step 3: Entropy validation
        logger.info("Validating quantum key entropy")
        entropy_result = self.entropy.estimate_entropy(quantum_key)

        if not entropy_result.sufficient:
            logger.warning(f"Insufficient entropy: {entropy_result.min_entropy:.4f}")
            warnings.append(f"Low entropy: {entropy_result.min_entropy:.4f}")

        # Step 4: Privacy amplification
        logger.info("Applying privacy amplification")
        amp_result = self.amplifier.amplify_with_entropy(
            quantum_key,
            min_entropy=entropy_result.min_entropy
        )

        if not amp_result.success:
            logger.error(f"Privacy amplification failed: {amp_result.error}")
            return False, None

        logger.info(f"Quantum channel established: {amp_result.output_length} bytes")
        return True, amp_result.amplified_key

    def initiate_exchange(self, responder_public_key: bytes,
                         noise_level: float = 0.0) -> CuKEMResult:
        """
        Initiate key exchange (initiator role).

        Args:
            responder_public_key: Responder's PQC public key
            noise_level: Simulated quantum channel noise

        Returns:
            CuKEMResult with shared key
        """
        warnings = []
        pqc_result = None
        bb84_result = None
        chsh_result = None
        entropy_result = None
        amp_result = None
        combiner_result = None
        fallback_used = False

        try:
            logger.info("=== CuKEM Key Exchange: INITIATOR ===")

            # Layer 1: Post-Quantum KEM
            if self.config.mode in [CuKEMMode.HYBRID, CuKEMMode.PQC_ONLY]:
                logger.info("Performing PQC encapsulation")
                pqc_result = self.pqc_kem.encapsulate(responder_public_key)

                if not pqc_result.success:
                    logger.error(f"PQC encapsulation failed: {pqc_result.error}")
                    if self.config.mode == CuKEMMode.PQC_ONLY:
                        return CuKEMResult(
                            success=False,
                            mode=self.config.mode,
                            error=f"PQC failed: {pqc_result.error}"
                        )
                    warnings.append("PQC encapsulation failed")

            # Layer 2-3: Quantum Key Distribution
            quantum_key = None
            if self.config.mode in [CuKEMMode.HYBRID, CuKEMMode.QKD_ONLY]:
                success, quantum_key = self.establish_quantum_channel(noise_level)

                if not success:
                    logger.warning("Quantum channel establishment failed")
                    if self.config.mode == CuKEMMode.QKD_ONLY:
                        return CuKEMResult(
                            success=False,
                            mode=self.config.mode,
                            error="QKD failed"
                        )
                    warnings.append("QKD failed - using PQC only")
                    fallback_used = True

            # Layer 4: Key Combination
            final_key = None

            if self.config.mode == CuKEMMode.HYBRID:
                if pqc_result and pqc_result.success and quantum_key:
                    logger.info("Combining PQC and quantum keys")
                    combiner_result = self.combiner.combine_keys(
                        pqc_key=pqc_result.shared_secret,
                        quantum_key=quantum_key,
                        output_length=self.config.output_key_length
                    )

                    if combiner_result.success:
                        final_key = combiner_result.combined_key
                    else:
                        logger.error("Key combination failed")
                        return CuKEMResult(
                            success=False,
                            mode=self.config.mode,
                            error="Key combination failed"
                        )
                elif pqc_result and pqc_result.success:
                    # Fallback to PQC only
                    logger.warning("Falling back to PQC-only mode")
                    final_key = pqc_result.shared_secret[:self.config.output_key_length]
                    fallback_used = True
                else:
                    return CuKEMResult(
                        success=False,
                        mode=self.config.mode,
                        error="Both PQC and QKD failed"
                    )

            elif self.config.mode == CuKEMMode.PQC_ONLY:
                final_key = pqc_result.shared_secret[:self.config.output_key_length]

            elif self.config.mode == CuKEMMode.QKD_ONLY:
                final_key = quantum_key[:self.config.output_key_length]

            logger.info(f"=== Key Exchange Complete: {len(final_key)} bytes ===")

            return CuKEMResult(
                success=True,
                mode=self.config.mode,
                shared_key=final_key,
                key_length=len(final_key),
                pqc_result=pqc_result,
                fallback_used=fallback_used,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Key exchange failed: {e}")
            return CuKEMResult(
                success=False,
                mode=self.config.mode,
                error=str(e),
                warnings=warnings
            )

    def respond_exchange(self, keypair: PQCKeyPair, ciphertext: bytes,
                        noise_level: float = 0.0) -> CuKEMResult:
        """
        Respond to key exchange (responder role).

        Args:
            keypair: Responder's PQC keypair
            ciphertext: PQC ciphertext from initiator
            noise_level: Simulated quantum channel noise

        Returns:
            CuKEMResult with shared key
        """
        warnings = []

        try:
            logger.info("=== CuKEM Key Exchange: RESPONDER ===")

            # Layer 1: Post-Quantum KEM Decapsulation
            pqc_result = None
            if self.config.mode in [CuKEMMode.HYBRID, CuKEMMode.PQC_ONLY]:
                logger.info("Performing PQC decapsulation")
                pqc_result = self.pqc_kem.decapsulate(ciphertext, keypair.secret_key)

                if not pqc_result.success:
                    logger.error(f"PQC decapsulation failed: {pqc_result.error}")
                    if self.config.mode == CuKEMMode.PQC_ONLY:
                        return CuKEMResult(
                            success=False,
                            mode=self.config.mode,
                            error=f"PQC failed: {pqc_result.error}"
                        )

            # Layer 2-3: Quantum Key Distribution
            quantum_key = None
            if self.config.mode in [CuKEMMode.HYBRID, CuKEMMode.QKD_ONLY]:
                success, quantum_key = self.establish_quantum_channel(noise_level)

                if not success and self.config.mode == CuKEMMode.QKD_ONLY:
                    return CuKEMResult(
                        success=False,
                        mode=self.config.mode,
                        error="QKD failed"
                    )

            # Layer 4: Key Combination (same logic as initiator)
            final_key = None

            if self.config.mode == CuKEMMode.HYBRID and pqc_result and quantum_key:
                combiner_result = self.combiner.combine_keys(
                    pqc_key=pqc_result.shared_secret,
                    quantum_key=quantum_key,
                    output_length=self.config.output_key_length
                )
                final_key = combiner_result.combined_key

            elif self.config.mode == CuKEMMode.PQC_ONLY and pqc_result:
                final_key = pqc_result.shared_secret[:self.config.output_key_length]

            elif self.config.mode == CuKEMMode.QKD_ONLY and quantum_key:
                final_key = quantum_key[:self.config.output_key_length]

            logger.info(f"=== Key Exchange Complete: {len(final_key)} bytes ===")

            return CuKEMResult(
                success=True,
                mode=self.config.mode,
                shared_key=final_key,
                key_length=len(final_key),
                pqc_result=pqc_result,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Key exchange failed: {e}")
            return CuKEMResult(
                success=False,
                mode=self.config.mode,
                error=str(e)
            )

    def get_statistics(self) -> dict:
        """Get comprehensive CuKEM statistics"""
        return {
            "config": {
                "mode": self.config.mode.value,
                "n_qubits": self.config.n_qubits,
                "min_entropy": self.config.min_entropy,
                "qber_threshold": self.config.qber_threshold,
                "output_key_length": self.config.output_key_length
            },
            "pqc": self.pqc_kem.get_details(),
            "bb84": self.bb84.get_statistics(),
            "chsh": self.chsh.get_theoretical_values(),
            "entropy": self.entropy.get_statistics(),
            "amplifier": self.amplifier.get_statistics(),
            "combiner": self.combiner.get_statistics()
        }
