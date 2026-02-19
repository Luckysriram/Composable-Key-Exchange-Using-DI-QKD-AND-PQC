"""
Layer 2: BB84 Quantum Key Distribution Simulator

Implements the BB84 quantum key distribution protocol using Qiskit
for quantum state simulation and key establishment.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import logging
import secrets

logger = logging.getLogger(__name__)


@dataclass
class BB84Result:
    """Result container for BB84 protocol execution"""
    success: bool
    raw_key: Optional[bytes] = None
    key_length: int = 0
    qber: float = 0.0  # Quantum Bit Error Rate
    sifted_key_length: int = 0
    error: Optional[str] = None
    metadata: Optional[dict] = None


class BB84Simulator:
    """
    BB84 Quantum Key Distribution protocol simulator.

    Implements the BB84 protocol with:
    - Random bit and basis generation
    - Quantum state preparation and measurement
    - Basis reconciliation
    - Error estimation (QBER)
    - Key sifting
    """

    RECTILINEAR_BASIS = 0  # |0⟩, |1⟩
    DIAGONAL_BASIS = 1     # |+⟩, |-⟩

    def __init__(self, n_qubits: int = 256, error_correction_threshold: float = 0.11):
        """
        Initialize BB84 simulator.

        Args:
            n_qubits: Number of qubits to transmit
            error_correction_threshold: Maximum acceptable QBER (typically 11%)
        """
        self.n_qubits = n_qubits
        self.error_threshold = error_correction_threshold
        self.simulator = AerSimulator()
        logger.info(f"Initialized BB84 simulator with {n_qubits} qubits")

    def generate_random_bits(self, length: int) -> List[int]:
        """Generate cryptographically secure random bits"""
        return [secrets.randbits(1) for _ in range(length)]

    def generate_random_bases(self, length: int) -> List[int]:
        """Generate random measurement bases (0=rectilinear, 1=diagonal)"""
        return [secrets.randbits(1) for _ in range(length)]

    def prepare_qubit_state(self, bit: int, basis: int) -> QuantumCircuit:
        """
        Prepare a qubit in the specified state and basis.

        Args:
            bit: 0 or 1
            basis: 0 (rectilinear) or 1 (diagonal)

        Returns:
            QuantumCircuit with prepared state
        """
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qr, cr)

        if bit == 1:
            qc.x(qr[0])  # Apply X gate for |1⟩

        if basis == self.DIAGONAL_BASIS:
            qc.h(qr[0])  # Apply Hadamard for diagonal basis

        return qc

    def measure_qubit(self, qc: QuantumCircuit, basis: int) -> QuantumCircuit:
        """
        Measure qubit in the specified basis.

        Args:
            qc: Quantum circuit with prepared state
            basis: Measurement basis

        Returns:
            QuantumCircuit with measurement
        """
        if basis == self.DIAGONAL_BASIS:
            qc.h(0)  # Rotate to diagonal basis before measurement

        qc.measure(0, 0)
        return qc

    def simulate_transmission(self, alice_bits: List[int], alice_bases: List[int],
                             bob_bases: List[int], noise_level: float = 0.0) -> List[int]:
        """
        Simulate quantum transmission from Alice to Bob.

        Args:
            alice_bits: Alice's random bits
            alice_bases: Alice's random bases
            bob_bases: Bob's random measurement bases
            noise_level: Simulated channel noise (0.0 to 1.0)

        Returns:
            Bob's measurement results
        """
        bob_results = []

        for bit, alice_basis, bob_basis in zip(alice_bits, alice_bases, bob_bases):
            # Prepare qubit
            qc = self.prepare_qubit_state(bit, alice_basis)

            # Add noise if specified
            if noise_level > 0 and np.random.random() < noise_level:
                qc.x(0)  # Bit flip error

            # Measure qubit
            qc = self.measure_qubit(qc, bob_basis)

            # Simulate
            job = self.simulator.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            measured_bit = int(list(counts.keys())[0])

            bob_results.append(measured_bit)

        return bob_results

    def sift_key(self, alice_bits: List[int], alice_bases: List[int],
                 bob_results: List[int], bob_bases: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform basis reconciliation and key sifting.

        Returns:
            Tuple of (alice_sifted_key, bob_sifted_key)
        """
        alice_sifted = []
        bob_sifted = []

        for a_bit, a_basis, b_bit, b_basis in zip(alice_bits, alice_bases, bob_results, bob_bases):
            if a_basis == b_basis:  # Matching bases
                alice_sifted.append(a_bit)
                bob_sifted.append(b_bit)

        logger.debug(f"Sifted key length: {len(alice_sifted)} from {len(alice_bits)} qubits")
        return alice_sifted, bob_sifted

    def estimate_qber(self, alice_key: List[int], bob_key: List[int],
                     sample_size: Optional[int] = None) -> float:
        """
        Estimate Quantum Bit Error Rate (QBER) by comparing subset of keys.

        Args:
            alice_key: Alice's sifted key
            bob_key: Bob's sifted key
            sample_size: Number of bits to compare (default: 20% of key)

        Returns:
            QBER as a fraction (0.0 to 1.0)
        """
        if len(alice_key) != len(bob_key):
            raise ValueError("Key lengths must match for QBER estimation")

        if len(alice_key) == 0:
            return 1.0

        if sample_size is None:
            sample_size = max(1, len(alice_key) // 5)  # 20% sample

        sample_size = min(sample_size, len(alice_key))

        # Random sampling
        indices = np.random.choice(len(alice_key), size=sample_size, replace=False)

        errors = sum(1 for i in indices if alice_key[i] != bob_key[i])
        qber = errors / sample_size

        logger.info(f"QBER: {qber:.4f} ({errors}/{sample_size} errors)")
        return qber

    def execute_protocol(self, noise_level: float = 0.0) -> BB84Result:
        """
        Execute full BB84 protocol.

        Args:
            noise_level: Simulated channel noise (0.0 to 1.0)

        Returns:
            BB84Result with final key and metadata
        """
        try:
            logger.info(f"Starting BB84 protocol with {self.n_qubits} qubits, noise={noise_level}")

            # Step 1: Alice generates random bits and bases
            alice_bits = self.generate_random_bits(self.n_qubits)
            alice_bases = self.generate_random_bases(self.n_qubits)

            # Step 2: Bob generates random measurement bases
            bob_bases = self.generate_random_bases(self.n_qubits)

            # Step 3: Quantum transmission and measurement
            bob_results = self.simulate_transmission(alice_bits, alice_bases, bob_bases, noise_level)

            # Step 4: Basis reconciliation and key sifting
            alice_sifted, bob_sifted = self.sift_key(alice_bits, alice_bases, bob_results, bob_bases)

            if len(alice_sifted) == 0:
                return BB84Result(success=False, error="No matching bases found")

            # Step 5: Error estimation
            qber = self.estimate_qber(alice_sifted, bob_sifted)

            # Step 6: Check QBER threshold
            if qber > self.error_threshold:
                logger.warning(f"QBER {qber:.4f} exceeds threshold {self.error_threshold}")
                return BB84Result(
                    success=False,
                    qber=qber,
                    sifted_key_length=len(alice_sifted),
                    error=f"QBER too high: {qber:.4f}"
                )

            # Step 7: Convert to bytes (use Bob's key as final)
            # In practice, error correction would be applied here
            key_bits = bob_sifted
            key_bytes = bytes(
                int(''.join(map(str, key_bits[i:i+8])), 2)
                for i in range(0, len(key_bits) - len(key_bits) % 8, 8)
            )

            result = BB84Result(
                success=True,
                raw_key=key_bytes,
                key_length=len(key_bytes),
                qber=qber,
                sifted_key_length=len(alice_sifted),
                metadata={
                    "n_qubits_sent": self.n_qubits,
                    "sifting_efficiency": len(alice_sifted) / self.n_qubits,
                    "noise_level": noise_level
                }
            )

            logger.info(f"BB84 protocol completed successfully: {len(key_bytes)} bytes, QBER={qber:.4f}")
            return result

        except Exception as e:
            logger.error(f"BB84 protocol failed: {e}")
            return BB84Result(success=False, error=str(e))

    def get_statistics(self) -> dict:
        """Get protocol statistics and parameters"""
        return {
            "n_qubits": self.n_qubits,
            "error_threshold": self.error_threshold,
            "expected_sifted_length": self.n_qubits // 2,
            "expected_key_bytes": self.n_qubits // 16
        }
