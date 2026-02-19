"""
Layer 2: CHSH Bell Test for Quantum Entanglement Verification

Implements the Clauser-Horne-Shimony-Holt (CHSH) inequality test
to verify quantum entanglement and detect eavesdropping.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import logging

logger = logging.getLogger(__name__)


@dataclass
class CHSHResult:
    """Result container for CHSH test"""
    success: bool
    chsh_value: float
    violation: bool  # True if quantum correlation detected
    correlations: dict
    n_shots: int
    error: Optional[str] = None


class CHSHBellTest:
    """
    CHSH inequality test for quantum entanglement verification.

    Classical limit: |S| ≤ 2
    Quantum maximum (Tsirelson bound): |S| ≤ 2√2 ≈ 2.828

    S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    where E(x,y) is the correlation between measurements x and y.
    """

    CLASSICAL_BOUND = 2.0
    QUANTUM_BOUND = 2.828  # 2 * sqrt(2)
    VIOLATION_THRESHOLD = 2.2  # Minimum for quantum behavior detection

    def __init__(self, n_shots: int = 8192):
        """
        Initialize CHSH test.

        Args:
            n_shots: Number of measurement shots per configuration
        """
        self.n_shots = n_shots
        self.simulator = AerSimulator()
        logger.info(f"Initialized CHSH Bell test with {n_shots} shots")

    def create_bell_state(self) -> QuantumCircuit:
        """
        Create a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2

        Returns:
            QuantumCircuit preparing Bell state
        """
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(qr, cr)

        # Create entanglement
        qc.h(qr[0])  # Hadamard on first qubit
        qc.cx(qr[0], qr[1])  # CNOT to entangle

        return qc

    def measure_in_basis(self, qc: QuantumCircuit, alice_angle: float,
                        bob_angle: float) -> QuantumCircuit:
        """
        Measure qubits in specified bases (angles).

        Args:
            qc: Quantum circuit with Bell state
            alice_angle: Alice's measurement angle (radians)
            bob_angle: Bob's measurement angle (radians)

        Returns:
            QuantumCircuit with measurements
        """
        # Rotate measurement bases
        qc.ry(2 * alice_angle, 0)  # Alice's basis
        qc.ry(2 * bob_angle, 1)    # Bob's basis

        # Measure both qubits
        qc.measure([0, 1], [0, 1])

        return qc

    def calculate_correlation(self, counts: dict) -> float:
        """
        Calculate correlation E(a,b) = P(same) - P(different)

        Args:
            counts: Measurement outcomes

        Returns:
            Correlation value between -1 and 1
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0

        same = counts.get('00', 0) + counts.get('11', 0)
        different = counts.get('01', 0) + counts.get('10', 0)

        correlation = (same - different) / total
        return correlation

    def run_measurement(self, alice_angle: float, bob_angle: float) -> float:
        """
        Run measurement with specified angles and calculate correlation.

        Args:
            alice_angle: Alice's measurement angle
            bob_angle: Bob's measurement angle

        Returns:
            Correlation value
        """
        qc = self.create_bell_state()
        qc = self.measure_in_basis(qc, alice_angle, bob_angle)

        # Run simulation
        job = self.simulator.run(qc, shots=self.n_shots)
        result = job.result()
        counts = result.get_counts()

        correlation = self.calculate_correlation(counts)
        return correlation

    def execute_chsh_test(self) -> CHSHResult:
        """
        Execute full CHSH test with optimal angles.

        CHSH angles (in radians):
        - a  = 0
        - a' = π/2
        - b  = π/4
        - b' = -π/4

        Returns:
            CHSHResult with test outcome
        """
        try:
            logger.info("Starting CHSH Bell test")

            # Define measurement angles
            a = 0
            a_prime = np.pi / 2
            b = np.pi / 4
            b_prime = -np.pi / 4

            # Run four measurement configurations
            logger.debug("Measuring E(a, b)")
            E_ab = self.run_measurement(a, b)

            logger.debug("Measuring E(a, b')")
            E_ab_prime = self.run_measurement(a, b_prime)

            logger.debug("Measuring E(a', b)")
            E_a_prime_b = self.run_measurement(a_prime, b)

            logger.debug("Measuring E(a', b')")
            E_a_prime_b_prime = self.run_measurement(a_prime, b_prime)

            # Calculate CHSH value: S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
            S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

            # Check for violation
            violation = abs(S) > self.VIOLATION_THRESHOLD
            exceeds_classical = abs(S) > self.CLASSICAL_BOUND

            correlations = {
                'E(a,b)': E_ab,
                'E(a,b_prime)': E_ab_prime,
                'E(a_prime,b)': E_a_prime_b,
                'E(a_prime,b_prime)': E_a_prime_b_prime
            }

            result = CHSHResult(
                success=True,
                chsh_value=S,
                violation=violation,
                correlations=correlations,
                n_shots=self.n_shots
            )

            if violation:
                logger.info(f"CHSH test PASSED: S = {S:.4f} (violation detected)")
            else:
                logger.warning(f"CHSH test inconclusive: S = {S:.4f}")

            return result

        except Exception as e:
            logger.error(f"CHSH test failed: {e}")
            return CHSHResult(
                success=False,
                chsh_value=0.0,
                violation=False,
                correlations={},
                n_shots=self.n_shots,
                error=str(e)
            )

    def verify_entanglement_quality(self, min_violation: float = 2.5) -> bool:
        """
        Run multiple CHSH tests to verify entanglement quality.

        Args:
            min_violation: Minimum CHSH value for acceptable quality

        Returns:
            True if entanglement quality is sufficient
        """
        result = self.execute_chsh_test()

        if not result.success:
            logger.error("CHSH test failed")
            return False

        if abs(result.chsh_value) >= min_violation:
            logger.info(f"Entanglement quality verified: |S| = {abs(result.chsh_value):.4f}")
            return True
        else:
            logger.warning(f"Poor entanglement quality: |S| = {abs(result.chsh_value):.4f}")
            return False

    def get_theoretical_values(self) -> dict:
        """Get theoretical CHSH values"""
        return {
            "classical_bound": self.CLASSICAL_BOUND,
            "quantum_bound": self.QUANTUM_BOUND,
            "violation_threshold": self.VIOLATION_THRESHOLD,
            "optimal_quantum_value": self.QUANTUM_BOUND
        }
