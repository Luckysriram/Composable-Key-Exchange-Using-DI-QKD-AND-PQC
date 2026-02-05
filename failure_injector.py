"""
Layer 8: Failure Injector

Provides controlled failure injection for testing CuKEM resilience
and adaptive behavior under adverse conditions.
"""

from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum
import random
import logging

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures to inject"""
    NETWORK_DELAY = "network_delay"
    PACKET_LOSS = "packet_loss"
    QUANTUM_NOISE = "quantum_noise"
    KEY_CORRUPTION = "key_corruption"
    TIMEOUT = "timeout"
    RANDOM_ERROR = "random_error"


@dataclass
class FailureScenario:
    """Configuration for failure injection scenario"""
    failure_type: FailureType
    probability: float = 0.1  # Probability of injection (0.0-1.0)
    duration: Optional[int] = None  # Duration in seconds (None = indefinite)
    severity: float = 0.5  # Severity level (0.0-1.0)
    target: Optional[str] = None  # Target component


@dataclass
class InjectionResult:
    """Result of failure injection"""
    injected: bool
    failure_type: FailureType
    severity: float
    description: str


class FailureInjector:
    """
    Failure injector for testing CuKEM resilience.

    Supports:
    - Network failures (delay, loss, partition)
    - Quantum channel noise
    - Key corruption
    - Component failures
    - Random errors
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize failure injector.

        Args:
            enabled: Enable/disable failure injection
        """
        self.enabled = enabled
        self.scenarios: list[FailureScenario] = []
        self.injection_count = 0
        logger.info(f"Initialized failure injector (enabled={enabled})")

    def add_scenario(self, scenario: FailureScenario):
        """Add failure scenario"""
        self.scenarios.append(scenario)
        logger.info(f"Added failure scenario: {scenario.failure_type.value} "
                   f"(p={scenario.probability}, severity={scenario.severity})")

    def remove_scenario(self, failure_type: FailureType):
        """Remove failure scenario by type"""
        self.scenarios = [s for s in self.scenarios if s.failure_type != failure_type]
        logger.info(f"Removed scenario: {failure_type.value}")

    def inject_network_delay(self, severity: float) -> InjectionResult:
        """
        Inject network delay.

        Args:
            severity: Delay severity (0.0-1.0)

        Returns:
            InjectionResult
        """
        # Delay in milliseconds: 0ms (0.0) to 1000ms (1.0)
        delay_ms = int(severity * 1000)

        logger.warning(f"INJECTING: Network delay {delay_ms}ms")

        # In practice, this would interact with network simulator
        # For now, just return the result

        return InjectionResult(
            injected=True,
            failure_type=FailureType.NETWORK_DELAY,
            severity=severity,
            description=f"Network delay: {delay_ms}ms"
        )

    def inject_packet_loss(self, severity: float) -> InjectionResult:
        """
        Inject packet loss.

        Args:
            severity: Loss rate (0.0-1.0)

        Returns:
            InjectionResult
        """
        loss_percent = severity * 100

        logger.warning(f"INJECTING: Packet loss {loss_percent:.1f}%")

        return InjectionResult(
            injected=True,
            failure_type=FailureType.PACKET_LOSS,
            severity=severity,
            description=f"Packet loss: {loss_percent:.1f}%"
        )

    def inject_quantum_noise(self, severity: float) -> float:
        """
        Inject quantum channel noise.

        Args:
            severity: Noise level (0.0-1.0)

        Returns:
            Noise level to apply to BB84
        """
        logger.warning(f"INJECTING: Quantum noise {severity:.3f}")
        self.injection_count += 1
        return severity

    def inject_key_corruption(self, key: bytes, severity: float) -> bytes:
        """
        Inject key corruption by flipping bits.

        Args:
            key: Original key
            severity: Corruption severity (0.0-1.0)

        Returns:
            Corrupted key
        """
        if not key:
            return key

        # Calculate number of bits to flip
        total_bits = len(key) * 8
        bits_to_flip = int(total_bits * severity)

        logger.warning(f"INJECTING: Key corruption ({bits_to_flip}/{total_bits} bits)")

        # Convert to bytearray for modification
        corrupted = bytearray(key)

        # Flip random bits
        for _ in range(bits_to_flip):
            byte_idx = random.randint(0, len(corrupted) - 1)
            bit_idx = random.randint(0, 7)
            corrupted[byte_idx] ^= (1 << bit_idx)

        self.injection_count += 1
        return bytes(corrupted)

    def should_inject(self, failure_type: FailureType) -> Optional[FailureScenario]:
        """
        Check if failure should be injected.

        Args:
            failure_type: Type of failure to check

        Returns:
            FailureScenario if should inject, None otherwise
        """
        if not self.enabled:
            return None

        for scenario in self.scenarios:
            if scenario.failure_type == failure_type:
                if random.random() < scenario.probability:
                    return scenario

        return None

    def inject_if_enabled(self, failure_type: FailureType,
                         data: Optional[any] = None) -> any:
        """
        Conditionally inject failure based on scenario.

        Args:
            failure_type: Type of failure
            data: Data to potentially corrupt (for key corruption)

        Returns:
            Modified data or injection result
        """
        scenario = self.should_inject(failure_type)

        if not scenario:
            return data

        if failure_type == FailureType.QUANTUM_NOISE:
            return self.inject_quantum_noise(scenario.severity)

        elif failure_type == FailureType.KEY_CORRUPTION and data:
            return self.inject_key_corruption(data, scenario.severity)

        elif failure_type == FailureType.NETWORK_DELAY:
            return self.inject_network_delay(scenario.severity)

        elif failure_type == FailureType.PACKET_LOSS:
            return self.inject_packet_loss(scenario.severity)

        return data

    def enable(self):
        """Enable failure injection"""
        self.enabled = True
        logger.info("Failure injection enabled")

    def disable(self):
        """Disable failure injection"""
        self.enabled = False
        logger.info("Failure injection disabled")

    def clear_scenarios(self):
        """Clear all failure scenarios"""
        self.scenarios.clear()
        logger.info("Cleared all failure scenarios")

    def get_statistics(self) -> dict:
        """Get injector statistics"""
        return {
            "enabled": self.enabled,
            "scenarios": len(self.scenarios),
            "injection_count": self.injection_count,
            "active_scenarios": [
                {
                    "type": s.failure_type.value,
                    "probability": s.probability,
                    "severity": s.severity
                }
                for s in self.scenarios
            ]
        }


# Predefined failure scenarios
def create_mild_noise_scenario() -> FailureScenario:
    """Create mild quantum noise scenario"""
    return FailureScenario(
        failure_type=FailureType.QUANTUM_NOISE,
        probability=0.3,
        severity=0.05  # 5% noise
    )


def create_severe_noise_scenario() -> FailureScenario:
    """Create severe quantum noise scenario"""
    return FailureScenario(
        failure_type=FailureType.QUANTUM_NOISE,
        probability=0.5,
        severity=0.15  # 15% noise
    )


def create_network_instability_scenario() -> FailureScenario:
    """Create network instability scenario"""
    return FailureScenario(
        failure_type=FailureType.PACKET_LOSS,
        probability=0.2,
        severity=0.05  # 5% packet loss
    )
