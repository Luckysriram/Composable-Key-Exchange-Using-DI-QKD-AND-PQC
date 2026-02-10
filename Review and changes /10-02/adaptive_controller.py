"""
Layer 7: Adaptive Controller

Implements state machine and health monitoring for CuKEM system.
Provides adaptive mode switching, failure detection, and recovery.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from enum import Enum
from datetime import datetime, timedelta
import logging
from transitions import Machine
from cukem import CuKEM, CuKEMConfig, CuKEMMode, CuKEMResult

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """CuKEM system states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    HYBRID_ACTIVE = "hybrid_active"
    PQC_ONLY = "pqc_only"
    QKD_ONLY = "qkd_only"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    pqc_success_rate: float = 1.0
    qkd_success_rate: float = 1.0
    avg_qber: float = 0.0
    avg_chsh_value: float = 0.0
    avg_entropy: float = 0.0
    total_exchanges: int = 0
    failed_exchanges: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive controller"""
    health_check_interval: int = 60  # seconds
    failure_threshold: int = 3
    recovery_threshold: int = 2
    min_success_rate: float = 0.8
    max_qber: float = 0.11
    min_chsh_value: float = 2.2
    min_entropy: float = 0.7
    auto_fallback: bool = True
    auto_recovery: bool = True


@dataclass
class ControllerEvent:
    """Event record for state transitions"""
    timestamp: datetime
    event_type: str
    from_state: SystemState
    to_state: SystemState
    reason: str
    metrics: Optional[HealthMetrics] = None


class AdaptiveController:
    """
    Adaptive controller for CuKEM system.

    Features:
    - State machine for system modes
    - Health monitoring and metrics
    - Automatic fallback on failures
    - Recovery and healing
    - Performance tracking
    """

    states = [state.value for state in SystemState]

    transitions = [
        # Initialization
        {'trigger': 'initialize', 'source': 'idle', 'dest': 'initializing'},
        {'trigger': 'init_complete', 'source': 'initializing', 'dest': 'hybrid_active'},

        # Normal operation
        {'trigger': 'switch_to_hybrid', 'source': ['pqc_only', 'qkd_only', 'degraded'],
         'dest': 'hybrid_active'},
        {'trigger': 'switch_to_pqc', 'source': ['hybrid_active', 'qkd_only', 'degraded'],
         'dest': 'pqc_only'},
        {'trigger': 'switch_to_qkd', 'source': ['hybrid_active', 'pqc_only', 'degraded'],
         'dest': 'qkd_only'},

        # Degradation
        {'trigger': 'degrade', 'source': ['hybrid_active', 'pqc_only', 'qkd_only'],
         'dest': 'degraded'},

        # Failure and recovery
        {'trigger': 'fail', 'source': '*', 'dest': 'failed'},
        {'trigger': 'start_recovery', 'source': ['failed', 'degraded'], 'dest': 'recovering'},
        {'trigger': 'recovery_success', 'source': 'recovering', 'dest': 'hybrid_active'},
        {'trigger': 'recovery_partial', 'source': 'recovering', 'dest': 'degraded'},

        # Reset
        {'trigger': 'reset', 'source': '*', 'dest': 'idle'}
    ]

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """
        Initialize adaptive controller.

        Args:
            config: Controller configuration
        """
        self.config = config or AdaptiveConfig()
        self.cukem: Optional[CuKEM] = None
        self.current_metrics = HealthMetrics()
        self.metrics_history: List[HealthMetrics] = []
        self.event_log: List[ControllerEvent] = []
        self.consecutive_failures = 0
        self.consecutive_successes = 0

        # Initialize state machine
        self.machine = Machine(
            model=self,
            states=self.states,
            transitions=self.transitions,
            initial=SystemState.IDLE.value,
            after_state_change='_on_state_change'
        )

        logger.info("Initialized adaptive controller")

    def _on_state_change(self):
        """Callback for state changes"""
        logger.info(f"State changed to: {self.state}")

    def initialize_system(self, cukem_config: Optional[CuKEMConfig] = None) -> bool:
        """
        Initialize CuKEM system.

        Args:
            cukem_config: CuKEM configuration

        Returns:
            True if successful
        """
        try:
            self.initialize()
            logger.info("Initializing CuKEM system")

            self.cukem = CuKEM(cukem_config or CuKEMConfig())

            self._log_event(
                event_type="initialization",
                from_state=SystemState.IDLE,
                to_state=SystemState.INITIALIZING,
                reason="System startup"
            )

            self.init_complete()
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.fail()
            return False

    def perform_exchange(self, role: str = "initiator",
                        noise_level: float = 0.0) -> CuKEMResult:
        """
        Perform key exchange with monitoring.

        Args:
            role: "initiator" or "responder"
            noise_level: Quantum channel noise

        Returns:
            CuKEMResult
        """
        if not self.cukem:
            raise RuntimeError("System not initialized")

        start_time = datetime.now()

        try:
            # Perform exchange based on current state
            if self.state == SystemState.HYBRID_ACTIVE.value:
                self.cukem.config.mode = CuKEMMode.HYBRID
            elif self.state == SystemState.PQC_ONLY.value:
                self.cukem.config.mode = CuKEMMode.PQC_ONLY
            elif self.state == SystemState.QKD_ONLY.value:
                self.cukem.config.mode = CuKEMMode.QKD_ONLY

            # Execute exchange
            if role == "initiator":
                keypair = self.cukem.generate_keypair()
                result = self.cukem.initiate_exchange(
                    responder_public_key=keypair.public_key,
                    noise_level=noise_level
                )
            else:
                keypair = self.cukem.generate_keypair()
                temp_cukem = CuKEM(self.cukem.config)
                init_result = temp_cukem.initiate_exchange(
                    responder_public_key=keypair.public_key,
                    noise_level=noise_level
                )
                result = self.cukem.respond_exchange(
                    keypair=keypair,
                    ciphertext=init_result.pqc_result.ciphertext,
                    noise_level=noise_level
                )

            # Calculate latency
            latency = (datetime.now() - start_time).total_seconds() * 1000

            # Update metrics
            self._update_metrics(result, latency)

            # Check health
            self._check_health(result)

            return result

        except Exception as e:
            logger.error(f"Exchange failed: {e}")
            self.consecutive_failures += 1
            self._handle_failure()
            raise

    def _update_metrics(self, result: CuKEMResult, latency_ms: float):
        """Update health metrics based on exchange result"""
        self.current_metrics.total_exchanges += 1

        if result.success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.current_metrics.failed_exchanges += 1

        # Update success rates
        if self.current_metrics.total_exchanges > 0:
            success_count = (self.current_metrics.total_exchanges -
                           self.current_metrics.failed_exchanges)
            self.current_metrics.pqc_success_rate = (
                success_count / self.current_metrics.total_exchanges
            )

        # Update quantum metrics if available
        if result.bb84_result:
            self.current_metrics.avg_qber = (
                (self.current_metrics.avg_qber * 0.9) +
                (result.bb84_result.qber * 0.1)
            )

        if result.chsh_result:
            self.current_metrics.avg_chsh_value = (
                (self.current_metrics.avg_chsh_value * 0.9) +
                (abs(result.chsh_result.chsh_value) * 0.1)
            )

        if result.entropy_result:
            self.current_metrics.avg_entropy = (
                (self.current_metrics.avg_entropy * 0.9) +
                (result.entropy_result.min_entropy * 0.1)
            )

        # Update latency
        self.current_metrics.avg_latency_ms = (
            (self.current_metrics.avg_latency_ms * 0.9) + (latency_ms * 0.1)
        )

        self.current_metrics.timestamp = datetime.now()

    def _check_health(self, result: CuKEMResult):
        """Check system health and trigger state changes if needed"""

        if not self.config.auto_fallback:
            return

        # Check for critical failures
        if self.consecutive_failures >= self.config.failure_threshold:
            logger.warning(f"Critical: {self.consecutive_failures} consecutive failures")
            self._handle_failure()
            return

        # Check QBER
        if self.current_metrics.avg_qber > self.config.max_qber:
            logger.warning(f"High QBER: {self.current_metrics.avg_qber:.4f}")
            if self.state == SystemState.HYBRID_ACTIVE.value:
                self._fallback_to_pqc("High QBER")

        # Check entropy
        if self.current_metrics.avg_entropy < self.config.min_entropy:
            logger.warning(f"Low entropy: {self.current_metrics.avg_entropy:.4f}")
            if self.state == SystemState.HYBRID_ACTIVE.value:
                self._fallback_to_pqc("Low entropy")

        # Check for recovery
        if self.config.auto_recovery:
            if self.consecutive_successes >= self.config.recovery_threshold:
                self._attempt_recovery()

    def _handle_failure(self):
        """Handle system failure"""
        if self.state == SystemState.HYBRID_ACTIVE.value:
            self._fallback_to_pqc("Multiple failures")
        elif self.state in [SystemState.PQC_ONLY.value, SystemState.QKD_ONLY.value]:
            self.degrade()
            self._log_event(
                event_type="degradation",
                from_state=SystemState[self.state.upper()],
                to_state=SystemState.DEGRADED,
                reason="Consecutive failures"
            )
        elif self.state == SystemState.DEGRADED.value:
            self.fail()
            self._log_event(
                event_type="failure",
                from_state=SystemState.DEGRADED,
                to_state=SystemState.FAILED,
                reason="System failure"
            )

    def _fallback_to_pqc(self, reason: str):
        """Fallback to PQC-only mode"""
        logger.warning(f"Falling back to PQC-only: {reason}")
        previous_state = SystemState[self.state.upper()]
        self.switch_to_pqc()
        self._log_event(
            event_type="fallback",
            from_state=previous_state,
            to_state=SystemState.PQC_ONLY,
            reason=reason
        )

    def _attempt_recovery(self):
        """Attempt to recover to hybrid mode"""
        if self.state in [SystemState.PQC_ONLY.value, SystemState.DEGRADED.value]:
            logger.info("Attempting recovery to hybrid mode")
            self.start_recovery()

            # Test quantum channel
            try:
                test_config = CuKEMConfig(mode=CuKEMMode.QKD_ONLY)
                test_cukem = CuKEM(test_config)
                keypair = test_cukem.generate_keypair()
                test_result = test_cukem.initiate_exchange(keypair.public_key)

                if test_result.success:
                    self.recovery_success()
                    self._log_event(
                        event_type="recovery",
                        from_state=SystemState.RECOVERING,
                        to_state=SystemState.HYBRID_ACTIVE,
                        reason="Quantum channel restored"
                    )
                else:
                    self.recovery_partial()

            except Exception as e:
                logger.error(f"Recovery failed: {e}")
                self.recovery_partial()

    def _log_event(self, event_type: str, from_state: SystemState,
                   to_state: SystemState, reason: str):
        """Log state transition event"""
        event = ControllerEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            metrics=HealthMetrics(**self.current_metrics.__dict__)
        )
        self.event_log.append(event)
        logger.info(f"Event: {event_type} - {from_state.value} → {to_state.value}: {reason}")

    def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        if self.state == SystemState.FAILED.value:
            return HealthStatus.CRITICAL

        if self.state == SystemState.DEGRADED.value:
            return HealthStatus.WARNING

        if self.current_metrics.pqc_success_rate < self.config.min_success_rate:
            return HealthStatus.WARNING

        if self.current_metrics.avg_qber > self.config.max_qber:
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "state": self.state,
            "health": self.get_health_status().value,
            "metrics": {
                "total_exchanges": self.current_metrics.total_exchanges,
                "failed_exchanges": self.current_metrics.failed_exchanges,
                "pqc_success_rate": self.current_metrics.pqc_success_rate,
                "qkd_success_rate": self.current_metrics.qkd_success_rate,
                "avg_qber": self.current_metrics.avg_qber,
                "avg_chsh_value": self.current_metrics.avg_chsh_value,
                "avg_entropy": self.current_metrics.avg_entropy,
                "avg_latency_ms": self.current_metrics.avg_latency_ms
            },
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "event_count": len(self.event_log)
        }
