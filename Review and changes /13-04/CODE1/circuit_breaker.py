"""
Layer 7: Circuit Breaker Pattern

Implements circuit breaker pattern for resilient CuKEM operations.
Prevents cascading failures and provides graceful degradation.
"""

from dataclasses import dataclass
from typing import Callable, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    half_open_max_calls: int = 3


@dataclass
class CircuitStats:
    """Circuit breaker statistics"""
    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: int = 0


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting CuKEM operations.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests rejected immediately
    - HALF_OPEN: Testing if system recovered, limited requests allowed

    Transitions:
    - CLOSED → OPEN: After threshold failures
    - OPEN → HALF_OPEN: After timeout period
    - HALF_OPEN → CLOSED: After threshold successes
    - HALF_OPEN → OPEN: On any failure
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None,
                 name: str = "default"):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
            name: Circuit breaker name for logging
        """
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats(state=self.state)
        self.half_open_calls = 0
        self.opened_at: Optional[datetime] = None

        logger.info(f"Initialized circuit breaker: {name}")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        self.stats.total_calls += 1

        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                logger.warning(f"Circuit {self.name} is OPEN, rejecting call")
                raise CircuitBreakerOpen(f"Circuit breaker {self.name} is open")

        # Check half-open call limit
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                logger.warning(f"Circuit {self.name} half-open call limit reached")
                raise CircuitBreakerOpen(f"Circuit breaker {self.name} half-open limit")
            self.half_open_calls += 1

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call"""
        self.stats.success_count += 1
        self.stats.total_successes += 1
        self.stats.last_success_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            if self.stats.success_count >= self.config.success_threshold:
                self._transition_to_closed()

    def _on_failure(self):
        """Handle failed call"""
        self.stats.failure_count += 1
        self.stats.total_failures += 1
        self.stats.last_failure_time = datetime.now()

        if self.state == CircuitState.CLOSED:
            if self.stats.failure_count >= self.config.failure_threshold:
                self._transition_to_open()

        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state reopens circuit
            self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.opened_at is None:
            return True

        elapsed = datetime.now() - self.opened_at
        return elapsed >= timedelta(seconds=self.config.timeout_seconds)

    def _transition_to_open(self):
        """Transition to OPEN state"""
        logger.warning(f"Circuit {self.name}: OPEN (failures: {self.stats.failure_count})")
        self.state = CircuitState.OPEN
        self.stats.state = CircuitState.OPEN
        self.stats.failure_count = 0
        self.stats.success_count = 0
        self.opened_at = datetime.now()
        self.stats.state_changes += 1

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        logger.info(f"Circuit {self.name}: HALF_OPEN (testing recovery)")
        self.state = CircuitState.HALF_OPEN
        self.stats.state = CircuitState.HALF_OPEN
        self.stats.failure_count = 0
        self.stats.success_count = 0
        self.half_open_calls = 0
        self.stats.state_changes += 1

    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        logger.info(f"Circuit {self.name}: CLOSED (recovered)")
        self.state = CircuitState.CLOSED
        self.stats.state = CircuitState.CLOSED
        self.stats.failure_count = 0
        self.stats.success_count = 0
        self.opened_at = None
        self.stats.state_changes += 1

    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        logger.info(f"Circuit {self.name}: Manual reset")
        self._transition_to_closed()

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state

    def get_statistics(self) -> dict:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "stats": {
                "total_calls": self.stats.total_calls,
                "total_successes": self.stats.total_successes,
                "total_failures": self.stats.total_failures,
                "failure_count": self.stats.failure_count,
                "success_count": self.stats.success_count,
                "state_changes": self.stats.state_changes,
                "last_failure": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
                "last_success": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds
            }
        }
