"""
Layer 3: NIST 800-90B Entropy Estimation

Implements entropy estimation methods from NIST SP 800-90B
for validating quantum random number quality.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from collections import Counter
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class EntropyResult:
    """Result container for entropy estimation"""
    success: bool
    min_entropy: float
    sufficient: bool  # True if entropy meets threshold
    estimations: dict
    data_length: int
    error: Optional[str] = None


class EntropyEstimator:
    """
    NIST SP 800-90B entropy estimation.

    Implements multiple entropy estimation methods:
    - Most Common Value (MCV)
    - Collision Test
    - Markov Test
    - Compression Test
    - Shannon Entropy (informational)
    """

    MIN_SAMPLES = 1000
    MIN_ENTROPY_THRESHOLD = 0.8  # Require 80% of theoretical maximum

    def __init__(self, min_entropy_per_bit: float = 0.8):
        """
        Initialize entropy estimator.

        Args:
            min_entropy_per_bit: Minimum required entropy per bit (0.0 to 1.0)
        """
        self.min_required = min_entropy_per_bit
        logger.info(f"Initialized entropy estimator (threshold: {min_entropy_per_bit})")

    def bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to bit list"""
        bits = []
        for byte in data:
            bits.extend([int(b) for b in format(byte, '08b')])
        return bits

    def shannon_entropy(self, data: List[int]) -> float:
        """
        Calculate Shannon entropy H(X) = -Σ p(x) log₂ p(x)

        Args:
            data: Input data sequence

        Returns:
            Shannon entropy in bits per symbol
        """
        if len(data) == 0:
            return 0.0

        counts = Counter(data)
        total = len(data)
        entropy = 0.0

        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def most_common_value_estimate(self, data: List[int]) -> float:
        """
        Most Common Value (MCV) estimate.

        Min-entropy = -log₂(p_max)
        where p_max is the probability of the most common value.

        Args:
            data: Input data sequence

        Returns:
            Min-entropy estimate per sample
        """
        if len(data) == 0:
            return 0.0

        counts = Counter(data)
        most_common_count = counts.most_common(1)[0][1]
        p_max = most_common_count / len(data)

        min_entropy = -math.log2(p_max) if p_max > 0 else 0.0
        logger.debug(f"MCV estimate: {min_entropy:.4f} (p_max={p_max:.4f})")
        return min_entropy

    def collision_estimate(self, data: List[int]) -> float:
        """
        Collision test estimate.

        Estimates entropy based on the mean time between collisions
        (repeated values).

        Args:
            data: Input data sequence

        Returns:
            Min-entropy estimate per sample
        """
        if len(data) < 2:
            return 0.0

        seen = set()
        collisions = 0
        positions = []

        for i, value in enumerate(data):
            if value in seen:
                collisions += 1
                positions.append(i)
            seen.add(value)

        if collisions == 0:
            # No collisions - very high entropy
            return math.log2(len(data))

        mean_collision_time = len(data) / collisions if collisions > 0 else len(data)

        # Estimate based on collision time
        # X ~ sqrt(2^H * pi/2) for collision time
        if mean_collision_time > 0:
            H = 2 * math.log2(mean_collision_time) - math.log2(math.pi / 2)
            min_entropy = max(0, H / len(data))
        else:
            min_entropy = 0.0

        logger.debug(f"Collision estimate: {min_entropy:.4f} ({collisions} collisions)")
        return min_entropy

    def markov_estimate(self, data: List[int]) -> float:
        """
        Markov test estimate.

        Estimates entropy considering first-order dependencies
        between consecutive symbols.

        Args:
            data: Input data sequence

        Returns:
            Min-entropy estimate per sample
        """
        if len(data) < 2:
            return 0.0

        # Build transition matrix
        unique_vals = sorted(set(data))
        val_to_idx = {v: i for i, v in enumerate(unique_vals)}
        n_states = len(unique_vals)

        transitions = np.zeros((n_states, n_states))

        for i in range(len(data) - 1):
            current = val_to_idx[data[i]]
            next_val = val_to_idx[data[i + 1]]
            transitions[current][next_val] += 1

        # Calculate conditional entropies
        total_entropy = 0.0
        total_weight = 0

        for i in range(n_states):
            row_sum = transitions[i].sum()
            if row_sum > 0:
                probs = transitions[i] / row_sum
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                total_entropy += entropy * row_sum
                total_weight += row_sum

        markov_entropy = total_entropy / total_weight if total_weight > 0 else 0.0
        logger.debug(f"Markov estimate: {markov_entropy:.4f}")
        return markov_entropy

    def compression_estimate(self, data: List[int]) -> float:
        """
        Simple compression-based estimate.

        Uses run-length encoding to estimate compressibility.

        Args:
            data: Input data sequence

        Returns:
            Entropy estimate based on compressibility
        """
        if len(data) == 0:
            return 0.0

        # Simple run-length encoding
        runs = []
        current_val = data[0]
        current_len = 1

        for val in data[1:]:
            if val == current_val:
                current_len += 1
            else:
                runs.append((current_val, current_len))
                current_val = val
                current_len = 1
        runs.append((current_val, current_len))

        # Estimate entropy from compression ratio
        compressed_size = len(runs) * 2  # (value, length) pairs
        original_size = len(data)
        compression_ratio = compressed_size / original_size

        # Higher compression ratio = lower entropy
        entropy = compression_ratio * math.log2(len(set(data)))
        logger.debug(f"Compression estimate: {entropy:.4f} (ratio={compression_ratio:.4f})")
        return entropy

    def estimate_entropy(self, data: bytes, symbol_size: int = 1) -> EntropyResult:
        """
        Perform comprehensive entropy estimation.

        Args:
            data: Input data to analyze
            symbol_size: Symbol size in bits (1 for binary)

        Returns:
            EntropyResult with min-entropy and detailed estimates
        """
        try:
            if len(data) < self.MIN_SAMPLES // 8:
                return EntropyResult(
                    success=False,
                    min_entropy=0.0,
                    sufficient=False,
                    estimations={},
                    data_length=len(data),
                    error=f"Insufficient data: need at least {self.MIN_SAMPLES // 8} bytes"
                )

            # Convert to bits for analysis
            bits = self.bytes_to_bits(data)
            logger.info(f"Analyzing {len(bits)} bits from {len(data)} bytes")

            # Run all estimation methods
            shannon = self.shannon_entropy(bits)
            mcv = self.most_common_value_estimate(bits)
            collision = self.collision_estimate(bits)
            markov = self.markov_estimate(bits)
            compression = self.compression_estimate(bits)

            estimations = {
                'shannon_entropy': shannon,
                'mcv_estimate': mcv,
                'collision_estimate': collision,
                'markov_estimate': markov,
                'compression_estimate': compression
            }

            # Conservative approach: use minimum of all estimates
            min_entropy = min(mcv, collision, markov, compression)

            # Check if entropy is sufficient
            sufficient = min_entropy >= self.min_required

            result = EntropyResult(
                success=True,
                min_entropy=min_entropy,
                sufficient=sufficient,
                estimations=estimations,
                data_length=len(data)
            )

            if sufficient:
                logger.info(f"Entropy sufficient: {min_entropy:.4f} >= {self.min_required:.4f}")
            else:
                logger.warning(f"Entropy insufficient: {min_entropy:.4f} < {self.min_required:.4f}")

            return result

        except Exception as e:
            logger.error(f"Entropy estimation failed: {e}")
            return EntropyResult(
                success=False,
                min_entropy=0.0,
                sufficient=False,
                estimations={},
                data_length=len(data),
                error=str(e)
            )

    def get_statistics(self) -> dict:
        """Get estimator configuration"""
        return {
            "min_required_entropy": self.min_required,
            "min_samples": self.MIN_SAMPLES,
            "methods": ["shannon", "mcv", "collision", "markov", "compression"]
        }
