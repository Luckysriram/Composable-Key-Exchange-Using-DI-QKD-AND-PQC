"""
Utility Functions

Common utility functions for CuKEM system.
"""

import hashlib
import secrets
import base64
import time
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def generate_random_bytes(length: int) -> bytes:
    """
    Generate cryptographically secure random bytes.

    Args:
        length: Number of bytes to generate

    Returns:
        Random bytes
    """
    return secrets.token_bytes(length)


def bytes_to_hex(data: bytes) -> str:
    """Convert bytes to hexadecimal string"""
    return data.hex()


def hex_to_bytes(hex_string: str) -> bytes:
    """Convert hexadecimal string to bytes"""
    return bytes.fromhex(hex_string)


def bytes_to_base64(data: bytes) -> str:
    """Convert bytes to base64 string"""
    return base64.b64encode(data).decode('utf-8')


def base64_to_bytes(b64_string: str) -> bytes:
    """Convert base64 string to bytes"""
    return base64.b64decode(b64_string.encode('utf-8'))


def compute_sha256(data: bytes) -> bytes:
    """Compute SHA-256 hash of data"""
    return hashlib.sha256(data).digest()


def compute_fingerprint(data: bytes, length: int = 16) -> str:
    """
    Compute short fingerprint of data.

    Args:
        data: Input data
        length: Fingerprint length in characters

    Returns:
        Hexadecimal fingerprint
    """
    hash_value = hashlib.sha256(data).hexdigest()
    return hash_value[:length]


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """
    XOR two byte sequences.

    Args:
        a: First byte sequence
        b: Second byte sequence

    Returns:
        XOR result (length = min(len(a), len(b)))
    """
    return bytes(x ^ y for x, y in zip(a, b))


def hamming_distance(a: bytes, b: bytes) -> int:
    """
    Calculate Hamming distance between two byte sequences.

    Args:
        a: First byte sequence
        b: Second byte sequence

    Returns:
        Hamming distance (number of differing bits)
    """
    if len(a) != len(b):
        raise ValueError("Byte sequences must have same length")

    xor_result = xor_bytes(a, b)
    return sum(bin(byte).count('1') for byte in xor_result)


def timing_decorator(func):
    """
    Decorator to measure function execution time.

    Usage:
        @timing_decorator
        def my_function():
            ...
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"{func.__name__} took {elapsed_ms:.2f}ms")
        return result
    return wrapper


def format_bytes(num_bytes: int) -> str:
    """
    Format byte count in human-readable form.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 KB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 30m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value on division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default


def exponential_backoff(attempt: int, base_delay: float = 1.0,
                       max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay.

    Args:
        attempt: Attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    # Add jitter
    jitter = secrets.randbelow(int(delay * 100)) / 100
    return delay + jitter


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger.info(f"Logging initialized at {level} level")


def create_summary_report(data: Dict, title: str = "Report") -> str:
    """
    Create formatted summary report.

    Args:
        data: Dictionary of data to report
        title: Report title

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f" {title}")
    lines.append("=" * 60)

    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"\n{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")

    lines.append("=" * 60)
    return "\n".join(lines)


def validate_key_material(key: bytes, min_length: int = 16) -> bool:
    """
    Validate key material.

    Args:
        key: Key bytes to validate
        min_length: Minimum key length

    Returns:
        True if valid
    """
    if not key:
        logger.error("Key is empty")
        return False

    if len(key) < min_length:
        logger.error(f"Key too short: {len(key)} < {min_length}")
        return False

    # Check for all-zero key
    if all(b == 0 for b in key):
        logger.error("Key is all zeros")
        return False

    return True


def rate_limiter(max_calls: int, time_window: int):
    """
    Rate limiter decorator.

    Args:
        max_calls: Maximum calls allowed
        time_window: Time window in seconds

    Usage:
        @rate_limiter(max_calls=10, time_window=60)
        def my_function():
            ...
    """
    calls = []

    def decorator(func):
        def wrapper(*args, **kwargs):
            now = time.time()

            # Remove old calls outside window
            calls[:] = [call_time for call_time in calls
                       if now - call_time < time_window]

            if len(calls) >= max_calls:
                logger.warning(f"Rate limit exceeded for {func.__name__}")
                raise RuntimeError("Rate limit exceeded")

            calls.append(now)
            return func(*args, **kwargs)

        return wrapper
    return decorator
