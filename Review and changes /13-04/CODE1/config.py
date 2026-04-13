"""
Configuration Management

Central configuration management for CuKEM system.
Supports YAML configuration files and environment variables.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import yaml
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Complete system configuration"""

    # CuKEM Configuration
    cukem_mode: str = "hybrid"  # "hybrid", "pqc_only", "qkd_only"
    n_qubits: int = 256
    min_entropy: float = 0.8
    qber_threshold: float = 0.11
    chsh_verification: bool = True
    output_key_length: int = 32

    # Adaptive Controller
    auto_fallback: bool = True
    auto_recovery: bool = True
    failure_threshold: int = 3
    recovery_threshold: int = 2
    health_check_interval: int = 60

    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    circuit_failure_threshold: int = 5
    circuit_success_threshold: int = 2
    circuit_timeout_seconds: int = 60

    # TLS Configuration
    tls_hostname: str = "localhost"
    tls_port: int = 8443
    tls_use_psk: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Testing
    failure_injection_enabled: bool = False
    network_simulation_enabled: bool = False


class ConfigManager:
    """
    Configuration manager for CuKEM system.

    Supports:
    - YAML configuration files
    - Environment variable overrides
    - Configuration validation
    """

    ENV_PREFIX = "CUKEM_"

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        self.config = SystemConfig()

        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

        self.load_from_env()

        logger.info("Configuration loaded")

    def load_from_file(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)

            if data:
                for key, value in data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

            logger.info(f"Loaded configuration from {config_file}")

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")

    def load_from_env(self):
        """Load configuration from environment variables"""
        for key in self.config.__annotations__.keys():
            env_var = f"{self.ENV_PREFIX}{key.upper()}"
            value = os.environ.get(env_var)

            if value is not None:
                # Convert to appropriate type
                field_type = self.config.__annotations__[key]
                try:
                    if field_type == bool:
                        converted_value = value.lower() in ('true', '1', 'yes')
                    elif field_type == int:
                        converted_value = int(value)
                    elif field_type == float:
                        converted_value = float(value)
                    else:
                        converted_value = value

                    setattr(self.config, key, converted_value)
                    logger.debug(f"Set {key} from environment: {converted_value}")

                except ValueError as e:
                    logger.warning(f"Invalid environment value for {key}: {value}")

    def save_to_file(self, config_file: str):
        """Save configuration to YAML file"""
        try:
            with open(config_file, 'w') as f:
                yaml.dump(asdict(self.config), f, default_flow_style=False)

            logger.info(f"Saved configuration to {config_file}")

        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")

    def get_config(self) -> SystemConfig:
        """Get system configuration"""
        return self.config

    def validate(self) -> bool:
        """Validate configuration"""
        errors = []

        # Validate ranges
        if not 0.0 <= self.config.min_entropy <= 1.0:
            errors.append("min_entropy must be between 0.0 and 1.0")

        if not 0.0 <= self.config.qber_threshold <= 1.0:
            errors.append("qber_threshold must be between 0.0 and 1.0")

        if self.config.n_qubits < 64:
            errors.append("n_qubits should be at least 64")

        if self.config.output_key_length < 16:
            errors.append("output_key_length should be at least 16 bytes")

        # Log errors
        for error in errors:
            logger.error(f"Configuration validation error: {error}")

        return len(errors) == 0


def load_config(config_file: Optional[str] = None) -> SystemConfig:
    """
    Load system configuration.

    Args:
        config_file: Path to configuration file

    Returns:
        SystemConfig instance
    """
    manager = ConfigManager(config_file)
    return manager.get_config()
