# Hybrid DIQKD-PQC

A quantum-safe cryptographic key exchange system combining Device-Independent Quantum Key Distribution (DIQKD) with Post-Quantum Cryptography (PQC).

## Overview

This project implements **CuKEM** (Custom Unified Key Encapsulation Mechanism) - a hybrid cryptographic framework that orchestrates multiple security layers to generate keys resistant to both classical and quantum computer attacks.

### Key Features

- **Hybrid Security**: Combines quantum (BB84 + CHSH) and post-quantum (ML-KEM-768) cryptography
- **NIST Compliance**: Uses NIST-standardized algorithms (ML-KEM-768, NIST 800-90B entropy validation)
- **Adaptive Fallback**: 4-tier fallback mechanism for resilient key exchange
- **TLS 1.3 Integration**: Quantum-safe transport layer with PSK mode
- **Production-Ready**: Circuit breaker pattern, health monitoring, comprehensive logging

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│         Key Manager │ Session Handler │ Public API           │
├─────────────────────────────────────────────────────────────┤
│                     PROTOCOL LAYER                           │
│      Hybrid Key Exchange │ Fallback Controller (4-tier)      │
├─────────────────────────────────────────────────────────────┤
│               CRYPTOGRAPHIC PRIMITIVES LAYER                 │
│  PQC (ML-KEM, ML-DSA) │ DIQKD (Bell, CHSH) │ Classical (AES) │
├─────────────────────────────────────────────────────────────┤
│                 CORE INFRASTRUCTURE LAYER                    │
│      Secure Memory │ RNG │ HKDF │ Logging & Metrics          │
└─────────────────────────────────────────────────────────────┘
```

## Security Layers

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | ML-KEM-768 | Post-quantum key encapsulation (NIST Level 3) |
| 2 | BB84 + CHSH | Quantum key distribution with entanglement verification |
| 3 | NIST 800-90B | Entropy validation and estimation |
| 4 | SHAKE-256 | Privacy amplification via 2-universal hashing |
| 5 | HKDF-SHA256 | Secure key combination and derivation |
| 6 | TLS 1.3 PSK | Quantum-safe transport layer |
| 7 | Adaptive Controller | State machine with health monitoring |

## Project Structure

```
PROJECT/
├── code file/
│   ├── config.py                 # Configuration management
│   ├── utils.py                  # Utility functions
│   ├── pqc_kem.py                # ML-KEM-768 implementation
│   ├── bb84_simulator.py         # BB84 QKD protocol
│   ├── chsh_bell_test.py         # CHSH entanglement verification
│   ├── entropy_estimator.py      # NIST 800-90B entropy validation
│   ├── privacy_amplification.py  # SHAKE-256 privacy amplification
│   ├── hkdf_combiner.py          # HKDF key combination
│   ├── cukem.py                  # CuKEM orchestration layer
│   ├── hybrid_tls_wrapper.py     # TLS 1.3 PSK integration
│   ├── adaptive_controller.py    # State machine & health monitoring
│   ├── circuit_breaker.py        # Circuit breaker pattern
│   ├── network_simulator.py      # Mininet network simulation
│   └── failure_injector.py       # Fault injection testing
│
└── files/
    ├── hybrid_diqkd_pqc_project_documentation.md
    ├── Real_World_Scenarios_Analysis.md
    └── INTRO.md
```

## Requirements

- Python 3.8+
- Qiskit & Qiskit Aer (quantum simulation)
- liboqs-python (post-quantum cryptography)
- cryptography (classical cryptography)
- transitions (state machines)
- NumPy
- PyYAML

## Installation

```bash
# Install dependencies
pip install qiskit qiskit-aer numpy liboqs cryptography pyyaml transitions
```

## Quick Start

### Basic Key Exchange

```python
from cukem import CuKEM, CuKEMConfig, CuKEMMode

# Initialize CuKEM with hybrid mode
config = CuKEMConfig(mode=CuKEMMode.HYBRID, n_qubits=256)
cukem = CuKEM(config)

# Responder: Generate keypair
keypair = cukem.generate_keypair()

# Initiator: Perform key exchange
init_result = cukem.initiate_exchange(keypair.public_key)

# Responder: Complete key exchange
resp_result = cukem.respond_exchange(keypair, init_result.pqc_result.ciphertext)

# Both parties now have the same shared key
shared_key = resp_result.shared_key  # 32 bytes
```

### With TLS Integration

```python
from hybrid_tls_wrapper import HybridTLSWrapper, TLSConfig

tls_wrapper = HybridTLSWrapper()
# Configure server and client connections with quantum-safe PSK
```

## Configuration

### Operating Modes

| Mode | Description |
|------|-------------|
| `HYBRID` | PQC + Quantum (default, maximum security) |
| `PQC_ONLY` | Post-quantum cryptography only |
| `QKD_ONLY` | Quantum key distribution only |
| `FALLBACK` | Automatic fallback between tiers |

### Key Parameters

```python
CuKEMConfig(
    mode="hybrid",           # Operating mode
    n_qubits=256,            # Number of qubits for QKD
    min_entropy=0.8,         # Minimum entropy threshold
    qber_threshold=0.11,     # QBER acceptance level
    chsh_verification=True,  # Enable entanglement verification
    output_key_length=32,    # Output key size (bytes)
    auto_fallback=True,      # Automatic mode switching
    auto_recovery=True       # Automatic recovery
)
```

### Environment Variables

```bash
export CUKEM_CUKEM_MODE="hybrid"
export CUKEM_N_QUBITS=256
export CUKEM_QBER_THRESHOLD=0.11
export CUKEM_AUTO_FALLBACK=true
```

## Fallback Mechanism

The system implements a 4-tier fallback for resilient operation:

| Tier | Protocol | Security Level |
|------|----------|----------------|
| 1 | DIQKD + PQC | Maximum (device-independent) |
| 2 | MDI-QKD + PQC | High (measurement-device-independent) |
| 3 | QKD + PQC | Standard quantum-safe |
| 4 | PQC Only | Post-quantum (final fallback) |

## Testing

```bash
# Run failure injection tests
python failure_injector.py

# Network simulation (Linux only)
python network_simulator.py
```

## Security Properties

- **Post-Quantum Resistant**: NIST-standardized ML-KEM-768 (~192-bit classical security)
- **Information-Theoretic Security**: Quantum entanglement-based key generation
- **Entropy Validated**: NIST 800-90B certified entropy estimation
- **Privacy Amplified**: SHAKE-256 universal hashing
- **Resilient**: Circuit breaker pattern prevents cascading failures
- **Auditable**: Comprehensive logging and metrics


## References

- NIST Post-Quantum Cryptography Standardization
- BB84 Quantum Key Distribution Protocol
- CHSH Bell Inequality Test
- NIST SP 800-90B Entropy Estimation
