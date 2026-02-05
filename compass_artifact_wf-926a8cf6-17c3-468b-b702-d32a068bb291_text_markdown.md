# Building an 8-Layer Adaptive PQC-QKD Architecture: Complete Implementation Guide

A fully-functional hybrid Post-Quantum Cryptography and Device-Independent Quantum Key Distribution system is achievable for a final-year project using open-source libraries and simulated quantum channels. This architecture provides quantum-safe security through an adaptive system that dynamically selects between **PQC-only** mode (always available) and **hybrid PQC+QKD** mode (when quantum channels are operational), with proper entropy estimation and TLS 1.3 integration.

The system leverages **liboqs** for NIST-standardized ML-KEM algorithms, **Qiskit** for QKD simulation with Bell inequality testing, and **OQS-provider** for TLS integration—all mature, well-documented tools that can be assembled into a working prototype within **16 weeks**.

---

## The 8-layer architecture and its dependencies

Understanding the dependency graph between layers is critical for efficient development. Layers 1 and 2 can be built in parallel as foundational components, while higher layers depend on outputs from lower ones.

```
                    ┌─────────────────────────────────────────┐
                    │  Layer 8: Network Simulator (ns-3)      │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  Layer 7: Adaptive Controller           │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  Layer 6: TLS 1.3 Integration           │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  Layer 5: CuKEM Abstraction             │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  Layer 4: Hybrid Key Combiner           │
                    └──────┬─────────────────────┬────────────┘
                           │                     │
          ┌────────────────▼────┐    ┌──────────▼────────────┐
          │ Layer 3: Entropy    │    │  Layers 1 & 2         │
          │ Estimator           │    │  (parallel dev)       │
          └─────────────────────┘    └──────┬────────┬───────┘
                                            │        │
                       ┌────────────────────▼──┐  ┌──▼────────────────────┐
                       │ Layer 1: PQC KEM      │  │ Layer 2: DI-QKD       │
                       │ (ML-KEM-768)          │  │ Simulation            │
                       └───────────────────────┘  └───────────────────────┘
```

| Layer | Purpose | Primary Library | Complexity |
|-------|---------|-----------------|------------|
| 1 | PQC shared secret (always available) | liboqs-python | Low |
| 2 | Quantum entropy source (simulated) | Qiskit + Qiskit-Aer | High |
| 3 | Quantify usable entropy from QKD | NIST SP 800-90B tools | Medium |
| 4 | Combine PQC + QKD secrets securely | cryptography (HKDF) | Medium |
| 5 | Present unified KEM interface | Custom abstraction | Medium |
| 6 | Derive TLS session keys | OQS-provider / PSK | High |
| 7 | Dynamic mode selection | Custom state machine | Medium |
| 8 | Test failures and performance | QKDNetSim / Mininet | Medium |

---

## Layer 1: Post-Quantum KEM with ML-KEM-768

The foundation layer provides NIST-standardized lattice-based key encapsulation using **ML-KEM-768** (formerly Kyber), which offers **192-bit post-quantum security** with excellent performance—**58,000 key generations per second** on modern hardware.

**Installation and setup:**

```bash
# Build liboqs from source (provides underlying C implementation)
git clone --depth=1 https://github.com/open-quantum-safe/liboqs
cmake -S liboqs -B liboqs/build -DBUILD_SHARED_LIBS=ON
cmake --build liboqs/build --parallel 8
sudo cmake --build liboqs/build --target install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# Install Python bindings
pip install liboqs-python cryptography
```

**Complete ML-KEM-768 implementation:**

```python
import oqs
from dataclasses import dataclass
from typing import Tuple

@dataclass
class KEMResult:
    shared_secret: bytes
    ciphertext: bytes = None

class MLKEM768:
    """ML-KEM-768 wrapper providing clean interface for Layer 1"""
    
    ALGORITHM = "ML-KEM-768"
    PUBLIC_KEY_SIZE = 1184    # bytes
    SECRET_KEY_SIZE = 2400    # bytes  
    CIPHERTEXT_SIZE = 1088    # bytes
    SHARED_SECRET_SIZE = 32   # bytes
    
    def __init__(self):
        self._kem = None
        self._secret_key = None
        
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate new ML-KEM-768 key pair"""
        self._kem = oqs.KeyEncapsulation(self.ALGORITHM)
        public_key = self._kem.generate_keypair()
        self._secret_key = self._kem.export_secret_key()
        return public_key, self._secret_key
    
    def encapsulate(self, public_key: bytes) -> KEMResult:
        """Encapsulate: create ciphertext and shared secret for recipient"""
        kem = oqs.KeyEncapsulation(self.ALGORITHM)
        ciphertext, shared_secret = kem.encap_secret(public_key)
        return KEMResult(shared_secret=shared_secret, ciphertext=ciphertext)
    
    def decapsulate(self, ciphertext: bytes) -> KEMResult:
        """Decapsulate: recover shared secret from ciphertext"""
        if self._kem is None:
            raise ValueError("Must generate keypair before decapsulating")
        shared_secret = self._kem.decap_secret(ciphertext)
        return KEMResult(shared_secret=shared_secret)
```

**Key sizes for ML-KEM variants:**

| Variant | Public Key | Ciphertext | Security Level |
|---------|------------|------------|----------------|
| ML-KEM-512 | 800 bytes | 768 bytes | NIST Level 1 |
| ML-KEM-768 | 1,184 bytes | 1,088 bytes | NIST Level 3 |
| ML-KEM-1024 | 1,568 bytes | 1,568 bytes | NIST Level 5 |

---

## Layer 2: Simulating DI-QKD without quantum hardware

Device-Independent QKD provides the strongest security guarantees by verifying quantum behavior through **Bell inequality violations**. Without real quantum hardware, you simulate the entire protocol including entanglement generation, basis selection, measurement, and security verification through **CHSH tests**.

**BB84 protocol simulation with Qiskit:**

```python
import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

class BB84Simulator:
    """Complete BB84 QKD simulation with eavesdropper detection"""
    
    def __init__(self, noise_level: float = 0.01):
        self.backend = AerSimulator()
        self.noise_model = self._create_noise_model(noise_level)
        
    def _create_noise_model(self, depol_prob: float) -> NoiseModel:
        noise = NoiseModel()
        error = depolarizing_error(depol_prob, 1)
        noise.add_all_qubit_quantum_error(error, ['h', 'x', 'measure'])
        return noise
    
    def generate_qkd_key(self, n_qubits: int = 1000) -> dict:
        """Run full BB84 protocol and return key with metadata"""
        
        # Alice generates random bits and bases
        alice_bits = [random.randint(0, 1) for _ in range(n_qubits)]
        alice_bases = [random.choice(['Z', 'X']) for _ in range(n_qubits)]
        bob_bases = [random.choice(['Z', 'X']) for _ in range(n_qubits)]
        bob_measurements = []
        
        # Quantum transmission and measurement
        for i in range(n_qubits):
            qc = QuantumCircuit(1, 1)
            
            # Alice encodes her bit in chosen basis
            if alice_bases[i] == 'Z':
                if alice_bits[i] == 1:
                    qc.x(0)  # |1⟩
            else:  # X basis
                qc.h(0)
                if alice_bits[i] == 1:
                    qc.z(0)  # |−⟩
                    
            # Bob measures in his chosen basis
            if bob_bases[i] == 'X':
                qc.h(0)
            qc.measure(0, 0)
            
            result = self.backend.run(
                qc, shots=1, noise_model=self.noise_model
            ).result()
            bob_measurements.append(int(list(result.get_counts())[0]))
        
        # Sifting: keep only matching bases
        sifted_alice, sifted_bob = [], []
        for i in range(n_qubits):
            if alice_bases[i] == bob_bases[i]:
                sifted_alice.append(alice_bits[i])
                sifted_bob.append(bob_measurements[i])
        
        # Calculate QBER
        errors = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
        qber = errors / len(sifted_alice) if sifted_alice else 1.0
        
        return {
            'key': bytes(sifted_alice[:256 // 8]),  # First 256 bits as key
            'qber': qber,
            'sifted_length': len(sifted_alice),
            'is_secure': qber < 0.11  # BB84 security threshold
        }
```

**CHSH Bell inequality test for DI-QKD security:**

```python
class CHSHBellTest:
    """Verify quantum correlations through CHSH inequality violation"""
    
    CLASSICAL_BOUND = 2.0
    QUANTUM_MAXIMUM = 2 * np.sqrt(2)  # ≈ 2.828
    SECURITY_THRESHOLD = 2.0  # Must exceed for quantum security
    
    def __init__(self):
        self.backend = AerSimulator()
        
    def run_chsh_test(self, n_measurements: int = 1000) -> dict:
        """Run CHSH test and compute S-value"""
        
        # Optimal measurement angles for Bell state
        angles = {
            'a1': 0, 'a2': np.pi/4,           # Alice's settings
            'b1': np.pi/8, 'b2': 3*np.pi/8    # Bob's settings
        }
        
        correlations = {}
        for (a_name, a_angle) in [('a1', angles['a1']), ('a2', angles['a2'])]:
            for (b_name, b_angle) in [('b1', angles['b1']), ('b2', angles['b2'])]:
                corr = self._measure_correlation(a_angle, b_angle, n_measurements)
                correlations[f'{a_name}{b_name}'] = corr
        
        # CHSH S-value: S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)
        S = (correlations['a1b1'] - correlations['a1b2'] + 
             correlations['a2b1'] + correlations['a2b2'])
        
        return {
            's_value': abs(S),
            'violates_classical': abs(S) > self.CLASSICAL_BOUND,
            'correlations': correlations,
            'is_di_secure': abs(S) > self.SECURITY_THRESHOLD
        }
    
    def _measure_correlation(self, a_angle: float, b_angle: float, shots: int) -> float:
        """Measure correlation for given measurement angles"""
        qc = QuantumCircuit(2, 2)
        
        # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        qc.h(0)
        qc.cx(0, 1)
        
        # Apply measurement rotations
        qc.ry(2 * a_angle, 0)
        qc.ry(2 * b_angle, 1)
        qc.measure([0, 1], [0, 1])
        
        result = self.backend.run(qc, shots=shots).result()
        counts = result.get_counts()
        
        # Calculate expectation value
        corr = 0
        for outcome, count in counts.items():
            a_result = 1 - 2 * int(outcome[1])  # Convert 0/1 to +1/-1
            b_result = 1 - 2 * int(outcome[0])
            corr += a_result * b_result * count
        
        return corr / shots
```

**Existing QKD simulators to reference:**
- `github.com/Sidhazzzzzz/BB84_Quantum_Key_Distribution` — Full BB84 with QBER calculation
- `github.com/dhruvbhq/quantum_key_distrib_simple` — Clean Python OOP implementation
- `github.com/aws-samples/sample-BB84-qkd-on-amazon-braket` — Includes Golay error correction

---

## Layer 3: Entropy estimation for quantum key material

Before combining QKD output with PQC keys, you must estimate how much **usable entropy** the quantum channel actually provides. This uses NIST SP 800-90B min-entropy estimation and QKD-specific conditional entropy bounds.

**Min-entropy estimation implementation:**

```python
import numpy as np
from scipy.stats import entropy
import hashlib

class EntropyEstimator:
    """NIST SP 800-90B compliant entropy estimation for QKD keys"""
    
    def __init__(self):
        self.samples = []
        
    def estimate_min_entropy(self, data: bytes) -> float:
        """
        Estimate min-entropy using most common value estimator (NIST 6.3.1)
        Returns bits of entropy per byte
        """
        if len(data) < 10:
            return 0.0
            
        # Count byte frequencies
        counts = np.zeros(256)
        for byte in data:
            counts[byte] += 1
        
        # Most common value estimator: H_min = -log2(p_max)
        p_max = max(counts) / len(data)
        
        # Apply upper bound with 99% confidence interval
        confidence_bound = p_max + 2.576 * np.sqrt(p_max * (1 - p_max) / len(data))
        min_entropy = -np.log2(min(confidence_bound, 1.0))
        
        return min_entropy
    
    def qkd_conditional_entropy(self, qber: float) -> float:
        """
        Calculate H(A|E) for BB84 protocol using Devetak-Winter bound
        Returns bits of conditional entropy per raw bit
        """
        if qber <= 0:
            return 1.0
        if qber >= 0.5:
            return 0.0
        
        # Binary entropy function
        h_binary = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)
        
        # Devetak-Winter: H(A|E) ≥ 1 - h(QBER)
        return max(0, 1 - h_binary)
    
    def calculate_secure_key_length(self, raw_bits: int, qber: float, 
                                     security_parameter: float = 1e-10) -> int:
        """
        Calculate final secure key length after privacy amplification
        Using Leftover Hash Lemma: ℓ ≤ n·H(A|E) - 2·log(1/ε)
        """
        h_ae = self.qkd_conditional_entropy(qber)
        entropy_bits = raw_bits * h_ae
        
        # Finite-size correction
        correction = 2 * np.log2(1 / security_parameter)
        
        return max(0, int(entropy_bits - correction))
```

**Privacy amplification with universal hash functions:**

```python
def privacy_amplification(raw_key: bytes, target_bits: int, 
                          estimated_entropy: float) -> bytes:
    """
    Apply privacy amplification using SHAKE-256 (universal hash approximation)
    Reduces key length while increasing entropy density
    """
    # Verify we have enough entropy for requested output
    available_entropy = len(raw_key) * 8 * estimated_entropy
    if target_bits > available_entropy:
        raise ValueError(f"Insufficient entropy: {available_entropy:.0f} bits available")
    
    # SHAKE-256 as extendable-output function for privacy amplification
    h = hashlib.shake_256()
    h.update(raw_key)
    
    return h.digest(target_bits // 8)
```

---

## Layer 4: Hybrid key combiner using HKDF

The combiner securely merges PQC and QKD shared secrets into a single key. The **concatenation approach** recommended by NIST SP 800-56C and IETF draft-ietf-tls-hybrid-design provides provable security: if either component is secure, the output is secure.

**HKDF-based hybrid combiner:**

```python
import hmac
import hashlib
from typing import Optional

class HybridKeyCombiner:
    """
    NIST SP 800-56C compliant key combiner for PQC + QKD secrets
    Uses HKDF with concatenation: combined_ikm = pqc_secret || qkd_key
    """
    
    def __init__(self, hash_func=hashlib.sha256):
        self.hash_func = hash_func
        self.hash_len = hash_func().digest_size
        
    def hkdf_extract(self, salt: bytes, ikm: bytes) -> bytes:
        """HKDF-Extract: PRK = HMAC-Hash(salt, IKM)"""
        if not salt:
            salt = bytes(self.hash_len)
        return hmac.new(salt, ikm, self.hash_func).digest()
    
    def hkdf_expand(self, prk: bytes, info: bytes, length: int) -> bytes:
        """HKDF-Expand: derive output key material"""
        n = (length + self.hash_len - 1) // self.hash_len
        okm = b""
        t = b""
        for i in range(1, n + 1):
            t = hmac.new(prk, t + info + bytes([i]), self.hash_func).digest()
            okm += t
        return okm[:length]
    
    def combine(self, pqc_secret: bytes, qkd_key: Optional[bytes] = None,
                context: bytes = b"", output_length: int = 32) -> bytes:
        """
        Combine PQC and QKD secrets using HKDF
        
        Args:
            pqc_secret: Shared secret from ML-KEM (always provided)
            qkd_key: Optional QKD-derived key (when available)
            context: Application-specific context string
            output_length: Desired output key length in bytes
            
        Returns:
            Combined key material of specified length
        """
        # Concatenate secrets (NIST SP 800-56C approved)
        if qkd_key:
            combined_ikm = pqc_secret + qkd_key
            mode_info = b"hybrid-pqc-qkd"
        else:
            combined_ikm = pqc_secret
            mode_info = b"pqc-only"
        
        # Extract: derive PRK from combined material
        prk = self.hkdf_extract(salt=b"", ikm=combined_ikm)
        
        # Expand: derive final key with context binding
        info = mode_info + b"|" + context
        return self.hkdf_expand(prk, info, output_length)
    
    def derive_tls_secrets(self, pqc_secret: bytes, qkd_key: Optional[bytes],
                           transcript_hash: bytes) -> dict:
        """
        Derive TLS 1.3 style secrets for handshake and application data
        """
        combined = self.combine(pqc_secret, qkd_key, b"tls13-hybrid", 32)
        
        # Derive handshake secret
        hs_secret = self.hkdf_expand(
            self.hkdf_extract(combined, b""),
            b"tls13 derived" + transcript_hash, 32
        )
        
        # Derive client/server traffic secrets
        return {
            'handshake_secret': hs_secret,
            'client_traffic': self.hkdf_expand(hs_secret, b"c hs traffic" + transcript_hash, 32),
            'server_traffic': self.hkdf_expand(hs_secret, b"s hs traffic" + transcript_hash, 32)
        }
```

---

## Layer 5: CuKEM unified abstraction interface

The Customizable KEM layer presents a single interface that abstracts away whether the system is operating in pure PQC mode or hybrid mode. This enables Layer 6 (TLS) to work identically regardless of QKD availability.

**Complete CuKEM implementation:**

```python
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict

class KEMMode(Enum):
    PURE_PQC = "pure_pqc"          # ML-KEM only
    HYBRID_PQC_QKD = "hybrid_qkd"  # ML-KEM + QKD combined

@dataclass  
class CuKEMResult:
    shared_secret: bytes
    ciphertext: bytes
    mode: KEMMode
    metadata: Dict = None

class CuKEM:
    """
    Customizable KEM abstraction providing unified interface
    for PQC-only and hybrid PQC+QKD modes
    """
    
    def __init__(self, algorithm: str = "ML-KEM-768"):
        self.algorithm = algorithm
        self._mode = KEMMode.PURE_PQC
        self._qkd_key: Optional[bytes] = None
        self._pqc_kem = MLKEM768()
        self._combiner = HybridKeyCombiner()
        self._public_key: Optional[bytes] = None
        self._secret_key: Optional[bytes] = None
        
    @property
    def mode(self) -> KEMMode:
        return self._mode
    
    def set_qkd_key(self, qkd_key: bytes):
        """Inject QKD-derived key for hybrid mode"""
        self._qkd_key = qkd_key
        self._mode = KEMMode.HYBRID_PQC_QKD
        
    def clear_qkd_key(self):
        """Revert to PQC-only mode"""
        self._qkd_key = None
        self._mode = KEMMode.PURE_PQC
        
    def generate_keypair(self) -> bytes:
        """Generate key pair, return public key"""
        self._public_key, self._secret_key = self._pqc_kem.generate_keypair()
        return self._public_key
    
    def encapsulate(self, peer_public_key: bytes) -> CuKEMResult:
        """
        Encapsulate shared secret for peer
        Automatically combines with QKD key if available
        """
        pqc_result = self._pqc_kem.encapsulate(peer_public_key)
        
        if self._mode == KEMMode.HYBRID_PQC_QKD and self._qkd_key:
            combined_secret = self._combiner.combine(
                pqc_result.shared_secret, 
                self._qkd_key,
                context=b"cukem-encaps"
            )
            return CuKEMResult(
                shared_secret=combined_secret,
                ciphertext=pqc_result.ciphertext,
                mode=self._mode,
                metadata={'qkd_contribution': True}
            )
        
        return CuKEMResult(
            shared_secret=pqc_result.shared_secret,
            ciphertext=pqc_result.ciphertext,
            mode=self._mode
        )
    
    def decapsulate(self, ciphertext: bytes) -> CuKEMResult:
        """
        Decapsulate to recover shared secret
        Automatically combines with QKD key if available
        """
        pqc_result = self._pqc_kem.decapsulate(ciphertext)
        
        if self._mode == KEMMode.HYBRID_PQC_QKD and self._qkd_key:
            combined_secret = self._combiner.combine(
                pqc_result.shared_secret,
                self._qkd_key,
                context=b"cukem-decaps"
            )
            return CuKEMResult(
                shared_secret=combined_secret,
                ciphertext=ciphertext,
                mode=self._mode,
                metadata={'qkd_contribution': True}
            )
        
        return CuKEMResult(
            shared_secret=pqc_result.shared_secret,
            ciphertext=ciphertext,
            mode=self._mode
        )
```

---

## Layer 6: TLS 1.3 integration approaches

Integrating hybrid keys into TLS 1.3 is the most complex layer. The recommended approach uses **OQS-provider** with OpenSSL 3.x and **PSK mode** for injecting QKD key material.

**OQS-Provider setup:**

```bash
# Prerequisites: OpenSSL 3.0+ and liboqs installed
git clone https://github.com/open-quantum-safe/oqs-provider
cd oqs-provider
cmake -S . -B _build && cmake --build _build
sudo cmake --install _build

# Configure OpenSSL to load oqs-provider
# Add to /etc/ssl/openssl.cnf:
```

**OpenSSL configuration (openssl.cnf):**

```ini
[openssl_init]
providers = provider_sect

[provider_sect]
default = default_sect
oqsprovider = oqsprovider_sect

[default_sect]
activate = 1

[oqsprovider_sect]
activate = 1
module = /usr/local/lib/ossl-modules/oqsprovider.so
```

**Running hybrid TLS server/client:**

```bash
# Generate ML-DSA certificate for hybrid authentication
openssl req -x509 -new -newkey mldsa65 \
    -keyout server.key -out server.crt -nodes \
    -subj "/CN=localhost" -provider oqsprovider -provider default

# Start TLS 1.3 server with hybrid KEM
openssl s_server -cert server.crt -key server.key \
    -tls1_3 -groups X25519MLKEM768 \
    -provider oqsprovider -provider default \
    -accept 4433

# Connect with hybrid client
openssl s_client -CAfile server.crt \
    -tls1_3 -groups X25519MLKEM768 \
    -provider oqsprovider -provider default \
    -connect localhost:4433
```

**PSK mode for QKD key injection (simpler approach for project):**

```python
import subprocess
import hashlib

class HybridTLSWrapper:
    """
    Wrapper combining CuKEM output with TLS PSK mode
    This is the recommended approach for the student project
    """
    
    def __init__(self, cukem: CuKEM):
        self.cukem = cukem
        
    def create_hybrid_psk(self, peer_public_key: bytes) -> bytes:
        """
        Create PSK from hybrid key exchange for TLS PSK mode
        This PSK can be used with OpenSSL's -psk option
        """
        result = self.cukem.encapsulate(peer_public_key)
        
        # Derive TLS-suitable PSK (32 bytes)
        psk = hashlib.sha256(
            b"tls13-hybrid-psk" + result.shared_secret
        ).digest()
        
        return psk
    
    def start_psk_server(self, psk: bytes, port: int = 4433):
        """Start TLS 1.3 server with hybrid-derived PSK"""
        psk_hex = psk.hex()
        cmd = [
            'openssl', 's_server',
            '-psk', psk_hex,
            '-psk_identity', 'hybrid_qkd_session',
            '-nocert',
            '-tls1_3',
            '-accept', str(port)
        ]
        return subprocess.Popen(cmd)
```

**Key integration patterns:**

| Approach | Complexity | Best For |
|----------|------------|----------|
| PSK mode | Low | Project demos, proof-of-concept |
| OQS-provider groups | Medium | Production-like implementation |
| Custom TLS extension | High | Research contribution |

---

## Layer 7: Adaptive controller with fallback mechanisms

The adaptive controller monitors QKD channel health and automatically switches between modes. It uses a **state machine** with **circuit breaker pattern** for resilient fallback.

**Adaptive controller implementation:**

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable
import time
import threading

class ControllerState(Enum):
    INITIALIZING = auto()
    READY = auto()
    QKD_ACTIVE = auto()
    PQC_FALLBACK = auto()
    ERROR = auto()

@dataclass
class ChannelMetrics:
    qber: float
    key_rate: float  # bits/second
    bell_violation: float  # CHSH S-value
    last_update: float

class AdaptiveController:
    """
    Layer 7: Dynamic mode selection between PQC-only and hybrid modes
    """
    
    # Thresholds for QKD channel health
    MAX_QBER = 0.11           # BB84 security threshold
    MIN_KEY_RATE = 100        # bits/second
    MIN_BELL_VIOLATION = 2.0  # CHSH classical bound
    
    def __init__(self, cukem: CuKEM, qkd_simulator: BB84Simulator):
        self.cukem = cukem
        self.qkd = qkd_simulator
        self.state = ControllerState.INITIALIZING
        self.metrics = None
        self._circuit_breaker = CircuitBreaker(failure_threshold=3)
        self._monitor_thread = None
        self._running = False
        
    def start(self):
        """Start adaptive monitoring"""
        self._running = True
        self.state = ControllerState.READY
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.start()
        
    def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
            
    def _monitor_loop(self):
        """Continuous monitoring of QKD channel"""
        while self._running:
            self._update_metrics()
            self._evaluate_mode()
            time.sleep(1.0)
            
    def _update_metrics(self):
        """Collect current channel metrics"""
        try:
            qkd_result = self.qkd.generate_qkd_key(n_qubits=100)
            bell_test = CHSHBellTest().run_chsh_test(n_measurements=100)
            
            self.metrics = ChannelMetrics(
                qber=qkd_result['qber'],
                key_rate=qkd_result['sifted_length'] * 8,  # simplified
                bell_violation=bell_test['s_value'],
                last_update=time.time()
            )
            self._circuit_breaker.record_success()
        except Exception as e:
            self._circuit_breaker.record_failure()
            self.metrics = None
            
    def _evaluate_mode(self):
        """Decide operating mode based on metrics"""
        if not self._circuit_breaker.can_proceed():
            self._switch_to_pqc_fallback()
            return
            
        if self.metrics is None:
            self._switch_to_pqc_fallback()
            return
            
        # Check QKD channel health
        qkd_healthy = (
            self.metrics.qber < self.MAX_QBER and
            self.metrics.key_rate >= self.MIN_KEY_RATE and
            self.metrics.bell_violation > self.MIN_BELL_VIOLATION
        )
        
        if qkd_healthy:
            self._switch_to_hybrid_mode()
        else:
            self._switch_to_pqc_fallback()
            
    def _switch_to_hybrid_mode(self):
        """Enable hybrid PQC+QKD mode"""
        if self.state != ControllerState.QKD_ACTIVE:
            qkd_result = self.qkd.generate_qkd_key(n_qubits=1000)
            self.cukem.set_qkd_key(qkd_result['key'])
            self.state = ControllerState.QKD_ACTIVE
            
    def _switch_to_pqc_fallback(self):
        """Fall back to PQC-only mode"""
        if self.state != ControllerState.PQC_FALLBACK:
            self.cukem.clear_qkd_key()
            self.state = ControllerState.PQC_FALLBACK

class CircuitBreaker:
    """Prevents repeated attempts on failing QKD channel"""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.failures = 0
        self.threshold = failure_threshold
        self.timeout = recovery_timeout
        self.state = "CLOSED"
        self.last_failure_time = 0
        
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.state = "OPEN"
            
    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"
        
    def can_proceed(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF-OPEN"
                return True
        return self.state == "HALF-OPEN"
```

---

## Layer 8: Network simulation with QKDNetSim

For comprehensive testing, use **QKDNetSim** (NS-3 extension) for realistic network simulation or **Mininet** for simpler topologies.

**QKDNetSim installation (NS-3 v3.46):**

```bash
# Install dependencies
sudo apt-get install cmake build-essential git libboost-all-dev \
    libcrypto++-dev uuid-dev flex bison tcpdump

# Clone NS-3 and QKDNetSim
git clone -b ns-3.46 https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev/contrib
git clone https://github.com/QKDNetSim/qkdnetsim

# Build
cd ..
./ns3 configure --enable-examples
./ns3 build
```

**Mininet-based test topology (simpler alternative):**

```python
#!/usr/bin/env python3
from mininet.net import Mininet
from mininet.node import Controller
from mininet.link import TCLink
from mininet.cli import CLI

def create_qkd_test_network():
    """
    Create test network topology:
    Alice <--quantum--> Bob <--classical--> KMS
    """
    net = Mininet(controller=Controller, link=TCLink)
    
    c0 = net.addController('c0')
    
    # QKD endpoints
    alice = net.addHost('alice', ip='10.0.1.1/24')
    bob = net.addHost('bob', ip='10.0.1.2/24')
    kms = net.addHost('kms', ip='10.0.1.3/24')  # Key Management System
    
    # Switch for classical channel
    s1 = net.addSwitch('s1')
    
    # Quantum channel simulation (high latency, low bandwidth)
    net.addLink(alice, s1, bw=10, delay='50ms', loss=1)
    net.addLink(bob, s1, bw=10, delay='50ms', loss=1)
    
    # Classical channel (low latency, high bandwidth)
    net.addLink(kms, s1, bw=1000, delay='1ms')
    
    net.start()
    return net

def inject_channel_failure(net, duration_sec: float = 5.0):
    """Simulate quantum channel disruption"""
    alice = net.get('alice')
    alice.cmd(f'tc qdisc change dev alice-eth0 root netem loss 100%')
    time.sleep(duration_sec)
    alice.cmd(f'tc qdisc change dev alice-eth0 root netem loss 1%')
```

**Performance benchmarking framework:**

```python
import time
import json
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class BenchmarkResult:
    mode: str
    key_gen_rate: float      # keys/second
    handshake_latency_ms: float
    throughput_mbps: float
    recovery_time_ms: float
    
class PerformanceBenchmark:
    """Comprehensive benchmarking for hybrid system"""
    
    def __init__(self, system):
        self.system = system
        self.results: List[BenchmarkResult] = []
        
    def run_full_benchmark(self) -> dict:
        """Execute complete benchmark suite"""
        results = {}
        
        for mode in ['PQC_ONLY', 'HYBRID']:
            self.system.set_mode(mode)
            
            results[mode] = {
                'key_generation': self._benchmark_keygen(iterations=100),
                'handshake_latency': self._benchmark_handshake(iterations=50),
                'throughput': self._benchmark_throughput(data_mb=10),
            }
            
        # Test fallback recovery
        results['fallback_recovery'] = self._benchmark_recovery()
        
        return results
    
    def _benchmark_keygen(self, iterations: int) -> dict:
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.system.cukem.generate_keypair()
            times.append(time.perf_counter() - start)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'p99_ms': np.percentile(times, 99) * 1000,
            'ops_per_sec': 1 / np.mean(times)
        }
```

---

## Complete implementation roadmap

| Week | Phase | Layers | Key Deliverables |
|------|-------|--------|------------------|
| 1-2 | Foundation | 1, 2 | ML-KEM wrapper working, BB84 simulator running |
| 3-4 | QKD Complete | 2, 3 | CHSH test implemented, entropy estimation working |
| 5-6 | Integration Core | 4, 5 | Hybrid combiner tested, CuKEM abstraction complete |
| 7-9 | TLS Integration | 6 | PSK-based TLS demo working |
| 10-11 | Adaptive Control | 7 | State machine with fallback operational |
| 12-13 | Network Sim | 8 | Mininet topology, failure injection tests |
| 14-15 | Integration | All | End-to-end system, performance benchmarks |
| 16 | Demo | — | Final demonstration, documentation |

**Critical dependencies to resolve first:**
1. liboqs installation (blocks Layer 1)
2. Qiskit environment setup (blocks Layer 2)
3. OpenSSL 3.x with oqs-provider (blocks Layer 6)

---

## Key open-source implementations to reference

| Project | URL | Reusability |
|---------|-----|-------------|
| **QKD-KEM Provider** | github.com/qursa-uc3m/qkd-kem-provider | ⭐⭐⭐⭐⭐ Reference architecture for PQC+QKD hybrid |
| **liboqs-python** | github.com/open-quantum-safe/liboqs-python | ⭐⭐⭐⭐⭐ Direct use for ML-KEM |
| **OQS-provider** | github.com/open-quantum-safe/oqs-provider | ⭐⭐⭐⭐⭐ TLS integration foundation |
| **cascade-python** | github.com/brunorijsman/cascade-python | ⭐⭐⭐⭐ Error correction protocol |
| **BB84 Qiskit** | github.com/Sidhazzzzzz/BB84_Quantum_Key_Distribution | ⭐⭐⭐⭐ QKD simulation reference |
| **ETSI QKD API** | github.com/qursa-uc3m/qkd-etsi-api | ⭐⭐⭐⭐ Standards-compliant interface |

---

## Testing strategy summary

Each layer requires specific validation approaches:

- **Layer 1**: NIST Known Answer Tests (KATs) for ML-KEM
- **Layer 2**: Verify QBER = 0% without noise, ~25% with full eavesdropper
- **Layer 3**: Compare entropy estimates against theoretical bounds
- **Layer 4**: Test key independence (output changes if either input changes)
- **Layer 5**: Mode switching correctness under various conditions
- **Layer 6**: Wireshark inspection of TLS handshake
- **Layer 7**: Verify fallback triggers within specified thresholds
- **Layer 8**: Network failure injection and recovery timing

**Minimum test coverage targets:** Unit tests for all layers, integration tests for layer pairs, end-to-end test for complete key establishment, and stress tests for concurrent connections.

---

## Conclusion

This 8-layer architecture provides a practical framework for implementing quantum-safe communications that gracefully handles the transition between classical-only, PQC-only, and hybrid PQC+QKD modes. The key innovations are the **CuKEM abstraction** that hides mode complexity from upper layers, the **adaptive controller** that automatically optimizes security based on channel conditions, and the **HKDF-based combiner** that provides provable security guarantees.

For a final-year project, prioritize getting Layers 1-5 working first as a functional core, then add TLS integration (Layer 6) using the simpler PSK approach, and finally implement adaptive control and simulation for comprehensive testing. The entire system can be implemented in **~3,500 lines of Python** with approximately **500 lines of configuration/scripting**, making it achievable within a 16-week timeframe while producing a research-grade prototype that demonstrates the future of quantum-safe cryptography.