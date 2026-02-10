# CuKEM: Hybrid Post-Quantum Cryptography + Quantum Key Distribution

A complete 8-layer hybrid cryptographic system combining **post-quantum cryptography (ML-KEM-768)** with **quantum key distribution (BB84)** for quantum-safe communications.

## 📋 Quick Start Guide

### Prerequisites
- **Python 3.9+** (3.10+ recommended)
- **pip** (Python package manager)
- Windows 10/11 or Linux system

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd "c:\Users\LENOVO\Desktop\PROJECT-2\PROJECT\code file"

# Install all required packages
pip install -r requirements.txt
```

**Common issues:**
- If `liboqs-python` fails to install, see [Installation Guide](#installation-troubleshooting)
- If `qiskit` fails, try: `pip install --upgrade qiskit qiskit-aer`

### Step 2: Configure the System

Edit `config.yml` to customize settings (optional - defaults work well):
```yaml
cukem_mode: "hybrid"      # pqc_only, qkd_only, or hybrid
n_qubits: 256             # Number of qubits for BB84
log_level: "INFO"         # DEBUG, INFO, WARNING, ERROR
```

### Step 3: Run the Demo

```bash
python demo.py
```

This will run all 6 demonstrations:
1. ✓ Post-Quantum KEM (ML-KEM-768)
2. ✓ Quantum Key Distribution (BB84)
3. ✓ Entropy Estimation (NIST 800-90B)
4. ✓ CuKEM Hybrid Mode (PQC + QKD)
5. ✓ Adaptive Controller
6. ✓ Circuit Breaker Pattern

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────┐
│  Layer 8: Network Simulator & Failure Injection │
├─────────────────────────────────────────────────┤
│  Layer 7: Adaptive Controller & Circuit Breaker │
├─────────────────────────────────────────────────┤
│  Layer 6: TLS 1.3 Integration (PSK Mode)        │
├─────────────────────────────────────────────────┤
│  Layer 5: CuKEM Unified Interface               │
├─────────────────────────────────────────────────┤
│  Layer 4: HKDF Key Combination                  │
├─────────────────────────────────────────────────┤
│  Layer 3: Entropy Estimation & Privacy Amp.     │
├─────────────────────────────────────────────────┤
│  Layer 2: BB84 QKD + CHSH Bell Test            │
├─────────────────────────────────────────────────┤
│  Layer 1: Post-Quantum KEM (ML-KEM-768)        │
└─────────────────────────────────────────────────┘
```

## 📦 Module Overview

| Layer | Module | Purpose |
|-------|--------|---------|
| 1 | `pqc_kem.py` | ML-KEM-768 post-quantum key encapsulation |
| 2 | `bb84_simulator.py` | BB84 quantum key distribution protocol |
| 2b | `chsh_bell_test.py` | CHSH entanglement verification |
| 3 | `entropy_estimator.py` | NIST 800-90B entropy estimation (5 methods) |
| 3b | `privacy_amplification.py` | SHAKE-256 privacy amplification |
| 4 | `hkdf_combiner.py` | HKDF-based hybrid key derivation |
| 5 | `cukem.py` | Unified KEM abstraction (all modes) |
| 6 | `hybrid_tls_wrapper.py` | TLS 1.3 integration with PSK |
| 7 | `adaptive_controller.py` | Health monitoring and state machine |
| 7b | `circuit_breaker.py` | Circuit breaker pattern for resilience |
| 8 | `network_simulator.py` | Mininet-based network simulation |
| 8b | `failure_injector.py` | Controlled failure injection for testing |
| Util | `utils.py` | Cryptographic utilities and helpers |
| Config | `config.py` | Configuration management |

## 🚀 Usage Examples

### Example 1: Basic Post-Quantum Key Exchange
```python
from pqc_kem import MLKEM768

kem = MLKEM768()
responder_keypair = kem.generate_keypair()
encap_result = kem.encapsulate(responder_keypair.public_key)
decap_result = kem.decapsulate(encap_result.ciphertext, 
                               responder_keypair.secret_key)
```

### Example 2: BB84 Quantum Key Distribution
```python
from bb84_simulator import BB84Simulator

bb84 = BB84Simulator(n_qubits=256)
result = bb84.execute_protocol(noise_level=0.0)
if result.success:
    print(f"Generated: {result.key_length} bytes")
    print(f"QBER: {result.qber:.4f}")
```

### Example 3: Hybrid CuKEM Exchange
```python
from cukem import CuKEM, CuKEMConfig, CuKEMMode

config = CuKEMConfig(mode=CuKEMMode.HYBRID, n_qubits=256)
cukem = CuKEM(config)

responder_keypair = cukem.generate_keypair()
result = cukem.initiate_exchange(responder_keypair.public_key)

if result.success:
    print(f"Hybrid key established: {result.key_length} bytes")
```

### Example 4: Adaptive Control with Health Monitoring
```python
from adaptive_controller import AdaptiveController

controller = AdaptiveController()
controller.initialize_system()

result = controller.perform_exchange(role="initiator")
health = controller.get_health_status()
stats = controller.get_statistics()

print(f"Health: {health.value}")
print(f"Success rate: {stats['metrics']['pqc_success_rate']:.2%}")
```

## 🔧 Installation Troubleshooting

### Issue: liboqs-python installation fails

**Solution 1 (Recommended):** Use pre-built wheels
```bash
pip install --only-binary :all: liboqs-python
```

**Solution 2:** Build from source (requires build tools)
```bash
# On Windows: Install Visual C++ Build Tools first
pip install --no-binary liboqs-python liboqs-python
```

**Solution 3:** Skip and use PQC-only mode
```bash
# Edit config.yml: cukem_mode: "pqc_only"
# Or omit liboqs: pip install -r requirements_lite.txt
```

### Issue: Qiskit installation fails

**Solution:**
```bash
pip install --upgrade qiskit qiskit-aer --no-cache-dir
```

### Issue: YAML library not found

**Solution:**
```bash
pip install PyYAML
```

## 📊 Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| ML-KEM-768 keypair generation | ~1-2ms | Fast lattice-based KEM |
| ML-KEM-768 encapsulation | ~1-2ms | |
| ML-KEM-768 decapsulation | ~1-2ms | |
| BB84 (256 qubits) | ~100-500ms | Simulated quantum channel |
| CHSH test (8192 shots) | ~200-400ms | Bell inequality verification |
| Entropy estimation | ~10-50ms | 5 estimation methods |
| HKDF key derivation | <1ms | Industry standard |
| Hybrid exchange (full) | ~200-1000ms | All layers combined |

## 🔐 Security Considerations

- **Post-Quantum Security:** ML-KEM-768 provides NIST Level 3 security (~192-bit)
- **Quantum Entropy:** BB84 simulation provides perfect entropy (simulated noiseless channel)
- **Privacy Amplification:** SHAKE-256 reduces adversary knowledge
- **Key Combination:** HKDF provides proven secure composition
- **Adaptive Resilience:** Automatic fallback on quantum channel failures

## 📝 Logging and Debugging

Enable debug logging:
```bash
# Edit config.yml
log_level: "DEBUG"
log_file: "cukem_debug.log"
```

Or set environment variable:
```bash
$env:CUKEM_LOG_LEVEL = "DEBUG"
python demo.py
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
pytest --cov  # With coverage report
```

Run specific demos:
```bash
python -c "from demo import demo_basic_pqc; demo_basic_pqc()"
```

## 📚 Additional Resources

- **NIST ML-KEM Standard:** https://csrc.nist.gov/publications/detail/fips/203/final
- **BB84 QKD Protocol:** Bennett & Brassard (1984)
- **CHSH Inequality:** Clauser et al. (1969)
- **HKDF RFC 5869:** https://tools.ietf.org/html/rfc5869
- **Mininet Documentation:** http://mininet.org/

## 🤝 Contributing

To extend the system:

1. **Add a new quantum protocol:** Create `new_protocol.py` implementing the protocol
2. **Add a resilience feature:** Extend `adaptive_controller.py`
3. **Add network conditions:** Extend `network_simulator.py` or `failure_injector.py`
4. **Improve entropy estimation:** Add methods to `entropy_estimator.py`

## 📄 License

This is a reference implementation for academic and research purposes.

## 🎓 References

- Post-quantum cryptography: NIST PQC Standardization Project
- Quantum key distribution: Bennett & Brassard (1984), Ekert (1991)
- Entropy: NIST SP 800-90B
- Key derivation: RFC 5869 (HKDF)

---

**Questions?** Check the architecture documentation in `compass_artifact_wf-926a8cf6-17c3-468b-b702-d32a068bb291_text_markdown.md`
