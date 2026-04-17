# 🚀 START HERE: Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies (2 minutes)
Open PowerShell in the project directory and run:

```powershell
cd "c:\Users\LENOVO\Desktop\PROJECT-2\PROJECT\code file"
python setup.py
```

This will automatically:
- ✓ Check Python version
- ✓ Install all required packages
- ✓ Verify dependencies
- ✓ Create necessary directories
- ✓ Test basic functionality

### Step 2: Verify Installation (1 minute)
If setup.py completes successfully, you'll see:
```
===============================================================
 ✓ SETUP COMPLETE!
===============================================================
```

### Step 3: Run the Demo (2 minutes)
```powershell
python demo.py
```

This will run 6 complete demonstrations:
1. Post-Quantum KEM (ML-KEM-768) - **PQC Key Exchange**
2. BB84 Quantum Key Distribution - **Quantum Protocol**
3. Entropy Estimation - **Security Validation**
4. CuKEM Hybrid Mode - **PQC + QKD Combined**
5. Adaptive Controller - **Health Monitoring**
6. Circuit Breaker Pattern - **Resilience Testing**

---

## 📋 Installation Options

### Option 1: Full Setup (Recommended)
```powershell
python setup.py
python demo.py
```

### Option 2: Manual Installation
```powershell
pip install -r requirements.txt
python demo.py
```

### Option 3: Minimal Installation (PQC-only mode)
```powershell
pip install cryptography PyYAML transitions python-dotenv numpy scipy
# Edit config.yml: cukem_mode: "pqc_only"
python demo.py
```

---

## ✅ Expected Output

When you run `python demo.py`, you should see:

```
======================================================================
 CuKEM HYBRID PQC-QKD SYSTEM - DEMONSTRATION
 8-Layer Architecture for Quantum-Safe Communication
======================================================================

======================================================================
 DEMO 1: Post-Quantum Key Exchange (ML-KEM-768)
======================================================================
INFO:__main__:Initialized ML-KEM-768
INFO:__main__:Generated responder keypair
INFO:__main__:✓ Encapsulation successful
INFO:__main__:✓ Decapsulation successful
INFO:__main__:✓ Shared secrets match!

======================================================================
 DEMO 2: Quantum Key Distribution (BB84)
======================================================================
INFO:__main__:Initialized BB84 simulator with 256 qubits
INFO:__main__:✓ BB84 protocol successful
INFO:__main__:  Generated key: 32 bytes
INFO:__main__:  Sifting efficiency: 25.00%
INFO:__main__:  QBER: 0.0000

[... more demos ...]

======================================================================
 DEMONSTRATION SUMMARY
======================================================================
============================================================
 CuKEM Demo Results
============================================================
Demo: Status
Post-Quantum KEM: ✓ PASS
Quantum Key Distribution: ✓ PASS
Entropy Estimation: ✓ PASS
CuKEM Hybrid Mode: ✓ PASS
Adaptive Controller: ✓ PASS
Circuit Breaker: ✓ PASS
============================================================

DEMO COMPLETE: 6/6 tests passed (100.0%)
```

---

## 🔧 Common Setup Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'liboqs'"
**Solution:** This is optional for PQC-only functionality
```powershell
pip install --only-binary :all: liboqs-python
# OR just skip to PQC-only mode
```

### Issue 2: "ModuleNotFoundError: No module named 'qiskit'"
**Solution:** 
```powershell
pip install --upgrade qiskit qiskit-aer --no-cache-dir
```

### Issue 3: "Python 3.8 is not supported"
**Solution:** Upgrade Python to 3.9 or later
```powershell
# Check version
python --version

# If you have multiple versions, use python3.10 or later
python3.10 setup.py
```

### Issue 4: Permission denied when installing
**Solution:** Use `--user` flag
```powershell
pip install --user -r requirements.txt
```

---

## 📝 Configuration (Optional)

Edit `config.yml` to customize behavior:

```yaml
# Run in PQC-only mode (skip quantum simulation)
cukem_mode: "pqc_only"

# Reduce quantum qubits for faster demo
n_qubits: 128

# Enable detailed logging
log_level: "DEBUG"
log_file: "cukem_debug.log"

# Enable failure injection for resilience testing
failure_injection_enabled: true
```

---

## 🎯 What Each Demo Does

### Demo 1: Post-Quantum KEM
Tests ML-KEM-768 key exchange:
- Generates keypair (1184 bytes public, 2400 bytes secret)
- Encapsulates shared secret
- Decapsulates and verifies match
- **Time:** ~1-2 ms

### Demo 2: BB84 QKD
Simulates quantum key distribution:
- Creates 256 quantum states
- Bob measures with random bases
- Performs sifting (achieves ~25% efficiency)
- Calculates QBER (should be ~0% without noise)
- **Time:** ~100-500 ms

### Demo 3: Entropy Estimation
Validates quantum key quality:
- Uses 5 entropy estimation methods
- Checks min-entropy against NIST 800-90B threshold
- Outputs Shannon entropy, MCV, collision, Markov, compression estimates
- **Time:** ~10-50 ms

### Demo 4: CuKEM Hybrid
Combines PQC + QKD:
- Performs both ML-KEM-768 and BB84
- Derives final key using HKDF
- Applies privacy amplification
- **Time:** ~200-1000 ms (all layers)

### Demo 5: Adaptive Controller
Tests health monitoring:
- Tracks success/failure rates
- Monitors QBER and entropy
- Triggers fallback on failures
- Attempts recovery
- **Time:** ~1-5 seconds

### Demo 6: Circuit Breaker
Tests resilience pattern:
- Simulates failing function
- Monitors failure rate
- Opens circuit when threshold exceeded
- Attempts recovery after timeout
- **Time:** ~1 second

---

## 🚀 Next Steps After Setup

### Try These Commands:

```powershell
# Run just the PQC demo
python -c "from demo import demo_basic_pqc; demo_basic_pqc()"

# Run just the BB84 demo
python -c "from demo import demo_quantum_key_distribution; demo_quantum_key_distribution()"

# View detailed logs
type cukem_demo.log

# Run in debug mode
python -c "$env:CUKEM_LOG_LEVEL='DEBUG'; exec(open('demo.py').read())"

# Try running with PQC-only mode (faster)
python -c "
from cukem import CuKEM, CuKEMConfig, CuKEMMode
config = CuKEMConfig(mode=CuKEMMode.PQC_ONLY)
cukem = CuKEM(config)
keypair = cukem.generate_keypair()
result = cukem.initiate_exchange(keypair.public_key)
print(f'Key: {result.key_length} bytes, Success: {result.success}')
"
```

---

## 📊 Performance Benchmarks

Typical performance on modern hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| ML-KEM-768 keypair | 1-2 ms | Fast |
| BB84 (256 qubits) | 100-500 ms | Quantum simulation |
| Entropy estimation | 10-50 ms | 5 methods run |
| Full hybrid exchange | 200-1000 ms | All layers |

---

## 📚 Documentation Structure

```
Project Directory/
├── demo.py                    # Main demonstrations (RUN THIS)
├── setup.py                   # Automated setup (RUN THIS FIRST)
├── config.yml                 # Configuration file
├── requirements.txt           # Python dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md              # This file
│
├── pqc_kem.py                # Layer 1: Post-quantum KEM
├── bb84_simulator.py         # Layer 2: Quantum key distribution
├── chsh_bell_test.py         # Layer 2b: Bell test
├── entropy_estimator.py      # Layer 3: Entropy analysis
├── privacy_amplification.py  # Layer 3b: Privacy amp
├── hkdf_combiner.py          # Layer 4: Key combination
├── cukem.py                  # Layer 5: Unified interface
├── hybrid_tls_wrapper.py     # Layer 6: TLS integration
├── adaptive_controller.py    # Layer 7: Adaptive control
├── circuit_breaker.py        # Layer 7b: Resilience
├── network_simulator.py      # Layer 8: Network sim
├── failure_injector.py       # Layer 8b: Failure injection
│
├── utils.py                  # Utilities & helpers
├── config.py                 # Configuration management
└── compass_artifact_*.md     # Full architecture doc
```

---

## ❓ Frequently Asked Questions

**Q: Do I need quantum hardware?**
A: No! The system uses quantum simulators (Qiskit). Perfect for development.

**Q: Will this work on Windows?**
A: Yes! All code is cross-platform. Some features (Mininet networking) need Linux.

**Q: Can I use just PQC without QKD?**
A: Yes! Set `cukem_mode: "pqc_only"` in config.yml

**Q: How do I enable debug logging?**
A: Set `log_level: "DEBUG"` in config.yml or via environment variable

**Q: What's the security level?**
A: NIST Level 3 (~192-bit post-quantum security from ML-KEM-768)

**Q: Can I modify the demos?**
A: Absolutely! demo.py is fully editable. Try changing n_qubits, noise levels, etc.

---

## 🎓 Learning Path

1. **Start:** Run `python setup.py` then `python demo.py`
2. **Understand:** Read this QUICKSTART.md
3. **Explore:** Read README.md and source files
4. **Customize:** Edit demo.py and config.yml
5. **Master:** Read the full architecture doc (compass_artifact_*.md)

---

**Ready? Let's go!**
```powershell
python setup.py
```

Then:
```powershell
python demo.py
```

Enjoy exploring the future of quantum-safe cryptography! 🚀
