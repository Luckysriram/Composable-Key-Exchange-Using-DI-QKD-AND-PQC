# DI-QKD Simulator - Quick Reference Guide

## 🚀 Getting Started (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Demo (Command-line)
```bash
python demo.py
```
This shows all features in action without needing the web interface.

### Step 3: Start Web Interface
```bash
# Terminal 1: Start backend
python -m backend.api

# Terminal 2: Open frontend
# Navigate to frontend/index.html in your browser
```

---

## 📊 Understanding the Results

### BB84 Metrics

| Metric | Interpretation |
|--------|-----------------|
| **Initial Bits** | Total quantum states sent |
| **Sifted Key** | After basis sifting (~50%) |
| **Final Key** | After error correction |
| **QBER** | Quantum Bit Error Rate |
| **Eve Detected** | Eavesdropping indication |

**QBER Threshold**: 11%
- Below 11%: Channel is secure
- Above 11%: Possible eavesdropping detected

**Sift Efficiency**: ~50% expected
- Alice and Bob match bases roughly half the time

### CHSH Metrics

| Metric | Classical | Quantum |
|--------|-----------|---------|
| **CHSH Value** | ≤ 2.0 | ≤ 2.828 |
| **Bell Inequality** | Cannot violate | Can violate |
| **Entanglement** | Not needed | Required |

**Interpretation**:
- S > 2.0: Bell violation (quantum advantage)
- S > 2.4: Strong entanglement (Strong robustness)
- S > 2.1: Moderate entanglement (Moderate robustness)
- S ≤ 2.0: No Bell violation (classical/separable)

### Security Levels

```
Very High (DI Certified)
  ↓
High
  ↓
Medium
  ↓
Medium-High
  ↓
Low
```

**Achieving "Very High"**:
1. CHSH value > 2.4 (strong violation)
2. QBER < 11% (no Eve)
3. Device-independent certification
4. Key length > 256 bits

---

## 💡 Common Use Cases

### Case 1: Verify BB84 Protocol Works
```python
from backend.bb84 import BB84

bb84 = BB84(key_size=256)
alice_states = bb84.alice_prepare_states()
bob_measurements = bb84.bob_measure_states()
sifted_key = bb84.sift_keys()
print(f"Sifted key: {len(sifted_key)} bits")
```

### Case 2: Test Bell Violation
```python
from backend.chsh import CHSHBellTest
from backend.quantum_simulator import QuantumSimulator

sim = QuantumSimulator()
state = sim.create_bell_pair('phi_plus')

chsh = CHSHBellTest(num_rounds=1000)
measurements = chsh.run_bell_test(state)
stats = chsh.get_statistics()

print(f"CHSH = {stats['chsh_value']:.6f}")
print(f"Violates Bell: {stats['violates_bell']}")
```

### Case 3: Full DI-QKD Simulation
```python
from backend.diqkd_simulator import DIQKDSimulator

sim = DIQKDSimulator(key_size=512, num_chsh_rounds=1000)
results = sim.run_full_simulation()

print(f"Final Key Length: {len(results['combined_key'])}")
print(f"Security: {results['security_certification']['overall_security_level']}")
```

### Case 4: Detect Eavesdropping
```python
from backend.bb84 import BB84

bb84 = BB84(key_size=256)
bb84.alice_prepare_states()
bb84.bob_measure_states()
bb84.sift_keys()

final_key, qber, _ = bb84.error_correction()
eve_stats = bb84.simulate_eve_eavesdropping()

print(f"QBER: {qber*100:.4f}%")
print(f"Eve Detected: {eve_stats['eve_detected']}")
```

---

## 🔧 Configuration Parameters

### In Web UI
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Key Size | 512 | 128-4096 | BB84 bits sent |
| CHSH Rounds | 1000 | 100-10000 | Bell test measurements |
| Bell State | Φ+ | Φ±, Ψ± | Entanglement type |

### In Code
```python
DIQKDSimulator(
    key_size=512,        # Bits for BB84
    num_chsh_rounds=1000 # Measurements for CHSH
)
```

---

## 📈 Optimization Tips

### Get Longer Keys
- Increase `key_size` parameter
- Use multiple rounds of simulation
- Combine multiple independent runs via XOR

### Better Security Certification
- Increase `num_chsh_rounds` for more stable statistics
- Use maximally entangled states (Φ+, Ψ+)
- Run multiple simulations and check consistency

### Faster Results
- Decrease parameters for quick testing
- Run on machine with multiple cores
- Use command-line demo for faster feedback

### Verify No Eavesdropping
- Check QBER < 11%
- Eve detection should show 'NO'
- Run multiple simulations - results should be consistent

---

## 🐛 Debugging Tips

### "CHSH value not high enough"
→ Increase `num_chsh_rounds` for better statistics

### "QBER too high"
→ Simulate channel noise, not an error (re-run)

### "Eve detected in simulation"
→ This is intentional in the Eve eavesdropping simulation

### "Browser won't connect to API"
→ Ensure backend is running: `python -m backend.api`

### "Import errors"
→ Install dependencies: `pip install -r requirements.txt`

---

## 📚 Key Concepts Quick Reference

### BB84 Protocol Flow
```
Alice: Generate random bits & bases
         ↓
Bob: Measure in random bases
         ↓
Public: Compare bases (not bits)
         ↓
Sift: Keep matching bases
         ↓
Error Check: QBER assessment
         ↓
Amplify: Privacy amplification
         ↓
Final Key: Secure key with Eve detection
```

### CHSH Bell Test Flow
```
Generate: Entangled quantum state
         ↓
Measure: Random measurement settings
         ↓
Correlate: Calculate E(a,b) values
         ↓
CHSH: S = |E(00) + E(01) + E(10) - E(11)|
         ↓
Certify: Device-independent security
```

### DI-QKD Combined Flow
```
BB84:  Generate key + check QBER
         ↓
CHSH: Verify entanglement + certify DI
         ↓
Combine: XOR both keys
         ↓
Assess: Security certification
         ↓
Output: Secure DI-QKD key
```

---

## 🎯 Expected Results

### Typical BB84 Run (256 bits)
```
Initial Bits:        256
Sifted Key:          125 (≈50%)
Final Key:           110 (after error correction)
QBER:                1-3% (typical)
Sift Efficiency:     ~49%
Eve Detected:        NO
```

### Typical CHSH Run (1000 rounds, Φ+ state)
```
CHSH Value:          2.70-2.76
Violates Bell:       YES
Violation Margin:    0.70-0.76
Device-Independent:  YES
Security Robustness: Strong
```

### Typical DI-QKD Results
```
BB84 QBER:           < 5%
CHSH Value:          > 2.6
Combined Key:        > 50 bits
Security Level:      High / Very High (DI Certified)
```

---

## 🔐 Security Checklist

Before using key for encryption:

- [ ] QBER < 11%
- [ ] Eve not detected
- [ ] CHSH > 2.0 (Bell violation)
- [ ] Device-independent certified
- [ ] Combined key length > 256 bits
- [ ] No anomalies in multiple runs
- [ ] Security level shows "High" or "Very High"

---

## 📞 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Backend running (`python -m backend.api`)
- [ ] Frontend accessible (frontend/index.html)
- [ ] First simulation completed
- [ ] Results showing secure key generation

---

## 🚦 Next Steps

1. **Run Demo** - See all features: `python demo.py`
2. **Web UI** - Interactive testing: `frontend/index.html`
3. **Customize** - Modify parameters for your use case
4. **Integrate** - Use simulator in your quantum computing project
5. **Extend** - Add new quantum state types or protocols

---

**Need Help?**
- Check README.md for detailed documentation
- Review demo.py for code examples
- Run tests: `python -m pytest test_simulator.py -v`

