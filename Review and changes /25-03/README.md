# Device-Independent Quantum Key Distribution (DI-QKD) Simulator

A comprehensive simulator combining **BB84** (Bennett-Brassard 1984) and **CHSH** (Clauser-Horne-Shimony-Holt) Bell test for device-independent quantum key distribution.

## 🎯 Features

- **BB84 Protocol**: Full implementation with basis sifting, error correction, and privacy amplification
- **CHSH Bell Test**: Quantum entanglement verification and device-independent certification
- **Quantum State Simulation**: Bell states, separable states, and measurement simulation
- **Eve Eavesdropping Detection**: Quantum bit error rate (QBER) monitoring
- **Web-based Frontend**: Interactive UI for running simulations
- **REST API Backend**: Flask-based API for protocol execution
- **Security Certification**: Device-independent security assessment

## 📋 Project Structure

```
code-2/
├── backend/
│   ├── quantum_simulator.py      # Quantum state simulation
│   ├── bb84.py                   # BB84 protocol implementation
│   ├── chsh.py                   # CHSH Bell test implementation
│   ├── diqkd_simulator.py        # Main DI-QKD simulator
│   └── api.py                    # Flask REST API
├── frontend/
│   ├── index.html                # Web UI
│   └── app.js                    # Frontend JavaScript
├── ml_kem.py                     # ML-KEM post-quantum crypto (optional)
├── ml_kem_768.py                 # ML-KEM-768 implementation (optional)
├── demo.py                       # Demonstration script
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Modern web browser

### Installation

1. **Clone or navigate to the project directory**
```bash
cd c:\Users\LENOVO\Desktop\code-2
```

2. **Create virtual environment** (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or: source venv/bin/activate  # On Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Web Interface (Recommended)

1. **Start the Flask backend**
```bash
python -m backend.api
```
The server will start on `http://localhost:5000`

2. **Open the web UI**
- Navigate to `frontend/index.html` in your browser
- Or serve with a local server:
```bash
# Using Python
python -m http.server 8000 --directory frontend
```
- Visit `http://localhost:8000` in your browser

#### Option 2: Command-line Demo

Run the demonstration script:
```bash
python demo.py
```

This runs:
- Basic BB84 protocol demo
- Quantum state operations
- CHSH Bell test
- Full DI-QKD simulation
- Eve eavesdropping resistance test

## 📚 Protocol Overview

### BB84 (Bennett-Brassard 1984)

**Purpose**: Secure quantum key distribution with eavesdropping detection

**Steps**:
1. **Preparation**: Alice prepares random bits in random bases (Z or X)
2. **Transmission**: Quantum states sent to Bob
3. **Measurement**: Bob measures in random bases
4. **Sifting**: Alice and Bob publicly compare bases (not bits)
5. **Error Correction**: QBER assessment for eavesdropping detection
6. **Privacy Amplification**: XOR-based key enhancement

**Security**: 
- QBER threshold: ~11% indicates no eavesdropping
- Sift efficiency: ~50% (only matching bases kept)

### CHSH Bell Test

**Purpose**: Device-independent security certification using quantum entanglement

**Principle**: Tests violation of classical inequalities
- Classical bound: S ≤ 2
- Quantum maximum: S ≤ 2√2 ≈ 2.828
- Violation indicates genuine quantum entanglement

**Formula**:
```
S = |E(0,0) + E(0,1) + E(1,0) - E(1,1)|
```

**Device-Independence**:
- Bell violation proves quantum advantage
- Does not require trust in measurement devices
- Provides bounds on extractable key rate

### DI-QKD Integration

Combines both protocols:
1. **BB84** for practical key distribution
2. **CHSH** for device-independent certification
3. **Key combination** via XOR of both keys
4. **Security assessment** merging both metrics

## 🔧 API Endpoints

### Simulator Management
- `POST /api/initialize` - Initialize simulator with parameters
- `POST /api/reset` - Reset simulator for new run

### Protocol Execution
- `POST /api/run_bb84` - Run BB84 protocol
- `POST /api/run_chsh` - Run CHSH Bell test
- `POST /api/run_full_simulation` - Run complete DI-QKD

### Data Retrieval
- `GET /api/get_execution_log` - Get execution log
- `GET /api/export_results` - Export results to JSON
- `GET /health` - Health check

### Detailed Analysis
- `POST /api/bb84_detailed` - Detailed BB84 step-by-step
- `POST /api/chsh_detailed` - Detailed CHSH measurements
- `POST /api/bell_state_test` - Test Bell state properties

## 💻 Code Examples

### Using the Simulator Programmatically

```python
from backend.diqkd_simulator import DIQKDSimulator

# Create simulator
simulator = DIQKDSimulator(key_size=512, num_chsh_rounds=1000)

# Run full simulation
results = simulator.run_full_simulation(chsh_state='entangled')

# Access results
print("BB84 QBER:", results['bb84_results']['qber'])
print("CHSH Value:", results['chsh_results']['chsh_value'])
print("Security Level:", results['security_certification']['overall_security_level'])

# Export results
simulator.export_results('simulation_results.json')
```

### Using BB84 Protocol

```python
from backend.bb84 import BB84

bb84 = BB84(key_size=256)

# Execute protocol
alice_states = bb84.alice_prepare_states()
bob_measurements = bb84.bob_measure_states()
sifted_key = bb84.sift_keys()

# Check security
final_key, qber, test_positions = bb84.error_correction()
print(f"QBER: {qber:.4f}")
print(f"Secure: {'YES' if qber < 0.11 else 'NO'}")

# Privacy amplification
amplified = bb84.privacy_amplification(final_key)
```

### Using CHSH Bell Test

```python
from backend.chsh import CHSHBellTest
from backend.quantum_simulator import QuantumSimulator

# Create entangled state
quantum_sim = QuantumSimulator()
state = quantum_sim.create_bell_pair('phi_plus')

# Run CHSH test
chsh = CHSHBellTest(num_rounds=1000)
measurements = chsh.run_bell_test(state)
stats = chsh.get_statistics()

# Check device independence
print(f"CHSH Value: {stats['chsh_value']:.6f}")
print(f"Bell Violation: {stats['violates_bell']}")

# Get certification
di_cert = chsh.device_independent_certification()
print(f"Device-Independent: {di_cert['device_independent']}")
```

## 📊 Parameters

### Simulator Configuration
- **key_size**: Bits to transmit in BB84 (default: 512)
- **num_chsh_rounds**: Number of CHSH measurements (default: 1000)
- **bell_state**: Type of Bell state (phi_plus, phi_minus, psi_plus, psi_minus)

### Security Thresholds
- **QBER threshold**: 11% (BB84 security limit)
- **Bell threshold**: 2.0 (classical bound)
- **Quantum maximum**: 2.828 (≈2√2)

## 🔐 Security Considerations

### BB84 Security
- **Eve Detection**: Eavesdropping causes QBER increase
- **Threshold**: If QBER > 11%, channel compromised
- **Eve Information**: Eavesdropping limited to ~25% of key

### CHSH Security
- **Device Independence**: No assumption about device internals
- **Entanglement Requirement**: Bell violation proves quantum advantage
- **Robustness Levels**:
  - Strong: S > 2.4
  - Moderate: S > 2.1
  - Weak: S ≤ 2.1

### Combined Security
- Key is secure if:
  - BB84: QBER < 11% and no Eve detected
  - CHSH: Bell violation (S > 2.0) and device-independent certified
  - Both conditions met → "Very High" security level

## 📈 Output Interpretation

### Key Metrics

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| QBER | Quantum Bit Error Rate | < 11% |
| CHSH Value | Bell test statistic | > 2.0 for violation |
| Sift Efficiency | Ratio of sifted to sent bits | ~50% |
| Key Length | Final usable key bits | > 256 |
| Eve Detected | Eavesdropping indication | No |

### Security Levels

| Level | Condition |
|-------|-----------|
| Very High (DI Certified) | CHSH > 2.4, QBER < 11%, no Eve |
| High | CHSH > 2.1, QBER < 11% |
| Medium | CHSH > 2.0, QBER < 13% |
| Medium-High | QBER < 11%, no DI cert |
| Low | QBER > 11% or CHSH ≤ 2 |

## 🧪 Testing

Run the comprehensive demo suite:
```bash
python demo.py
```

Includes:
- Basic BB84 protocol test
- Quantum state operations
- CHSH Bell test comparison
- Full DI-QKD simulation
- Eve eavesdropping detection

## 📖 References

- Bennett, C. H., & Brassard, G. (1984). "Quantum cryptography: Public key distribution and coin tossing"
- Clauser, J. F., et al. (1969). "Proposed Experiment to Test Local Hidden-Variable Theories"
- Arnon-Friedman, R., et al. (2021). "Simple and Low-Noise Measurement-Device-Independent Quantum Random-Number Generation"
- FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism Standard

## 🛠️ Troubleshooting

### API Connection Issues
```
Error: Cannot connect to API
Solution: Ensure Flask backend is running (python -m backend.api)
```

### Import Errors
```
Error: ModuleNotFoundError
Solution: Install dependencies (pip install -r requirements.txt)
```

### Port Already in Use
```
Error: Address already in use
Solution: Change port in api.py or kill process using port 5000
```

## 📝 License

This project is for educational and research purposes.

## 👨‍💻 Contributing

For improvements and bug reports, please feel free to modify and extend the simulator.

## 📧 Support

For questions and issues, refer to the inline documentation in source files.

---

**Last Updated**: 2026-03-25
**Version**: 1.0
**Status**: Production Ready
