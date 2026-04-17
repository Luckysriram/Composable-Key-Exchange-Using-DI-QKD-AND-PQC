#!/usr/bin/env python3
"""
CuKEM Full Working Model Showcase
==================================
Demonstrates the complete 8-layer hybrid PQC-QKD system with
detailed output for each layer, end-to-end flow, and resilience testing.

Run: python -X utf8 full_showcase.py
"""

import sys
import time
import logging
from utils import setup_logging, compute_fingerprint, format_bytes

# Configure logging - only WARNING+ to keep console clean
setup_logging(level="WARNING", log_file="cukem_showcase.log")
logger = logging.getLogger(__name__)

# ── Formatting helpers ──────────────────────────────────────────────

W = 72  # output width

def banner(title, char="="):
    print(f"\n{char * W}")
    print(f"  {title}")
    print(f"{char * W}")

def section(title):
    print(f"\n  --- {title} ---")

def ok(msg):
    print(f"  [OK] {msg}")

def info(msg):
    print(f"       {msg}")

def warn(msg):
    print(f"  [!!] {msg}")

def fail(msg):
    print(f"  [FAIL] {msg}")

def kv(key, value, indent=7):
    print(f"{' ' * indent}{key:.<36s} {value}")

def hexpreview(data: bytes, n=32):
    """Return short hex preview of bytes."""
    h = data.hex()
    return h[:n] + "..." if len(h) > n else h


# ════════════════════════════════════════════════════════════════════
#  LAYER 1 : Post-Quantum KEM (ML-KEM-768 / Kyber768)
# ════════════════════════════════════════════════════════════════════

def showcase_layer1():
    banner("LAYER 1 : Post-Quantum KEM  (ML-KEM-768 / Kyber768)")
    from pqc_kem import MLKEM768

    kem = MLKEM768()
    details = kem.get_details()

    section("Algorithm Details")
    kv("Algorithm", details["algorithm"])
    kv("NIST Security Level", str(details["claimed_nist_level"]))
    kv("Public key size", f"{details['public_key_bytes']} bytes")
    kv("Secret key size", f"{details['secret_key_bytes']} bytes")
    kv("Ciphertext size", f"{details['ciphertext_bytes']} bytes")
    kv("Shared secret size", f"{details['shared_secret_bytes']} bytes")

    section("Key Exchange Flow")
    t0 = time.time()

    # Responder generates keypair
    keypair = kem.generate_keypair()
    t_keygen = (time.time() - t0) * 1000
    ok(f"Responder generated keypair ({t_keygen:.1f} ms)")
    kv("Public key fingerprint", compute_fingerprint(keypair.public_key))
    kv("Secret key fingerprint", compute_fingerprint(keypair.secret_key))

    # Initiator encapsulates
    t0 = time.time()
    encap = kem.encapsulate(keypair.public_key)
    t_encap = (time.time() - t0) * 1000
    ok(f"Initiator encapsulated shared secret ({t_encap:.1f} ms)")
    kv("Ciphertext fingerprint", compute_fingerprint(encap.ciphertext))
    kv("Shared secret (init)", hexpreview(encap.shared_secret))

    # Responder decapsulates
    t0 = time.time()
    decap = kem.decapsulate(encap.ciphertext, keypair.secret_key)
    t_decap = (time.time() - t0) * 1000
    ok(f"Responder decapsulated shared secret ({t_decap:.1f} ms)")
    kv("Shared secret (resp)", hexpreview(decap.shared_secret))

    match = encap.shared_secret == decap.shared_secret
    if match:
        ok("Shared secrets MATCH - key exchange successful")
    else:
        fail("Shared secrets DO NOT match")

    return match


# ════════════════════════════════════════════════════════════════════
#  LAYER 2a : BB84 Quantum Key Distribution
# ════════════════════════════════════════════════════════════════════

def showcase_layer2a():
    banner("LAYER 2a: Quantum Key Distribution  (BB84 Protocol)")
    from bb84_simulator import BB84Simulator

    section("Protocol Parameters")
    n_qubits = 256
    kv("Qubits transmitted", str(n_qubits))
    kv("Expected sifted bits", f"~{n_qubits // 2}")
    kv("Error threshold (QBER)", "11%")
    kv("Noise level", "0.0  (ideal channel)")

    bb84 = BB84Simulator(n_qubits=n_qubits, error_correction_threshold=0.11)

    section("Protocol Execution")
    info("Alice: generating random bits and bases...")
    info("Alice: preparing qubits and transmitting over quantum channel...")
    info("Bob:   measuring qubits in random bases...")
    info("Both:  comparing bases over classical channel (sifting)...")

    t0 = time.time()
    result = bb84.execute_protocol(noise_level=0.0)
    elapsed = (time.time() - t0) * 1000

    if result.success:
        ok(f"BB84 protocol completed ({elapsed:.0f} ms)")
        kv("Sifted key length", f"{result.sifted_key_length} bits")
        kv("Final key length", f"{result.key_length} bytes ({result.key_length * 8} bits)")
        kv("QBER", f"{result.qber:.4f}  (threshold: 0.11)")
        kv("Sifting efficiency", f"{result.metadata['sifting_efficiency']:.2%}")
        kv("Key fingerprint", hexpreview(result.raw_key, 24))
    else:
        fail(f"BB84 failed: {result.error}")

    # Also show a noisy channel run
    section("Noisy Channel Test (5% noise)")
    result_noisy = bb84.execute_protocol(noise_level=0.05)
    if result_noisy.success:
        ok(f"Noisy BB84 succeeded, QBER = {result_noisy.qber:.4f}")
    else:
        warn(f"Noisy BB84 failed: {result_noisy.error}")
        info("(Expected - high noise causes QBER to exceed threshold)")

    return result.success


# ════════════════════════════════════════════════════════════════════
#  LAYER 2b : CHSH Bell Inequality Test
# ════════════════════════════════════════════════════════════════════

def showcase_layer2b():
    banner("LAYER 2b: Entanglement Verification  (CHSH Bell Test)")
    from chsh_bell_test import CHSHBellTest
    import numpy as np

    chsh = CHSHBellTest(n_shots=8192)

    section("Theory")
    kv("Classical bound |S|", "<= 2.0")
    kv("Quantum bound |S|", f"<= 2*sqrt(2) = {2 * np.sqrt(2):.4f}")
    kv("Violation threshold", ">= 2.2")

    section("Measurement Angles")
    kv("Alice a", "0 rad")
    kv("Alice a'", "pi/2 rad")
    kv("Bob   b", "pi/4 rad")
    kv("Bob   b'", "-pi/4 rad")

    section("Running CHSH Test (8192 shots per angle pair)")
    t0 = time.time()
    result = chsh.execute_chsh_test()
    elapsed = (time.time() - t0) * 1000

    if result.success:
        ok(f"CHSH test completed ({elapsed:.0f} ms)")
        kv("S value", f"{result.chsh_value:.4f}")
        for k, v in result.correlations.items():
            kv(f"  {k}", f"{v:.4f}")
        if result.violation:
            ok(f"|S| = {abs(result.chsh_value):.4f} > 2.0 => Bell inequality VIOLATED")
            info("Quantum entanglement confirmed!")
        else:
            warn(f"|S| = {abs(result.chsh_value):.4f} - no violation detected")
            info("(Simulated result - real quantum hardware would show violation)")
    else:
        fail(f"CHSH test failed: {result.error}")

    return result.success


# ════════════════════════════════════════════════════════════════════
#  LAYER 3a : Entropy Estimation (NIST 800-90B)
# ════════════════════════════════════════════════════════════════════

def showcase_layer3a():
    banner("LAYER 3a: Entropy Estimation  (NIST SP 800-90B)")
    from bb84_simulator import BB84Simulator
    from entropy_estimator import EntropyEstimator

    estimator = EntropyEstimator(min_entropy_per_bit=0.8)

    section("Generating Quantum Key Material (2048 qubits)")
    bb84 = BB84Simulator(n_qubits=2048)
    bb84_result = bb84.execute_protocol(noise_level=0.0)

    if not (bb84_result.success and bb84_result.raw_key):
        fail("Could not generate key material for entropy test")
        return False

    data = bb84_result.raw_key
    ok(f"Generated {len(data)} bytes ({len(data) * 8} bits) of quantum key material")

    section("Running 5 Entropy Estimation Methods")
    t0 = time.time()
    result = estimator.estimate_entropy(data)
    elapsed = (time.time() - t0) * 1000

    if result.success:
        ok(f"Entropy estimation completed ({elapsed:.1f} ms)")
        kv("Shannon entropy", f"{result.estimations['shannon_entropy']:.4f} bits/symbol")
        kv("MCV estimate", f"{result.estimations['mcv_estimate']:.4f} bits/symbol")
        kv("Collision estimate (Renyi H2)", f"{result.estimations['collision_estimate']:.4f} bits/symbol")
        kv("Markov estimate", f"{result.estimations['markov_estimate']:.4f} bits/symbol")
        kv("Compression estimate", f"{result.estimations['compression_estimate']:.4f} bits/symbol")
        print()
        kv("Min-entropy (conservative)", f"{result.min_entropy:.4f} bits/symbol")
        kv("Threshold", f"{estimator.min_required:.4f}")
        kv("Sufficient?", "YES" if result.sufficient else "NO")
    else:
        fail(f"Entropy estimation failed: {result.error}")

    return result.success and result.sufficient


# ════════════════════════════════════════════════════════════════════
#  LAYER 3b : Privacy Amplification (SHAKE-256)
# ════════════════════════════════════════════════════════════════════

def showcase_layer3b():
    banner("LAYER 3b: Privacy Amplification  (SHAKE-256)")
    from bb84_simulator import BB84Simulator
    from privacy_amplification import PrivacyAmplifier

    amplifier = PrivacyAmplifier(security_parameter=256)

    section("Concept")
    info("Privacy amplification compresses a partially-secret key")
    info("to a shorter key about which the adversary has negligible info.")
    info("Formula: output_len = min_entropy - adversary_info - security_param")

    section("Generating Key Material")
    bb84 = BB84Simulator(n_qubits=256)
    bb84_result = bb84.execute_protocol(noise_level=0.0)

    if not (bb84_result.success and bb84_result.raw_key):
        fail("Could not generate key material")
        return False

    raw_key = bb84_result.raw_key
    ok(f"Raw quantum key: {len(raw_key)} bytes")

    section("Direct Amplification (fixed output)")
    amp_result = amplifier.amplify(raw_key, output_length=32)
    if amp_result.success:
        ok(f"Amplified: {len(raw_key)}B -> {amp_result.output_length}B")
        kv("Compression ratio", f"{amp_result.compression_ratio:.2f}x")
        kv("Amplified key", hexpreview(amp_result.amplified_key))
    else:
        fail(f"Amplification failed: {amp_result.error}")

    section("SHAKE-256 Properties")
    kv("Hash function", "SHAKE-256 (XOF)")
    kv("Security parameter", "256 bits")
    kv("Mode", "2-universal hash family")

    return amp_result.success if amp_result else False


# ════════════════════════════════════════════════════════════════════
#  LAYER 4 : HKDF Key Combination
# ════════════════════════════════════════════════════════════════════

def showcase_layer4():
    banner("LAYER 4 : HKDF Key Combination  (RFC 5869)")
    from hkdf_combiner import HKDFCombiner
    import secrets

    combiner = HKDFCombiner()

    section("Concept")
    info("HKDF combines PQC key + quantum key into a single hybrid key.")
    info("Security: if EITHER source is secure, the combined key is secure.")
    info("HKDF-Extract(salt, IKM) -> PRK, then HKDF-Expand(PRK, info, L) -> OKM")

    section("Key Combination Demo")
    pqc_key = secrets.token_bytes(32)
    qkd_key = secrets.token_bytes(16)

    kv("PQC key (simulated)", f"{len(pqc_key)}B  {hexpreview(pqc_key)}")
    kv("QKD key (simulated)", f"{len(qkd_key)}B  {hexpreview(qkd_key)}")

    result = combiner.combine_keys(
        pqc_key=pqc_key,
        quantum_key=qkd_key,
        output_length=32,
        info=b"CuKEM-hybrid-key"
    )

    if result.success:
        ok(f"Combined hybrid key: {result.key_length} bytes")
        kv("Hybrid key", hexpreview(result.combined_key))
        kv("Hash algorithm", "SHA-256")
        kv("Info string", "CuKEM-hybrid-key")
    else:
        fail(f"Combination failed: {result.error}")

    section("Multi-Purpose Key Derivation")
    derived = combiner.derive_multiple_keys(
        master_key=result.combined_key,
        key_purposes=["encryption", "authentication", "iv_generation"],
        key_length=32
    )
    for purpose, key in derived.items():
        kv(f"  {purpose}", hexpreview(key))
    ok(f"Derived {len(derived)} purpose-specific keys from master")

    return result.success


# ════════════════════════════════════════════════════════════════════
#  LAYER 5 : CuKEM Unified Interface (End-to-End)
# ════════════════════════════════════════════════════════════════════

def showcase_layer5():
    banner("LAYER 5 : CuKEM Unified Key Exchange  (End-to-End)")
    from cukem import CuKEM, CuKEMConfig, CuKEMMode

    # ── Hybrid mode ──
    section("Mode 1: HYBRID (PQC + QKD)")
    config = CuKEMConfig(
        mode=CuKEMMode.HYBRID,
        n_qubits=256,
        min_entropy=0.8,
        qber_threshold=0.11,
        chsh_verification=False,
        output_key_length=32
    )
    cukem = CuKEM(config)
    keypair = cukem.generate_keypair()

    t0 = time.time()
    result = cukem.initiate_exchange(keypair.public_key, noise_level=0.0)
    elapsed = (time.time() - t0) * 1000

    if result.success:
        ok(f"Hybrid exchange complete ({elapsed:.0f} ms)")
        kv("Final key length", f"{result.key_length} bytes")
        kv("Final key", hexpreview(result.shared_key))
        kv("Fallback used?", "Yes (PQC-only)" if result.fallback_used else "No (true hybrid)")
        if result.warnings:
            for w in result.warnings:
                warn(w)
    else:
        fail(f"Hybrid exchange failed: {result.error}")

    # ── PQC-only mode ──
    section("Mode 2: PQC_ONLY")
    config2 = CuKEMConfig(mode=CuKEMMode.PQC_ONLY, output_key_length=32)
    cukem2 = CuKEM(config2)
    kp2 = cukem2.generate_keypair()
    t0 = time.time()
    r2 = cukem2.initiate_exchange(kp2.public_key)
    t2 = (time.time() - t0) * 1000
    if r2.success:
        ok(f"PQC-only exchange complete ({t2:.0f} ms), key: {r2.key_length}B")
    else:
        fail(f"PQC-only failed: {r2.error}")

    return result.success


# ════════════════════════════════════════════════════════════════════
#  LAYER 6 : TLS 1.3 Integration (PSK)
# ════════════════════════════════════════════════════════════════════

def showcase_layer6():
    banner("LAYER 6 : TLS 1.3 Integration  (PSK Mode)")
    from hybrid_tls_wrapper import HybridTLSWrapper

    section("Concept")
    info("CuKEM-derived keys are injected into TLS 1.3 as Pre-Shared Keys.")
    info("This provides quantum-safe protection at the transport layer.")

    section("TLS Wrapper Configuration")
    tls = HybridTLSWrapper()
    kv("Hostname", "localhost")
    kv("Port", "8443")
    kv("Protocol", "TLS 1.3 with PSK")
    kv("Key source", "CuKEM hybrid key")

    stats = tls.get_statistics()
    kv("Status", stats.get("status", "configured"))

    ok("TLS 1.3 PSK wrapper initialized (network connection not demonstrated)")
    info("In production: CuKEM key -> TLS PSK -> encrypted channel")

    return True


# ════════════════════════════════════════════════════════════════════
#  LAYER 7a : Adaptive Controller & State Machine
# ════════════════════════════════════════════════════════════════════

def showcase_layer7a():
    banner("LAYER 7a: Adaptive Controller  (State Machine)")
    from adaptive_controller import AdaptiveController

    section("State Machine")
    info("States: IDLE -> INITIALIZING -> HYBRID_ACTIVE")
    info("        HYBRID_ACTIVE -> PQC_ONLY  (on QKD failure)")
    info("        PQC_ONLY -> RECOVERING -> HYBRID_ACTIVE  (auto-recovery)")
    info("        Any -> DEGRADED -> FAILED  (cascade)")

    controller = AdaptiveController()

    section("Initialization")
    ok(f"Initial state: {controller.state}")
    success = controller.initialize_system()
    if success:
        ok(f"State after init: {controller.state}")
    else:
        fail("Initialization failed")
        return False

    section("Key Exchange with Health Monitoring")
    result = controller.perform_exchange(role="initiator", noise_level=0.0)
    if result.success:
        ok("Key exchange successful")

    health = controller.get_health_status()
    stats = controller.get_statistics()

    kv("Health status", health.value)
    kv("Current state", stats["state"])
    kv("Total exchanges", str(stats["metrics"]["total_exchanges"]))
    kv("PQC success rate", f"{stats['metrics']['pqc_success_rate']:.0%}")
    kv("Avg QBER", f"{stats['metrics']['avg_qber']:.4f}")
    kv("Avg latency", f"{stats['metrics']['avg_latency_ms']:.1f} ms")
    kv("Consecutive failures", str(stats["consecutive_failures"]))
    kv("Event log entries", str(stats["event_count"]))

    return success


# ════════════════════════════════════════════════════════════════════
#  LAYER 7b : Circuit Breaker Pattern
# ════════════════════════════════════════════════════════════════════

def showcase_layer7b():
    banner("LAYER 7b: Circuit Breaker  (Resilience Pattern)")
    from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState

    section("Concept")
    info("CLOSED  -> normal operation")
    info("OPEN    -> too many failures, fast-reject all calls")
    info("HALF_OPEN -> timeout elapsed, test with limited calls")

    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=2
    )
    breaker = CircuitBreaker(config, name="showcase")

    section("Simulating Unreliable Service")
    call_count = 0

    def unreliable():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("Service unavailable")
        return f"Success (call #{call_count})"

    for i in range(1, 7):
        try:
            result = breaker.call(unreliable)
            ok(f"Call {i}: {result}  [state={breaker.state.value}]")
        except Exception as e:
            warn(f"Call {i}: {e}  [state={breaker.state.value}]")

    section("Circuit Breaker Statistics")
    stats = breaker.get_statistics()
    kv("Final state", stats["state"])
    kv("Total calls", str(stats["stats"]["total_calls"]))
    kv("Total successes", str(stats["stats"]["total_successes"]))
    kv("Total failures", str(stats["stats"]["total_failures"]))
    kv("State changes", str(stats["stats"]["state_changes"]))

    return True


# ════════════════════════════════════════════════════════════════════
#  LAYER 8a : Network Simulator (Mininet)
# ════════════════════════════════════════════════════════════════════

def showcase_layer8a():
    banner("LAYER 8a: Network Simulator  (Mininet Topology)")
    from network_simulator import (
        NetworkSimulator, create_simple_topology, create_complex_topology
    )

    section("Simple Topology")
    simple = create_simple_topology()
    kv("Name", simple.name)
    kv("Nodes", str(len(simple.nodes)))
    for n in simple.nodes:
        kv(f"  {n.name}", f"IP={n.ip}, role={n.role}")
    for l in simple.links:
        kv(f"  {l.node1}<->{l.node2}", f"bw={l.bandwidth}, delay={l.delay}")

    section("Complex Multi-Hop Topology")
    cplx = create_complex_topology()
    kv("Name", cplx.name)
    kv("Nodes", str(len(cplx.nodes)))
    kv("Links", str(len(cplx.links)))
    for n in cplx.nodes:
        kv(f"  {n.name}", f"IP={n.ip}, role={n.role}")

    sim = NetworkSimulator()
    kv("Mininet available", str(sim.mininet_available))
    if not sim.mininet_available:
        info("(Mininet requires Linux - topology defined but not started)")

    ok("Network topologies configured successfully")
    return True


# ════════════════════════════════════════════════════════════════════
#  LAYER 8b : Failure Injection
# ════════════════════════════════════════════════════════════════════

def showcase_layer8b():
    banner("LAYER 8b: Failure Injection  (Chaos Testing)")
    from failure_injector import (
        FailureInjector, FailureScenario, FailureType,
        create_mild_noise_scenario, create_severe_noise_scenario,
        create_network_instability_scenario
    )

    injector = FailureInjector(enabled=True)

    section("Available Failure Types")
    for ft in FailureType:
        kv(f"  {ft.value}", ft.name)

    section("Predefined Scenarios")
    mild = create_mild_noise_scenario()
    kv("Mild noise", f"p={mild.probability}, severity={mild.severity}")
    severe = create_severe_noise_scenario()
    kv("Severe noise", f"p={severe.probability}, severity={severe.severity}")
    net = create_network_instability_scenario()
    kv("Network instability", f"p={net.probability}, severity={net.severity}")

    section("Key Corruption Demo")
    import secrets
    original_key = secrets.token_bytes(32)
    kv("Original key", hexpreview(original_key))
    corrupted = injector.inject_key_corruption(original_key, severity=0.1)
    kv("Corrupted key (10%)", hexpreview(corrupted))

    # Count bit differences
    diff_bits = sum(bin(a ^ b).count('1') for a, b in zip(original_key, corrupted))
    kv("Bits flipped", f"{diff_bits} / {len(original_key) * 8}")

    section("Quantum Noise Injection")
    noise = injector.inject_quantum_noise(severity=0.08)
    kv("Injected noise level", f"{noise:.3f}")

    stats = injector.get_statistics()
    kv("Total injections", str(stats["injection_count"]))

    ok("Failure injection framework operational")
    return True


# ════════════════════════════════════════════════════════════════════
#  FULL END-TO-END FLOW
# ════════════════════════════════════════════════════════════════════

def showcase_end_to_end():
    banner("END-TO-END: Complete CuKEM Hybrid Key Exchange", char="*")
    from pqc_kem import MLKEM768
    from bb84_simulator import BB84Simulator
    from entropy_estimator import EntropyEstimator
    from privacy_amplification import PrivacyAmplifier
    from hkdf_combiner import HKDFCombiner

    print()
    info("This demonstrates the full data flow through all cryptographic layers:")
    info("  PQC Keygen -> Encapsulation -> BB84 QKD -> Entropy Check")
    info("  -> Privacy Amplification -> HKDF Combination -> Final Key")
    print()

    t_start = time.time()

    # Layer 1: PQC
    section("Step 1: Post-Quantum KEM (ML-KEM-768)")
    kem = MLKEM768()
    keypair = kem.generate_keypair()
    encap = kem.encapsulate(keypair.public_key)
    decap = kem.decapsulate(encap.ciphertext, keypair.secret_key)
    assert encap.shared_secret == decap.shared_secret
    pqc_key = encap.shared_secret
    ok(f"PQC shared secret: {len(pqc_key)}B  [{hexpreview(pqc_key)}]")

    # Layer 2: QKD
    section("Step 2: BB84 Quantum Key Distribution")
    bb84 = BB84Simulator(n_qubits=2048)
    bb84_result = bb84.execute_protocol(noise_level=0.0)
    assert bb84_result.success
    ok(f"Quantum raw key:   {len(bb84_result.raw_key)}B  [{hexpreview(bb84_result.raw_key)}]")
    kv("QBER", f"{bb84_result.qber:.4f}")

    # Layer 3a: Entropy
    section("Step 3: Entropy Validation")
    estimator = EntropyEstimator(min_entropy_per_bit=0.8)
    entropy_result = estimator.estimate_entropy(bb84_result.raw_key)
    assert entropy_result.success
    ok(f"Min-entropy: {entropy_result.min_entropy:.4f} bits/symbol  (threshold: 0.8)")

    # Layer 3b: Privacy Amplification
    section("Step 4: Privacy Amplification")
    amplifier = PrivacyAmplifier(security_parameter=256)
    amp_result = amplifier.amplify(bb84_result.raw_key, output_length=32)
    assert amp_result.success
    quantum_key = amp_result.amplified_key
    ok(f"Amplified QKD key: {len(quantum_key)}B  [{hexpreview(quantum_key)}]")

    # Layer 4: HKDF Combination
    section("Step 5: HKDF Key Combination")
    combiner = HKDFCombiner()
    combine_result = combiner.combine_keys(
        pqc_key=pqc_key,
        quantum_key=quantum_key,
        output_length=32,
        info=b"CuKEM-hybrid-key"
    )
    assert combine_result.success
    final_key = combine_result.combined_key

    t_total = (time.time() - t_start) * 1000

    section("Result")
    print()
    kv("PQC component", f"{len(pqc_key)}B from ML-KEM-768")
    kv("QKD component", f"{len(quantum_key)}B from BB84 + SHAKE-256")
    kv("Combination", "HKDF-SHA256(PQC || QKD)")
    print()
    print(f"       {'=' * 56}")
    print(f"       FINAL HYBRID KEY ({len(final_key)} bytes):")
    print(f"       {final_key.hex()}")
    print(f"       {'=' * 56}")
    print()
    ok(f"End-to-end key exchange completed in {t_total:.0f} ms")
    info("This key is quantum-safe: secure if EITHER PQC or QKD holds.")

    return True


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    banner("CuKEM FULL WORKING MODEL SHOWCASE", char="*")
    print("  8-Layer Hybrid PQC-QKD Cryptographic System")
    print("  Post-Quantum Security + Quantum Key Distribution")
    print("*" * W)

    layers = [
        ("Layer 1:  Post-Quantum KEM",        showcase_layer1),
        ("Layer 2a: BB84 QKD",                 showcase_layer2a),
        ("Layer 2b: CHSH Bell Test",           showcase_layer2b),
        ("Layer 3a: Entropy Estimation",       showcase_layer3a),
        ("Layer 3b: Privacy Amplification",    showcase_layer3b),
        ("Layer 4:  HKDF Key Combination",     showcase_layer4),
        ("Layer 5:  CuKEM Unified Exchange",   showcase_layer5),
        ("Layer 6:  TLS 1.3 Integration",      showcase_layer6),
        ("Layer 7a: Adaptive Controller",       showcase_layer7a),
        ("Layer 7b: Circuit Breaker",           showcase_layer7b),
        ("Layer 8a: Network Simulator",         showcase_layer8a),
        ("Layer 8b: Failure Injection",         showcase_layer8b),
        ("End-to-End Hybrid Exchange",          showcase_end_to_end),
    ]

    results = {}
    for name, func in layers:
        try:
            results[name] = func()
        except Exception as e:
            results[name] = False
            fail(f"{name} raised exception: {e}")

    # ── Summary ──
    banner("SHOWCASE SUMMARY", char="*")
    print()
    passed = 0
    for name, ok_flag in results.items():
        status = "[OK]  " if ok_flag else "[FAIL]"
        print(f"  {status} {name}")
        if ok_flag:
            passed += 1

    total = len(results)
    pct = (passed / total) * 100
    print()
    print(f"  Result: {passed}/{total} components demonstrated ({pct:.0f}%)")
    print("*" * W)

    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
