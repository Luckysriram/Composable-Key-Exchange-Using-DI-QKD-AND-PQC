#!/usr/bin/env python3
"""
CuKEM Quick Demo - Works without liboqs
Demonstrates quantum key distribution and entropy estimation
"""

import sys
import logging
from utils import setup_logging, create_summary_report

# Configure logging
setup_logging(level="INFO", log_file="cukem_quick_demo.log")
logger = logging.getLogger(__name__)


def demo_quantum_key_distribution():
    """Demonstrate BB84 Quantum Key Distribution"""
    print("\n" + "="*70)
    print(" DEMO 1: Quantum Key Distribution (BB84)")
    print("="*70)
    
    try:
        from bb84_simulator import BB84Simulator
        
        bb84 = BB84Simulator(n_qubits=256, error_correction_threshold=0.11)
        logger.info("Initialized BB84 simulator with 256 qubits")
        
        # Execute BB84 protocol without noise
        logger.info("Executing BB84 protocol (no noise)...")
        result = bb84.execute_protocol(noise_level=0.0)
        
        if result.success:
            logger.info(f"[OK] BB84 protocol successful")
            logger.info(f"  Generated key: {result.key_length} bytes")
            logger.info(f"  Sifted key length: {result.sifted_key_length} qubits")
            logger.info(f"  QBER: {result.qber:.4f}")
            logger.info(f"  Sifting efficiency: {result.metadata['sifting_efficiency']:.2%}")
            return True
        else:
            logger.error(f"[FAIL] BB84 failed: {result.error}")
            return False
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_entropy_estimation():
    """Demonstrate Entropy Estimation"""
    print("\n" + "="*70)
    print(" DEMO 2: Entropy Estimation (NIST 800-90B)")
    print("="*70)
    
    try:
        from bb84_simulator import BB84Simulator
        from entropy_estimator import EntropyEstimator
        
        estimator = EntropyEstimator(min_entropy_per_bit=0.8)
        logger.info("Initialized entropy estimator")
        
        # Generate some quantum key material and estimate entropy
        # Need at least 125 bytes of data for entropy estimation
        bb84 = BB84Simulator(n_qubits=2048)
        bb84_result = bb84.execute_protocol(noise_level=0.0)
        
        if bb84_result.success and bb84_result.raw_key:
            logger.info(f"Analyzing entropy of {len(bb84_result.raw_key)} bytes...")
            entropy_result = estimator.estimate_entropy(bb84_result.raw_key)
            
            if entropy_result.success:
                logger.info(f"[OK] Entropy estimation complete")
                logger.info(f"  Min-entropy: {entropy_result.min_entropy:.4f} bits/symbol")
                logger.info(f"  Shannon entropy: {entropy_result.estimations['shannon_entropy']:.4f}")
                logger.info(f"  MCV estimate: {entropy_result.estimations['mcv_estimate']:.4f}")
                logger.info(f"  Collision estimate: {entropy_result.estimations['collision_estimate']:.4f}")
                logger.info(f"  Sufficient: {'Yes' if entropy_result.sufficient else 'No'}")
                return entropy_result.sufficient
            else:
                logger.error(f"[FAIL] Entropy estimation failed: {entropy_result.error}")
                return False
        else:
            logger.error("BB84 key generation failed")
            return False
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_entanglement_verification():
    """Demonstrate CHSH Bell Test"""
    print("\n" + "="*70)
    print(" DEMO 3: Bell Inequality Test (CHSH)")
    print("="*70)
    
    try:
        from chsh_bell_test import CHSHBellTest
        
        chsh = CHSHBellTest(n_shots=8192)
        logger.info("Initialized CHSH Bell test")
        
        logger.info("Running CHSH test...")
        result = chsh.execute_chsh_test()
        
        if result.success:
            logger.info(f"✓ CHSH test complete")
            logger.info(f"  CHSH value: {result.chsh_value:.4f}")
            logger.info(f"  Classical bound: {chsh.CLASSICAL_BOUND}")
            logger.info(f"  Quantum bound: {chsh.QUANTUM_BOUND:.3f}")
            logger.info(f"  Violation detected: {result.violation}")
            logger.info(f"  Correlations:")
            for key, val in result.correlations.items():
                logger.info(f"    {key}: {val:.4f}")
            return result.violation
        else:
            logger.error(f"✗ CHSH test failed: {result.error}")
            return False
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_privacy_amplification():
    """Demonstrate Privacy Amplification"""
    print("\n" + "="*70)
    print(" DEMO 4: Privacy Amplification (SHAKE-256)")
    print("="*70)
    
    try:
        from privacy_amplification import PrivacyAmplifier
        from bb84_simulator import BB84Simulator
        
        amplifier = PrivacyAmplifier(security_parameter=256)
        logger.info("Initialized privacy amplifier")
        
        # Generate a key to amplify
        bb84 = BB84Simulator(n_qubits=256)
        bb84_result = bb84.execute_protocol(noise_level=0.0)
        
        if bb84_result.success and bb84_result.raw_key:
            logger.info(f"Amplifying key ({len(bb84_result.raw_key)} bytes)...")
            amp_result = amplifier.amplify(bb84_result.raw_key, output_length=32)
            
            if amp_result.success:
                logger.info(f"✓ Privacy amplification successful")
                logger.info(f"  Input: {len(bb84_result.raw_key)} bytes")
                logger.info(f"  Output: {amp_result.output_length} bytes")
                logger.info(f"  Compression ratio: {amp_result.compression_ratio:.2f}")
                logger.info(f"  Fingerprint: {amp_result.amplified_key.hex()[:32]}...")
                return True
            else:
                logger.error(f"✗ Amplification failed: {amp_result.error}")
                return False
        else:
            logger.error("BB84 key generation failed")
            return False
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_circuit_breaker():
    """Demonstrate Circuit Breaker Pattern"""
    print("\n" + "="*70)
    print(" DEMO 5: Circuit Breaker Pattern")
    print("="*70)
    
    try:
        from circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=5
        )
        
        breaker = CircuitBreaker(config, name="demo_breaker")
        logger.info("Initialized circuit breaker")
        
        # Define a function that sometimes fails
        call_count = 0
        
        def unreliable_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Simulated failure")
            return f"Success on call {call_count}"
        
        # Try calling through circuit breaker
        for attempt in range(1, 6):
            try:
                result = breaker.call(unreliable_function)
                logger.info(f"  Attempt {attempt}: [OK] {result}")
            except Exception as e:
                logger.warning(f"  Attempt {attempt}: [FAIL] {str(e)}")
        
        stats = breaker.get_statistics()
        logger.info(f"[OK] Circuit breaker demo complete")
        logger.info(f"  State: {stats['state']}")
        logger.info(f"  Total calls: {stats['stats']['total_calls']}")
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the quick demo"""
    print("\n" + "="*70)
    print(" CuKEM QUICK DEMO")
    print(" Hybrid PQC-QKD System - Core Functionality")
    print("="*70)
    
    logger.info("Starting CuKEM quick demo")
    
    # Track results
    results = {
        "BB84 QKD": demo_quantum_key_distribution(),
        "Entropy Estimation": demo_entropy_estimation(),
        "Bell Test (CHSH)": demo_entanglement_verification(),
        "Privacy Amplification": demo_privacy_amplification(),
        "Circuit Breaker": demo_circuit_breaker(),
    }
    
    # Print summary
    print("\n" + "="*70)
    print(" DEMO SUMMARY")
    print("="*70)
    
    summary_data = {
        "Demo": "Status",
        **{k: ("✓ PASS" if v else "✗ FAIL") for k, v in results.items()}
    }
    
    summary = create_summary_report(summary_data, title="CuKEM Quick Demo Results")
    print(summary)
    
    # Calculate success rate
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print()
    print(f"{'='*70}")
    print(f"DEMO COMPLETE: {passed}/{total} tests passed ({success_rate:.1f}%)")
    print(f"{'='*70}")
    print("\n📝 Note: Post-Quantum KEM demo skipped (requires liboqs C library)")
    print("   For full demos with PQC, install liboqs from source:")
    print("   https://github.com/open-quantum-safe/liboqs")
    print()
    
    logger.info(f"Demo complete: {passed}/{total} passed")
    return 0 if passed >= 4 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
