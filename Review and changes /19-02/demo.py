#!/usr/bin/env python3
"""
CuKEM Demo: Hybrid PQC-QKD Key Exchange Demonstration

This script demonstrates the complete CuKEM system including:
- Layer 1: Post-Quantum KEM (ML-KEM-768)
- Layer 2: BB84 Quantum Key Distribution
- Layer 3: Entropy Estimation
- Layer 4: HKDF Key Combination
- Layer 5: CuKEM Abstraction
- Layer 7: Adaptive Controller
"""

import sys
import logging
from config import ConfigManager, SystemConfig
from cukem import CuKEM, CuKEMConfig, CuKEMMode
from pqc_kem import MLKEM768
from bb84_simulator import BB84Simulator
from entropy_estimator import EntropyEstimator
from adaptive_controller import AdaptiveController
from circuit_breaker import CircuitBreaker
from utils import setup_logging, create_summary_report

# Configure logging
setup_logging(level="INFO", log_file="cukem_demo.log")
logger = logging.getLogger(__name__)


def demo_basic_pqc():
    """Demonstrate basic Post-Quantum Cryptography (ML-KEM-768)"""
    print("\n" + "="*70)
    print(" DEMO 1: Post-Quantum Key Exchange (ML-KEM-768)")
    print("="*70)
    
    try:
        kem = MLKEM768()
        logger.info("Initialized ML-KEM-768")
        
        # Generate keypair for responder
        responder_keypair = kem.generate_keypair()
        logger.info(f"Generated responder keypair")
        logger.info(f"  Public key size: {len(responder_keypair.public_key)} bytes")
        logger.info(f"  Secret key size: {len(responder_keypair.secret_key)} bytes")
        
        # Initiator encapsulates shared secret
        encap_result = kem.encapsulate(responder_keypair.public_key)
        if encap_result.success:
            logger.info(f"✓ Encapsulation successful")
            logger.info(f"  Ciphertext size: {len(encap_result.ciphertext)} bytes")
            logger.info(f"  Shared secret size: {len(encap_result.shared_secret)} bytes")
            logger.info(f"  Fingerprint: {encap_result.get_key_fingerprint()}")
        
        # Responder decapsulates shared secret
        decap_result = kem.decapsulate(encap_result.ciphertext, responder_keypair.secret_key)
        if decap_result.success:
            logger.info(f"✓ Decapsulation successful")
            
            # Verify shared secrets match
            if encap_result.shared_secret == decap_result.shared_secret:
                logger.info("✓ Shared secrets match!")
                return True
            else:
                logger.error("✗ Shared secrets don't match!")
                return False
        else:
            logger.error(f"✗ Decapsulation failed: {decap_result.error}")
            return False
            
    except Exception as e:
        logger.error(f"Demo 1 failed: {e}")
        return False


def demo_quantum_key_distribution():
    """Demonstrate BB84 Quantum Key Distribution"""
    print("\n" + "="*70)
    print(" DEMO 2: Quantum Key Distribution (BB84)")
    print("="*70)
    
    try:
        bb84 = BB84Simulator(n_qubits=256, error_correction_threshold=0.11)
        logger.info("Initialized BB84 simulator with 256 qubits")
        
        # Execute BB84 protocol without noise
        logger.info("Executing BB84 protocol (no noise)...")
        result = bb84.execute_protocol(noise_level=0.0)
        
        if result.success:
            logger.info(f"✓ BB84 protocol successful")
            logger.info(f"  Generated key: {result.key_length} bytes")
            logger.info(f"  Sifted key length: {result.sifted_key_length} qubits")
            logger.info(f"  QBER: {result.qber:.4f}")
            logger.info(f"  Sifting efficiency: {result.metadata['sifting_efficiency']:.2%}")
            return True
        else:
            logger.error(f"✗ BB84 failed: {result.error}")
            return False
            
    except Exception as e:
        logger.error(f"Demo 2 failed: {e}")
        return False


def demo_entropy_estimation():
    """Demonstrate Entropy Estimation"""
    print("\n" + "="*70)
    print(" DEMO 3: Entropy Estimation (NIST 800-90B)")
    print("="*70)
    
    try:
        estimator = EntropyEstimator(min_entropy_per_bit=0.8)
        logger.info("Initialized entropy estimator")
        
        # Generate some quantum key material and estimate entropy
        bb84 = BB84Simulator(n_qubits=2048)
        bb84_result = bb84.execute_protocol(noise_level=0.0)

        if bb84_result.success and bb84_result.raw_key:
            logger.info(f"Analyzing entropy of {len(bb84_result.raw_key)} bytes...")
            entropy_result = estimator.estimate_entropy(bb84_result.raw_key)
            
            if entropy_result.success:
                logger.info(f"✓ Entropy estimation complete")
                logger.info(f"  Min-entropy: {entropy_result.min_entropy:.4f} bits/symbol")
                logger.info(f"  Shannon entropy: {entropy_result.estimations['shannon_entropy']:.4f}")
                logger.info(f"  MCV estimate: {entropy_result.estimations['mcv_estimate']:.4f}")
                logger.info(f"  Collision estimate: {entropy_result.estimations['collision_estimate']:.4f}")
                logger.info(f"  Sufficient: {'Yes' if entropy_result.sufficient else 'No'}")
                return entropy_result.sufficient
            else:
                logger.error(f"✗ Entropy estimation failed: {entropy_result.error}")
                return False
        else:
            logger.error("BB84 key generation failed")
            return False
            
    except Exception as e:
        logger.error(f"Demo 3 failed: {e}")
        return False


def demo_cukem_hybrid():
    """Demonstrate CuKEM Hybrid Mode (PQC + QKD)"""
    print("\n" + "="*70)
    print(" DEMO 4: CuKEM Hybrid Mode (PQC + QKD)")
    print("="*70)
    
    try:
        config = CuKEMConfig(
            mode=CuKEMMode.HYBRID,
            n_qubits=256,
            min_entropy=0.8,
            qber_threshold=0.11,
            chsh_verification=False,  # Disable CHSH for faster demo
            output_key_length=32
        )
        
        cukem = CuKEM(config)
        logger.info("Initialized CuKEM in HYBRID mode")
        
        # Responder generates keypair
        responder_keypair = cukem.generate_keypair()
        logger.info("Responder generated keypair")
        
        # Initiator performs key exchange
        logger.info("Initiator performing key exchange...")
        result = cukem.initiate_exchange(
            responder_public_key=responder_keypair.public_key,
            noise_level=0.0
        )
        
        if result.success:
            logger.info(f"✓ Key exchange successful!")
            logger.info(f"  Mode: {result.mode.value}")
            logger.info(f"  Key length: {result.key_length} bytes")
            logger.info(f"  Fallback used: {result.fallback_used}")
            if result.warnings:
                logger.warning(f"  Warnings: {result.warnings}")
            return True
        else:
            logger.error(f"✗ Key exchange failed: {result.error}")
            return False
            
    except Exception as e:
        logger.error(f"Demo 4 failed: {e}")
        return False


def demo_adaptive_controller():
    """Demonstrate Adaptive Controller with Health Monitoring"""
    print("\n" + "="*70)
    print(" DEMO 5: Adaptive Controller & Health Monitoring")
    print("="*70)
    
    try:
        controller = AdaptiveController()
        logger.info("Initialized adaptive controller")
        
        # Initialize system
        logger.info("Initializing CuKEM system...")
        success = controller.initialize_system()
        
        if success:
            logger.info("✓ System initialized")
            
            # Perform a key exchange
            logger.info("Performing key exchange...")
            result = controller.perform_exchange(role="initiator", noise_level=0.0)
            
            if result.success:
                logger.info(f"✓ Key exchange successful")
                
                # Get health status
                health = controller.get_health_status()
                stats = controller.get_statistics()
                
                logger.info(f"  Health status: {health.value}")
                logger.info(f"  Total exchanges: {stats['metrics']['total_exchanges']}")
                logger.info(f"  PQC success rate: {stats['metrics']['pqc_success_rate']:.2%}")
                
                return True
            else:
                logger.error(f"Exchange failed: {result.error}")
                return False
        else:
            logger.error("Failed to initialize system")
            return False
            
    except Exception as e:
        logger.error(f"Demo 5 failed: {e}")
        return False


def demo_circuit_breaker():
    """Demonstrate Circuit Breaker Pattern"""
    print("\n" + "="*70)
    print(" DEMO 6: Circuit Breaker Pattern")
    print("="*70)
    
    try:
        from circuit_breaker import CircuitBreakerConfig
        
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
                logger.info(f"  Attempt {attempt}: ✓ {result}")
            except Exception as e:
                logger.warning(f"  Attempt {attempt}: ✗ {str(e)}")
        
        stats = breaker.get_statistics()
        logger.info(f"✓ Circuit breaker demo complete")
        logger.info(f"  State: {stats['state']}")
        logger.info(f"  Total calls: {stats['stats']['total_calls']}")
        return True
        
    except Exception as e:
        logger.error(f"Demo 6 failed: {e}")
        return False


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" CuKEM HYBRID PQC-QKD SYSTEM - DEMONSTRATION")
    print(" 8-Layer Architecture for Quantum-Safe Communication")
    print("="*70)
    
    logger.info("Starting CuKEM demonstration suite")
    
    # Track results
    results = {
        "Post-Quantum KEM": demo_basic_pqc(),
        "Quantum Key Distribution": demo_quantum_key_distribution(),
        "Entropy Estimation": demo_entropy_estimation(),
        "CuKEM Hybrid Mode": demo_cukem_hybrid(),
        "Adaptive Controller": demo_adaptive_controller(),
        "Circuit Breaker": demo_circuit_breaker(),
    }
    
    # Print summary
    print("\n" + "="*70)
    print(" DEMONSTRATION SUMMARY")
    print("="*70)
    
    summary = create_summary_report(
        {
            "Demo": "Status",
            **{k: ("✓ PASS" if v else "✗ FAIL") for k, v in results.items()}
        },
        title="CuKEM Demo Results"
    )
    print(summary)
    
    # Calculate success rate
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    success_rate = (passed / total) * 100
    
    logger.info(f"\n{'='*70}")
    logger.info(f"DEMO COMPLETE: {passed}/{total} tests passed ({success_rate:.1f}%)")
    logger.info(f"{'='*70}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
