"""
DI-QKD Simulator - Demonstration and Testing Script
Shows how to use the simulator programmatically
"""

from backend.diqkd_simulator import DIQKDSimulator
from backend.bb84 import BB84
from backend.chsh import CHSHBellTest
from backend.quantum_simulator import QuantumSimulator, QuantumState


def demo_basic_bb84():
    """Demonstrate basic BB84 protocol"""
    print("\n" + "="*60)
    print("DEMO 1: Basic BB84 Protocol")
    print("="*60)
    
    bb84 = BB84(key_size=256)
    
    # Step 1: Alice prepares states
    print("\n1. Alice prepares quantum states...")
    states = bb84.alice_prepare_states()
    print(f"   Prepared {len(states)} quantum states")
    print(f"   Sample bases: {bb84.alice_bases[:10]}")
    print(f"   Sample bits: {bb84.alice_bits[:10]}")
    
    # Step 2: Bob measures
    print("\n2. Bob measures states in random bases...")
    measurements = bb84.bob_measure_states()
    print(f"   Measured {len(measurements)} states")
    print(f"   Sample bases: {bb84.bob_bases[:10]}")
    print(f"   Sample results: {measurements[:10]}")
    
    # Step 3: Sift keys
    print("\n3. Sifting keys...")
    sifted = bb84.sift_keys()
    print(f"   Sifted key length: {len(sifted)} bits")
    print(f"   Sift efficiency: {len(sifted)/256*100:.2f}%")
    
    # Step 4: Error correction
    print("\n4. Error correction and QBER assessment...")
    final_key, qber, _ = bb84.error_correction()
    print(f"   Final key length: {len(final_key)} bits")
    print(f"   QBER: {qber*100:.4f}%")
    print(f"   Secure: {'YES' if qber < 0.11 else 'NO'}")
    
    # Step 5: Privacy amplification
    print("\n5. Privacy amplification...")
    amplified = bb84.privacy_amplification(final_key)
    print(f"   Amplified key length: {len(amplified)} bits")
    
    # Eve eavesdropping
    print("\n6. Eve eavesdropping simulation...")
    eve_stats = bb84.simulate_eve_eavesdropping()
    print(f"   Eve detected: {eve_stats['eve_detected']}")
    print(f"   Eve error rate: {eve_stats['eve_error_rate']*100:.4f}%")


def demo_quantum_states():
    """Demonstrate quantum state operations"""
    print("\n" + "="*60)
    print("DEMO 2: Quantum State Operations")
    print("="*60)
    
    # Bell states
    print("\n1. Bell States (Maximally Entangled):")
    for state_name in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
        state = QuantumState.bell_state(state_name)
        corr = state.correlation(num_measurements=100)
        print(f"   {state_name}: correlation = {corr:.4f}")
    
    # Separable states
    print("\n2. Separable States (Non-Entangled):")
    state = QuantumState.product_state('0', '1')
    corr = state.correlation(num_measurements=100)
    print(f"   |0⟩|1⟩: correlation = {corr:.4f}")


def demo_chsh_bell_test():
    """Demonstrate CHSH Bell test"""
    print("\n" + "="*60)
    print("DEMO 3: CHSH Bell Test")
    print("="*60)
    
    # Test with entangled state
    print("\n1. CHSH with Maximally Entangled State:")
    quantum_sim = QuantumSimulator()
    entangled_state = quantum_sim.create_bell_pair('phi_plus')
    
    chsh = CHSHBellTest(num_rounds=1000)
    measurements = chsh.run_bell_test(entangled_state)
    stats = chsh.get_statistics()
    
    print(f"   CHSH Value: {stats['chsh_value']:.6f}")
    print(f"   Violates Bell Inequality: {stats['violates_bell']}")
    print(f"   Violation Margin: {stats['violation_margin']:.6f}")
    print(f"   Quantum Advantage: {stats['quantum_advantage']:.6f}")
    
    # Device-independent certification
    di_cert = chsh.device_independent_certification()
    print(f"   Device-Independent Certified: {di_cert['device_independent']}")
    print(f"   Security Robustness: {di_cert['security_robustness']}")
    print(f"   Certified Key Rate: {di_cert['certified_key_rate']:.6f}")
    
    # Test with separable state
    print("\n2. CHSH with Separable State (Classical):")
    separable_state = quantum_sim.create_separable_pair()
    
    chsh2 = CHSHBellTest(num_rounds=1000)
    measurements2 = chsh2.run_bell_test(separable_state)
    stats2 = chsh2.get_statistics()
    
    print(f"   CHSH Value: {stats2['chsh_value']:.6f}")
    print(f"   Violates Bell Inequality: {stats2['violates_bell']}")


def demo_full_diqkd():
    """Demonstrate full DI-QKD simulator"""
    print("\n" + "="*60)
    print("DEMO 4: Full DI-QKD Simulation")
    print("="*60)
    
    simulator = DIQKDSimulator(key_size=512, num_chsh_rounds=1000)
    results = simulator.run_full_simulation(chsh_state='entangled')
    
    print("\n" + "="*60)
    print("SIMULATION RESULTS SUMMARY")
    print("="*60)
    
    # BB84 results
    bb84_res = results['bb84_results']
    print("\nBB84 Protocol:")
    print(f"  Initial bits: {bb84_res['initial_bits']}")
    print(f"  Sifted key: {bb84_res['sifted_key_length']} bits")
    print(f"  Final key: {bb84_res['final_key_length']} bits")
    print(f"  QBER: {bb84_res['qber']*100:.4f}%")
    print(f"  Eve detected: {bb84_res['eve_detected']}")
    
    # CHSH results
    chsh_res = results['chsh_results']
    print("\nCHSH Bell Test:")
    print(f"  CHSH value: {chsh_res['chsh_value']:.6f}")
    print(f"  Bell violation: {chsh_res['violates_bell']}")
    print(f"  Device-independent certified: {chsh_res['device_independent']}")
    print(f"  Security robustness: {chsh_res['security_robustness']}")
    
    # Security assessment
    sec_cert = results['security_certification']
    print("\nSecurity Assessment:")
    print(f"  Overall security level: {sec_cert['overall_security_level']}")
    print(f"  Device-independent: {sec_cert['device_independent_certified']}")
    print(f"  Combined key length: {sec_cert['key_size']} bits")
    print(f"  Quantum advantage: {sec_cert['quantum_advantage']}")
    
    # Export results
    filename = simulator.export_results()
    print(f"\nResults exported to: {filename}")


def demo_eve_resistance():
    """Demonstrate resistance to Eve eavesdropping"""
    print("\n" + "="*60)
    print("DEMO 5: Eve Eavesdropping Resistance")
    print("="*60)
    
    bb84 = BB84(key_size=512)
    
    # Run protocol
    alice_states = bb84.alice_prepare_states()
    bob_measurements = bb84.bob_measure_states()
    sifted_key = bb84.sift_keys()
    final_key, qber, _ = bb84.error_correction()
    
    print(f"\n1. Without eavesdropping:")
    print(f"   QBER: {qber*100:.4f}%")
    print(f"   Secure: {'YES' if qber < 0.11 else 'NO'}")
    
    # Simulate Eve
    print(f"\n2. With Eve eavesdropping:")
    eve_stats = bb84.simulate_eve_eavesdropping()
    print(f"   Eve detected: {eve_stats['eve_detected']}")
    print(f"   Eve error rate: {eve_stats['eve_error_rate']*100:.4f}%")
    print(f"   Detection threshold: 5%")
    print(f"   Status: {'DETECTED ⚠️' if eve_stats['eve_detected'] else 'NOT DETECTED'}")


def run_all_demos():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "DI-QKD SIMULATOR - DEMONSTRATION SUITE" + " "*10 + "║")
    print("╚" + "="*58 + "╝")
    
    demo_basic_bb84()
    demo_quantum_states()
    demo_chsh_bell_test()
    demo_full_diqkd()
    demo_eve_resistance()
    
    print("\n" + "="*60)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("="*60)


if __name__ == '__main__':
    run_all_demos()
