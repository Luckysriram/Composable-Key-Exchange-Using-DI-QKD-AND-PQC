"""
Comprehensive Test Suite for DI-QKD Simulator
Tests all components to ensure correctness
"""

import pytest
import json
from backend.bb84 import BB84
from backend.chsh import CHSHBellTest
from backend.quantum_simulator import QuantumSimulator, QuantumState
from backend.diqkd_simulator import DIQKDSimulator


class TestBB84:
    """Tests for BB84 protocol"""
    
    def test_initialization(self):
        """Test BB84 initialization"""
        bb84 = BB84(key_size=256)
        assert bb84.key_size == 256
        assert len(bb84.alice_bits) == 0
    
    def test_state_preparation(self):
        """Test Alice state preparation"""
        bb84 = BB84(key_size=128)
        states = bb84.alice_prepare_states()
        
        assert len(states) == 128
        assert len(bb84.alice_bits) == 128
        assert len(bb84.alice_bases) == 128
        assert all(b in [0, 1] for b in bb84.alice_bits)
        assert all(b in ['z', 'x'] for b in bb84.alice_bases)
    
    def test_bob_measurement(self):
        """Test Bob measurements"""
        bb84 = BB84(key_size=128)
        bb84.alice_prepare_states()
        measurements = bb84.bob_measure_states()
        
        assert len(measurements) == 128
        assert len(bb84.bob_bases) == 128
        assert all(m in [0, 1] for m in measurements)
    
    def test_sifting(self):
        """Test key sifting"""
        bb84 = BB84(key_size=256)
        bb84.alice_prepare_states()
        bb84.bob_measure_states()
        sifted = bb84.sift_keys()
        
        # Sifted key should be roughly 50% of original
        assert 0.3 < len(sifted) / 256 < 0.7
        assert all(b in [0, 1] for b in sifted)
    
    def test_qber_calculation(self):
        """Test QBER calculation"""
        bb84 = BB84(key_size=256)
        bb84.alice_prepare_states()
        bb84.bob_measure_states()
        bb84.sift_keys()
        final_key, qber, _ = bb84.error_correction()
        
        # QBER should be between 0 and 1
        assert 0 <= qber <= 1
        # With no eavesdropping, QBER should be low
        assert qber < 0.15  # Allow some measurement noise
    
    def test_privacy_amplification(self):
        """Test privacy amplification"""
        bb84 = BB84(key_size=256)
        key = [0, 1, 0, 1, 1, 0]
        amplified = bb84.privacy_amplification(key, chunk_size=2)
        
        # Amplified key should be smaller
        assert len(amplified) < len(key)
        assert all(b in [0, 1] for b in amplified)
    
    def test_eve_simulation(self):
        """Test Eve eavesdropping simulation"""
        bb84 = BB84(key_size=256)
        bb84.alice_prepare_states()
        bb84.bob_measure_states()
        bb84.sift_keys()
        
        eve_stats = bb84.simulate_eve_eavesdropping()
        
        assert 'eve_detected' in eve_stats
        assert 'eve_error_rate' in eve_stats
        assert 0 <= eve_stats['eve_error_rate'] <= 1
    
    def test_statistics(self):
        """Test statistics generation"""
        bb84 = BB84(key_size=256)
        bb84.alice_prepare_states()
        stats = bb84.get_statistics()
        
        assert 'total_bits_sent' in stats
        assert 'sifted_key_length' in stats
        assert stats['total_bits_sent'] == 256


class TestQuantumState:
    """Tests for Quantum State operations"""
    
    def test_bell_state_creation(self):
        """Test Bell state creation"""
        for state_type in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
            state = QuantumState.bell_state(state_type)
            assert state.amplitudes is not None
            # Check normalization
            assert abs(sum(abs(a)**2 for a in state.amplitudes) - 1.0) < 1e-10
    
    def test_product_state_creation(self):
        """Test separable state creation"""
        state = QuantumState.product_state('0', '1')
        assert state.amplitudes is not None
    
    def test_measurement(self):
        """Test quantum measurement"""
        state = QuantumState.bell_state('phi_plus')
        result_a, result_b = state.measure('z', 'z')
        
        assert result_a in [0, 1]
        assert result_b in [0, 1]
    
    def test_correlation(self):
        """Test correlation measurement"""
        # Entangled state should have high anti-correlation
        state = QuantumState.bell_state('phi_plus')
        corr = state.correlation(num_measurements=100)
        
        # Should be close to -1 for Phi+ state
        assert -1.1 <= corr <= -0.8


class TestCHSH:
    """Tests for CHSH Bell test"""
    
    def test_initialization(self):
        """Test CHSH initialization"""
        chsh = CHSHBellTest(num_rounds=100)
        assert chsh.num_rounds == 100
        assert len(chsh.measurements) == 0
    
    def test_bell_test_with_entanglement(self):
        """Test CHSH with entangled state"""
        quantum_sim = QuantumSimulator()
        state = quantum_sim.create_bell_pair('phi_plus')
        
        chsh = CHSHBellTest(num_rounds=500)
        measurements = chsh.run_bell_test(state)
        
        assert len(measurements) == 500
        assert all(m.alice_output in [0, 1] for m in measurements)
        assert all(m.bob_output in [0, 1] for m in measurements)
    
    def test_chsh_value_calculation(self):
        """Test CHSH value calculation"""
        quantum_sim = QuantumSimulator()
        state = quantum_sim.create_bell_pair('phi_plus')
        
        chsh = CHSHBellTest(num_rounds=1000)
        chsh.run_bell_test(state)
        s_value = chsh.calculate_chsh_value()
        
        # With entanglement, should see Bell violation
        assert s_value > 2.0
        assert s_value <= 2 * 2**0.5 + 0.1  # Allow some numerical error
    
    def test_chsh_with_separable_state(self):
        """Test CHSH with non-entangled state"""
        quantum_sim = QuantumSimulator()
        state = quantum_sim.create_separable_pair()
        
        chsh = CHSHBellTest(num_rounds=500)
        chsh.run_bell_test(state)
        s_value = chsh.calculate_chsh_value()
        
        # Non-entangled should not violate Bell
        assert s_value <= 2.1  # Allow some statistical fluctuation
    
    def test_device_independent_certification(self):
        """Test DI certification"""
        quantum_sim = QuantumSimulator()
        state = quantum_sim.create_bell_pair('phi_plus')
        
        chsh = CHSHBellTest(num_rounds=1000)
        chsh.run_bell_test(state)
        di_cert = chsh.device_independent_certification()
        
        assert 'device_independent' in di_cert
        assert 'certified_key_rate' in di_cert
        assert di_cert['device_independent'] == True


class TestQuantumSimulator:
    """Tests for Quantum Simulator"""
    
    def test_bell_pair_creation(self):
        """Test Bell pair creation"""
        sim = QuantumSimulator()
        state = sim.create_bell_pair('phi_plus')
        assert state is not None
    
    def test_separable_pair_creation(self):
        """Test separable pair creation"""
        sim = QuantumSimulator()
        state = sim.create_separable_pair()
        assert state is not None


class TestDIQKDSimulator:
    """Tests for full DI-QKD Simulator"""
    
    def test_initialization(self):
        """Test simulator initialization"""
        sim = DIQKDSimulator(key_size=256, num_chsh_rounds=500)
        assert sim.key_size == 256
        assert sim.num_chsh_rounds == 500
    
    def test_bb84_execution(self):
        """Test BB84 execution"""
        sim = DIQKDSimulator(key_size=128, num_chsh_rounds=100)
        results = sim.run_bb84_protocol()
        
        assert results['initial_bits'] == 128
        assert 'final_key_length' in results
        assert 'qber' in results
    
    def test_chsh_execution(self):
        """Test CHSH execution"""
        sim = DIQKDSimulator(key_size=128, num_chsh_rounds=200)
        results = sim.run_chsh_bell_test(state_type='entangled')
        
        assert 'chsh_value' in results
        assert 'violates_bell' in results
        assert 'device_independent' in results
    
    def test_full_simulation(self):
        """Test full DI-QKD simulation"""
        sim = DIQKDSimulator(key_size=128, num_chsh_rounds=200)
        results = sim.run_full_simulation(chsh_state='entangled')
        
        assert results['bb84_results'] is not None
        assert results['chsh_results'] is not None
        assert results['security_certification'] is not None
    
    def test_security_assessment(self):
        """Test security assessment"""
        sim = DIQKDSimulator(key_size=128, num_chsh_rounds=200)
        sim.run_full_simulation()
        
        sec_cert = sim.results['security_certification']
        assert 'overall_security_level' in sec_cert
        assert 'device_independent_certified' in sec_cert
        assert 'recommendations' in sec_cert


class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow(self):
        """Test complete workflow"""
        # Initialize
        sim = DIQKDSimulator(key_size=256, num_chsh_rounds=500)
        
        # Run full simulation
        results = sim.run_full_simulation(chsh_state='entangled')
        
        # Verify results structure
        assert results['timestamp'] is not None
        assert results['bb84_results'] is not None
        assert results['chsh_results'] is not None
        assert results['combined_key'] is not None
        assert results['security_certification'] is not None
        
        # Verify key was generated
        assert len(results['combined_key']) > 0
        
        # Verify security certification
        cert = results['security_certification']
        assert cert['overall_security_level'] in [
            'Low',
            'Medium-High',
            'Medium',
            'High',
            'Very High (Device-Independent Certified)'
        ]


def run_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running DI-QKD Simulator Test Suite")
    print("="*60 + "\n")
    
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
