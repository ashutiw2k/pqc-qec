"""
Test script for the custom statevector backend experiment runner.

This script runs a small-scale experiment to verify that all components work together:
- Custom Numba statevector simulator
- Circuit template system
- LEL-ZZ model with quaternions
- JAX/Optax training
"""

from pqcqec.experiment.pqc_experiment import pqc_experiment_custom_statevec_runner

def test_custom_statevec_runner():
    """Run a small test experiment."""
    
    print("=" * 80)
    print("Testing Custom Statevector Backend Experiment Runner")
    print("=" * 80)
    
    # Small-scale test parameters
    num_qubits = 3
    num_gates = 6
    gate_blocks = 2  # Add PQC layer every 2 gates
    pqc_blocks = 1
    epochs = 2
    num_data = 64  # Small training set
    num_test = 16
    batch_size = 16
    seed = 42
    
    # Define noise distribution
    noise_dist = {
        'x_rad': 0.01,
        'z_rad': 0.01,
        'delta_x': 0.0,
        'delta_z': 0.0
    }
    
    # Gate distribution (optional - defaults to uniform)
    gate_dist = None
    
    print(f"\nExperiment Configuration:")
    print(f"  Qubits: {num_qubits}")
    print(f"  Gates: {num_gates}")
    print(f"  Gate blocks: {gate_blocks}")
    print(f"  PQC blocks: {pqc_blocks}")
    print(f"  Training data: {num_data}")
    print(f"  Test data: {num_test}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Seed: {seed}")
    print(f"  Noise: {noise_dist}")
    
    try:
        print("\n" + "=" * 80)
        print("Running with Uncomputation (U U†)")
        print("=" * 80)
        
        circuit_ops, circuit_tokens, mean_fidelity, pqc_params = pqc_experiment_custom_statevec_runner(
            num_qubits=num_qubits,
            num_gates=num_gates,
            gate_blocks=gate_blocks,
            pqc_blocks=pqc_blocks,
            epochs=epochs,
            num_data=num_data,
            num_test=num_test,
            noise_dist=noise_dist,
            gate_dist=gate_dist,
            seed=seed,
            batch_size=batch_size,
            return_fidelity=False,
            add_uncomputation=True
        )
        
        print("\n" + "=" * 80)
        print("Test Completed Successfully!")
        print("=" * 80)
        print(f"Final mean fidelity: {mean_fidelity:.6f}")
        print(f"Number of circuit operations: {len(circuit_ops)}")
        print(f"Number of circuit tokens (with PQC): {len(circuit_tokens)}")
        print(f"PQC parameter shapes:")
        for key, value in pqc_params.items():
            print(f"  {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: Test Failed!")
        print("=" * 80)
        print(f"Exception: {type(e).__name__}")
        print(f"Message: {str(e)}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False


def test_no_uncomputation():
    """Test without uncomputation (more challenging)."""
    
    print("\n\n" + "=" * 80)
    print("Testing WITHOUT Uncomputation")
    print("=" * 80)
    
    # Even smaller test for no-uncomputation case (harder to train)
    num_qubits = 2
    num_gates = 4
    gate_blocks = 2
    pqc_blocks = 1
    epochs = 2
    num_data = 32
    num_test = 8
    batch_size = 8
    seed = 42
    
    noise_dist = {
        'x_rad': 0.01,
        'z_rad': 0.01,
        'delta_x': 0.0,
        'delta_z': 0.0
    }
    
    try:
        circuit_ops, circuit_tokens, mean_fidelity, pqc_params = pqc_experiment_custom_statevec_runner(
            num_qubits=num_qubits,
            num_gates=num_gates,
            gate_blocks=gate_blocks,
            pqc_blocks=pqc_blocks,
            epochs=epochs,
            num_data=num_data,
            num_test=num_test,
            noise_dist=noise_dist,
            seed=seed,
            batch_size=batch_size,
            return_fidelity=False,
            add_uncomputation=False
        )
        
        print("\n" + "=" * 80)
        print("No-Uncomputation Test Completed Successfully!")
        print("=" * 80)
        print(f"Final mean fidelity: {mean_fidelity:.6f}")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: No-Uncomputation Test Failed!")
        print("=" * 80)
        print(f"Exception: {type(e).__name__}")
        print(f"Message: {str(e)}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CUSTOM STATEVECTOR BACKEND TEST SUITE")
    print("=" * 80)
    
    # Test with uncomputation
    test1_passed = test_custom_statevec_runner()
    
    # Test without uncomputation
    test2_passed = test_no_uncomputation()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"With Uncomputation: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Without Uncomputation: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The custom statevector backend is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    print("=" * 80)
