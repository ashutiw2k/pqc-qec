"""
Test script for progressive block-by-block training.

This script runs a small-scale experiment to verify that the progressive
training implementation works correctly.
"""

from pqcqec.experiment.pqc_experiment import pqc_experiment_progressive_custom_statevec_runner

def test_progressive_training():
    """Run a small test of progressive block-by-block training."""
    
    print("=" * 80)
    print("Testing Progressive Block-by-Block Training")
    print("=" * 80)
    
    # Small-scale test parameters
    num_qubits = 5
    num_gates = 50  # Will create 1 PQC layers with gate_blocks=10
    gate_blocks = 10
    pqc_blocks = 1
    epochs_per_block = 5  # Small number for quick testing
    num_data = 5000
    num_test = 32
    batch_size = 50
    seed = 42
    
    # Define noise distribution
    noise_dist = {
        'x_rad': 0.1,
        'z_rad': 0.1,
        'delta_x': 0.0,
        'delta_z': 0.0
    }
    
    print(f"\nTest Configuration:")
    print(f"  Qubits: {num_qubits}")
    print(f"  Gates: {num_gates}")
    print(f"  Gate blocks: {gate_blocks}")
    print(f"  PQC blocks: {pqc_blocks}")
    print(f"  Expected PQC layers: {int(pqc_blocks * (num_gates / gate_blocks))}")
    print(f"  Training data: {num_data}")
    print(f"  Test data: {num_test}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs per block: {epochs_per_block}")
    print(f"  Seed: {seed}")
    print()
    
    try:
        # Run progressive training
        circuit_ops, circuit_tokens, mean_fidelity, pqc_params = \
            pqc_experiment_progressive_custom_statevec_runner(
                num_qubits=num_qubits,
                num_gates=num_gates,
                gate_blocks=gate_blocks,
                pqc_blocks=pqc_blocks,
                epochs_per_block=epochs_per_block,
                num_data=num_data,
                num_test=num_test,
                noise_dist=noise_dist,
                gate_dist=None,
                seed=seed,
                batch_size=batch_size,
                return_fidelity=False,
                add_uncomputation=False
            )
        
        print("\n" + "=" * 80)
        print("TEST PASSED!")
        print("=" * 80)
        print(f"\nFinal Results:")
        print(f"  Circuit operations: {len(circuit_ops)}")
        print(f"  Circuit tokens: {len(circuit_tokens)}")
        print(f"  Mean fidelity: {mean_fidelity:.6f}")
        print(f"  PQC parameter keys: {list(pqc_params.keys())}")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_progressive_training()
    exit(0 if success else 1)
