"""
Quick test to verify PyTorch transformer training system integration.
"""

import torch
import json
import tempfile
import os

from pqcqec.experiment.pytorch_transformer_experiment import run_pytorch_transformer_experiment


def test_basic_integration():
    """Test that all components work together."""
    
    print("=" * 60)
    print("Testing PyTorch Transformer Training Integration")
    print("=" * 60)
    
    # Create a tiny synthetic dataset
    temp_dir = tempfile.mkdtemp()
    data_path = os.path.join(temp_dir, "test_circuits.jsonl")
    
    # Create 5 simple circuits (using legacy format)
    circuits = []
    for i in range(5):
        circuit = {
            "idx": i,
            "n_qubits": 3,
            "base_gates": ['h', 'x', 'z', 'h', 'x', 'z'],  # 6 gates = 2 blocks of 3
            "base_qubits": [
                [0, 1, 2, 0, 1, 2],  # wire1s
                [-1, -1, -1, -1, -1, -1],  # wire2s
            ],
        }
        circuits.append(circuit)
    
    # Save to JSONL
    with open(data_path, 'w') as f:
        for circuit in circuits:
            f.write(json.dumps(circuit) + '\n')
    
    print(f"\nCreated test dataset with {len(circuits)} circuits at: {data_path}")
    
    # Test parameters
    n_qubits = 3
    gate_blocks = 3  # 3 gates per PQC block
    k_random = 2  # Use 2 random initial states
    noise_x_rad = 0.01
    noise_z_rad = 0.01
    epochs = 1  # Just 1 epoch for quick test
    
    checkpoint_dir = os.path.join(temp_dir, "checkpoints")
    
    # Test progressive mode
    print("\n" + "=" * 60)
    print("Testing PROGRESSIVE training mode")
    print("=" * 60)
    
    try:
        results_prog = run_pytorch_transformer_experiment(
            data_path=data_path,
            n_qubits=n_qubits,
            gate_blocks=gate_blocks,
            k_random=k_random,
            noise_x_rad=noise_x_rad,
            noise_z_rad=noise_z_rad,
            epochs=epochs,
            batch_size=1,
            learning_rate=1e-4,
            device=torch.device('cpu'),  # Use CPU for testing
            checkpoint_dir=os.path.join(checkpoint_dir, 'progressive'),
            mode='progressive',
            seed=0,
            train_split=0.8,
        )
        
        print("\n✓ Progressive mode test PASSED")
        print(f"  Test fidelity: {results_prog['test_mean_fidelity']:.6f}")
        print(f"  Test loss: {results_prog['test_mean_loss']:.6f}")
        
    except Exception as e:
        print(f"\n✗ Progressive mode test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test individual mode
    print("\n" + "=" * 60)
    print("Testing INDIVIDUAL training mode")
    print("=" * 60)
    
    try:
        results_indiv = run_pytorch_transformer_experiment(
            data_path=data_path,
            n_qubits=n_qubits,
            gate_blocks=gate_blocks,
            k_random=k_random,
            noise_x_rad=noise_x_rad,
            noise_z_rad=noise_z_rad,
            epochs=epochs,
            batch_size=1,
            learning_rate=1e-4,
            device=torch.device('cpu'),
            checkpoint_dir=os.path.join(checkpoint_dir, 'individual'),
            mode='individual',
            seed=0,
            train_split=0.8,
        )
        
        print("\n✓ Individual mode test PASSED")
        print(f"  Test fidelity: {results_indiv['test_mean_fidelity']:.6f}")
        print(f"  Test loss: {results_indiv['test_mean_loss']:.6f}")
        
    except Exception as e:
        print(f"\n✗ Individual mode test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nSystem is ready for training on real datasets.")
    print("\nNext steps:")
    print("1. Prepare your circuit dataset in JSON/JSONL format")
    print("2. Run run_pytorch_transformer_experiment() with your data")
    print("3. Choose 'progressive' or 'individual' training mode")
    print("4. Monitor fidelity improvement during training")
    
    return True


if __name__ == "__main__":
    success = test_basic_integration()
    exit(0 if success else 1)
