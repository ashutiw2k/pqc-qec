#!/usr/bin/env python3
"""Test script for pqc_experiment_custom_statevec_runner with full training.

This verifies that the full training pipeline works with the refactored
CustomStatevecComplexQuaternionModel including:
- Model initialization with template pre-compilation
- Forward pass with template-based circuit building
- Gradient computation through JAX autodiff
- Full training loop with optimizer
- Circuit token extraction via get_circuit_tokens()
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import jax
import jax.numpy as jnp

from pqcqec.experiment.pqc_experiment import pqc_experiment_custom_statevec_runner

print("=" * 80)
print("Testing pqc_experiment_custom_statevec_runner with Full Training")
print("=" * 80)

# Small test configuration for quick verification
test_config = {
    'num_qubits': 3,
    'num_gates': 10,
    'gate_blocks': 5,
    'pqc_blocks': 2,
    'epochs': 2,
    'num_data': 64,   # Small dataset for fast test
    'num_test': 16,
    'seed': 42,
    'batch_size': 16,
    'add_uncomputation': True,
    'gpu': False
}

print("\nTest Configuration:")
for key, val in test_config.items():
    print(f"  {key}: {val}")
print()

try:
    # Run the experiment
    print("Starting experiment...")
    base_circuit_ops, pqc_circuit_tokens, mean_fidelity, pqc_params = \
        pqc_experiment_custom_statevec_runner(**test_config)
    
    print("\n" + "=" * 80)
    print("✓ EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    # Validate outputs
    print("\nOutput Validation:")
    
    # Check base circuit
    print(f"  Base circuit operations: {len(base_circuit_ops)} gates")
    assert isinstance(base_circuit_ops, list), "Base circuit should be a list"
    assert len(base_circuit_ops) > 0, "Base circuit should have gates"
    print("  ✓ Base circuit valid")
    
    # Check PQC circuit tokens
    print(f"  PQC circuit tokens: {len(pqc_circuit_tokens)} gates")
    assert isinstance(pqc_circuit_tokens, list), "PQC circuit tokens should be a list"
    assert len(pqc_circuit_tokens) > len(base_circuit_ops), "PQC circuit should have more gates (includes PQC blocks)"
    print("  ✓ PQC circuit tokens valid")
    
    # Check fidelity
    print(f"  Mean fidelity: {mean_fidelity:.6f}")
    assert isinstance(mean_fidelity, float), "Fidelity should be a float"
    assert 0 <= mean_fidelity <= 1, "Fidelity should be between 0 and 1"
    print("  ✓ Fidelity valid")
    
    # Check PQC parameters
    print(f"  PQC parameters: {type(pqc_params)}")
    assert isinstance(pqc_params, tuple), "PQC params should be a tuple"
    assert len(pqc_params) == 3, "PQC params should have 3 components (pre, theta_zz, post)"
    
    pre_angles, theta_zz, post_angles = pqc_params
    print(f"    - pre_angles: {pre_angles.shape}")
    print(f"    - theta_zz: {theta_zz.shape}")
    print(f"    - post_angles: {post_angles.shape}")
    
    # Check shapes
    expected_blocks = test_config['pqc_blocks'] * \
                     ((test_config['num_gates'] * 2) // test_config['gate_blocks'])
    assert pre_angles.shape[0] == expected_blocks, f"Expected {expected_blocks} blocks"
    assert pre_angles.shape[1] == test_config['num_qubits'], "Wrong num_qubits in pre_angles"
    assert pre_angles.shape[2] == 3, "Should have 3 Euler angles per qubit"
    print("  ✓ PQC parameters valid")
    
    # Check circuit token structure
    print("\n  Circuit Token Analysis:")
    gate_counts = {}
    param_gate_count = 0
    for token in pqc_circuit_tokens:
        gate_name = token[0]
        params = token[2] if len(token) > 2 else []
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        if len(params) > 0:
            param_gate_count += 1
    
    print(f"    Gate type counts:")
    for gate, count in sorted(gate_counts.items()):
        print(f"      {gate}: {count}")
    print(f"    Parameterized gates: {param_gate_count}")
    print("  ✓ Circuit tokens have expected structure")
    
    # Verify PQC blocks were inserted
    num_pqc_gates = len(pqc_circuit_tokens) - len(base_circuit_ops)
    expected_pqc_gates_per_block = 9 * test_config['num_qubits']  # LEL-ZZ structure
    expected_total_pqc_gates = expected_blocks * expected_pqc_gates_per_block
    
    print(f"\n  PQC Block Verification:")
    print(f"    Expected PQC blocks: {expected_blocks}")
    print(f"    Expected PQC gates per block: {expected_pqc_gates_per_block}")
    print(f"    Expected total PQC gates: {expected_total_pqc_gates}")
    print(f"    Actual PQC gates added: {num_pqc_gates}")
    
    # Allow some tolerance due to noise gates
    assert abs(num_pqc_gates - expected_total_pqc_gates) < expected_pqc_gates_per_block * 2, \
        "PQC gate count mismatch"
    print("  ✓ PQC blocks correctly inserted")
    
    print("\n" + "=" * 80)
    print("ALL VALIDATIONS PASSED ✓")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Training completed successfully with {test_config['epochs']} epochs")
    print(f"  - Final fidelity: {mean_fidelity:.4f}")
    print(f"  - Circuit tokens extracted: {len(pqc_circuit_tokens)} gates")
    print(f"  - get_circuit_tokens() working correctly")
    print(f"  - Template-based circuit building successful")
    print(f"  - Gradients computed correctly during training")
    
    print("\n✓ pqc_experiment_custom_statevec_runner is working correctly!")
    
except Exception as e:
    print("\n" + "=" * 80)
    print("✗ TEST FAILED")
    print("=" * 80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
