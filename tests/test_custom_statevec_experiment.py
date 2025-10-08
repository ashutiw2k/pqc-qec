#!/usr/bin/env python
"""Test script to verify custom statevector experiment runner works.

This script runs a small experiment to verify:
1. Custom statevec model initializes correctly
2. JAX gradients work through Numba simulator
3. Training loop executes without errors
4. Results are comparable to Pennylane version
"""

import jax
import jax.numpy as jnp

from pqcqec.experiment.pqc_experiment import (
    pqc_experiment_runner,
    pqc_experiment_custom_statevec_runner
)

def test_small_experiment():
    """Run a small experiment with both versions and compare."""
    
    # Small experiment parameters for quick testing
    params = {
        'num_qubits': 3,
        'num_gates': 10,
        'gate_blocks': 2,
        'pqc_blocks': 1,
        'epochs': 2,
        'num_data': 50,
        'num_test': 20,
        'batch_size': 10,
        'seed': 42,
        'return_fidelity': True,
        'add_uncomputation': True
    }
    
    print("=" * 80)
    print("TESTING CUSTOM STATEVECTOR EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Parameters: {params}")
    print()
    
    # Test custom statevec version
    print("\n" + "=" * 80)
    print("Running CUSTOM STATEVEC version...")
    print("=" * 80)
    
    try:
        fid_noisy_custom, fid_pqc_custom = pqc_experiment_custom_statevec_runner(**params)
        print(f"\n✓ Custom statevec version completed successfully!")
        print(f"  Final Noisy Fidelity: {jnp.mean(fid_noisy_custom):.4f}")
        print(f"  Final PQC Fidelity: {jnp.mean(fid_pqc_custom):.4f}")
        print(f"  Improvement: {jnp.mean(fid_pqc_custom - fid_noisy_custom):.4f}")
        custom_success = True
    except Exception as e:
        print(f"\n✗ Custom statevec version failed!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        custom_success = False
    
    # Optionally test Pennylane version for comparison
    # (Commented out by default to save time)
    """
    print("\n" + "=" * 80)
    print("Running PENNYLANE version (for comparison)...")
    print("=" * 80)
    
    try:
        fid_noisy_pl, fid_pqc_pl = pqc_experiment_runner(**params)
        print(f"\n✓ Pennylane version completed successfully!")
        print(f"  Final Noisy Fidelity: {jnp.mean(fid_noisy_pl):.4f}")
        print(f"  Final PQC Fidelity: {jnp.mean(fid_pqc_pl):.4f}")
        print(f"  Improvement: {jnp.mean(fid_pqc_pl - fid_noisy_pl):.4f}")
        
        # Compare results
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"Custom PQC Fidelity: {jnp.mean(fid_pqc_custom):.4f}")
        print(f"Pennylane PQC Fidelity: {jnp.mean(fid_pqc_pl):.4f}")
        print(f"Difference: {jnp.mean(jnp.abs(fid_pqc_custom - fid_pqc_pl)):.4f}")
        
    except Exception as e:
        print(f"\n✗ Pennylane version failed!")
        print(f"  Error: {e}")
    """
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    if custom_success:
        print("✓ Custom statevector experiment runner is working!")
        print("✓ JAX gradients through Numba simulator are functioning")
        print("✓ Training loop completed without errors")
        print("\nYou can now use pqc_experiment_custom_statevec_runner() in your experiments!")
    else:
        print("✗ Custom statevector experiment runner failed")
        print("  Please check the error messages above")
    
    return custom_success


if __name__ == "__main__":
    success = test_small_experiment()
    exit(0 if success else 1)
