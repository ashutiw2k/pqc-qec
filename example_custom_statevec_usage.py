#!/usr/bin/env python
"""Example usage of custom statevector experiment runner.

This script shows how to use the new pqc_experiment_custom_statevec_runner()
which uses the Numba simulator with JAX gradients instead of Pennylane.
"""

from pqcqec.experiment.pqc_experiment import pqc_experiment_custom_statevec_runner

def main():
    """Run a simple PQC experiment with custom statevector simulator."""
    
    print("=" * 80)
    print("PQC EXPERIMENT WITH CUSTOM NUMBA STATEVECTOR SIMULATOR")
    print("=" * 80)
    print()
    
    # Experiment parameters
    params = {
        'num_qubits': 5,
        'num_gates': 50,
        'gate_blocks': 5,
        'pqc_blocks': 1,
        'epochs': 5,
        'num_data': 500,
        'num_test': 100,
        'batch_size': 32,
        'seed': 42,
        'add_uncomputation': True,
        'return_fidelity': False
    }
    
    print("Experiment Parameters:")
    for key, val in params.items():
        print(f"  {key:20s}: {val}")
    print()
    
    # Run experiment
    print("Starting experiment...")
    print("-" * 80)
    
    circuit_ops, circuit_tokens, final_fidelity, pqc_params = \
        pqc_experiment_custom_statevec_runner(**params)
    
    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Final PQC Fidelity: {final_fidelity:.6f}")
    print(f"Number of circuit operations: {len(circuit_ops)}")
    
    # Extract PQC parameters
    pre_angles, theta_zz, post_angles = pqc_params
    print(f"\nPQC Parameters:")
    print(f"  Pre-local angles shape: {pre_angles.shape}")
    print(f"  ZZ angles shape: {theta_zz.shape}")
    print(f"  Post-local angles shape: {post_angles.shape}")
    
    print("\n✓ Experiment completed successfully!")
    print("✓ Custom Numba statevector simulator with JAX gradients is working!")


if __name__ == "__main__":
    main()
