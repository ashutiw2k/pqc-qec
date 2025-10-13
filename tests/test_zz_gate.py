"""
Test the ZZ gate implementation directly.
"""

import jax.numpy as jnp
import numpy as np

from pqcqec.simulate.jax_statevector import (
    create_zero_state, apply_cx, apply_rz, run_circuit_with_state, build_jax_circuit
)


def test_zz_gate():
    """Test if the ZZ gate pattern works correctly."""
    
    print("="*60)
    print("Testing ZZ Gate Implementation")
    print("="*60)
    
    # Create a 2-qubit test state
    n_qubits = 2
    state = create_zero_state(n_qubits)
    
    print(f"\nInitial state |00⟩: {state}")
    
    # Apply H to first qubit to create |+0⟩
    from pqcqec.simulate.jax_statevector import apply_h
    state = apply_h(state, n_qubits, 0)
    print(f"After H on qubit 0: {state}")
    print(f"  State is: (|00⟩ + |10⟩)/√2")
    
    # Now apply ZZ gate with θ=π/4
    theta = jnp.pi / 4
    
    print(f"\nApplying ZZ gate with θ={float(theta):.4f}:")
    print(f"  Pattern: CNOT(0,1) - RZ(1,{float(theta):.4f}) - CNOT(0,1)")
    
    # ZZ pattern
    state = apply_cx(state, n_qubits, 0, 1)  # CNOT
    print(f"  After CNOT(0,1): {state}")
    
    state = apply_rz(state, n_qubits, 1, theta)  # RZ on target
    print(f"  After RZ(1,θ): {state}")
    
    state = apply_cx(state, n_qubits, 0, 1)  # CNOT again
    print(f"  After CNOT(0,1): {state}")
    
    # The state should have changed!
    print(f"\nFinal state norm: {jnp.linalg.norm(state):.6f}")
    
    # Compare with no ZZ (theta=0)
    print(f"\n" + "="*60)
    print("Comparing with θ=0 (no rotation)")
    print("="*60)
    
    state_no_zz = create_zero_state(n_qubits)
    state_no_zz = apply_h(state_no_zz, n_qubits, 0)
    
    state_no_zz = apply_cx(state_no_zz, n_qubits, 0, 1)
    state_no_zz = apply_rz(state_no_zz, n_qubits, 1, 0.0)  # θ=0
    state_no_zz = apply_cx(state_no_zz, n_qubits, 0, 1)
    
    print(f"State with θ=0: {state_no_zz}")
    print(f"State with θ=π/4: {state}")
    
    diff = jnp.linalg.norm(state - state_no_zz)
    print(f"\nDifference: {diff:.6e}")
    
    if diff < 1e-10:
        print("✗ States are identical! ZZ gate is not working!")
    else:
        print("✓ States are different. ZZ gate is working.")
    
    # Test with circuit builder
    print(f"\n" + "="*60)
    print("Testing via build_jax_circuit")
    print("="*60)
    
    # Create circuit operations for ZZ
    circuit_ops = [
        ('h', [0], []),
        ('cnot', [0, 1], []),
        ('rz', [1], [theta]),
        ('cnot', [0, 1], []),
    ]
    
    gate_ids, wire1s, wire2s, thetas = build_jax_circuit(circuit_ops)
    
    print(f"Circuit: {len(gate_ids)} gates")
    print(f"Thetas: {thetas}")
    
    initial_state = create_zero_state(n_qubits)
    final_state = run_circuit_with_state(initial_state, n_qubits, gate_ids, wire1s, wire2s, thetas)
    
    print(f"Final state: {final_state}")
    print(f"Matches manual application: {jnp.allclose(final_state, state)}")
    
    # Now test in a ring (3 qubits)
    print(f"\n" + "="*60)
    print("Testing ZZ Ring (3 qubits)")
    print("="*60)
    
    n_qubits_ring = 3
    theta_vals = jnp.array([0.1, 0.2, 0.3])
    
    # Create a superposition state
    state_ring = create_zero_state(n_qubits_ring)
    for q in range(n_qubits_ring):
        state_ring = apply_h(state_ring, n_qubits_ring, q)
    
    print(f"Initial state (all |+⟩): norm = {jnp.linalg.norm(state_ring):.6f}")
    
    # Apply ZZ ring
    for q in range(n_qubits_ring):
        j = (q + 1) % n_qubits_ring
        theta = theta_vals[q]
        
        state_ring = apply_cx(state_ring, n_qubits_ring, q, j)
        state_ring = apply_rz(state_ring, n_qubits_ring, j, theta)
        state_ring = apply_cx(state_ring, n_qubits_ring, q, j)
    
    print(f"After ZZ ring: norm = {jnp.linalg.norm(state_ring):.6f}")
    
    # Compare with theta=0
    state_ring_zero = create_zero_state(n_qubits_ring)
    for q in range(n_qubits_ring):
        state_ring_zero = apply_h(state_ring_zero, n_qubits_ring, q)
    
    for q in range(n_qubits_ring):
        j = (q + 1) % n_qubits_ring
        state_ring_zero = apply_cx(state_ring_zero, n_qubits_ring, q, j)
        state_ring_zero = apply_rz(state_ring_zero, n_qubits_ring, j, 0.0)
        state_ring_zero = apply_cx(state_ring_zero, n_qubits_ring, q, j)
    
    diff_ring = jnp.linalg.norm(state_ring - state_ring_zero)
    print(f"Difference from θ=0: {diff_ring:.6e}")
    
    if diff_ring < 1e-10:
        print("✗ Ring ZZ has no effect!")
    else:
        print("✓ Ring ZZ changes the state")
    
    print("="*60)


if __name__ == "__main__":
    test_zz_gate()
