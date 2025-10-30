"""
Test what happens when theta_zz values are all the same (like [0,0,0]).
"""

import jax
import jax.numpy as jnp
import numpy as np

from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit
from pqcqec.models.pqc_models import LELZZInterleavedQuaternionCustomStatevecModel
from pqcqec.simulate.simulate import get_input_data


def test_identical_theta_values():
    """Test gradient flow when all theta_zz values are identical."""
    
    print("="*60)
    print("Testing Identical Theta_ZZ Values")
    print("="*60)
    
    # Setup
    num_qubits = 3
    num_gates = 5
    gate_blocks = 5
    pqc_blocks = 1
    seed = 42
    
    # Generate circuit
    qiskit_circuit = generate_random_circuit(
        num_qubits=num_qubits,
        num_gates=num_gates,
        seed=seed
    )
    
    qiskit_uncomp = qiskit_circuit.compose(qiskit_circuit.inverse())
    circuit_ops = tokenize_qiskit_circuit(qiskit_uncomp)
    
    # Create noise
    x_noise = np.random.uniform(0.05, 0.15, (len(circuit_ops),)).astype(np.float32)
    z_noise = np.random.uniform(0.05, 0.15, (len(circuit_ops),)).astype(np.float32)
    
    # Initialize model
    model = LELZZInterleavedQuaternionCustomStatevecModel(
        base_circuit_ops=circuit_ops,
        num_qubits=num_qubits,
        x_noise=x_noise,
        z_noise=z_noise,
        pqc_blocks=pqc_blocks,
        gate_blocks=gate_blocks,
        seed=seed
    )
    
    # Get parameters
    params = model.get_model_params_to_store()
    
    # Create test input
    test_input = get_input_data(num_qubits, 2, seed=seed)
    
    # Test 1: All zeros (default initialization)
    print(f"\nTest 1: theta_zz = [0, 0, 0]")
    print("="*40)
    
    theta_zeros = jnp.array([0.0, 0.0, 0.0])
    
    def loss_fn_zeros(theta):
        output = model.run_model_batch(test_input, params['pre_quaternions'], theta, params['post_quaternions'])
        return jnp.sum(jnp.abs(output)**2)
    
    loss_zeros = loss_fn_zeros(theta_zeros)
    grad_zeros = jax.grad(loss_fn_zeros)(theta_zeros)
    
    print(f"Loss: {loss_zeros:.6f}")
    print(f"Gradient: {grad_zeros}")
    print(f"Gradient norm: {jnp.linalg.norm(grad_zeros):.6e}")
    
    # Test 2: All same non-zero value
    print(f"\nTest 2: theta_zz = [0.5, 0.5, 0.5]")
    print("="*40)
    
    theta_same = jnp.array([0.5, 0.5, 0.5])
    
    def loss_fn_same(theta):
        output = model.run_model_batch(test_input, params['pre_quaternions'], theta, params['post_quaternions'])
        return jnp.sum(jnp.abs(output)**2)
    
    loss_same = loss_fn_same(theta_same)
    grad_same = jax.grad(loss_fn_same)(theta_same)
    
    print(f"Loss: {loss_same:.6f}")
    print(f"Gradient: {grad_same}")
    print(f"Gradient norm: {jnp.linalg.norm(grad_same):.6e}")
    
    # Test 3: Different values
    print(f"\nTest 3: theta_zz = [0.3, 0.5, 0.7]")
    print("="*40)
    
    theta_diff = jnp.array([0.3, 0.5, 0.7])
    
    def loss_fn_diff(theta):
        output = model.run_model_batch(test_input, params['pre_quaternions'], theta, params['post_quaternions'])
        return jnp.sum(jnp.abs(output)**2)
    
    loss_diff = loss_fn_diff(theta_diff)
    grad_diff = jax.grad(loss_fn_diff)(theta_diff)
    
    print(f"Loss: {loss_diff:.6f}")
    print(f"Gradient: {grad_diff}")
    print(f"Gradient norm: {jnp.linalg.norm(grad_diff):.6e}")
    
    # Test 4: Check if outputs change
    print(f"\nTest 4: Output Changes")
    print("="*40)
    
    output_zeros = model.run_model_batch(test_input, params['pre_quaternions'], theta_zeros, params['post_quaternions'])
    output_same = model.run_model_batch(test_input, params['pre_quaternions'], theta_same, params['post_quaternions'])
    output_diff = model.run_model_batch(test_input, params['pre_quaternions'], theta_diff, params['post_quaternions'])
    
    print(f"Output norm (zeros):  {jnp.linalg.norm(output_zeros):.6f}")
    print(f"Output norm (same):   {jnp.linalg.norm(output_same):.6f}")
    print(f"Output norm (diff):   {jnp.linalg.norm(output_diff):.6f}")
    
    diff_zeros_same = jnp.linalg.norm(output_zeros - output_same)
    diff_zeros_diff = jnp.linalg.norm(output_zeros - output_diff)
    diff_same_diff = jnp.linalg.norm(output_same - output_diff)
    
    print(f"\nOutput differences:")
    print(f"  zeros vs same: {diff_zeros_same:.6e}")
    print(f"  zeros vs diff: {diff_zeros_diff:.6e}")
    print(f"  same vs diff:  {diff_same_diff:.6e}")
    
    # Analysis
    print(f"\n" + "="*60)
    print("Analysis")
    print("="*60)
    
    if jnp.allclose(grad_zeros, 0.0) and jnp.allclose(grad_same, 0.0):
        print("\n✗ PROBLEM: Gradients are zero for both uniform values!")
        print("This suggests:")
        print("1. The ZZ layer creates a symmetric effect")
        print("2. Gradients vanish when all theta_zz are equal")
        print("3. Need to initialize with different values or")
        print("4. The circuit topology makes individual thetas unobservable")
    elif jnp.allclose(grad_zeros, 0.0):
        print("\n⚠️  Gradients are zero only for theta=[0,0,0]")
        print("This is a special case (identity gate)")
    else:
        print("\n✓ Gradients are non-zero")
    
    if diff_zeros_same < 1e-10 and diff_zeros_diff < 1e-10:
        print("\n✗ CRITICAL: Outputs don't change with different theta_zz!")
        print("The ZZ layer is not affecting the circuit output at all.")
    elif diff_zeros_same < 1e-10:
        print("\n⚠️  Output same for uniform theta_zz values")
        print("Different values DO change output")
    else:
        print("\n✓ Outputs change with different theta_zz values")
    
    print("="*60)


if __name__ == "__main__":
    test_identical_theta_values()
