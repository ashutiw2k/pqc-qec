"""
Test to understand why theta_zz gradients are zero in the actual model.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit
from pqcqec.models.pqc_models import LELZZInterleavedQuaternionCustomStatevecModel
from pqcqec.simulate.simulate import get_input_data
from pqcqec.training.jax_loss_functions import jax_fidelity_loss, jax_mse_complex_loss_aligned


def test_theta_zz_gradients():
    """Test why theta_zz gradients are zero."""
    
    print("="*60)
    print("Testing Theta_ZZ Gradient Flow")
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
    params = model.get_model_params()
    
    print(f"\nInitial theta_zz: {params['theta_zz']}")
    print(f"Are all values the same? {jnp.all(params['theta_zz'] == params['theta_zz'][0])}")
    
    # Create test input
    test_input = get_input_data(num_qubits, 1, seed=seed)
    
    # Test 1: Can we differentiate through run_model_batch directly?
    print(f"\n" + "="*60)
    print("Test 1: Direct differentiation of theta_zz")
    print("="*60)
    
    def simple_loss(theta):
        """Simple loss that only depends on theta_zz."""
        output = model.run_model_batch(
            test_input, 
            params['pre_quaternions'], 
            theta, 
            params['post_quaternions']
        )
        # Just sum the output (arbitrary loss)
        return jnp.sum(jnp.abs(output)**2)
    
    loss_val = simple_loss(params['theta_zz'])
    grad_theta = jax.grad(simple_loss)(params['theta_zz'])
    
    print(f"Loss: {loss_val:.6f}")
    print(f"Gradient: {grad_theta}")
    print(f"Gradient norm: {jnp.linalg.norm(grad_theta):.6e}")
    print(f"Gradient is non-zero: {not jnp.allclose(grad_theta, 0.0)}")
    
    # Test 2: Vary only theta_zz and see output change
    print(f"\n" + "="*60)
    print("Test 2: Output sensitivity to theta_zz changes")
    print("="*60)
    
    # Try different theta_zz values
    theta_tests = [
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([0.1, 0.1, 0.1]),
        jnp.array([0.0, 0.1, 0.2]),  # Different values
        jnp.array([1.0, 1.0, 1.0]),
    ]
    
    outputs = []
    for theta in theta_tests:
        output = model.run_model_batch(
            test_input,
            params['pre_quaternions'],
            theta,
            params['post_quaternions']
        )
        outputs.append(output)
        print(f"theta={theta} → output norm: {jnp.linalg.norm(output):.6f}")
    
    # Check if outputs are different
    print(f"\nOutputs are all the same: {all(jnp.allclose(outputs[0], out) for out in outputs[1:])}")
    
    if all(jnp.allclose(outputs[0], out) for out in outputs[1:]):
        print("⚠️  WARNING: Changing theta_zz doesn't change the output!")
        print("This explains why gradients are zero.")
    
    # Test 3: Check circuit structure
    print(f"\n" + "="*60)
    print("Test 3: Inspect circuit structure")
    print("="*60)
    
    circuit_tokens = model.get_circuit_tokens()
    print(f"Full circuit has {len(circuit_tokens)} operations")
    
    # Find ZZ-related operations
    zz_ops = []
    for i, op in enumerate(circuit_tokens):
        gate, qubits, params_list = op
        if gate == 'cnot':
            # Check if next gate is RZ (potential ZZ layer)
            if i + 1 < len(circuit_tokens):
                next_gate, next_qubits, next_params = circuit_tokens[i + 1]
                if next_gate == 'rz':
                    zz_ops.append((i, i+1, qubits, next_qubits, next_params))
    
    print(f"\nFound {len(zz_ops)} potential ZZ patterns:")
    for idx, (cnot_idx, rz_idx, cnot_q, rz_q, rz_param) in enumerate(zz_ops[:5]):  # Show first 5
        print(f"  Pattern {idx}: CNOT{cnot_q} → RZ{rz_q} with param {rz_param}")
    
    # Test 4: Loss function sensitivity
    print(f"\n" + "="*60)
    print("Test 4: Loss function gradient w.r.t. theta_zz")
    print("="*60)
    
    def fidelity_loss_fn(theta):
        output = model.run_model_batch(
            test_input,
            params['pre_quaternions'],
            theta,
            params['post_quaternions']
        )
        return jax_fidelity_loss(test_input, output)
    
    fid_loss = fidelity_loss_fn(params['theta_zz'])
    fid_grad = jax.grad(fidelity_loss_fn)(params['theta_zz'])
    
    print(f"Fidelity loss: {fid_loss:.6f}")
    print(f"Fidelity gradient: {fid_grad}")
    print(f"Fidelity gradient norm: {jnp.linalg.norm(fid_grad):.6e}")
    
    def mse_loss_fn(theta):
        output = model.run_model_batch(
            test_input,
            params['pre_quaternions'],
            theta,
            params['post_quaternions']
        )
        return jax_mse_complex_loss_aligned(test_input, output)
    
    mse_loss = mse_loss_fn(params['theta_zz'])
    mse_grad = jax.grad(mse_loss_fn)(params['theta_zz'])
    
    print(f"\nMSE loss: {mse_loss:.6f}")
    print(f"MSE gradient: {mse_grad}")
    print(f"MSE gradient norm: {jnp.linalg.norm(mse_grad):.6e}")
    
    # Summary
    print(f"\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if jnp.allclose(grad_theta, 0.0) and all(jnp.allclose(outputs[0], out) for out in outputs[1:]):
        print("\n✗ PROBLEM IDENTIFIED:")
        print("  Theta_zz changes don't affect the output")
        print("  This means the ZZ layer is not being applied correctly")
        print("  or the circuit doesn't contain ZZ gates")
    elif jnp.allclose(grad_theta, 0.0):
        print("\n✗ PROBLEM:")
        print("  Gradients are zero even though output changes")
        print("  This suggests a gradient flow issue")
    else:
        print("\n✓ Theta_zz gradients ARE flowing")
        print("  The issue must be elsewhere in the training loop")
    
    print("="*60)


if __name__ == "__main__":
    test_theta_zz_gradients()
