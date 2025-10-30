"""
Test script to verify gradient flow through the PQC model.

This script checks if gradients properly flow from loss to quaternion parameters.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit
from pqcqec.models.pqc_models import LELZZInterleavedQuaternionCustomStatevecModel
from pqcqec.simulate.simulate import get_input_data
from pqcqec.training.jax_loss_functions import jax_fidelity_loss, jax_mse_complex_loss_aligned


def test_gradient_flow():
    """Test if gradients flow properly through the model."""
    
    print("="*60)
    print("Testing Gradient Flow Through PQC Model")
    print("="*60)
    
    # Setup
    num_qubits = 3
    num_gates = 5
    gate_blocks = 5
    pqc_blocks = 1
    seed = 42
    
    # Generate a simple circuit
    qiskit_circuit = generate_random_circuit(
        num_qubits=num_qubits,
        num_gates=num_gates,
        seed=seed
    )
    
    # Add uncomputation
    qiskit_adjoint = qiskit_circuit.inverse()
    qiskit_uncomp = qiskit_circuit.compose(qiskit_adjoint)
    circuit_ops = tokenize_qiskit_circuit(qiskit_uncomp)
    
    print(f"\n✓ Generated circuit with {len(circuit_ops)} operations")
    
    # Create noise arrays
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
    
    print(f"✓ Initialized model")
    
    # Get initial parameters
    params = model.get_model_params_to_store()
    print(f"\nInitial parameters:")
    print(f"  Pre-quaternions shape: {params['pre_quaternions'].shape}")
    print(f"  Theta_zz shape: {params['theta_zz'].shape}")
    print(f"  Post-quaternions shape: {params['post_quaternions'].shape}")
    
    # Create test input (single state)
    test_input = get_input_data(num_qubits, 1, seed=seed)
    print(f"\n✓ Created test input with shape: {test_input.shape}")
    
    # Define loss function
    def loss_fn(pre_q, theta, post_q):
        output = model.run_model_batch(test_input, pre_q, theta, post_q)
        return jax_fidelity_loss(test_input, output)
    
    print("\n" + "="*60)
    print("Testing with Fidelity Loss")
    print("="*60)
    
    # Compute loss and gradients
    loss_value, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(
        params['pre_quaternions'],
        params['theta_zz'],
        params['post_quaternions']
    )
    
    print(f"\nLoss value: {loss_value:.6f}")
    
    # Check gradients
    pre_grad_norm = jnp.linalg.norm(grads[0])
    theta_grad_norm = jnp.linalg.norm(grads[1])
    post_grad_norm = jnp.linalg.norm(grads[2])
    
    print(f"\nGradient norms:")
    print(f"  Pre-quaternions:  {pre_grad_norm:.6e}")
    print(f"  Theta_zz:         {theta_grad_norm:.6e}")
    print(f"  Post-quaternions: {post_grad_norm:.6e}")
    
    # Check for zeros
    pre_has_grad = not jnp.allclose(grads[0], 0.0, atol=1e-10)
    theta_has_grad = not jnp.allclose(grads[1], 0.0, atol=1e-10)
    post_has_grad = not jnp.allclose(grads[2], 0.0, atol=1e-10)
    
    print(f"\nGradient flow check:")
    print(f"  Pre-quaternions:  {'✓ FLOWING' if pre_has_grad else '✗ BLOCKED'}")
    print(f"  Theta_zz:         {'✓ FLOWING' if theta_has_grad else '✗ BLOCKED'}")
    print(f"  Post-quaternions: {'✓ FLOWING' if post_has_grad else '✗ BLOCKED'}")
    
    # Check for NaNs
    pre_has_nan = jnp.any(jnp.isnan(grads[0]))
    theta_has_nan = jnp.any(jnp.isnan(grads[1]))
    post_has_nan = jnp.any(jnp.isnan(grads[2]))
    
    print(f"\nNaN check:")
    print(f"  Pre-quaternions:  {'✗ HAS NaN' if pre_has_nan else '✓ No NaN'}")
    print(f"  Theta_zz:         {'✗ HAS NaN' if theta_has_nan else '✓ No NaN'}")
    print(f"  Post-quaternions: {'✗ HAS NaN' if post_has_nan else '✓ No NaN'}")
    
    # Test with MSE loss
    print("\n" + "="*60)
    print("Testing with MSE Loss (Phase Aligned)")
    print("="*60)
    
    def loss_fn_mse(pre_q, theta, post_q):
        output = model.run_model_batch(test_input, pre_q, theta, post_q)
        return jax_mse_complex_loss_aligned(test_input, output)
    
    loss_value_mse, grads_mse = jax.value_and_grad(loss_fn_mse, argnums=(0, 1, 2))(
        params['pre_quaternions'],
        params['theta_zz'],
        params['post_quaternions']
    )
    
    print(f"\nLoss value: {loss_value_mse:.6f}")
    
    pre_grad_norm_mse = jnp.linalg.norm(grads_mse[0])
    theta_grad_norm_mse = jnp.linalg.norm(grads_mse[1])
    post_grad_norm_mse = jnp.linalg.norm(grads_mse[2])
    
    print(f"\nGradient norms:")
    print(f"  Pre-quaternions:  {pre_grad_norm_mse:.6e}")
    print(f"  Theta_zz:         {theta_grad_norm_mse:.6e}")
    print(f"  Post-quaternions: {post_grad_norm_mse:.6e}")
    
    pre_has_grad_mse = not jnp.allclose(grads_mse[0], 0.0, atol=1e-10)
    theta_has_grad_mse = not jnp.allclose(grads_mse[1], 0.0, atol=1e-10)
    post_has_grad_mse = not jnp.allclose(grads_mse[2], 0.0, atol=1e-10)
    
    print(f"\nGradient flow check:")
    print(f"  Pre-quaternions:  {'✓ FLOWING' if pre_has_grad_mse else '✗ BLOCKED'}")
    print(f"  Theta_zz:         {'✓ FLOWING' if theta_has_grad_mse else '✗ BLOCKED'}")
    print(f"  Post-quaternions: {'✓ FLOWING' if post_has_grad_mse else '✗ BLOCKED'}")
    
    # Test parameter update
    print("\n" + "="*60)
    print("Testing Parameter Update")
    print("="*60)
    
    # Simulate one gradient descent step
    learning_rate = 0.01
    new_pre = params['pre_quaternions'] - learning_rate * grads_mse[0]
    new_theta = params['theta_zz'] - learning_rate * grads_mse[1]
    new_post = params['post_quaternions'] - learning_rate * grads_mse[2]
    
    # Compute loss with new parameters
    new_loss = loss_fn_mse(new_pre, new_theta, new_post)
    loss_change = new_loss - loss_value_mse
    
    print(f"\nAfter gradient step (LR={learning_rate}):")
    print(f"  Old loss: {loss_value_mse:.6f}")
    print(f"  New loss: {new_loss:.6f}")
    print(f"  Change:   {loss_change:.6f} ({'✓ DECREASING' if loss_change < 0 else '✗ INCREASING'})")
    
    # Check parameter change magnitude
    pre_change = jnp.linalg.norm(new_pre - params['pre_quaternions'])
    theta_change = jnp.linalg.norm(new_theta - params['theta_zz'])
    post_change = jnp.linalg.norm(new_post - params['post_quaternions'])
    
    print(f"\nParameter changes:")
    print(f"  Pre-quaternions:  {pre_change:.6e}")
    print(f"  Theta_zz:         {theta_change:.6e}")
    print(f"  Post-quaternions: {post_change:.6e}")
    
    # Overall assessment
    print("\n" + "="*60)
    print("Overall Assessment")
    print("="*60)
    
    all_gradients_flowing = pre_has_grad_mse and theta_has_grad_mse and post_has_grad_mse
    no_nans = not (pre_has_nan or theta_has_nan or post_has_nan)
    loss_decreasing = loss_change < 0
    
    if all_gradients_flowing and no_nans and loss_decreasing:
        print("\n✓✓✓ GRADIENT FLOW IS WORKING CORRECTLY ✓✓✓")
        print("The model should be able to train properly.")
    else:
        print("\n✗✗✗ GRADIENT FLOW HAS ISSUES ✗✗✗")
        if not all_gradients_flowing:
            print("  - Some gradients are not flowing")
        if not no_nans:
            print("  - NaN values detected in gradients")
        if not loss_decreasing:
            print("  - Loss is not decreasing with gradient step")
        print("\nThis explains why training is not working.")
    
    print("="*60)
    
    return all_gradients_flowing and no_nans and loss_decreasing


if __name__ == "__main__":
    success = test_gradient_flow()
    exit(0 if success else 1)
