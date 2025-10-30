"""
Test ZXZ model training with single-parameter tuple.

This test verifies that the ZXZ model (local-only PQC) works correctly
with the training functions that use flexible parameter unpacking.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import optax

from pqcqec.models.pqc_models import ZXZInterleavedQuaternionCustomStatevecModel
from pqcqec.training.jax_train_functions import train_lel_zz_custom_statevec_no_uncomp
from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.simulate.simulate import get_input_data


def test_zxz_model_parameter_structure():
    """Test that ZXZ model returns single-element tuple from get_model_params()."""
    
    # Create simple circuit
    num_qubits = 2
    num_gates = 4
    base_ops = generate_random_circuit(num_qubits, num_gates, seed=42, backend='list')
    
    # Create noise arrays
    x_noise = np.random.uniform(-0.05, 0.05, num_gates).astype(np.float32)
    z_noise = np.random.uniform(-0.05, 0.05, num_gates).astype(np.float32)
    
    # Create model
    model = ZXZInterleavedQuaternionCustomStatevecModel(
        base_circuit_ops=base_ops,
        num_qubits=num_qubits,
        x_noise=x_noise,
        z_noise=z_noise,
        pqc_blocks=1,
        gate_blocks=2,
        seed=100,
        pqc_type='zxz'
    )
    
    # Check parameter structure
    params = model.get_model_params()
    
    # Should be a tuple
    assert isinstance(params, tuple), f"Expected tuple, got {type(params)}"
    
    # Should have single element
    assert len(params) == 1, f"Expected 1 element, got {len(params)}"
    
    # Element should be JAX array (quaternions)
    assert isinstance(params[0], jnp.ndarray), f"Expected JAX array, got {type(params[0])}"
    
    # Should have correct shape (num_layers, num_qubits, 4)
    expected_layers = 2  # 4 gates / 2 gates_per_block
    assert params[0].shape == (expected_layers, num_qubits, 4), \
        f"Expected shape {(expected_layers, num_qubits, 4)}, got {params[0].shape}"


def test_zxz_model_run_model_batch():
    """Test that ZXZ model's run_model_batch accepts single parameter."""
    
    # Create simple circuit
    num_qubits = 2
    num_gates = 4
    base_ops = generate_random_circuit(num_qubits, num_gates, seed=42, backend='list')
    
    # Create noise arrays
    x_noise = np.random.uniform(-0.05, 0.05, num_gates).astype(np.float32)
    z_noise = np.random.uniform(-0.05, 0.05, num_gates).astype(np.float32)
    
    # Create model
    model = ZXZInterleavedQuaternionCustomStatevecModel(
        base_circuit_ops=base_ops,
        num_qubits=num_qubits,
        x_noise=x_noise,
        z_noise=z_noise,
        pqc_blocks=1,
        gate_blocks=2,
        seed=100,
        pqc_type='zxz'
    )
    
    # Create input data
    batch_size = 4
    input_states = get_input_data(num_qubits, batch_size, seed=42)
    
    # Get parameters
    params = model.get_model_params()
    
    # Run with unpacked parameters
    output_states = model.run_model_batch(input_states, *params)
    
    # Check output shape
    assert output_states.shape == input_states.shape, \
        f"Expected shape {input_states.shape}, got {output_states.shape}"
    
    # Check output is normalized
    norms = jnp.linalg.norm(output_states, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5), \
        f"Output states not normalized: {norms}"


def test_zxz_model_basic_training():
    """Test that ZXZ model trains correctly with no-uncomp training function."""
    
    # Create simple circuit
    num_qubits = 2
    num_gates = 4
    base_ops = generate_random_circuit(num_qubits, num_gates, seed=42, backend='list')
    
    # Create noise arrays (small noise for quick convergence)
    x_noise = np.random.uniform(-0.02, 0.02, num_gates).astype(np.float32)
    z_noise = np.random.uniform(-0.02, 0.02, num_gates).astype(np.float32)
    
    # Create model
    model = ZXZInterleavedQuaternionCustomStatevecModel(
        base_circuit_ops=base_ops,
        num_qubits=num_qubits,
        x_noise=x_noise,
        z_noise=z_noise,
        pqc_blocks=1,
        gate_blocks=2,
        seed=100,
        pqc_type='zxz'
    )
    
    # Create training data
    batch_size = 8
    input_states = get_input_data(num_qubits, batch_size, seed=42)
    
    # Simulate ideal circuit (no noise) for targets
    from pqcqec.circuits.templates import build_pqc_circuit_template
    from pqcqec.simulate.jax_statevector import build_jax_circuit, jax_run_many_states
    
    ideal_template = build_pqc_circuit_template(
        base_ops=base_ops,
        num_qubits=num_qubits,
        num_gate_blocks=2,
        add_noise=False,
        pqc_type='none'
    )
    ideal_ops = ideal_template.instantiate({
        'base': np.array([op[2][0] if len(op[2]) > 0 else 0.0 for op in base_ops], dtype=np.float32)
    })
    gate_ids, wire1s, wire2s, thetas = build_jax_circuit(ideal_ops)
    target_states = jax_run_many_states(num_qubits, gate_ids, wire1s, wire2s, thetas, input_states)
    
    # Create simple dataloader
    dataloader = [(input_states, target_states)]
    
    # Create optimizer
    learning_rate = 0.01
    optimizer = optax.adam(learning_rate)
    schedule = lambda step: learning_rate  # Constant schedule
    
    # Get initial fidelity
    from pqcqec.training.jax_loss_functions import jax_pure_state_fidelity
    initial_params = model.get_model_params()
    initial_output = model.run_model_batch(input_states, *initial_params)
    initial_fidelity = jax_pure_state_fidelity(target_states, initial_output)
    
    print(f"\nInitial fidelity: {initial_fidelity:.4f}")
    
    # Train for a few epochs
    final_fidelity = train_lel_zz_custom_statevec_no_uncomp(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        schedule=schedule,
        epochs=3
    )
    
    print(f"Final fidelity: {final_fidelity:.4f}")
    
    # Check that training improved fidelity
    assert final_fidelity > initial_fidelity, \
        f"Training did not improve fidelity: {initial_fidelity:.4f} -> {final_fidelity:.4f}"
    
    # With small noise, should achieve reasonably high fidelity
    assert final_fidelity > 0.8, \
        f"Final fidelity too low: {final_fidelity:.4f}"


if __name__ == "__main__":
    # Run tests
    test_zxz_model_parameter_structure()
    print("✓ Parameter structure test passed")
    
    test_zxz_model_run_model_batch()
    print("✓ run_model_batch test passed")
    
    test_zxz_model_basic_training()
    print("✓ Basic training test passed")
    
    print("\n✓ All ZXZ model tests passed!")
