"""
Test script for ZXZInterleavedAngleCustomStatevecModel (angle-based parametrization).

This tests the angle-based PQC model to ensure it works correctly without quaternions.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import optax

from pqcqec.models.pqc_models import ZXZInterleavedAngleCustomStatevecModel
from pqcqec.training.jax_train_functions import train_lel_zz_custom_statevec_no_uncomp
from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.simulate.simulate import get_input_data
from pqcqec.circuits.templates import build_pqc_circuit_template
from pqcqec.simulate.jax_statevector import build_jax_circuit, jax_run_many_states
from pqcqec.training.jax_loss_functions import jax_pure_state_fidelity
from pqcqec.utils.jax_utils import JAXStateMeasuredDataset, JAXDataLoader


def test_angle_model_parameter_structure():
    """Test that angle model returns single-element tuple with shape (N, M, 3)."""
    
    # Create simple circuit
    num_qubits = 2
    num_gates = 4
    base_ops = generate_random_circuit(num_qubits, num_gates, seed=42, backend='list')
    
    # Create noise arrays
    x_noise = np.random.uniform(-0.05, 0.05, num_gates).astype(np.float32)
    z_noise = np.random.uniform(-0.05, 0.05, num_gates).astype(np.float32)
    
    # Create model
    model = ZXZInterleavedAngleCustomStatevecModel(
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
    
    # Element should be JAX array (angles)
    assert isinstance(params[0], jnp.ndarray), f"Expected JAX array, got {type(params[0])}"
    
    # Should have correct shape (num_layers, num_qubits, 3) - 3 angles per qubit
    expected_layers = 2  # 4 gates / 2 gates_per_block
    assert params[0].shape == (expected_layers, num_qubits, 3), \
        f"Expected shape {(expected_layers, num_qubits, 3)}, got {params[0].shape}"
    
    print(f"✓ Parameter structure correct: {params[0].shape}")


def test_angle_model_run_model_batch():
    """Test that angle model's run_model_batch works without conversion."""
    
    # Create simple circuit
    num_qubits = 2
    num_gates = 4
    base_ops = generate_random_circuit(num_qubits, num_gates, seed=42, backend='list')
    
    # Create noise arrays
    x_noise = np.random.uniform(-0.05, 0.05, num_gates).astype(np.float32)
    z_noise = np.random.uniform(-0.05, 0.05, num_gates).astype(np.float32)
    
    # Create model
    model = ZXZInterleavedAngleCustomStatevecModel(
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
    
    print(f"✓ run_model_batch works correctly")


def test_angle_model_basic_training():
    """Test that angle model trains correctly."""
    
    # Create simple circuit with only single-qubit gates for stability
    num_qubits = 2
    num_gates = 4
    # Use gate distribution that only includes single-qubit gates
    gate_dist = {'h': 1.0, 'x': 1.0, 'z': 1.0}
    base_ops = generate_random_circuit(num_qubits, num_gates, seed=42, backend='list', gate_dist=gate_dist)
    
    # Create noise arrays (very small noise for easier training)
    x_noise = np.random.uniform(-0.01, 0.01, num_gates).astype(np.float32)
    z_noise = np.random.uniform(-0.01, 0.01, num_gates).astype(np.float32)
    
    # Create model
    model = ZXZInterleavedAngleCustomStatevecModel(
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
    
    print(f"✓ Training improved fidelity from {initial_fidelity:.4f} to {final_fidelity:.4f}")


def test_angle_model_no_conversion():
    """Verify that angle model doesn't have quaternion conversion overhead."""
    
    # Create simple circuit with single-qubit gates only
    num_qubits = 2  # Changed from 1 to 2 to avoid sampling errors
    num_gates = 4
    gate_dist = {'h': 1.0, 'x': 1.0}  # Only single-qubit gates
    base_ops = generate_random_circuit(num_qubits, num_gates, seed=42, backend='list', gate_dist=gate_dist)
    
    # Create noise arrays
    x_noise = np.random.uniform(-0.02, 0.02, num_gates).astype(np.float32)
    z_noise = np.random.uniform(-0.02, 0.02, num_gates).astype(np.float32)
    
    # Create model
    model = ZXZInterleavedAngleCustomStatevecModel(
        base_circuit_ops=base_ops,
        num_qubits=num_qubits,
        x_noise=x_noise,
        z_noise=z_noise,
        pqc_blocks=1,
        gate_blocks=4,  # All gates in one block
        seed=100,
        pqc_type='zxz'
    )
    
    # Check that model doesn't have quaternion conversion method or function
    # (The attribute may not exist at all, which is fine)
    
    # Check that model stores angles, not quaternions
    assert hasattr(model, 'pre_angles'), "Model should have pre_angles attribute"
    assert not hasattr(model, 'pre_quaternions'), "Model should not have pre_quaternions attribute"
    
    # Check shape is (N, M, 3) not (N, M, 4)
    assert model.pre_angles.shape[-1] == 3, \
        f"Expected 3 angles per qubit, got {model.pre_angles.shape[-1]}"
    
    print(f"✓ Model uses direct angle parametrization (no quaternion conversion)")


if __name__ == "__main__":
    # Run tests
    test_angle_model_parameter_structure()
    test_angle_model_run_model_batch()
    test_angle_model_no_conversion()
    test_angle_model_basic_training()
    
    print("\n✓ All angle model tests passed!")
