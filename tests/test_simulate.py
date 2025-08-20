import jax
import jax.numpy as jnp

from pqcqec.simulate.simulate import get_input_data, run_circuit_with_noise_model
from pqcqec.noise.simple_noise import PennylaneNoisyGates


def test_get_input_data_shape_and_normalization_and_repro():
    num_qubits, num_vals = 3, 7
    data1 = get_input_data(num_qubits, num_vals, seed=123)
    data2 = get_input_data(num_qubits, num_vals, seed=123)
    assert data1.shape == (num_vals, 2 ** num_qubits)
    # Row-wise norms near 1
    norms = jnp.linalg.norm(data1, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-6)
    # Deterministic for same seed
    assert jnp.allclose(data1, data2)


def test_run_circuit_with_noise_model_batched_identity_x():
    # Circuit: apply X on wire 0, with zero noise -> acts like ideal X
    noise = PennylaneNoisyGates(x_rad=0.0, z_rad=0.0, delta_x=0.0, delta_z=0.0)
    circuit_ops = [("x", [0], [])]

    num_qubits = 1
    # Batch of two states: |0>, |1>
    e0 = jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
    e1 = jnp.array([0.0 + 0.0j, 1.0 + 0.0j])
    inputs = jnp.stack([e0, e1], axis=0)

    out = run_circuit_with_noise_model(
        circuit_ops=circuit_ops,
        input_state=inputs,
        noise_model=noise,
        num_qubits=num_qubits,
        batched=True,
    )

    # Expect flipped states (up to global phase)
    expected = jnp.stack([e1, e0], axis=0)
    # Compare via fidelity per sample
    overlap = jnp.vdot(out.reshape(-1), out.reshape(-1))  # ensure array realized
    # Per-sample comparison
    for i in range(inputs.shape[0]):
        psi = out[i] / (jnp.linalg.norm(out[i]) + 1e-12)
        phi = expected[i]
        phi = phi / (jnp.linalg.norm(phi) + 1e-12)
        fid = jnp.abs(jnp.vdot(phi, psi)) ** 2
        assert float(fid) > 1 - 1e-6

