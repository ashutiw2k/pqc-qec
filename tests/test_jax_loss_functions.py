import math
import jax.numpy as jnp

from pqcqec.training.jax_loss_functions import (
    jax_pure_state_fidelity,
    jax_mixed_state_fidelity,
    jax_mse_complex_loss,
    jax_mse_complex_loss_aligned,
    jax_l2_loss_ignore_global_phase,
    jax_density_trace_loss,
)


def test_pure_state_fidelity_basic_cases():
    psi = jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
    phi = jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
    f_same = float(jax_pure_state_fidelity(psi, phi))
    assert abs(f_same - 1.0) < 1e-6

    phi_orth = jnp.array([0.0 + 0.0j, 1.0 + 0.0j])
    f_orth = float(jax_pure_state_fidelity(psi, phi_orth))
    assert abs(f_orth - 0.0) < 1e-6

    # Non-normalized inputs should be normalized internally
    psi_scaled = 2.0 * psi
    phi_scaled = 3.0 * phi
    f_scaled = float(jax_pure_state_fidelity(psi_scaled, phi_scaled))
    assert abs(f_scaled - 1.0) < 1e-6


def test_mse_complex_loss_and_aligned():
    a = jnp.array([1.0 + 1.0j, 0.0 + 0.0j])
    b = jnp.array([0.0 + 0.0j, 1.0 + 1.0j])

    # Manual expected MSE across real and imaginary parts
    real_diff = jnp.mean((jnp.real(a) - jnp.real(b)) ** 2)
    imag_diff = jnp.mean((jnp.imag(a) - jnp.imag(b)) ** 2)
    expected = float(real_diff + imag_diff)
    got = float(jax_mse_complex_loss(a, b))
    assert abs(got - expected) < 1e-7

    # Phase-invariant: a and e^{iθ} a should produce ~0 aligned loss
    theta = jnp.pi / 3
    phase = jnp.exp(1j * jnp.array(theta))
    a_phased = a * phase
    aligned_loss = float(jax_mse_complex_loss_aligned(a, a_phased))
    assert aligned_loss < 1e-6


def test_l2_loss_ignore_global_phase():
    psi = jnp.array([0.5 + 0.5j, -0.5 + 0.5j])
    theta = math.pi / 7
    phase = jnp.exp(1j * jnp.array(theta))
    phi = psi * phase
    loss = float(jax_l2_loss_ignore_global_phase(psi, phi))
    assert loss < 1e-6


def test_mixed_and_trace_losses():
    # |0><0|
    ket0 = jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
    rho0 = jnp.outer(ket0, jnp.conj(ket0))

    # |1><1|
    ket1 = jnp.array([0.0 + 0.0j, 1.0 + 0.0j])
    rho1 = jnp.outer(ket1, jnp.conj(ket1))

    f_same = float(jax_mixed_state_fidelity(rho0, rho0))
    assert abs(f_same - 1.0) < 1e-6

    f_orth = float(jax_mixed_state_fidelity(rho0, rho1))
    assert abs(f_orth - 0.0) < 1e-6

    # Trace-distance derived loss: identical -> 0, orthogonal pure -> 1
    td_same = float(jax_density_trace_loss(ket0, ket0))
    assert td_same < 1e-7

    td_orth = float(jax_density_trace_loss(ket0, ket1))
    assert abs(td_orth - 1.0) < 1e-6

