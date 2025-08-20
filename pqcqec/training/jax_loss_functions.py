import jax.numpy as jnp
import jax

@jax.jit
def jax_mixed_state_fidelity(rho: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Uhlmann fidelity between two density matrices ρ and σ:
        F(ρ, σ) = (Tr sqrt(sqrt(ρ) σ sqrt(ρ)))²
    
    Args:
        rho, sigma: density matrices of shape (2^n, 2^n)

    Returns:
        Scalar fidelity ∈ [0, 1]
    """
    # Ensure Hermitian
    rho = 0.5 * (rho + rho.conj().T)
    sigma = 0.5 * (sigma + sigma.conj().T)

    # Eigen decomposition of sqrt(ρ)
    evals, evecs = jnp.linalg.eigh(rho)
    sqrt_rho = (evecs * jnp.sqrt(jnp.maximum(evals, 0.0))) @ evecs.conj().T

    # M = sqrt(ρ) σ sqrt(ρ)
    M = sqrt_rho @ sigma @ sqrt_rho

    # sqrt of M via eigendecomposition
    m_evals, m_evecs = jnp.linalg.eigh(M)
    sqrt_M = (m_evecs * jnp.sqrt(jnp.maximum(m_evals, 0.0))) @ m_evecs.conj().T

    return jnp.real(jnp.trace(sqrt_M)) ** 2

@jax.jit
def jax_pure_state_fidelity(psi: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """
    Compute fidelity F = |⟨ψ|φ⟩|² between two (possibly unnormalized) state vectors.
    Args:
        psi: Complex array of shape (2**n,)
        phi: Complex array of shape (2**n,)
    Returns:
        Scalar fidelity value ∈ [0, 1]
    """
    # Ensure complex64 for JAX compatibility and performance
    psi = psi.astype(jnp.complex64)
    phi = phi.astype(jnp.complex64)

    # Normalize both states (safe for noisy output states)
    psi /= jnp.linalg.norm(psi) + 1e-12
    phi /= jnp.linalg.norm(phi) + 1e-12

    overlap = jnp.vdot(psi, phi)
    fidelity = jnp.abs(overlap) ** 2

    return fidelity.real  # Ensure scalar float32 (avoids accidental complex gradients)

@jax.jit
def jax_l2_loss_ignore_global_phase(psi, phi):
    # Align global phase
    overlap = jnp.vdot(phi, psi)
    phase = overlap / jnp.abs(overlap + 1e-12)
    psi_aligned = psi * phase.conj()

    return jnp.sum(jnp.abs(psi_aligned - phi) ** 2)

@jax.jit
def jax_mse_complex_loss(psi: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    # psi = psi.astype(jnp.complex128)
    # phi = phi.astype(jnp.complex128)

    real_loss = jnp.mean((jnp.real(psi) - jnp.real(phi)) ** 2)
    imag_loss = jnp.mean((jnp.imag(psi) - jnp.imag(phi)) ** 2)
    return real_loss + imag_loss

@jax.jit
def jax_mse_complex_loss_aligned(psi: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    psi = psi / (jnp.linalg.norm(psi) + 1e-12)
    phi = phi / (jnp.linalg.norm(phi) + 1e-12)

    # Align global phase
    overlap = jnp.vdot(phi, psi)
    phase = overlap / (jnp.abs(overlap) + 1e-12)
    psi = psi * phase.conj()

    real_loss = jnp.mean((jnp.real(psi) - jnp.real(phi)) ** 2)
    imag_loss = jnp.mean((jnp.imag(psi) - jnp.imag(phi)) ** 2)
    return real_loss + imag_loss

@jax.jit
def jax_fidelity_loss(ideal: jnp.ndarray, measured: jnp.ndarray) -> jnp.ndarray:
    """
    Fidelity-based loss: L = 1 - F(ideal, measured)
    """
    return 1.0 - jax_pure_state_fidelity(ideal, measured)

@jax.jit
def jax_mixed_fidelity_loss(ideal: jnp.ndarray, measured: jnp.ndarray) -> jnp.ndarray:
    """
    Fidelity-based loss: L = 1 - F(ideal, measured)
    """
    psi = ideal / jnp.linalg.norm(ideal)
    phi = measured / jnp.linalg.norm(measured)
    rho = jnp.outer(psi, jnp.conj(psi))
    sigma = jnp.outer(phi, jnp.conj(phi))

    return 1.0 - jax_mixed_state_fidelity(rho, sigma)

@jax.jit
def jax_density_trace_loss(psi: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """
    Loss based on trace distance between pure-state density matrices.
    Equivalent to sqrt(1 - fidelity) for pure states.
    """
    # Normalize
    psi = psi / (jnp.linalg.norm(psi) + 1e-12)
    phi = phi / (jnp.linalg.norm(phi) + 1e-12)

    fidelity = jnp.abs(jnp.vdot(psi, phi)) ** 2
    return jnp.sqrt(jnp.maximum(0.0, 1.0 - fidelity))

@jax.jit
def jax_hilbert_schmidt_density_loss(psi, phi):
    psi = psi / (jnp.linalg.norm(psi) + 1e-12)
    phi = phi / (jnp.linalg.norm(phi) + 1e-12)

    rho = jnp.outer(psi, jnp.conj(psi))
    sigma = jnp.outer(phi, jnp.conj(phi))
    diff = rho - sigma
    # HS-norm squared
    return jnp.real(jnp.trace(diff @ diff))