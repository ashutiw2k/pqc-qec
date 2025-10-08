"""JAX-compatible wrapper for custom Numba statevector simulator.

This module provides JAX integration for the high-performance Numba simulator,
enabling automatic differentiation while maintaining speed. It uses:
- jax.pure_callback for forward pass (calls Numba)
- Custom VJP (vector-jacobian product) for gradients via finite differences
- Zero-copy conversions between JAX and NumPy where possible

The gradient computation uses finite differences, which is:
- Simple to implement
- Reasonably fast for small circuits
- Can be replaced with parameter-shift rule for better accuracy
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from .statevector import run_many_states, run_circuit_with_state


def _statevec_forward_numpy(states_in, gate_ids, wire1, wire2, theta, num_qubits):
    """Pure NumPy/Numba forward pass (called from JAX via pure_callback).
    
    Args:
        states_in: Input states [batch, 2^n] (will be converted to NumPy)
        gate_ids: Gate type identifiers [num_gates]
        wire1: Primary qubit indices [num_gates]
        wire2: Secondary qubit indices [num_gates]
        theta: Gate parameters [num_gates]
        num_qubits: Number of qubits
    
    Returns:
        Output states [batch, 2^n] as NumPy array
    """
    # Convert to NumPy (JAX arrays on CPU are zero-copy views)
    states_np = np.asarray(states_in, dtype=np.complex64)
    gate_ids_np = np.asarray(gate_ids, dtype=np.int32)
    wire1_np = np.asarray(wire1, dtype=np.int32)
    wire2_np = np.asarray(wire2, dtype=np.int32)
    theta_np = np.asarray(theta, dtype=np.float32)
    
    # Allocate output
    batch_size = states_np.shape[0]
    dim = 2 ** num_qubits
    states_out = np.empty((batch_size, dim), dtype=np.complex64)
    
    # Run Numba simulator (parallel batched execution)
    run_many_states(num_qubits, gate_ids_np, wire1_np, wire2_np, theta_np, 
                    states_np, states_out)
    
    return states_out


def _statevec_backward_finite_diff(states_in, theta, cotangent, gate_ids, wire1, wire2, 
                                    num_qubits, eps=1e-5):
    """Compute gradients w.r.t. theta using finite differences.
    
    This is called during backprop to compute dL/dtheta given dL/doutput (cotangent).
    Uses central finite differences: f'(x) = (f(x+eps) - f(x-eps)) / (2*eps)
    
    NOTE: Argument order matches how it's called from statevec_simulate_jax_bwd:
          states_in, theta, cotangent are passed positionally from pure_callback,
          gate_ids, wire1, wire2, num_qubits are passed via partial().
    
    Args:
        states_in: Input states [batch, 2^n]
        theta: Gate parameters [num_gates]
        cotangent: Gradient w.r.t. output [batch, 2^n]
        gate_ids: Gate identifiers [num_gates]
        wire1: Qubit indices [num_gates]
        wire2: Qubit indices [num_gates]
        num_qubits: Number of qubits
        eps: Finite difference step size
    
    Returns:
        Gradients w.r.t. theta [num_gates]
    """
    # Convert to NumPy
    states_np = np.asarray(states_in, dtype=np.complex64)
    theta_np = np.asarray(theta, dtype=np.float32)
    cotangent_np = np.asarray(cotangent, dtype=np.complex64)
    gate_ids_np = np.asarray(gate_ids, dtype=np.int32)
    wire1_np = np.asarray(wire1, dtype=np.int32)
    wire2_np = np.asarray(wire2, dtype=np.int32)
    
    batch_size = states_np.shape[0]
    dim = 2 ** num_qubits
    n_params = len(theta_np)
    
    # Storage for gradients
    grad_theta = np.zeros(n_params, dtype=np.float32)
    
    # Temporary buffers for finite differences (reused to save allocations)
    states_plus = np.empty((batch_size, dim), dtype=np.complex64)
    states_minus = np.empty((batch_size, dim), dtype=np.complex64)
    
    # For each parameter, compute finite difference gradient
    for i in range(n_params):
        # Skip non-parameterized gates (theta=0 gates like H, X, Z, CX, CZ)
        if theta_np[i] == 0.0:
            continue
            
        # Compute f(θ + ε)
        theta_plus = theta_np.copy()
        theta_plus[i] += eps
        run_many_states(num_qubits, gate_ids_np, wire1_np, wire2_np, 
                       theta_plus, states_np, states_plus)
        
        # Compute f(θ - ε)
        theta_minus = theta_np.copy()
        theta_minus[i] -= eps
        run_many_states(num_qubits, gate_ids_np, wire1_np, wire2_np,
                       theta_minus, states_np, states_minus)
        
        # Gradient via chain rule: ∂L/∂θᵢ = Re[∑ⱼ (∂L/∂outⱼ)* · ∂outⱼ/∂θᵢ]
        # Finite difference: ∂out/∂θᵢ ≈ (out(θ+ε) - out(θ-ε)) / (2ε)
        d_out = (states_plus - states_minus) / (2 * eps)
        
        # Dot product with cotangent (complex inner product)
        # For complex gradients: <a, b> = sum(conj(a) * b)
        grad_theta[i] = np.sum(np.real(np.conj(cotangent_np) * d_out))
    
    return grad_theta


# Register custom VJP (gradient rule) for JAX
@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 5))
def statevec_simulate_jax(states_in, gate_ids, wire1, wire2, theta, num_qubits):
    """JAX-differentiable statevector simulation using Numba backend.
    
    This is the main entry point for running circuits with JAX autodiff.
    The function is marked with custom_vjp to provide custom gradient rules.
    
    Args:
        states_in: Input states [batch, 2^n] (JAX array, differentiable)
        gate_ids: Gate identifiers (non-differentiable, circuit structure)
        wire1: Qubit indices (non-differentiable, circuit structure)
        wire2: Qubit indices (non-differentiable, circuit structure)
        theta: Gate parameters [num_gates] (differentiable!)
        num_qubits: Number of qubits (non-differentiable)
    
    Returns:
        Output states [batch, 2^n] (JAX array)
    
    Note:
        - Only theta is differentiable (PQC parameters)
        - Circuit structure (gates, wires) is fixed during training
        - Gradients computed via finite differences in backward pass
    """
    # Shape function tells JAX the output shape and dtype
    result_shape = jax.ShapeDtypeStruct(states_in.shape, jnp.complex64)
    
    # Call NumPy implementation via pure_callback
    states_out = jax.pure_callback(
        partial(_statevec_forward_numpy, num_qubits=num_qubits),
        result_shape,
        states_in, gate_ids, wire1, wire2, theta
    )
    
    return states_out


def statevec_simulate_jax_fwd(states_in, gate_ids, wire1, wire2, theta, num_qubits):
    """Forward pass for custom VJP.
    
    Computes output and saves values needed for backward pass.
    """
    out = statevec_simulate_jax(states_in, gate_ids, wire1, wire2, theta, num_qubits)
    # Save values needed for backward pass (inputs + circuit structure)
    residuals = (states_in, theta)
    return out, residuals


def statevec_simulate_jax_bwd(gate_ids, wire1, wire2, num_qubits, residuals, cotangent):
    """Backward pass for custom VJP (computes gradients).
    
    This is called automatically by JAX during backpropagation.
    
    Args:
        gate_ids, wire1, wire2, num_qubits: Circuit structure (from nondiff_argnums)
        residuals: Saved values from forward pass (states_in, theta)
        cotangent: Gradient w.r.t. output (∂L/∂output)
    
    Returns:
        Tuple of gradients for differentiable args: (grad_states_in, grad_theta)
    """
    states_in, theta = residuals
    
    # Compute gradients w.r.t. theta using finite differences
    grad_theta_shape = jax.ShapeDtypeStruct(theta.shape, jnp.float32)
    grad_theta = jax.pure_callback(
        partial(_statevec_backward_finite_diff,
                gate_ids=gate_ids, wire1=wire1, wire2=wire2,
                num_qubits=num_qubits),
        grad_theta_shape,
        states_in, theta, cotangent
    )
    
    # Return gradients for differentiable args: (states_in, theta)
    # Gradient w.r.t. states_in is rarely needed for PQC training, return zeros
    # (In PQC training, we only optimize gate parameters, not input states)
    grad_states = jnp.zeros_like(states_in)
    
    return grad_states, grad_theta


# Register the custom VJP with JAX
statevec_simulate_jax.defvjp(statevec_simulate_jax_fwd, statevec_simulate_jax_bwd)


# High-level convenience wrappers

def run_circuit_batch_jax(circuit_ops, input_states, num_qubits, noise_x=None, noise_z=None):
    """High-level JAX interface for batched circuit simulation with optional noise.
    
    This function handles circuit compilation and noise injection, providing
    a simple interface for running circuits in JAX code.
    
    Args:
        circuit_ops: List of (gate, qubits, params) tuples
        input_states: JAX array [batch, 2^n]
        num_qubits: Number of qubits
        noise_x: Optional X-noise array [num_gates] (applied as RX gates)
        noise_z: Optional Z-noise array [num_gates] (applied as RZ gates)
    
    Returns:
        Output states [batch, 2^n] as JAX array
    
    Example:
        >>> ops = [('h', [0], []), ('cx', [0, 1], [])]
        >>> states = jnp.ones((32, 4), dtype=jnp.complex64) / 2.0
        >>> output = run_circuit_batch_jax(ops, states, 2)
    """
    from ..noise.builder import build_regular_noisy_circuit, build_circuit
    
    # Build circuit with noise if provided
    if noise_x is not None and noise_z is not None:
        gate_ids, wire1, wire2, theta = build_regular_noisy_circuit(
            circuit_ops,
            np.asarray(noise_x, dtype=np.float32),
            np.asarray(noise_z, dtype=np.float32),
            return_tagged=False
        )
    else:
        gate_ids, wire1, wire2, theta = build_circuit(circuit_ops)
    
    # Convert circuit structure to JAX arrays
    # (These are constants during optimization, so no need for gradients)
    gate_ids_jax = jnp.asarray(gate_ids, dtype=jnp.int32)
    wire1_jax = jnp.asarray(wire1, dtype=jnp.int32)
    wire2_jax = jnp.asarray(wire2, dtype=jnp.int32)
    theta_jax = jnp.asarray(theta, dtype=jnp.float32)
    
    # Run simulation
    return statevec_simulate_jax(input_states, gate_ids_jax, wire1_jax, 
                                  wire2_jax, theta_jax, num_qubits)


@jax.jit
def run_circuit_batch_jax_jitted(gate_ids, wire1, wire2, theta, input_states, num_qubits):
    """JIT-compiled version for repeated calls with same circuit structure.
    
    Use this when you have pre-compiled circuit arrays and want maximum speed.
    The circuit structure (gate_ids, wire1, wire2) should be constant, only
    theta (gate parameters) and input_states can vary.
    
    Args:
        gate_ids: Gate identifiers [num_gates] (JAX array)
        wire1: Qubit indices [num_gates] (JAX array)
        wire2: Qubit indices [num_gates] (JAX array)
        theta: Gate parameters [num_gates] (JAX array, differentiable)
        input_states: Input states [batch, 2^n] (JAX array)
        num_qubits: Number of qubits (Python int)
    
    Returns:
        Output states [batch, 2^n]
    """
    return statevec_simulate_jax(input_states, gate_ids, wire1, wire2, theta, num_qubits)
