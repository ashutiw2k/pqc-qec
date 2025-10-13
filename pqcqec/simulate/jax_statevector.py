"""
JAX-based statevector simulator for quantum circuits.

This module provides a pure JAX implementation of quantum state vector simulation,
enabling automatic differentiation through quantum circuits. Unlike the Numba version,
this simulator is fully compatible with JAX's autodiff and JIT compilation.

Key features:
- Full JAX autodiff support (no custom gradients needed)
- JIT-compilable for performance
- Batched operations via vmap
- Clean integration with JAX-based training
"""

import jax
import jax.numpy as jnp
from typing import Tuple, List
from functools import partial

from ..utils.constants import GateEnums, GATE_DICT


# Gate enums
GATE_X = GateEnums.GATE_X
GATE_Z = GateEnums.GATE_Z
GATE_H = GateEnums.GATE_H
GATE_RX = GateEnums.GATE_RX
GATE_RY = GateEnums.GATE_RY
GATE_RZ = GateEnums.GATE_RZ
GATE_CX = GateEnums.GATE_CX
GATE_CZ = GateEnums.GATE_CZ


@partial(jax.jit, static_argnums=(1, 2))
def _apply_1q_unitary(state: jnp.ndarray, n_qubits: int, q: int, 
                     a: complex, b: complex, c: complex, d: complex) -> jnp.ndarray:
    """
    Apply a general 1-qubit 2x2 unitary matrix [[a,b],[c,d]] to qubit q.
    
    Args:
        state: Quantum state vector of shape (2^n,)
        n_qubits: Number of qubits
        q: Target qubit index
        a, b, c, d: Matrix elements of the unitary
    
    Returns:
        Updated state vector
    """
    dim = state.shape[0]
    bit_pos = n_qubits - 1 - q
    mask = 1 << bit_pos
    
    # Create indices for |0⟩ and |1⟩ on qubit q
    indices_0 = jnp.arange(dim) & ~mask
    indices_1 = indices_0 | mask
    
    # Get amplitudes
    u0 = state[indices_0]
    u1 = state[indices_1]
    
    # Apply unitary
    new_state = state.at[indices_0].set(a * u0 + b * u1)
    new_state = new_state.at[indices_1].set(c * u0 + d * u1)
    
    return new_state


@partial(jax.jit, static_argnums=(1, 2))
def apply_x(state: jnp.ndarray, n_qubits: int, q: int) -> jnp.ndarray:
    """Apply Pauli-X gate to qubit q."""
    return _apply_1q_unitary(state, n_qubits, q,
                             0.0+0.0j, 1.0+0.0j,
                             1.0+0.0j, 0.0+0.0j)


@partial(jax.jit, static_argnums=(1, 2))
def apply_z(state: jnp.ndarray, n_qubits: int, q: int) -> jnp.ndarray:
    """Apply Pauli-Z gate to qubit q."""
    dim = state.shape[0]
    bit_pos = n_qubits - 1 - q
    mask = 1 << bit_pos
    
    # Flip sign of |1⟩ components
    phases = jnp.where((jnp.arange(dim) & mask) != 0, -1.0, 1.0)
    return state * phases


@partial(jax.jit, static_argnums=(1, 2))
def apply_h(state: jnp.ndarray, n_qubits: int, q: int) -> jnp.ndarray:
    """Apply Hadamard gate to qubit q."""
    s = 1.0 / jnp.sqrt(2.0) + 0.0j
    return _apply_1q_unitary(state, n_qubits, q, s, s, s, -s)


@partial(jax.jit, static_argnums=(1, 2))
def apply_rx(state: jnp.ndarray, n_qubits: int, q: int, theta: float) -> jnp.ndarray:
    """Apply X-rotation gate to qubit q with angle theta."""
    half_theta = 0.5 * theta
    ct = jnp.cos(half_theta)
    st = jnp.sin(half_theta)
    
    a = ct + 0.0j
    b = 0.0 - 1j * st
    c = 0.0 - 1j * st
    d = ct + 0.0j
    
    return _apply_1q_unitary(state, n_qubits, q, a, b, c, d)


@partial(jax.jit, static_argnums=(1, 2))
def apply_ry(state: jnp.ndarray, n_qubits: int, q: int, theta: float) -> jnp.ndarray:
    """Apply Y-rotation gate to qubit q with angle theta."""
    half_theta = 0.5 * theta
    ct = jnp.cos(half_theta)
    st = jnp.sin(half_theta)
    
    a = ct + 0.0j
    b = -st + 0.0j
    c = st + 0.0j
    d = ct + 0.0j
    
    return _apply_1q_unitary(state, n_qubits, q, a, b, c, d)


@partial(jax.jit, static_argnums=(1, 2))
def apply_rz(state: jnp.ndarray, n_qubits: int, q: int, theta: float) -> jnp.ndarray:
    """Apply Z-rotation gate to qubit q with angle theta."""
    half_theta = 0.5 * theta
    
    # Phase factors for |0⟩ and |1⟩
    e0 = jnp.exp(-1j * half_theta)
    e1 = jnp.exp(1j * half_theta)
    
    dim = state.shape[0]
    bit_pos = n_qubits - 1 - q
    mask = 1 << bit_pos
    
    # Apply phase based on qubit state
    phases = jnp.where((jnp.arange(dim) & mask) != 0, e1, e0)
    return state * phases


@partial(jax.jit, static_argnums=(1, 2, 3))
def apply_cx(state: jnp.ndarray, n_qubits: int, control: int, target: int) -> jnp.ndarray:
    """Apply CNOT gate with specified control and target qubits."""
    dim = state.shape[0]
    control_bit_pos = n_qubits - 1 - control
    target_bit_pos = n_qubits - 1 - target
    
    mc = 1 << control_bit_pos
    mt = 1 << target_bit_pos
    
    # Create a new state by swapping appropriate pairs
    indices = jnp.arange(dim)
    
    # For each index, determine where its amplitude should come from
    # If control=1, swap target bit; otherwise keep same
    source_indices = jnp.where(
        (indices & mc) != 0,  # control=1
        indices ^ mt,          # flip target bit
        indices                # keep same
    )
    
    return state[source_indices]


@partial(jax.jit, static_argnums=(1, 2, 3))
def apply_cz(state: jnp.ndarray, n_qubits: int, control: int, target: int) -> jnp.ndarray:
    """Apply controlled-Z gate with specified control and target qubits."""
    dim = state.shape[0]
    control_bit_pos = n_qubits - 1 - control
    target_bit_pos = n_qubits - 1 - target
    
    mc = 1 << control_bit_pos
    mt = 1 << target_bit_pos
    
    # Flip phase only when both qubits are |1⟩
    indices = jnp.arange(dim)
    both_1 = (indices & mc != 0) & (indices & mt != 0)
    phases = jnp.where(both_1, -1.0, 1.0)
    
    return state * phases


@partial(jax.jit, static_argnums=(1, 2))
def apply_gate(state: jnp.ndarray, n_qubits: int, gate_id: int, 
               wire1: int, wire2: int, theta: float) -> jnp.ndarray:
    """
    Apply a single gate to the state vector.
    
    Args:
        state: Current state vector
        n_qubits: Number of qubits
        gate_id: Gate type identifier (1-8 from GateEnums)
        wire1: First wire (target for 1q, control for 2q)
        wire2: Second wire (unused for 1q, target for 2q)
        theta: Rotation angle (unused for non-parametric gates)
    
    Returns:
        Updated state vector
    """
    # Use lax.switch for efficient branching in JIT
    # Note: gate_id starts from 1 (enum.auto()), so subtract 1 for 0-based indexing
    return jax.lax.switch(
        gate_id - 1,  # Convert from 1-based enum to 0-based index
        [
            lambda s: apply_x(s, n_qubits, wire1),      # GATE_X = 1 -> index 0
            lambda s: apply_z(s, n_qubits, wire1),      # GATE_Z = 2 -> index 1
            lambda s: apply_h(s, n_qubits, wire1),      # GATE_H = 3 -> index 2
            lambda s: apply_rx(s, n_qubits, wire1, theta),  # GATE_RX = 4 -> index 3
            lambda s: apply_ry(s, n_qubits, wire1, theta),  # GATE_RY = 5 -> index 4
            lambda s: apply_rz(s, n_qubits, wire1, theta),  # GATE_RZ = 6 -> index 5
            lambda s: apply_cx(s, n_qubits, wire1, wire2),  # GATE_CX = 7 -> index 6
            lambda s: apply_cz(s, n_qubits, wire1, wire2),  # GATE_CZ = 8 -> index 7
        ],
        state
    )


@partial(jax.jit, static_argnums=(1,))
def run_circuit_with_state(state: jnp.ndarray, n_qubits: int,
                           gate_ids: jnp.ndarray, wire1s: jnp.ndarray,
                           wire2s: jnp.ndarray, thetas: jnp.ndarray) -> jnp.ndarray:
    """
    Execute a quantum circuit on a state vector.
    
    Args:
        state: Initial state vector of shape (2^n,)
        n_qubits: Number of qubits
        gate_ids: Array of gate type identifiers
        wire1s: Array of first wire indices
        wire2s: Array of second wire indices (unused for 1q gates)
        thetas: Array of rotation angles (unused for non-parametric gates)
    
    Returns:
        Final state vector after applying all gates
    """
    def apply_single_gate(state, gate_info):
        gate_id, wire1, wire2, theta = gate_info
        return apply_gate(state, n_qubits, gate_id, wire1, wire2, theta), None
    
    # Use scan for efficient sequential application
    final_state, _ = jax.lax.scan(
        apply_single_gate,
        state,
        (gate_ids, wire1s, wire2s, thetas)
    )
    
    return final_state


@partial(jax.jit, static_argnums=(0,))
def run_many_states(n_qubits: int, gate_ids: jnp.ndarray, 
                   wire1s: jnp.ndarray, wire2s: jnp.ndarray,
                   thetas: jnp.ndarray, states_in: jnp.ndarray) -> jnp.ndarray:
    """
    Execute the same quantum circuit on a batch of input states.
    
    Args:
        n_qubits: Number of qubits
        gate_ids: Array of gate type identifiers
        wire1s: Array of first wire indices
        wire2s: Array of second wire indices
        thetas: Array of rotation angles
        states_in: Batch of input states of shape (batch_size, 2^n)
    
    Returns:
        Batch of output states of shape (batch_size, 2^n)
    """
    # Use vmap to vectorize over the batch dimension
    batched_run = jax.vmap(
        lambda state: run_circuit_with_state(state, n_qubits, gate_ids, wire1s, wire2s, thetas),
        in_axes=0
    )
    
    return batched_run(states_in)


def build_jax_circuit(circuit_ops: List[Tuple], dtype=jnp.float32) -> Tuple[jnp.ndarray, ...]:
    """
    Convert a high-level circuit description into JAX arrays for the executor.
    
    This is the JAX equivalent of build_numba_circuit. It produces JAX arrays
    instead of NumPy arrays for seamless integration with JAX training.
    
    Args:
        circuit_ops: List of tuples (gate_name, qubits, params)
        dtype: Data type for angles (float32 or float64)
    
    Returns:
        Tuple of (gate_ids, wire1s, wire2s, thetas) as JAX arrays
    """
    gate_ids, w1, w2, th = [], [], [], []
    
    for op in circuit_ops:
        gate, qubits, param = op
        g = GATE_DICT[gate]
        
        # Single-qubit gates without parameters
        if g in (GATE_X, GATE_Z, GATE_H):
            gate_ids.append(g)
            w1.append(qubits[0])
            w2.append(-1)
            th.append(jnp.array(0.0, dtype=dtype))
        
        # Parameterized single-qubit rotation gates
        elif g in (GATE_RX, GATE_RY, GATE_RZ):
            gate_ids.append(g)
            w1.append(qubits[0])
            w2.append(-1)
            # Ensure param[0] is a JAX scalar array
            p = jnp.asarray(param[0], dtype=dtype)
            # Reshape only if not already scalar
            if p.shape != ():
                p = p.reshape(())
            th.append(p)
        
        # Two-qubit controlled gates
        elif g in (GATE_CX, GATE_CZ):
            gate_ids.append(g)
            w1.append(qubits[0])
            w2.append(qubits[1])
            th.append(jnp.array(0.0, dtype=dtype))
        
        else:
            raise ValueError(f"Unknown gate code: {g}")
    
    # Stack into JAX arrays
    return (
        jnp.asarray(gate_ids, dtype=jnp.int32),
        jnp.asarray(w1, dtype=jnp.int32),
        jnp.asarray(w2, dtype=jnp.int32),
        jnp.stack(th),  # Use stack since all elements are already JAX scalars
    )


@partial(jax.jit, static_argnums=(0,))
def create_zero_state(n_qubits: int) -> jnp.ndarray:
    """Create the |0...0⟩ computational basis state."""
    state = jnp.zeros((2**n_qubits,), dtype=jnp.complex64)
    state = state.at[0].set(1.0 + 0.0j)
    return state


@partial(jax.jit, static_argnums=(0,))
def create_ones_state(n_qubits: int) -> jnp.ndarray:
    """Create the |1...1⟩ computational basis state."""
    state = jnp.zeros((2**n_qubits,), dtype=jnp.complex64)
    state = state.at[-1].set(1.0 + 0.0j)
    return state


# Keep these for backwards compatibility
run_circuit_with_state_jit = run_circuit_with_state
run_many_states_jit = run_many_states
