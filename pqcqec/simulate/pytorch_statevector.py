"""
PyTorch-based statevector simulator for quantum circuits.

This module provides a pure PyTorch implementation of quantum state vector simulation,
enabling automatic differentiation through quantum circuits. Similar to the JAX version,
this simulator is fully compatible with PyTorch's autograd.

Key features:
- Full PyTorch autograd support
- GPU acceleration support
- Batched operations via vmap-like functionality
- Clean integration with PyTorch-based training
"""

import torch
from typing import Tuple, List, Optional
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


def _apply_1q_unitary(state: torch.Tensor, n_qubits: int, q: int,
                      a: complex, b: complex, c: complex, d: complex) -> torch.Tensor:
    """
    Apply a general 1-qubit 2x2 unitary matrix [[a,b],[c,d]] to qubit q.
    
    Args:
        state: Quantum state vector of shape (2^n,) or (K, 2^n)
        n_qubits: Number of qubits
        q: Target qubit index
        a, b, c, d: Matrix elements of the unitary
    
    Returns:
        Updated state vector
    """
    dim = state.shape[-1]  # Get state space dimension (works for both shapes)
    bit_pos = n_qubits - 1 - q
    mask = 1 << bit_pos
    
    # Create indices for |0⟩ and |1⟩ on qubit q
    indices = torch.arange(dim, device=state.device)
    indices_0 = indices & ~mask
    indices_1 = indices_0 | mask
    
    # Get amplitudes - handle both single and batched states
    if state.ndim == 1:
        # Single state [2^n]
        u0 = state[indices_0]
        u1 = state[indices_1]
    else:
        # Batched states [K, 2^n]
        u0 = state[:, indices_0]
        u1 = state[:, indices_1]
    
    # Convert scalars to tensors if needed
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=state.dtype, device=state.device)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=state.dtype, device=state.device)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=state.dtype, device=state.device)
    if not isinstance(d, torch.Tensor):
        d = torch.tensor(d, dtype=state.dtype, device=state.device)
    
    # Apply unitary
    new_state = state.clone()
    if state.ndim == 1:
        new_state[indices_0] = a * u0 + b * u1
        new_state[indices_1] = c * u0 + d * u1
    else:
        new_state[:, indices_0] = a * u0 + b * u1
        new_state[:, indices_1] = c * u0 + d * u1
    
    return new_state


def apply_x(state: torch.Tensor, n_qubits: int, q: int) -> torch.Tensor:
    """Apply Pauli-X gate to qubit q."""
    return _apply_1q_unitary(state, n_qubits, q,
                             0.0+0.0j, 1.0+0.0j,
                             1.0+0.0j, 0.0+0.0j)


def apply_z(state: torch.Tensor, n_qubits: int, q: int) -> torch.Tensor:
    """Apply Pauli-Z gate to qubit q."""
    dim = state.shape[0]
    bit_pos = n_qubits - 1 - q
    mask = 1 << bit_pos
    
    # Flip sign of |1⟩ components
    indices = torch.arange(dim, device=state.device)
    phases = torch.where((indices & mask) != 0, 
                        torch.tensor(-1.0, device=state.device, dtype=state.dtype),
                        torch.tensor(1.0, device=state.device, dtype=state.dtype))
    return state * phases


def apply_h(state: torch.Tensor, n_qubits: int, q: int) -> torch.Tensor:
    """Apply Hadamard gate to qubit q."""
    s = 1.0 / torch.sqrt(torch.tensor(2.0, device=state.device)) + 0.0j
    return _apply_1q_unitary(state, n_qubits, q, s, s, s, -s)


def apply_rx(state: torch.Tensor, n_qubits: int, q: int, theta: torch.Tensor) -> torch.Tensor:
    """Apply X-rotation gate to qubit q with angle theta."""
    half_theta = 0.5 * theta
    ct = torch.cos(half_theta)
    st = torch.sin(half_theta)
    
    a = ct + 0.0j
    b = 0.0 - 1j * st
    c = 0.0 - 1j * st
    d = ct + 0.0j
    
    return _apply_1q_unitary(state, n_qubits, q, a, b, c, d)


def apply_ry(state: torch.Tensor, n_qubits: int, q: int, theta: torch.Tensor) -> torch.Tensor:
    """Apply Y-rotation gate to qubit q with angle theta."""
    half_theta = 0.5 * theta
    ct = torch.cos(half_theta)
    st = torch.sin(half_theta)
    
    a = ct + 0.0j
    b = -st + 0.0j
    c = st + 0.0j
    d = ct + 0.0j
    
    return _apply_1q_unitary(state, n_qubits, q, a, b, c, d)


def apply_rz(state: torch.Tensor, n_qubits: int, q: int, theta: torch.Tensor) -> torch.Tensor:
    """Apply Z-rotation gate to qubit q with angle theta."""
    half_theta = 0.5 * theta
    
    # Phase factors for |0⟩ and |1⟩
    e0 = torch.exp(-1j * half_theta)
    e1 = torch.exp(1j * half_theta)
    
    # Handle batched states [K, 2^n] or single state [2^n]
    dim = state.shape[-1]  # Get state space dimension
    bit_pos = n_qubits - 1 - q
    mask = 1 << bit_pos
    
    # Apply phase based on qubit state
    indices = torch.arange(dim, device=state.device)
    phases = torch.where((indices & mask) != 0, e1, e0)
    
    # Broadcast phases to match state shape
    if state.ndim > 1:
        phases = phases.unsqueeze(0)  # [1, 2^n] for broadcasting
    
    return state * phases


def apply_cx(state: torch.Tensor, n_qubits: int, control: int, target: int) -> torch.Tensor:
    """Apply CX gate with specified control and target qubits."""
    dim = state.shape[-1]
    control_bit_pos = n_qubits - 1 - control
    target_bit_pos = n_qubits - 1 - target
    
    mc = 1 << control_bit_pos
    mt = 1 << target_bit_pos
    
    # Create a new state by swapping appropriate pairs
    indices = torch.arange(dim, device=state.device)
    
    # For each index, determine where its amplitude should come from
    # If control=1, swap target bit; otherwise keep same
    source_indices = torch.where(
        (indices & mc) != 0,  # control=1
        indices ^ mt,          # flip target bit
        indices                # keep same
    )
    
    # Handle both single and batched states
    if state.ndim == 1:
        return state[source_indices]
    else:
        return state[:, source_indices]


def apply_cz(state: torch.Tensor, n_qubits: int, control: int, target: int) -> torch.Tensor:
    """Apply controlled-Z gate with specified control and target qubits."""
    dim = state.shape[-1]
    control_bit_pos = n_qubits - 1 - control
    target_bit_pos = n_qubits - 1 - target
    
    mc = 1 << control_bit_pos
    mt = 1 << target_bit_pos
    
    # Flip phase only when both qubits are |1⟩
    indices = torch.arange(dim, device=state.device)
    both_1 = ((indices & mc) != 0) & ((indices & mt) != 0)
    phases = torch.where(both_1, 
                        torch.tensor(-1.0, device=state.device, dtype=state.dtype),
                        torch.tensor(1.0, device=state.device, dtype=state.dtype))
    
    # Broadcast phases for batched states
    if state.ndim > 1:
        phases = phases.unsqueeze(0)
    
    return state * phases


def apply_gate(state: torch.Tensor, n_qubits: int, gate_id: int,
               wire1: int, wire2: int, theta: torch.Tensor) -> torch.Tensor:
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
    # PyTorch doesn't have a direct equivalent to jax.lax.switch
    # Use if-elif chain instead
    if gate_id == GATE_X:
        return apply_x(state, n_qubits, wire1)
    elif gate_id == GATE_Z:
        return apply_z(state, n_qubits, wire1)
    elif gate_id == GATE_H:
        return apply_h(state, n_qubits, wire1)
    elif gate_id == GATE_RX:
        return apply_rx(state, n_qubits, wire1, theta)
    elif gate_id == GATE_RY:
        return apply_ry(state, n_qubits, wire1, theta)
    elif gate_id == GATE_RZ:
        return apply_rz(state, n_qubits, wire1, theta)
    elif gate_id == GATE_CX:
        return apply_cx(state, n_qubits, wire1, wire2)
    elif gate_id == GATE_CZ:
        return apply_cz(state, n_qubits, wire1, wire2)
    else:
        raise ValueError(f"Unknown gate_id: {gate_id}")


def torch_run_circuit_with_state(state: torch.Tensor, n_qubits: int,
                                  gate_ids: torch.Tensor, wire1s: torch.Tensor,
                                  wire2s: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
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
    current_state = state
    n_gates = gate_ids.shape[0]
    
    for i in range(n_gates):
        gate_id = int(gate_ids[i].item())
        wire1 = int(wire1s[i].item())
        wire2 = int(wire2s[i].item())
        theta = thetas[i]
        
        current_state = apply_gate(current_state, n_qubits, gate_id, wire1, wire2, theta)
    
    return current_state


def torch_run_many_states(n_qubits: int, gate_ids: torch.Tensor,
                          wire1s: torch.Tensor, wire2s: torch.Tensor,
                          thetas: torch.Tensor, states_in: torch.Tensor) -> torch.Tensor:
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
    batch_size = states_in.shape[0]
    results = []
    
    for i in range(batch_size):
        result = torch_run_circuit_with_state(
            states_in[i], n_qubits, gate_ids, wire1s, wire2s, thetas
        )
        results.append(result)
    
    return torch.stack(results)


def build_torch_circuit(circuit_ops: List[Tuple], 
                        dtype: torch.dtype = torch.float32,
                        device: Optional[torch.device] = None) -> Tuple[torch.Tensor, ...]:
    """
    Convert a high-level circuit description into PyTorch tensors for the executor.
    
    This is the PyTorch equivalent of build_jax_circuit. It produces PyTorch tensors
    for seamless integration with PyTorch training.
    
    Args:
        circuit_ops: List of tuples (gate_name, qubits, params)
        dtype: Data type for angles (float32 or float64)
        device: Device to place tensors on (CPU or CUDA)
    
    Returns:
        Tuple of (gate_ids, wire1s, wire2s, thetas) as PyTorch tensors
    """
    if device is None:
        device = torch.device('cpu')
    
    gate_ids, w1, w2, th = [], [], [], []
    
    for op in circuit_ops:
        gate, qubits, param = op
        g = GATE_DICT[gate]
        
        # Single-qubit gates without parameters
        if g in (GATE_X, GATE_Z, GATE_H):
            gate_ids.append(g)
            w1.append(qubits[0])
            w2.append(-1)
            th.append(torch.tensor(0.0, dtype=dtype, device=device))
        
        # Parameterized single-qubit rotation gates
        elif g in (GATE_RX, GATE_RY, GATE_RZ):
            gate_ids.append(g)
            w1.append(qubits[0])
            w2.append(-1)
            # Ensure param[0] is a PyTorch scalar tensor
            p = torch.as_tensor(param[0], dtype=dtype, device=device)
            # Ensure it's a scalar
            if p.dim() > 0:
                p = p.squeeze()
            th.append(p)
        
        # Two-qubit controlled gates
        elif g in (GATE_CX, GATE_CZ):
            gate_ids.append(g)
            w1.append(qubits[0])
            w2.append(qubits[1])
            th.append(torch.tensor(0.0, dtype=dtype, device=device))
        
        else:
            raise ValueError(f"Unknown gate code: {g}")
    
    # Stack into PyTorch tensors
    return (
        torch.tensor(gate_ids, dtype=torch.int32, device=device),
        torch.tensor(w1, dtype=torch.int32, device=device),
        torch.tensor(w2, dtype=torch.int32, device=device),
        torch.stack(th),
    )


def torch_create_zero_state(n_qubits: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create the |0...0⟩ computational basis state."""
    if device is None:
        device = torch.device('cpu')
    
    state = torch.zeros((2**n_qubits,), dtype=torch.complex64, device=device)
    state[0] = 1.0 + 0.0j
    return state


def torch_create_ones_state(n_qubits: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create the |1...1⟩ computational basis state."""
    if device is None:
        device = torch.device('cpu')
    
    state = torch.zeros((2**n_qubits,), dtype=torch.complex64, device=device)
    state[-1] = 1.0 + 0.0j
    return state


# Aliases for consistency with JAX version
run_circuit_with_state = torch_run_circuit_with_state
run_many_states = torch_run_many_states
