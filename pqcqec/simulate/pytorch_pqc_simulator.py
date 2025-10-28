"""
PyTorch PQC Simulator with LEL-ZZ architecture.

This module extends pytorch_statevector.py to support:
- Noise gate insertion (RX, RZ after each base gate)
- LEL-ZZ PQC block application (pre-local, ZZ-ring, post-local)
- Fidelity-based loss computation
"""

import torch
from typing import List, Tuple, Optional
import math

from .pytorch_statevector import (
    torch_run_circuit_with_state,
    torch_run_many_states,
    build_torch_circuit,
    apply_rx,
    apply_rz,
    apply_cx,
)


def add_noise_to_circuit_ops(
    circuit_ops: List[Tuple],
    x_noise: torch.Tensor,
    z_noise: torch.Tensor
) -> List[Tuple]:
    """
    Add noise gates after each base gate.
    
    Args:
        circuit_ops: List of (gate_name, qubits, params) tuples
        x_noise: [num_gates] X-rotation noise per gate
        z_noise: [num_gates] Z-rotation noise per gate
    
    Returns:
        circuit_ops_with_noise: Extended circuit with noise gates
    """
    noisy_ops = []
    
    for i, (gate, qubits, params) in enumerate(circuit_ops):
        # Add base gate
        noisy_ops.append((gate, qubits, params))
        
        # Add noise on each qubit this gate acts on
        for q in qubits:
            noisy_ops.append(('rx', [q], [float(x_noise[i])]))
            noisy_ops.append(('rz', [q], [float(z_noise[i])]))
    
    return noisy_ops


def apply_lelzz_pqc_block(
    states: torch.Tensor,
    n_qubits: int,
    pre_angles: torch.Tensor,
    theta_zz: torch.Tensor,
    post_angles: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Apply one LEL-ZZ PQC block: pre-local, ZZ-ring, post-local.
    
    Architecture:
    - Pre-local: RZ-RX-RZ per qubit (3*n_qubits angles)
    - ZZ-ring: CNOT-RZ-CNOT between adjacent pairs (n_qubits angles)
    - Post-local: RZ-RX-RZ per qubit (3*n_qubits angles)
    
    Args:
        states: [K, 2^n] quantum states
        n_qubits: number of qubits
        pre_angles: [n_qubits, 3] pre-local angles (rz1, rx, rz2)
        theta_zz: [n_qubits] ZZ-ring angles
        post_angles: [n_qubits, 3] post-local angles (rz1, rx, rz2)
        device: torch device
    
    Returns:
        states: Updated quantum states
    """
    K = states.shape[0]
    
        # 1. Pre-local rotations: RZ-RX-RZ per qubit
    for q in range(n_qubits):
        rz1 = pre_angles[q, 0]
        rx = pre_angles[q, 1]
        rz2 = pre_angles[q, 2]
        
        # Apply RZ(rz1)
        states = apply_rz(states, n_qubits, q, rz1)
        # Apply RX(rx)
        states = apply_rx(states, n_qubits, q, rx)
        # Apply RZ(rz2)
        states = apply_rz(states, n_qubits, q, rz2)
    
    # 2. ZZ-ring: CNOT-RZ-CNOT for adjacent pairs in a ring
    for q in range(n_qubits):
        q0 = q
        q1 = (q + 1) % n_qubits
        
        # Apply CNOT(q0, q1)
        states = apply_cx(states, n_qubits, q0, q1)
        
        # Apply RZ(theta) on q1
        states = apply_rz(states, n_qubits, q1, theta_zz[q])
        
        # Apply CNOT(q0, q1) again
        states = apply_cx(states, n_qubits, q0, q1)
    
    # 3. Post-local rotations: RZ-RX-RZ per qubit
    for q in range(n_qubits):
        rz1 = post_angles[q, 0]
        rx = post_angles[q, 1]
        rz2 = post_angles[q, 2]
        
        # Apply RZ(rz1)
        states = apply_rz(states, n_qubits, q, rz1)
        # Apply RX(rx)
        states = apply_rx(states, n_qubits, q, rx)
        # Apply RZ(rz2)
        states = apply_rz(states, n_qubits, q, rz2)
    
    return states
    
    # 2. ZZ-ring: CNOT-RZ-CNOT for adjacent pairs in a ring
    for q in range(n_qubits):
        q0 = q
        q1 = (q + 1) % n_qubits
        
        # Apply CNOT(q0, q1)
        apply_cx(states, n_qubits, q0, q1)
        
        # Apply RZ(theta) on q1
        apply_rz(states, n_qubits, q1, theta_zz[q])
        
        # Apply CNOT(q0, q1) again
        apply_cx(states, n_qubits, q0, q1)
    
    # 3. Post-local rotations: RZ-RX-RZ per qubit
    for q in range(n_qubits):
        rz1 = post_angles[q, 0]
        rx = post_angles[q, 1]
        rz2 = post_angles[q, 2]
        
        apply_rz(states, n_qubits, q, rz1)
        apply_rx(states, n_qubits, q, rx)
        apply_rz(states, n_qubits, q, rz2)
    
    return states


def compute_state_fidelity(state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
    """
    Compute fidelity between two quantum states.
    
    Fidelity = |<state1|state2>|^2
    
    Args:
        state1: [2^n] complex quantum state
        state2: [2^n] complex quantum state
    
    Returns:
        fidelity: scalar in [0, 1]
    """
    overlap = torch.sum(torch.conj(state1) * state2)
    fidelity = torch.abs(overlap) ** 2
    return fidelity


def simulate_block_progressive(
    input_states: torch.Tensor,
    block_idx: int,
    gate_blocks: int,
    n_qubits: int,
    circuit_ops: List[Tuple],
    x_noise: torch.Tensor,
    z_noise: torch.Tensor,
    prev_pqc_angles: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    curr_pqc_angles: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """
    Simulate progressive training for a single block.
    
    Progressive: Start from input, apply all previous blocks + current block.
    
    Args:
        input_states: [K, 2^n] initial states
        block_idx: current block index (0-based)
        gate_blocks: number of base gates per block
        n_qubits: number of qubits
        circuit_ops: full circuit operations
        x_noise: [num_gates] X-noise array
        z_noise: [num_gates] Z-noise array
        prev_pqc_angles: list of (pre, theta_zz, post) for previous blocks
        curr_pqc_angles: (pre, theta_zz, post) for current block
        device: torch device
    
    Returns:
        output_states: [K, 2^n] states after all blocks up to current
    """
    states = input_states.clone()
    K = states.shape[0]
    
    # Apply all blocks up to and including current block
    for blk_i in range(block_idx + 1):
        gate_start = blk_i * gate_blocks
        gate_end = min((blk_i + 1) * gate_blocks, len(circuit_ops))
        
        if gate_start >= len(circuit_ops):
            break
        
        # Get gates for this block
        block_gates = circuit_ops[gate_start:gate_end]
        
        # Add noise to gates
        block_x_noise = x_noise[gate_start:gate_end]
        block_z_noise = z_noise[gate_start:gate_end]
        noisy_block_gates = add_noise_to_circuit_ops(block_gates, block_x_noise, block_z_noise)
        
        # Build circuit and run
        gate_ids, wire1s, wire2s, thetas = build_torch_circuit(noisy_block_gates, device=device)
        states = torch_run_many_states(n_qubits, gate_ids, wire1s, wire2s, thetas, states)
        
        # Apply PQC block
        if blk_i < block_idx:
            # Previous block (frozen)
            pre, theta_zz, post = prev_pqc_angles[blk_i]
        else:
            # Current block (trainable)
            pre, theta_zz, post = curr_pqc_angles
        
        states = apply_lelzz_pqc_block(states, n_qubits, pre, theta_zz, post, device)
    
    return states


def simulate_block_individual(
    input_states: torch.Tensor,
    block_idx: int,
    gate_blocks: int,
    n_qubits: int,
    circuit_ops: List[Tuple],
    x_noise: torch.Tensor,
    z_noise: torch.Tensor,
    curr_pqc_angles: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """
    Simulate individual training for a single block.
    
    Individual: Start from input, apply only current block's gates.
    
    Args:
        input_states: [K, 2^n] initial states
        block_idx: current block index (0-based)
        gate_blocks: number of base gates per block
        n_qubits: number of qubits
        circuit_ops: full circuit operations
        x_noise: [num_gates] X-noise array
        z_noise: [num_gates] Z-noise array
        curr_pqc_angles: (pre, theta_zz, post) for current block
        device: torch device
    
    Returns:
        output_states: [K, 2^n] states after current block only
    """
    states = input_states.clone()
    
    # Get gates for this block only
    gate_start = block_idx * gate_blocks
    gate_end = min((block_idx + 1) * gate_blocks, len(circuit_ops))
    
    if gate_start >= len(circuit_ops):
        # No gates for this block, just apply PQC
        pre, theta_zz, post = curr_pqc_angles
        states = apply_lelzz_pqc_block(states, n_qubits, pre, theta_zz, post, device)
        return states
    
    block_gates = circuit_ops[gate_start:gate_end]
    
    # Add noise to gates
    block_x_noise = x_noise[gate_start:gate_end]
    block_z_noise = z_noise[gate_start:gate_end]
    noisy_block_gates = add_noise_to_circuit_ops(block_gates, block_x_noise, block_z_noise)
    
    # Build circuit and run
    gate_ids, wire1s, wire2s, thetas = build_torch_circuit(noisy_block_gates, device=device)
    states = torch_run_many_states(n_qubits, gate_ids, wire1s, wire2s, thetas, states)
    
    # Apply PQC block
    pre, theta_zz, post = curr_pqc_angles
    states = apply_lelzz_pqc_block(states, n_qubits, pre, theta_zz, post, device)
    
    return states


def compute_target_states_progressive(
    input_states: torch.Tensor,
    block_idx: int,
    gate_blocks: int,
    n_qubits: int,
    circuit_ops: List[Tuple],
    device: torch.device
) -> torch.Tensor:
    """
    Compute target states for progressive training.
    
    Target = noiseless output after gates [0:(block_idx+1)*gate_blocks]
    
    Args:
        input_states: [K, 2^n] initial states
        block_idx: current block index
        gate_blocks: gates per block
        n_qubits: number of qubits
        circuit_ops: full circuit (no noise)
        device: torch device
    
    Returns:
        target_states: [K, 2^n] ideal noiseless states
    """
    gate_end = min((block_idx + 1) * gate_blocks, len(circuit_ops))
    target_gates = circuit_ops[:gate_end]
    
    if len(target_gates) == 0:
        return input_states.clone()
    
    gate_ids, wire1s, wire2s, thetas = build_torch_circuit(target_gates, device=device)
    target_states = torch_run_many_states(n_qubits, gate_ids, wire1s, wire2s, thetas, input_states)
    
    return target_states


def compute_target_states_individual(
    input_states: torch.Tensor,
    block_idx: int,
    gate_blocks: int,
    n_qubits: int,
    circuit_ops: List[Tuple],
    device: torch.device
) -> torch.Tensor:
    """
    Compute target states for individual training.
    
    Target = noiseless output of only gates [block_idx*gate_blocks:(block_idx+1)*gate_blocks]
    
    Args:
        input_states: [K, 2^n] initial states
        block_idx: current block index
        gate_blocks: gates per block
        n_qubits: number of qubits
        circuit_ops: full circuit (no noise)
        device: torch device
    
    Returns:
        target_states: [K, 2^n] ideal noiseless states
    """
    gate_start = block_idx * gate_blocks
    gate_end = min((block_idx + 1) * gate_blocks, len(circuit_ops))
    
    if gate_start >= len(circuit_ops):
        return input_states.clone()
    
    target_gates = circuit_ops[gate_start:gate_end]
    
    if len(target_gates) == 0:
        return input_states.clone()
    
    gate_ids, wire1s, wire2s, thetas = build_torch_circuit(target_gates, device=device)
    target_states = torch_run_many_states(n_qubits, gate_ids, wire1s, wire2s, thetas, input_states)
    
    return target_states


def compute_fidelity_loss(
    predicted_states: torch.Tensor,
    target_states: torch.Tensor
) -> torch.Tensor:
    """
    Compute fidelity loss: 1 - mean(fidelity).
    
    Args:
        predicted_states: [K, 2^n] predicted states
        target_states: [K, 2^n] target states
    
    Returns:
        loss: scalar loss (lower is better)
    """
    K = predicted_states.shape[0]
    fidelities = []
    
    for k in range(K):
        fid = compute_state_fidelity(target_states[k], predicted_states[k])
        fidelities.append(fid)
    
    mean_fidelity = torch.mean(torch.stack(fidelities))
    loss = 1.0 - mean_fidelity
    
    return loss
