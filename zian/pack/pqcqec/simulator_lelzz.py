#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZZ-Ring PQC Simulator Extension

Extends simulator_core with ZZ-ring PQC block application:
- Pre-local: RZ-RX-RZ per qubit (3Q angles)
- ZZ-ring: CNOT-RZ-CNOT between adjacent pairs (Q angles)
- Post-local: RZ-RX-RZ per qubit (3Q angles)

Total: 7Q angles per block for Q qubits
"""
from typing import Dict, Optional
import torch
from .simulator_core import (
    Batch, _split_indices, _get_two_qubit_struct,
    _apply_rzrxrz_fused_pairs, _apply_cx, _apply_rz,
    simulate_loss as _base_simulate_loss,
)


def _apply_lelzz_pqc_block(
    states: torch.Tensor,
    angles_block: torch.Tensor,
    n_qubits: int,
    splits: list,
    cx_swap: dict,
    device: torch.device
):
    """Apply one ZZ-ring PQC block: pre-local, ZZ-ring, post-local.
    
    Args:
        states: [B, K, 2^n] quantum states
        angles_block: [B, 7*n_qubits] angles layout:
            [0:3*Q]       : pre_angles (rz1, rx, rz2) × Q qubits
            [3*Q:4*Q]     : theta_zz × Q pairs
            [4*Q:7*Q]     : post_angles (rz1, rx, rz2) × Q qubits
        n_qubits: number of qubits
        splits: qubit split indices
        cx_swap: CX swap indices
        device: torch device
    """
    B = states.size(0)
    Q = n_qubits
    
    # Extract angle groups
    pre_angles = angles_block[:, :3*Q].view(B, Q, 3)      # [B, Q, 3]
    theta_zz = angles_block[:, 3*Q:4*Q]                    # [B, Q]
    post_angles = angles_block[:, 4*Q:7*Q].view(B, Q, 3)  # [B, Q, 3]
    
    # 1. Pre-local rotations: RZ-RX-RZ per qubit
    for q in range(Q):
        i0, i1 = splits[q]
        a_rz1 = pre_angles[:, q, 0]
        a_rx  = pre_angles[:, q, 1]
        a_rz2 = pre_angles[:, q, 2]
        _apply_rzrxrz_fused_pairs(states, i0, i1, a_rz1, a_rx, a_rz2)
    
    # 2. ZZ-ring: CNOT-RZ-CNOT for adjacent pairs in a ring
    for q in range(Q):
        q0 = q
        q1 = (q + 1) % Q
        # Apply CNOT(q0, q1)
        _apply_cx(states, q0, q1)
        # Apply RZ(theta) on q1
        # theta_zz[:, q] has shape [B], need to expand to [B, K] for all K states
        K = states.size(1)
        theta_expanded = theta_zz[:, q].unsqueeze(1).expand(B, K).reshape(B * K)
        # Reshape states to [B*K, 2^n] for RZ application
        orig_shape = states.shape
        states_flat = states.view(B * K, -1)
        # Apply RZ with scalar broadcast (will be applied to each of B*K states)
        i0, i1 = splits[q1]
        em = torch.exp(-0.5j * theta_expanded).unsqueeze(-1)
        ep = torch.exp(0.5j * theta_expanded).unsqueeze(-1)
        states_flat[..., i0] *= em
        states_flat[..., i1] *= ep
        # Reshape back
        states.copy_(states_flat.view(orig_shape))
        # Apply CNOT(q0, q1) again
        _apply_cx(states, q0, q1)
    
    # 3. Post-local rotations: RZ-RX-RZ per qubit
    for q in range(Q):
        i0, i1 = splits[q]
        a_rz1 = post_angles[:, q, 0]
        a_rx  = post_angles[:, q, 1]
        a_rz2 = post_angles[:, q, 2]
        _apply_rzrxrz_fused_pairs(states, i0, i1, a_rz1, a_rx, a_rz2)


def simulate_loss_lelzz_blocks(
    batch: Batch,
    logits: torch.Tensor,
    init_cache: Dict[int, torch.Tensor],
    ref_cache: dict,
    noise_schedules: dict,
    gate_blocks: int,
    device: Optional[torch.device] = None,
    detach_base_noise: bool = True
) -> torch.Tensor:
    """Simulate with ZZ-ring PQC blocks.
    
    Args:
        batch: Batch of circuits (must have uniform n_qubits)
        logits: [B, blocks_needed*7*n_qubits, 1] predicted angles
        init_cache: initial states per n_qubits
        ref_cache: reference states
        noise_schedules: noise parameters
        gate_blocks: number of base gates per block
        device: torch device
        detach_base_noise: detach gradients through base+noise
    
    Returns:
        loss: 1 - fidelity
    """
    if device is None:
        device = logits.device
    
    B = batch.base_g.size(0)
    n = int(batch.n_qubits[0].item())
    
    # Verify uniform n_qubits
    assert (batch.n_qubits == n).all(), "All circuits must have same n_qubits for lelzz mode"
    
    # Initialize states
    states = init_cache[n].to(device).unsqueeze(0).expand(B, -1, -1).clone()
    
    # Get reference states
    rows = torch.tensor([ref_cache['idx2row'][int(i.item())] for i in batch.idx], device=device)
    ref = ref_cache['tensor'].index_select(0, rows)
    
    # Compute blocks needed
    Lb = int(batch.base_len[0].item())
    import math
    blocks_needed = math.ceil(Lb / max(1, gate_blocks)) if Lb > 0 else 1
    
    # Reshape logits to [B, blocks_needed, 7*n]
    expected_angles = blocks_needed * 7 * n
    angles_flat = logits[:, :expected_angles, 0]  # [B, expected_angles]
    
    # Pad if needed
    if angles_flat.size(1) < expected_angles:
        pad = torch.zeros(B, expected_angles - angles_flat.size(1), device=device, dtype=angles_flat.dtype)
        angles_flat = torch.cat([angles_flat, pad], dim=1)
    
    angles_blk = angles_flat.view(B, blocks_needed, 7 * n)  # [B, blocks_needed, 7*n]
    
    # Get base circuit info
    gate_ids = batch.base_g[:, :Lb].to(device)
    q1 = batch.base_q1[:, :Lb].to(device)
    q2 = batch.base_q2[:, :Lb].to(device)
    
    # Get noise info
    noise_rows = torch.tensor([noise_schedules['idx2row'][int(i.item())] for i in batch.idx], device=device)
    
    # Get split and CX indices
    splits = _split_indices(n, device)
    cx_swap, cz_mask = _get_two_qubit_struct(n, device)
    
    # Apply base gates + noise + PQC blocks
    from .simulator_core import _apply_base_step_batched, _apply_noise_step_batched, _try_fused_base_noise_segment, PAD_ID
    
    t = 0
    blk_idx = 0
    first_block = True
    
    while t < Lb:
        t_end = min(Lb, (blk_idx + 1) * gate_blocks)
        seg_len = t_end - t
        
        # Apply base gates + noise for this segment
        if seg_len > 0 and noise_schedules.get('use_noise', False):
            g_seg = gate_ids[:, t:t_end].contiguous()
            q1_seg = q1[:, t:t_end].contiguous()
            q2_seg = q2[:, t:t_end].contiguous()
            rz1_seg = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, t:t_end].contiguous()
            rx1_seg = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, t:t_end].contiguous()
            rz2_seg = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, t:t_end].contiguous()
            rx2_seg = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, t:t_end].contiguous()
            
            # Try fused kernel
            used = _try_fused_base_noise_segment(states, g_seg, q1_seg, q2_seg, rz1_seg, rx1_seg, rz2_seg, rx2_seg)
            
            if not used:
                # Fallback: apply base + noise sequentially
                for tt in range(t, t_end):
                    g_t = gate_ids[:, tt]
                    if (g_t == PAD_ID).all():
                        break
                    q1_t = q1[:, tt]
                    q2_t = q2[:, tt]
                    _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap, cz_mask)
                    
                    rx1_t = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, tt]
                    rz1_t = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, tt]
                    rx2_t = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, tt]
                    rz2_t = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, tt]
                    _apply_noise_step_batched(states, q1_t, q2_t, rx1_t, rz1_t, rx2_t, rz2_t, splits)
        else:
            # No noise
            for tt in range(t, t_end):
                g_t = gate_ids[:, tt]
                if (g_t == PAD_ID).all():
                    break
                q1_t = q1[:, tt]
                q2_t = q2[:, tt]
                _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap, cz_mask)
                
                if noise_schedules.get('use_noise', False):
                    rx1_t = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, tt]
                    rz1_t = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, tt]
                    rx2_t = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, tt]
                    rz2_t = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, tt]
                    _apply_noise_step_batched(states, q1_t, q2_t, rx1_t, rz1_t, rx2_t, rz2_t, splits)
        
        t = t_end
        
        # Detach after first base segment if requested
        if t < Lb and detach_base_noise and first_block:
            states = states.detach()
            first_block = False
        
        # Apply ZZ-ring PQC block
        if t < Lb or blk_idx == 0:  # Always apply at least one block
            angs_block = angles_blk[:, blk_idx]  # [B, 7*n]
            _apply_lelzz_pqc_block(states, angs_block, n, splits, cx_swap, device)
            blk_idx += 1
    
    # Apply final block if we have more predicted angles
    if blk_idx < blocks_needed:
        angs_block = angles_blk[:, blk_idx]
        _apply_lelzz_pqc_block(states, angs_block, n, splits, cx_swap, device)
    
    # Compute fidelity loss
    ov = (ref.conj() * states).sum(-1)  # [B, K]
    F = (ov.abs() ** 2).mean()
    
    return 1 - F
