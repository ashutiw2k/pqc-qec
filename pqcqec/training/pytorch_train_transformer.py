"""
PyTorch Transformer model for PQC angle prediction.

This module contains:
- ZZRingAnglePredictorPyTorch: Transformer that predicts LEL-ZZ PQC angles
- Training functions for progressive and individual modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

from ..simulate.pytorch_pqc_simulator import (
    simulate_block_progressive,
    simulate_block_individual,
    compute_target_states_progressive,
    compute_target_states_individual,
    compute_fidelity_loss,
)
from ..simulate.pytorch_statevector import build_torch_circuit
from ..utils.pytorch_utils import (
    create_circuit_ops_from_data,
    pad_gate_sequence,
)


# Model hyperparameters (matching Zian's configuration)
HID_DIM = 768
N_LAYERS = 8
N_HEADS = 12
FF_DIM = HID_DIM * 4
DROP = 0.15
PREV_K = 4


class ZZRingAnglePredictorPyTorch(nn.Module):
    """
    Transformer model that predicts 7*n_qubits LEL-ZZ angles per PQC block.
    
    Input features per block:
    - Gate sequence: gate_ids, wire1s, wire2s (flattened)
    - Block statistics: gate_count, cumulative_count, block_index
    - Previous angles: angles from previous K blocks
    
    Output:
    - 7*n_qubits angles per block (S¹ representation converted to angles)
    """
    
    def __init__(self, gate_blocks: int, n_qubits: int, max_blocks: int = 100):
        """
        Initialize the transformer model.
        
        Args:
            gate_blocks: Number of base gates per PQC block
            n_qubits: Number of qubits (fixed for this model)
            max_blocks: Maximum number of blocks to support
        """
        super().__init__()
        self.gate_blocks = int(gate_blocks)
        self.n_qubits = int(n_qubits)
        self.angles_per_block = 7 * n_qubits
        self.max_blocks = max_blocks
        
        # Input features:
        # - Gate sequence: gate_blocks * 3 (gate_id, wire1, wire2)
        # - Block stats: 3 (gate_count, cumulative, block_idx)
        # - Previous angles: PREV_K * angles_per_block
        self.feat_dim = gate_blocks * 3 + 3 + PREV_K * self.angles_per_block
        
        # Input projection
        self.in_proj = nn.Sequential(
            nn.Linear(self.feat_dim, HID_DIM),
            nn.GELU(),
            nn.Dropout(DROP),
            nn.LayerNorm(HID_DIM)
        )
        
        # Positional embeddings for block positions
        self.pos_emb = nn.Embedding(self.max_blocks, HID_DIM)
        
        # Causal transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            HID_DIM, N_HEADS, FF_DIM, DROP,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        
        # Output head: 2 values (x, y on unit circle) per angle
        self.head_ln = nn.LayerNorm(HID_DIM)
        self.head = nn.Linear(HID_DIM, 2 * self.angles_per_block)
        
        # Cache for attention masks
        self._attn_mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        
        # Initialize to predict identity (all angles = 0)
        self._init_head()
    
    def _init_head(self):
        """Initialize the output head to predict angle 0 (x=1, y=0)."""
        with torch.no_grad():
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            if self.head.bias.numel() >= 2 * self.angles_per_block:
                b = self.head.bias.view(self.angles_per_block, 2)
                b[:, 0] = 1.0  # x=1 -> angle=0
                b[:, 1] = 0.0  # y=0
    
    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Get or create cached causal attention mask."""
        cache_key = (size, device)
        if cache_key not in self._attn_mask_cache:
            mask = torch.triu(
                torch.ones((size, size), dtype=torch.bool, device=device),
                diagonal=1
            )
            self._attn_mask_cache[cache_key] = mask
        return self._attn_mask_cache[cache_key]
    
    def _angles_from_s1(self, logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Convert S¹ representation to angles [-π, π].
        
        Args:
            logits: [..., 2*angles_per_block] raw outputs
            eps: numerical stability epsilon
        
        Returns:
            angles: [..., angles_per_block] in [-π, π]
        """
        in_shape = logits.shape
        all_dims = in_shape[:-1]
        
        xy = logits.view(*all_dims, self.angles_per_block, 2)  # [..., A, 2]
        x, y = xy[..., 0], xy[..., 1]
        
        # Normalize to unit circle
        r = torch.hypot(x, y).clamp(min=eps)
        x_norm = x / r
        y_norm = y / r
        
        # Convert to angle
        theta = torch.atan2(y_norm, x_norm)  # Already in [-π, π]
        
        # Handle any numerical issues
        theta = torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
        
        return theta
    
    def extract_block_features(
        self,
        circuit_ops: List[Tuple],
        block_idx: int,
        prev_angles: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Extract features for a single block.
        
        Args:
            circuit_ops: Full circuit operations
            block_idx: Which block (0-indexed)
            prev_angles: [PREV_K * angles_per_block] previous predictions
            device: torch device
        
        Returns:
            features: [feat_dim] feature vector for this block
        """
        # Get gates for this block
        gate_start = block_idx * self.gate_blocks
        gate_end = min((block_idx + 1) * self.gate_blocks, len(circuit_ops))
        block_gates = circuit_ops[gate_start:gate_end]
        
        # Build gate features
        if len(block_gates) > 0:
            gate_ids, wire1s, wire2s, _ = build_torch_circuit(block_gates, device=device)
            # Pad to gate_blocks length
            gate_ids, wire1s, wire2s, actual_len = pad_gate_sequence(
                gate_ids, wire1s, wire2s, self.gate_blocks
            )
        else:
            # Empty block
            gate_ids = torch.full((self.gate_blocks,), -1, dtype=torch.int32, device=device)
            wire1s = torch.full((self.gate_blocks,), -1, dtype=torch.int32, device=device)
            wire2s = torch.full((self.gate_blocks,), -1, dtype=torch.int32, device=device)
            actual_len = 0
        
        # Flatten gate features: [gate_blocks * 3]
        gate_features = torch.cat([
            gate_ids.float(),
            wire1s.float(),
            wire2s.float()
        ])
        
        # Block statistics
        gate_count = float(actual_len)
        cumulative = gate_end
        block_index_val = float(block_idx)
        stats = torch.tensor([gate_count, cumulative, block_index_val], device=device)
        
        # Concatenate all features
        features = torch.cat([gate_features, stats, prev_angles])
        
        return features
    
    def forward_single_block(
        self,
        circuit_ops: List[Tuple],
        block_idx: int,
        prev_angles_buffer: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Predict angles for a single block (autoregressive).
        
        Args:
            circuit_ops: Full circuit operations
            block_idx: Current block index
            prev_angles_buffer: [PREV_K, angles_per_block] previous angles
            device: torch device
        
        Returns:
            angles: [angles_per_block] predicted angles
        """
        # Flatten previous angles
        prev_angles_flat = prev_angles_buffer.flatten()  # [PREV_K * angles_per_block]
        
        # Extract features for this block
        features = self.extract_block_features(
            circuit_ops, block_idx, prev_angles_flat, device
        )  # [feat_dim]
        
        # Add batch dimension
        features = features.unsqueeze(0)  # [1, feat_dim]
        
        # Project and add positional embedding
        x = self.in_proj(features)  # [1, HID_DIM]
        pos = self.pos_emb(torch.tensor([block_idx], device=device))  # [1, HID_DIM]
        x = x + pos
        
        # For single block, no need for causal mask
        h = self.encoder(x)  # [1, HID_DIM]
        
        # Predict angles
        h_norm = self.head_ln(h)  # [1, HID_DIM]
        logits = self.head(h_norm)  # [1, 2*angles_per_block]
        
        # Convert to angles
        angles = self._angles_from_s1(logits)  # [1, angles_per_block]
        
        return angles.squeeze(0)  # [angles_per_block]


def train_transformer_progressive(
    model: ZZRingAnglePredictorPyTorch,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    gate_blocks: int,
    n_qubits: int,
    k_random: int,
    noise_x_rad: float,
    noise_z_rad: float,
    epochs: int,
    device: torch.device,
    seed: int = 0
):
    """
    Train transformer in progressive mode.
    
    Progressive: Each block is trained on cumulative gates from start.
    """
    from ..utils.pytorch_utils import (
        generate_random_initial_states,
        generate_fixed_noise
    )
    
    model.train()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_losses = []
        epoch_fidelities = []
        
        pbar = tqdm(dataloader, desc="Training")
        
        for batch in pbar:
            for circuit in batch:
                # Extract circuit data
                circuit_ops = create_circuit_ops_from_data(circuit)
                num_gates = len(circuit_ops)
                
                # Calculate number of blocks
                num_blocks = math.ceil(num_gates / gate_blocks) if num_gates > 0 else 1
                
                # Generate initial states
                input_states = generate_random_initial_states(
                    n_qubits, k_random, device, seed=seed
                )
                
                # Generate fixed noise for this circuit
                x_noise, z_noise = generate_fixed_noise(
                    num_gates, noise_x_rad, noise_z_rad, seed=seed + circuit['idx']
                )
                x_noise = x_noise.to(device)
                z_noise = z_noise.to(device)
                
                # Previous angles buffer
                prev_angles_buffer = torch.zeros(
                    (PREV_K, model.angles_per_block), device=device
                )
                
                # Store predicted angles for all blocks
                all_predicted_angles = []
                
                # Train block by block
                for block_idx in range(num_blocks):
                    optimizer.zero_grad()
                    
                    # Predict angles for this block
                    predicted_angles = model.forward_single_block(
                        circuit_ops, block_idx, prev_angles_buffer, device
                    )  # [angles_per_block]
                    
                    # DEBUG: Check if angles have gradients
                    #print(f"predicted_angles requires_grad: {predicted_angles.requires_grad}")
                    
                    all_predicted_angles.append(predicted_angles)
                    
                    # Reshape to LEL-ZZ format
                    pre_angles = predicted_angles[:3*n_qubits].view(n_qubits, 3)
                    theta_zz = predicted_angles[3*n_qubits:4*n_qubits]
                    post_angles = predicted_angles[4*n_qubits:7*n_qubits].view(n_qubits, 3)
                    
                    # Collect previous blocks' angles (frozen)
                    prev_pqc_angles = []
                    for prev_idx in range(block_idx):
                        prev_ang = all_predicted_angles[prev_idx].detach()
                        prev_pre = prev_ang[:3*n_qubits].view(n_qubits, 3)
                        prev_theta = prev_ang[3*n_qubits:4*n_qubits]
                        prev_post = prev_ang[4*n_qubits:7*n_qubits].view(n_qubits, 3)
                        prev_pqc_angles.append((prev_pre, prev_theta, prev_post))
                    
                    # Simulate progressive
                    predicted_states = simulate_block_progressive(
                        input_states, block_idx, gate_blocks, n_qubits,
                        circuit_ops, x_noise, z_noise,
                        prev_pqc_angles,
                        (pre_angles, theta_zz, post_angles),
                        device
                    )
                    
                    # Compute target states
                    target_states = compute_target_states_progressive(
                        input_states, block_idx, gate_blocks, n_qubits,
                        circuit_ops, device
                    )
                    
                    # Compute loss
                    loss = compute_fidelity_loss(predicted_states, target_states)
                    
                    # DEBUG: Check if loss has gradients
                    if not loss.requires_grad:
                        print(f"WARNING: loss doesn't require grad!")
                        print(f"predicted_angles.requires_grad: {predicted_angles.requires_grad}")
                        print(f"predicted_states.requires_grad: {predicted_states.requires_grad}")
                        print(f"pre_angles.requires_grad: {pre_angles.requires_grad}")
                    
                    # Backward
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    # Update previous angles buffer
                    prev_angles_buffer = torch.roll(prev_angles_buffer, shifts=-1, dims=0)
                    prev_angles_buffer[-1] = predicted_angles.detach()
                    
                    # Track metrics
                    fidelity = 1.0 - loss.item()
                    epoch_losses.append(loss.item())
                    epoch_fidelities.append(fidelity)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': np.mean(epoch_losses[-10:]),
                    'fid': np.mean(epoch_fidelities[-10:])
                })
        
        # Step scheduler
        scheduler.step()
        
        # Epoch summary
        print(f"Epoch {epoch+1} - Loss: {np.mean(epoch_losses):.6f}, "
              f"Fidelity: {np.mean(epoch_fidelities):.6f}")


def train_transformer_individual(
    model: ZZRingAnglePredictorPyTorch,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    gate_blocks: int,
    n_qubits: int,
    k_random: int,
    noise_x_rad: float,
    noise_z_rad: float,
    epochs: int,
    device: torch.device,
    seed: int = 0
):
    """
    Train transformer in individual mode.
    
    Individual: Each block is trained independently on its own gates.
    """
    from ..utils.pytorch_utils import (
        generate_random_initial_states,
        generate_fixed_noise
    )
    
    model.train()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_losses = []
        epoch_fidelities = []
        
        pbar = tqdm(dataloader, desc="Training")
        
        for batch in pbar:
            for circuit in batch:
                # Extract circuit data
                circuit_ops = create_circuit_ops_from_data(circuit)
                num_gates = len(circuit_ops)
                
                # Calculate number of blocks
                num_blocks = math.ceil(num_gates / gate_blocks) if num_gates > 0 else 1
                
                # Generate initial states
                input_states = generate_random_initial_states(
                    n_qubits, k_random, device, seed=seed
                )
                
                # Generate fixed noise for this circuit
                x_noise, z_noise = generate_fixed_noise(
                    num_gates, noise_x_rad, noise_z_rad, seed=seed + circuit['idx']
                )
                x_noise = x_noise.to(device)
                z_noise = z_noise.to(device)
                
                # Previous angles buffer (for context, not used in simulation)
                prev_angles_buffer = torch.zeros(
                    (PREV_K, model.angles_per_block), device=device
                )
                
                # Train block by block
                for block_idx in range(num_blocks):
                    optimizer.zero_grad()
                    
                    # Predict angles for this block
                    predicted_angles = model.forward_single_block(
                        circuit_ops, block_idx, prev_angles_buffer, device
                    )  # [angles_per_block]
                    
                    # Reshape to LEL-ZZ format
                    pre_angles = predicted_angles[:3*n_qubits].view(n_qubits, 3)
                    theta_zz = predicted_angles[3*n_qubits:4*n_qubits]
                    post_angles = predicted_angles[4*n_qubits:7*n_qubits].view(n_qubits, 3)
                    
                    # Simulate individual (isolated block)
                    predicted_states = simulate_block_individual(
                        input_states, block_idx, gate_blocks, n_qubits,
                        circuit_ops, x_noise, z_noise,
                        (pre_angles, theta_zz, post_angles),
                        device
                    )
                    
                    # Compute target states (individual)
                    target_states = compute_target_states_individual(
                        input_states, block_idx, gate_blocks, n_qubits,
                        circuit_ops, device
                    )
                    
                    # Compute loss
                    loss = compute_fidelity_loss(predicted_states, target_states)
                    
                    # Backward
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    # Update previous angles buffer (for context)
                    prev_angles_buffer = torch.roll(prev_angles_buffer, shifts=-1, dims=0)
                    prev_angles_buffer[-1] = predicted_angles.detach()
                    
                    # Track metrics
                    fidelity = 1.0 - loss.item()
                    epoch_losses.append(loss.item())
                    epoch_fidelities.append(fidelity)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': np.mean(epoch_losses[-10:]),
                    'fid': np.mean(epoch_fidelities[-10:])
                })
        
        # Step scheduler
        scheduler.step()
        
        # Epoch summary
        print(f"Epoch {epoch+1} - Loss: {np.mean(epoch_losses):.6f}, "
              f"Fidelity: {np.mean(epoch_fidelities):.6f}")
