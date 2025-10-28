#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for ZZ-Ring PQC Architecture - Apple Silicon MPS Optimized

This is an MPS (Metal Performance Shaders) optimized version specifically for Apple Silicon.

Key MPS Optimizations:
- Uses torch.mps device for GPU acceleration on Apple Silicon
- Disables AMP (not fully supported on MPS)
- Uses MPS-optimized tensor operations
- Implements custom MPS-friendly gradient clipping
- Uses larger batch sizes to leverage unified memory architecture
- Optimized for Metal framework compute kernels

PQC Architecture per block:
- Pre-local: RZ-RX-RZ on each qubit (3*Q angles)
- ZZ-ring: CNOT-RZ-CNOT between adjacent pairs in a ring (Q angles)
- Post-local: RZ-RX-RZ on each qubit (3*Q angles)

Total: 7*Q angles per block for Q qubits

ENHANCED MODEL FEATURES:
The transformer now receives rich gate information per block instead of just gate counts:
1. Gate type histograms (distribution of H, X, Z, CX, CZ gates)
2. Qubit usage patterns (which qubits are active in each block)
3. Entanglement connectivity (2-qubit gate interaction matrix)
4. Traditional statistics (count, cumulative, block index)
5. Previous block angles (autoregressive context)

This gives the model actual structural information about the base circuit
to better predict optimal PQC angles.
"""

# Enable MPS fallback for unsupported operations (must be set before importing torch)
import os as _os
if 'PYTORCH_ENABLE_MPS_FALLBACK' not in _os.environ:
    _os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    print("[MPS] Auto-enabling PYTORCH_ENABLE_MPS_FALLBACK=1 for complex gradient support")

from typing import Optional
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .simulator_core import (
    CircuitDataset, Batch, collate,
    build_base_cache_vectorized, NoiseConfig,
)
from .simulator_lelzz_mps import simulate_loss_lelzz_blocks_mps
from .dataset import MAX_BASE_LEN, MAX_PARAM, MAX_QUBITS

# Model hyperparameters - tuned for Apple Silicon
HID_DIM = 768
N_LAYERS = 8
N_HEADS = 12
FF_DIM = HID_DIM * 4
DROP = 0.15
PREV_K = 2  # Increased from 1 for better context

# MPS device configuration
def get_mps_device():
    """Get MPS device if available, otherwise fallback to CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        print("[MPS] Warning: MPS not available, falling back to CPU")
        return torch.device('cpu')

DEVICE = get_mps_device()
MAX_SEQ = MAX_BASE_LEN + MAX_PARAM

print(f"[MPS] Using device: {DEVICE}")
print(f"[MPS] MPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    print(f"[MPS] MPS built: {torch.backends.mps.is_built()}")


class ZZRingAnglePredictorMPS(nn.Module):
    """MPS-optimized Transformer model for ZZ-Ring PQC angle prediction.
    
    Architecture optimizations for Apple Silicon:
    - Uses contiguous memory layouts for Metal kernels
    - Avoids operations with poor MPS support
    - Batched operations for unified memory efficiency
    - FP32 only (MPS AMP support is limited)
    
    Enhanced with actual gate information:
    - Gate type histograms (5 base gates: H, X, Z, CX, CZ)
    - Qubit usage patterns per block
    - Entanglement structure (2-qubit gate connectivity)
    """
    
    def __init__(self, gate_blocks: int, n_qubits: int):
        super().__init__()
        self.gate_blocks = int(gate_blocks)
        self.n_qubits = int(n_qubits)
        self.angles_per_block = 7 * n_qubits
        
        import math
        self.max_blocks = math.ceil(MAX_BASE_LEN / max(1, gate_blocks))
        
        # Gate embeddings (5 gate types: H, X, Z, CX, CZ)
        self.n_gate_types = 5
        self.gate_emb_dim = 32
        self.gate_emb = nn.Embedding(self.n_gate_types, self.gate_emb_dim)
        
        # Input features per block:
        # - Gate type histogram: 5 (one per gate type)
        # - Qubit usage: n_qubits (binary usage per qubit)
        # - Entanglement features: n_qubits * n_qubits (connectivity matrix flattened)
        # - Gate count statistics: 3 (count, cumulative, block_index)
        # - Previous angles: angles_per_block * PREV_K
        gate_hist_dim = self.n_gate_types
        qubit_usage_dim = n_qubits
        connectivity_dim = n_qubits * n_qubits
        stats_dim = 3
        prev_angles_dim = self.angles_per_block * PREV_K
        
        feat_dim = gate_hist_dim + qubit_usage_dim + connectivity_dim + stats_dim + prev_angles_dim
        
        # Input projection with MPS-friendly LayerNorm
        self.in_proj = nn.Sequential(
            nn.Linear(feat_dim, HID_DIM),
            nn.GELU(),
            nn.Dropout(DROP),
            nn.LayerNorm(HID_DIM)
        )
        
        self.pos_emb = nn.Embedding(self.max_blocks, HID_DIM)
        
        # Causal transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            HID_DIM, N_HEADS, FF_DIM, DROP,
            batch_first=True, norm_first=True
        )
        try:
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=N_LAYERS,
                enable_nested_tensor=False  # MPS doesn't support nested tensors well
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        
        # Output head: 2 values (x,y on unit circle) per angle
        self.head_ln = nn.LayerNorm(HID_DIM)
        self.head = nn.Linear(HID_DIM, 2 * self.angles_per_block)
        
        # Initialize to predict identity (all angles = 0)
        with torch.no_grad():
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            if self.head.bias.numel() >= 2 * self.angles_per_block:
                b = self.head.bias.view(self.angles_per_block, 2)
                b[:, 0] = 1.0  # x=1 -> angle=0
                b[:, 1] = 0.0  # y=0
    
    def _extract_block_features(self, batch: Batch, device: torch.device, max_blocks: int) -> torch.Tensor:
        """Extract rich gate features per block from the batch.
        
        MPS-optimized: Uses contiguous tensors and efficient operations.
        
        Args:
            batch: Batch of circuits
            device: torch device (MPS)
            max_blocks: Maximum number of blocks across batch
            
        Returns:
            structural_feats: [B, max_blocks, feat_dim] structural features
            counts: [B, max_blocks] gate counts per block
        """
        import math
        B = batch.base_g.size(0)
        
        # Initialize feature tensors on device
        gate_hist = torch.zeros(B, max_blocks, self.n_gate_types, device=device, dtype=torch.float32)
        qubit_usage = torch.zeros(B, max_blocks, self.n_qubits, device=device, dtype=torch.float32)
        connectivity = torch.zeros(B, max_blocks, self.n_qubits, self.n_qubits, device=device, dtype=torch.float32)
        counts = torch.zeros(B, max_blocks, device=device, dtype=torch.float32)
        
        # Process each sample in batch
        for i in range(B):
            Lb = int(batch.base_len[i].item())
            T = math.ceil(Lb / max(1, self.gate_blocks))
            
            # Extract gates and qubits for this sample (already on device)
            gates_i = batch.base_g[i, :Lb]
            q1_i = batch.base_q1[i, :Lb]
            q2_i = batch.base_q2[i, :Lb]
            
            # Process each block
            for t in range(T):
                s = t * self.gate_blocks
                e = min(Lb, (t + 1) * self.gate_blocks)
                block_size = e - s
                
                if block_size == 0:
                    continue
                
                # Extract block gates and qubits
                block_gates = gates_i[s:e]
                block_q1 = q1_i[s:e]
                block_q2 = q2_i[s:e]
                
                # Count gate types (histogram)
                for g_idx in range(self.n_gate_types):
                    gate_hist[i, t, g_idx] = (block_gates == g_idx).sum().float()
                
                # Track qubit usage and connectivity
                for j in range(block_size):
                    q1 = int(block_q1[j].item())
                    q2 = int(block_q2[j].item())
                    
                    if 0 <= q1 < self.n_qubits:
                        qubit_usage[i, t, q1] = 1.0
                    
                    # For 2-qubit gates, track connectivity
                    if q2 >= 0 and q2 < self.n_qubits and 0 <= q1 < self.n_qubits:
                        connectivity[i, t, q1, q2] += 1.0
                        connectivity[i, t, q2, q1] += 1.0  # Symmetric
                        qubit_usage[i, t, q2] = 1.0
                
                # Gate count
                counts[i, t] = float(block_size)
        
        # Flatten connectivity matrix
        connectivity_flat = connectivity.reshape(B, max_blocks, self.n_qubits * self.n_qubits)
        
        # Normalize features (MPS-friendly operations)
        gate_hist = gate_hist / (gate_hist.sum(dim=-1, keepdim=True) + 1e-8)
        connectivity_flat = connectivity_flat / (connectivity_flat.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Concatenate all structural features
        structural_feats = torch.cat([
            gate_hist,           # [B, max_blocks, n_gate_types]
            qubit_usage,         # [B, max_blocks, n_qubits]
            connectivity_flat,   # [B, max_blocks, n_qubits^2]
        ], dim=-1).contiguous()  # Ensure contiguous for MPS
        
        return structural_feats, counts
    
    def _angles_from_s1(self, logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Convert S¹ representation to angles [-π, π].
        
        MPS-optimized: Uses contiguous operations.
        
        Args:
            logits: [B, 2*angles_per_block] raw outputs
            eps: numerical stability epsilon
            
        Returns:
            angles: [B, angles_per_block] in [-π, π]
        """
        B = logits.size(0)
        xy = logits.view(B, self.angles_per_block, 2).contiguous()  # Ensure contiguous for MPS
        x, y = xy[..., 0], xy[..., 1]
        
        # Normalize to unit circle
        r = torch.sqrt(x*x + y*y + eps*eps)
        x_norm = x / r
        y_norm = y / r
        
        # Convert to angle - atan2 is well-supported on MPS
        theta = torch.atan2(y_norm, x_norm)
        return theta
    
    def forward(self, batch: Batch, device: torch.device) -> torch.Tensor:
        """Predict angles for all blocks in batch.
        
        MPS-optimized: Minimizes device transfers, uses contiguous tensors.
        
        Args:
            batch: Batch of circuits (must have uniform n_qubits)
            device: torch device (should be 'mps')
            
        Returns:
            logits: [B, max_blocks*angles_per_block, 1] predicted angles
        """
        import math
        B = batch.base_g.size(0)
        Lb_max = int(batch.base_len.max().item())
        
        # Compute number of blocks per sample
        max_blocks = math.ceil(Lb_max / max(1, self.gate_blocks))
        
        # Extract rich gate features per block
        structural_feats, counts = self._extract_block_features(batch, device, max_blocks)
        # structural_feats: [B, max_blocks, n_gate_types + n_qubits + n_qubits^2]
        
        # Cumulative counts
        cum = counts.cumsum(dim=1)
        
        # Block indices - create directly on device
        idx_seq = torch.arange(max_blocks, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        
        # Previous angles buffer (autoregressive)
        prev_buf = torch.zeros(B, PREV_K, self.angles_per_block, device=device, dtype=torch.float32)
        prev_seq = torch.zeros(B, max_blocks, self.angles_per_block * PREV_K, device=device, dtype=torch.float32)
        
        # Outputs
        Y = torch.zeros(B, max_blocks, self.angles_per_block, device=device, dtype=torch.float32)
        
        # Causal mask - create once on device
        attn_mask = torch.triu(torch.ones((max_blocks, max_blocks), dtype=torch.bool, device=device), diagonal=1)
        
        # Autoregressive loop over blocks
        for t in range(max_blocks):
            L = t + 1
            
            # Store current prev window
            prev_seq[:, t, :] = prev_buf.reshape(B, self.angles_per_block * PREV_K)
            
            # Build features for blocks [0, t] - all operations on device
            feats = torch.cat([
                structural_feats[:, :L, :],       # Rich gate features (histogram, qubit usage, connectivity)
                counts[:, :L].unsqueeze(-1),      # gate count
                cum[:, :L].unsqueeze(-1),         # cumulative
                idx_seq[:, :L].unsqueeze(-1),     # block index
                prev_seq[:, :L, :],               # previous angles
            ], dim=-1).contiguous()  # Ensure contiguous for MPS
            
            # Project and encode
            x = self.in_proj(feats)
            pos_indices = torch.arange(L, device=device)
            x = x + self.pos_emb(pos_indices).unsqueeze(0)
            
            # Apply causal transformer
            h = self.encoder(x, mask=attn_mask[:L, :L])
            
            # Predict angles from last position
            h_last = self.head_ln(h[:, -1, :])
            logits_t = self.head(h_last)
            
            # Convert to angles
            y_t = self._angles_from_s1(logits_t)
            
            # Sanitize - MPS handles these well
            y_t = torch.nan_to_num(y_t, nan=0.0, posinf=0.0, neginf=0.0)
            y_t = y_t.clamp(min=-torch.pi + 1e-6, max=torch.pi - 1e-6)
            
            # Store
            Y[:, t, :] = y_t
            
            # Update prev buffer for next step
            prev_buf = torch.roll(prev_buf, shifts=-1, dims=1)
            prev_buf[:, -1, :] = y_t
        
        # Reshape to [B, max_blocks*angles_per_block, 1]
        return Y.reshape(B, max_blocks * self.angles_per_block, 1).contiguous()


def mps_clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0):
    """MPS-optimized gradient clipping.
    
    Standard torch.nn.utils.clip_grad_norm_ can be slow on MPS due to
    device transfers. This version keeps everything on device.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    device = parameters[0].grad.device
    
    if norm_type == float('inf'):
        norms = [p.grad.detach().abs().max() for p in parameters]
        total_norm = max(norms)
    else:
        # Compute total norm on device
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type)
                for p in parameters
            ]),
            norm_type
        )
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    # Apply clipping on device
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped)
    
    return total_norm


def train_lelzz_mps(
    data_path: str,
    n_qubits: int = 2,
    batch_size: int = 64,  # Larger batch size for unified memory
    epochs: int = 100,
    lr: float = 1e-3,
    k_random: int = 32,
    num_sample: Optional[int] = None,
    noise: Optional[NoiseConfig] = None,
    device: Optional[torch.device] = None,
    gate_blocks: int = 5,
    detach_base_noise: bool = True,
    resume_checkpoint: Optional[str] = None,
):
    """Train ZZ-ring PQC model on Apple Silicon with MPS.
    
    Args:
        data_path: Path to dataset
        n_qubits: Number of qubits (must match data)
        batch_size: Batch size (default 64 for MPS)
        epochs: Number of training epochs
        lr: Learning rate
        k_random: Number of random initial states
        num_sample: Limit dataset size (None = use all)
        noise: Noise configuration
        device: torch device (None = auto-detect MPS)
        gate_blocks: Base gates per PQC block
        detach_base_noise: Detach gradients through base+noise
    """
    if device is None:
        device = DEVICE
    
    print(f"[MPS-LELZZ] Training ZZ-ring PQC on Apple Silicon")
    print(f"[MPS-LELZZ] Device: {device}")
    print(f"[MPS-LELZZ] n_qubits={n_qubits}, gate_blocks={gate_blocks}")
    print(f"[MPS-LELZZ] Angles per block: 7*{n_qubits} = {7*n_qubits}")
    print(f"[MPS-LELZZ] Batch size: {batch_size} (optimized for unified memory)")
    print(f"[MPS-LELZZ] PREV_K: {PREV_K} (context window)")
    
    # MPS complex gradient limitation check
    if device.type == 'mps':
        print("\n" + "="*70)
        print("[MPS] IMPORTANT: Complex Number Gradient Limitation")
        print("="*70)
        print("MPS does not support complex number gradients (quantum states).")
        print("Solution: Model runs on MPS, quantum simulation on CPU.")
        print("This is a hybrid approach for maximum compatibility.")
        print("="*70 + "\n")
        
        # Use CPU for quantum simulation, MPS only for model
        sim_device = torch.device('cpu')
        print(f"[MPS-LELZZ] Using hybrid mode:")
        print(f"  - Model device: {device} (MPS)")
        print(f"  - Simulation device: {sim_device} (CPU)")
    else:
        sim_device = device
    
    # Load dataset
    ds_full = CircuitDataset(data_path, num_sample=num_sample)
    
    # Normalize n_qubits: promote all circuits to target n_qubits
    # This allows training on circuits with fewer qubits by treating unused qubits as |0⟩
    normalized_count = 0
    for item in ds_full.items:
        if item['n_qubits'] < n_qubits:
            # Verify that all gates only use qubits 0 to n_qubits-1
            max_qubit_used = max(
                [q for q in item['base_q1'] if q >= 0] + 
                [q for q in item['base_q2'] if q >= 0]
            ) if item['base_q1'] else 0
            
            if max_qubit_used < n_qubits:
                item['n_qubits'] = n_qubits  # Promote to target n_qubits
                normalized_count += 1
            else:
                raise RuntimeError(
                    f"Circuit idx={item['idx']} has n_qubits={item['n_qubits']} "
                    f"but uses qubit {max_qubit_used}, cannot normalize to {n_qubits} qubits"
                )
    
    if normalized_count > 0:
        print(f"[LELZZ] Normalized {normalized_count} circuits with fewer qubits to {n_qubits} qubits")
    
    
    # Filter to only circuits with n_qubits
    filtered_items = [item for item in ds_full.items if item['n_qubits'] == n_qubits]
    
    if len(filtered_items) == 0:
        raise RuntimeError(f"No circuits with n_qubits={n_qubits} found in dataset")
    
    # Create filtered dataset
    class FilteredDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            return self.items[i]
    
    ds = FilteredDataset(filtered_items)
    print(f"[MPS-LELZZ] Filtered dataset: {len(ds)} circuits with {n_qubits} qubits")
    
    # Build caches
    # Note: torch.unique with dim parameter not supported on MPS, so build on CPU then transfer
    print(f"[MPS-LELZZ] Building caches on {sim_device}...")
    init_cache, ref_cache, noise_schedules = build_base_cache_vectorized(
        ds, k_random=k_random, device=sim_device, noise=noise
    )
    print(f"[MPS-LELZZ] Caches built on {sim_device}")
    
    # Train/val split
    N = len(ds)
    val_cnt = max(1, N // 10) if N > 1 else 0
    train_cnt = N - val_cnt
    
    indices = list(range(N))
    import random
    random.shuffle(indices)
    
    ds_train = Subset(ds, indices[:train_cnt])
    ds_val = Subset(ds, indices[train_cnt:]) if val_cnt > 0 else None
    
    print(f"[MPS-LELZZ] Train: {len(ds_train)}, Val: {len(ds_val) if ds_val else 0}")
    
    # Data loaders
    collate_fn = lambda xs: collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=MAX_QUBITS)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if ds_val else None
    
    # Model
    print(f"[MPS-LELZZ] Creating model...")
    model = ZZRingAnglePredictorMPS(gate_blocks=gate_blocks, n_qubits=n_qubits).to(device)
    
    print(f"[MPS-LELZZ] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[MPS-LELZZ] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Print angle configuration details
    print("\n" + "="*70)
    print("[MPS-LELZZ] PQC ANGLE CONFIGURATION")
    print("="*70)
    
    import math
    avg_base_len = sum(len(item['base_gates']) for item in ds.items) / len(ds.items)
    typical_blocks = math.ceil(avg_base_len / gate_blocks)
    
    angles_per_block = 7 * n_qubits
    total_angles_per_circuit = typical_blocks * angles_per_block
    
    print(f"\n1. TOTAL ANGLES TO OPTIMIZE (per circuit):")
    print(f"   - Average base gates per circuit: {avg_base_len:.1f}")
    print(f"   - Typical number of PQC blocks: {typical_blocks}")
    print(f"   - Angles per block: {angles_per_block}")
    print(f"   - Total angles per circuit: {total_angles_per_circuit}")
    
    print(f"\n2. ANGLES PER PQC BLOCK:")
    print(f"   - Total: {angles_per_block} angles")
    print(f"   - Layout: [pre_local:{3*n_qubits}, zz_ring:{n_qubits}, post_local:{3*n_qubits}]")
    
    print(f"\n3. MPS OPTIMIZATION FEATURES:")
    print(f"   - Contiguous tensor operations for Metal kernels")
    print(f"   - Unified memory batching (batch_size={batch_size})")
    print(f"   - FP32 precision (MPS AMP has limited support)")
    print(f"   - On-device gradient clipping")
    print(f"   - Minimized CPU↔GPU transfers")
    
    print("\n" + "="*70)
    print()
    
    # Optimizer with weight decay - works well on MPS
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
        fused=False  # Fused optimizer not available on MPS
    )
    
    # LR scheduler
    warmup_ep = min(50, epochs // 10)
    min_lr_ratio = 0.001
    
    def lr_lambda(ep_idx):
        if ep_idx < warmup_ep:
            return (ep_idx + 1) / warmup_ep
        t = ep_idx - warmup_ep
        T = max(1, epochs - warmup_ep)
        cos_inner = math.pi * t / T
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(cos_inner))
    
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    
    # NO AMP on MPS (limited support, can cause issues)
    print("[MPS-LELZZ] Note: AMP disabled for MPS (using FP32)")
    
    # Early stopping and checkpointing
    best_val_fid = 0.0
    patience = 100
    patience_counter = 0
    start_epoch = 1
    checkpoint_dir = "checkpoints_lelzz_mps"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_losses = []
    val_fids = []
    
    # Resume from checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[MPS-LELZZ] Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        sched.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_fid = checkpoint.get('best_val_fid', 0.0)
        train_losses = checkpoint.get('train_losses', [])
        val_fids = checkpoint.get('val_fids', [])
        print(f"[MPS-LELZZ] Resumed from epoch {checkpoint['epoch']}, best_val_fid={best_val_fid:.6f}")
    
    # Evaluation function
    def evaluate():
        if val_loader is None:
            return float('nan')
        model.eval()
        total_fid = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch, device)
                
                # Transfer to sim_device for simulation
                logits_sim = logits.to(sim_device) if sim_device != device else logits
                batch_sim = batch.to(sim_device) if sim_device != device else batch
                
                loss = simulate_loss_lelzz_blocks_mps(
                    batch_sim, logits_sim, init_cache, ref_cache,
                    noise_schedules, gate_blocks, sim_device, detach_base_noise
                )
                
                fid = 1.0 - float(loss.detach().cpu())
                total_fid += fid * batch.base_g.size(0)
                count += batch.base_g.size(0)
        
        model.train()
        return total_fid / max(1, count)
    
    # Training loop
    print(f"\n[MPS-LELZZ] Starting training for {epochs} epochs...")
    
    for ep in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        count = 0
        batch_times = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()
            
            batch = batch.to(device)
            
            # Forward pass (model on device, which may be MPS)
            logits = model(batch, device)
            
            # Compute loss (simulation on sim_device - CPU if using MPS)
            # Transfer logits to sim_device if needed
            logits_sim = logits.to(sim_device) if sim_device != device else logits
            batch_sim = batch.to(sim_device) if sim_device != device else batch
            
            loss = simulate_loss_lelzz_blocks_mps(
                batch_sim, logits_sim, init_cache, ref_cache,
                noise_schedules, gate_blocks, sim_device, detach_base_noise
            )
            
            # Backward pass
            opt.zero_grad(set_to_none=True)
            
            if torch.isfinite(loss):
                loss.backward()
                
                # MPS-optimized gradient clipping
                mps_clip_grad_norm_(model.parameters(), 1.0)
                
                opt.step()
                
                total_loss += float(loss.detach().cpu()) * batch.base_g.size(0)
                count += batch.base_g.size(0)
            else:
                print(f"[MPS-LELZZ] Warning: non-finite loss at epoch {ep}, batch {batch_idx}")
            
            batch_times.append(time.time() - batch_start)
        
        # Step scheduler
        sched.step()
        
        # Evaluate
        avg_loss = total_loss / max(1, count)
        train_fid = 1.0 - avg_loss
        val_fid = evaluate()
        cur_lr = opt.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        
        train_losses.append(avg_loss)
        val_fids.append(val_fid)
        
        # Print progress with detailed timing
        print(f"[MPS-LELZZ] Epoch {ep:4d}/{epochs} | "
              f"Time={epoch_time:.2f}s (avg batch={avg_batch_time*1000:.1f}ms) | "
              f"LR={cur_lr:.6f} | "
              f"Train Loss={avg_loss:.6f} (Fid={train_fid:.6f}) | "
              f"Val Fid={val_fid:.6f}")
        
        # Early stopping and checkpointing
        if val_fid > best_val_fid:
            best_val_fid = val_fid
            patience_counter = 0
            
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{n_qubits}q_gb{gate_blocks}.pt")
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': sched.state_dict(),
                'best_val_fid': best_val_fid,
                'train_losses': train_losses,
                'val_fids': val_fids,
            }, checkpoint_path)
            print(f"[MPS-LELZZ] ✓ New best Val Fid: {best_val_fid:.6f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[MPS-LELZZ] Early stopping triggered after {patience} epochs")
                break
        
        # Periodic checkpoint
        if ep % 100 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{ep}_{n_qubits}q_gb{gate_blocks}.pt")
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': sched.state_dict(),
                'best_val_fid': best_val_fid,
                'train_losses': train_losses,
                'val_fids': val_fids,
            }, checkpoint_path)
            print(f"[MPS-LELZZ] Checkpoint saved at epoch {ep}")
    
    # Final summary
    print(f"\n[MPS-LELZZ] Training complete!")
    print(f"[MPS-LELZZ] Best Val Fid: {best_val_fid:.6f}")
    print(f"[MPS-LELZZ] Final Train Fid: {train_fid:.6f}")
    print(f"[MPS-LELZZ] Final Val Fid: {val_fid:.6f}")
    
    return model


def _build_argparser():
    p = argparse.ArgumentParser(description="Train ZZ-ring PQC on Apple Silicon (MPS)")
    p.add_argument("--data-path", type=str, required=True, help="Path to dataset")
    p.add_argument("--n-qubits", type=int, default=2, help="Number of qubits")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size (default 64 for MPS)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--k-random", type=int, default=32, help="Number of random initial states")
    p.add_argument("--gate-blocks", type=int, default=5, help="Base gates per PQC block")
    p.add_argument("--num-sample", type=int, default=None, help="Limit dataset size")
    p.add_argument("--no-detach-base-noise", action="store_false", dest="detach_base_noise")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    train_lelzz_mps(
        data_path=args.data_path,
        n_qubits=args.n_qubits,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        k_random=args.k_random,
        num_sample=args.num_sample,
        noise=None,
        device=None,  # Auto-detect MPS
        gate_blocks=args.gate_blocks,
        detach_base_noise=args.detach_base_noise,
        resume_checkpoint=args.resume,
    )
