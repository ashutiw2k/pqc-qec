#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for ZZ-Ring PQC Architecture

PQC Architecture per block:
- Pre-local: RZ-RX-RZ on each qubit (3*Q angles)
- ZZ-ring: CNOT-RZ-CNOT between adjacent pairs in a ring (Q angles)
- Post-local: RZ-RX-RZ on each qubit (3*Q angles)

Total: 7*Q angles per block for Q qubits

This trains on full multi-qubit circuits (not subcircuits).
"""
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
from .simulator_lelzz import simulate_loss_lelzz_blocks
from .dataset import MAX_BASE_LEN, MAX_PARAM, MAX_QUBITS
from .precision import get_amp_settings, make_grad_scaler

# Model hyperparameters
HID_DIM = 768
N_LAYERS = 8
N_HEADS = 12
FF_DIM = HID_DIM * 4
DROP = 0.15  # Increased dropout for better regularization
PREV_K = 1  # Sliding window for more context
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ = MAX_BASE_LEN + MAX_PARAM


class ZZRingAnglePredictor(nn.Module):
    """Transformer model that predicts 7*n_qubits angles per PQC block.
    
    Architecture:
    - Input: per-block gate statistics + previous block angles
    - Encoder: Causal transformer with positional embeddings
    - Output: 7*n_qubits angles via S¹ (circle) representation
    """
    
    def __init__(self, gate_blocks: int, n_qubits: int):
        super().__init__()
        self.gate_blocks = int(gate_blocks)
        self.n_qubits = int(n_qubits)
        self.angles_per_block = 7 * n_qubits
        
        import math
        self.max_blocks = math.ceil(MAX_BASE_LEN / max(1, gate_blocks))
        
        # Input features: [gate_count, cumulative_count, block_index, prev_angles_flattened]
        feat_dim = 3 + self.angles_per_block * PREV_K
        
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
                enable_nested_tensor=False
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
    
    def _angles_from_s1(self, logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Convert S¹ representation to angles [-π, π].
        
        Args:
            logits: [B, 2*angles_per_block] raw outputs
            eps: numerical stability epsilon
            
        Returns:
            angles: [B, angles_per_block] in [-π, π]
        """
        B = logits.size(0)
        xy = logits.view(B, self.angles_per_block, 2)  # [B, angles, 2]
        x, y = xy[..., 0], xy[..., 1]
        
        # Normalize to unit circle
        r = torch.sqrt(x*x + y*y + eps*eps)
        x_norm = x / r
        y_norm = y / r
        
        # Convert to angle
        theta = torch.atan2(y_norm, x_norm)  # [-π, π]
        return theta
    
    def forward(self, batch: Batch, device: torch.device) -> torch.Tensor:
        """Predict angles for all blocks in batch.
        
        Args:
            batch: Batch of circuits (must have uniform n_qubits)
            device: torch device
            
        Returns:
            logits: [B, max_blocks*angles_per_block, 1] predicted angles
        """
        B = batch.base_g.size(0)
        Lb_max = int(batch.base_len.max().item())
        
        # Compute number of blocks per sample
        import math
        max_blocks = math.ceil(Lb_max / max(1, self.gate_blocks))
        
        # Compute gate counts per block
        counts = torch.zeros(B, max_blocks, device=device)
        for i in range(B):
            Lb = int(batch.base_len[i].item())
            T = math.ceil(Lb / max(1, self.gate_blocks))
            for t in range(T):
                s = t * self.gate_blocks
                e = min(Lb, (t + 1) * self.gate_blocks)
                counts[i, t] = float(e - s)
        
        # Cumulative counts
        cum = counts.cumsum(dim=1)
        
        # Block indices
        idx_seq = torch.arange(max_blocks, device=device).unsqueeze(0).expand(B, -1).float()
        
        # Previous angles buffer (autoregressive)
        prev_buf = torch.zeros(B, PREV_K, self.angles_per_block, device=device)
        prev_seq = torch.zeros(B, max_blocks, self.angles_per_block * PREV_K, device=device)
        
        # Outputs
        Y = torch.zeros(B, max_blocks, self.angles_per_block, device=device)
        
        # Causal mask
        attn_mask = torch.triu(torch.ones((max_blocks, max_blocks), dtype=torch.bool, device=device), diagonal=1)
        
        # Autoregressive loop over blocks
        for t in range(max_blocks):
            L = t + 1
            
            # Store current prev window
            prev_seq[:, t, :] = prev_buf.reshape(B, self.angles_per_block * PREV_K)
            
            # Build features for blocks [0, t]
            feats = torch.cat([
                counts[:, :L].unsqueeze(-1),      # gate count
                cum[:, :L].unsqueeze(-1),         # cumulative
                idx_seq[:, :L].unsqueeze(-1),     # block index
                prev_seq[:, :L, :],               # previous angles
            ], dim=-1)
            
            # Project and encode
            x = self.in_proj(feats)
            x = x + self.pos_emb(torch.arange(L, device=device)).unsqueeze(0)
            
            # Apply causal transformer
            h = self.encoder(x, mask=attn_mask[:L, :L])
            
            # Predict angles from last position
            h_last = self.head_ln(h[:, -1, :])
            logits_t = self.head(h_last)  # [B, 2*angles_per_block]
            
            # Convert to angles
            y_t = self._angles_from_s1(logits_t)  # [B, angles_per_block]
            
            # Sanitize
            y_t = torch.nan_to_num(y_t, nan=0.0, posinf=0.0, neginf=0.0)
            y_t = y_t.clamp(min=-torch.pi + 1e-6, max=torch.pi - 1e-6)
            
            # Store
            Y[:, t, :] = y_t
            
            # Update prev buffer for next step
            prev_buf = torch.roll(prev_buf, shifts=-1, dims=1)
            prev_buf[:, -1, :] = y_t
        
        # Reshape to [B, max_blocks*angles_per_block, 1] to match expected format
        return Y.reshape(B, max_blocks * self.angles_per_block, 1)


def train_lelzz(
    data_path: str,
    n_qubits: int = 2,
    batch_size: int = 32,
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
    """Train ZZ-ring PQC model on multi-qubit circuits.
    
    Args:
        data_path: Path to dataset
        n_qubits: Number of qubits (must match data)
        batch_size: Batch size
        epochs: Number of training epochs
        lr: Learning rate
        k_random: Number of random initial states
        num_sample: Limit dataset size (None = use all)
        noise: Noise configuration
        device: torch device (None = auto)
        gate_blocks: Base gates per PQC block
        detach_base_noise: Detach gradients through base+noise
    """
    if device is None:
        device = DEVICE
    
    print(f"[LELZZ] Training ZZ-ring PQC: n_qubits={n_qubits}, gate_blocks={gate_blocks}")
    print(f"[LELZZ] Angles per block: 7*{n_qubits} = {7*n_qubits}")
    
    # Load dataset
    ds_full = CircuitDataset(data_path, num_sample=num_sample)
    
    # Filter to only circuits with the target n_qubits (no normalization)
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
    print(f"[LELZZ] Filtered dataset: {len(ds)} circuits with {n_qubits} qubits")
    
    # Build caches
    init_cache, ref_cache, noise_schedules = build_base_cache_vectorized(
        ds, k_random=k_random, device=device, noise=noise
    )
    
    # Train/val split
    N = len(ds)
    val_cnt = max(1, N // 10) if N > 1 else 0
    train_cnt = N - val_cnt
    
    indices = list(range(N))
    import random
    random.shuffle(indices)
    
    ds_train = Subset(ds, indices[:train_cnt])
    ds_val = Subset(ds, indices[train_cnt:]) if val_cnt > 0 else None
    
    print(f"[LELZZ] Train: {len(ds_train)}, Val: {len(ds_val) if ds_val else 0}")
    
    # Data loaders
    collate_fn = lambda xs: collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=MAX_QUBITS)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if ds_val else None
    
    # Model
    model = ZZRingAnglePredictor(gate_blocks=gate_blocks, n_qubits=n_qubits).to(device)
    
    print(f"[LELZZ] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[LELZZ] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Print angle configuration details
    print("\n" + "="*70)
    print("[LELZZ] PQC ANGLE CONFIGURATION")
    print("="*70)
    
    # Calculate typical number of blocks for the dataset
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
    print(f"   - Overall params shape: [batch_size, {typical_blocks} blocks × {angles_per_block} angles, 1]")
    print(f"                         = [B, {total_angles_per_circuit}, 1]")
    
    print(f"\n2. ANGLES PER PQC BLOCK:")
    print(f"   - Total: {angles_per_block} angles")
    print(f"   - Layout: [pre_local, zz_ring, post_local]")
    print(f"   - Indices: [0:{3*n_qubits}] + [{3*n_qubits}:{4*n_qubits}] + [{4*n_qubits}:{7*n_qubits}]")
    
    print(f"\n3. ANGLE BREAKDOWN PER BLOCK:")
    print(f"   a) Pre-local angles: [{n_qubits} qubits × 3 angles] = {3*n_qubits} angles")
    print(f"      Shape: [{n_qubits}, 3]  → (qubits, [RZ1, RX, RZ2])")
    print(f"      Purpose: Local single-qubit rotations before entanglement")
    for q in range(min(n_qubits, 3)):  # Show first 3 qubits
        start_idx = q * 3
        print(f"        Qubit {q}: angles[{start_idx}:{start_idx+3}] = [RZ1, RX, RZ2]")
    if n_qubits > 3:
        print(f"        ... ({n_qubits - 3} more qubits)")
    
    print(f"\n   b) ZZ-ring angles: [{n_qubits} pairs] = {n_qubits} angles")
    print(f"      Shape: [{n_qubits}]  → (theta_zz for each adjacent pair)")
    print(f"      Purpose: Entangling ZZ gates in ring topology")
    print(f"      Ring structure:")
    for q in range(min(n_qubits, 3)):  # Show first 3 pairs
        q_next = (q + 1) % n_qubits
        zz_idx = 3*n_qubits + q
        print(f"        Pair {q}: qubit {q}→{q_next}, angle[{zz_idx}] = theta_zz[{q}]")
        print(f"                Gates: CNOT({q},{q_next}) - RZ(theta) on q{q_next} - CNOT({q},{q_next})")
    if n_qubits > 3:
        print(f"        ... ({n_qubits - 3} more pairs)")
    
    print(f"\n   c) Post-local angles: [{n_qubits} qubits × 3 angles] = {3*n_qubits} angles")
    print(f"      Shape: [{n_qubits}, 3]  → (qubits, [RZ1, RX, RZ2])")
    print(f"      Purpose: Local single-qubit rotations after entanglement")
    for q in range(min(n_qubits, 3)):  # Show first 3 qubits
        start_idx = 4*n_qubits + q * 3
        print(f"        Qubit {q}: angles[{start_idx}:{start_idx+3}] = [RZ1, RX, RZ2]")
    if n_qubits > 3:
        print(f"        ... ({n_qubits - 3} more qubits)")
    
    print(f"\n4. COMPLETE BLOCK STRUCTURE:")
    print(f"   Block angles[{angles_per_block}] =")
    print(f"     [pre[0,0], pre[0,1], pre[0,2],  # Qubit 0 pre")
    if n_qubits > 1:
        print(f"      pre[1,0], pre[1,1], pre[1,2],  # Qubit 1 pre")
    if n_qubits > 2:
        print(f"      ...,")
        print(f"      theta_zz[0], theta_zz[1], ..., theta_zz[{n_qubits-1}],  # ZZ-ring")
        print(f"      post[0,0], post[0,1], post[0,2],  # Qubit 0 post")
        print(f"      post[1,0], post[1,1], post[1,2],  # Qubit 1 post")
        print(f"      ...]")
    
    print("\n" + "="*70)
    print()
    
    # Optimizer with weight decay
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        betas=(0.9, 0.999),  # More stable second moment
        weight_decay=0.01,    # L2 regularization
        eps=1e-8
    )
    
    # LR scheduler: linear warmup then cosine decay with restarts
    import math
    warmup_ep = min(50, epochs // 10)  # Adaptive warmup
    min_lr_ratio = 0.001  # Lower minimum for fine-tuning
    
    def lr_lambda(ep_idx):
        if ep_idx < warmup_ep:
            # Smooth warmup
            return (ep_idx + 1) / warmup_ep
        # Cosine annealing
        t = ep_idx - warmup_ep
        T = max(1, epochs - warmup_ep)
        cos_inner = math.pi * t / T
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(cos_inner))
    
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    
    # AMP setup
    amp_enabled, amp_dtype = get_amp_settings()
    scaler = make_grad_scaler(amp_enabled, amp_dtype)
    
    # Early stopping and checkpointing
    best_val_fid = 0.0
    patience = 100
    patience_counter = 0
    start_epoch = 1
    checkpoint_dir = "checkpoints_lelzz"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Track metrics
    train_losses = []
    val_fids = []
    
    # Resume from checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[LELZZ] Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        sched.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_fid = checkpoint.get('best_val_fid', 0.0)
        train_losses = checkpoint.get('train_losses', [])
        val_fids = checkpoint.get('val_fids', [])
        print(f"[LELZZ] Resumed from epoch {checkpoint['epoch']}, best_val_fid={best_val_fid:.6f}")
    
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
                
                if amp_enabled and amp_dtype and torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits = model(batch, device)
                else:
                    logits = model(batch, device)
                
                # Compute loss in FP32
                if amp_enabled and torch.cuda.is_available():
                    with torch.amp.autocast('cuda', enabled=False):
                        loss = simulate_loss_lelzz_blocks(
                            batch, logits.float(), init_cache, ref_cache,
                            noise_schedules, gate_blocks, device, detach_base_noise
                        )
                else:
                    loss = simulate_loss_lelzz_blocks(
                        batch, logits, init_cache, ref_cache,
                        noise_schedules, gate_blocks, device, detach_base_noise
                    )
                
                fid = 1.0 - float(loss.detach())
                total_fid += fid * batch.base_g.size(0)
                count += batch.base_g.size(0)
        
        model.train()
        return total_fid / max(1, count)
    
    # Training loop
    for ep in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # Forward pass
            if amp_enabled and amp_dtype and torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(batch, device)
            else:
                logits = model(batch, device)
            
            # Compute loss in FP32
            if amp_enabled and torch.cuda.is_available():
                with torch.amp.autocast('cuda', enabled=False):
                    loss = simulate_loss_lelzz_blocks(
                        batch, logits.float(), init_cache, ref_cache,
                        noise_schedules, gate_blocks, device, detach_base_noise
                    )
            else:
                loss = simulate_loss_lelzz_blocks(
                    batch, logits, init_cache, ref_cache,
                    noise_schedules, gate_blocks, device, detach_base_noise
                )
            
            # Backward pass
            opt.zero_grad(set_to_none=True)
            
            if torch.isfinite(loss):
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # Gradient clipping before unscaling
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                
                total_loss += float(loss.detach()) * batch.base_g.size(0)
                count += batch.base_g.size(0)
            else:
                print(f"[LELZZ] Warning: non-finite loss at epoch {ep}, skipping batch")
        
        # Step scheduler
        sched.step()
        
        # Evaluate
        avg_loss = total_loss / max(1, count)
        train_fid = 1.0 - avg_loss
        val_fid = evaluate()
        cur_lr = opt.param_groups[0]['lr']
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Track metrics
        train_losses.append(avg_loss)
        val_fids.append(val_fid)
        
        # Print progress with timing
        print(f"[LELZZ] Epoch {ep:4d}/{epochs} | Time={epoch_time:.2f}s | LR={cur_lr:.6f} | "
              f"Train Loss={avg_loss:.6f} (Fid={train_fid:.6f}) | Val Fid={val_fid:.6f}")
        
        # Early stopping and checkpointing
        if val_fid > best_val_fid:
            best_val_fid = val_fid
            patience_counter = 0
            
            # Save best model
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
            print(f"[LELZZ] ✓ New best Val Fid: {best_val_fid:.6f} (saved to {checkpoint_path})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[LELZZ] Early stopping triggered after {patience} epochs without improvement")
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
            print(f"[LELZZ] Checkpoint saved at epoch {ep}")
    
    # Final summary
    print(f"\n[LELZZ] Training complete!")
    print(f"[LELZZ] Best Val Fid: {best_val_fid:.6f}")
    print(f"[LELZZ] Final Train Fid: {train_fid:.6f}")
    print(f"[LELZZ] Final Val Fid: {val_fid:.6f}")
    
    return model


def _build_argparser():
    p = argparse.ArgumentParser(description="Train ZZ-ring PQC architecture")
    p.add_argument("--data-path", type=str, required=True, help="Path to dataset")
    p.add_argument("--n-qubits", type=int, default=2, help="Number of qubits")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    p.add_argument("--k-random", type=int, default=32, help="Number of random initial states")
    p.add_argument("--gate-blocks", type=int, default=5, help="Base gates per PQC block")
    p.add_argument("--num-sample", type=int, default=None, help="Limit dataset size (None=all)")
    p.add_argument("--no-detach-base-noise", action="store_false", dest="detach_base_noise",
                   help="Don't detach gradients through base circuit")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    train_lelzz(
        data_path=args.data_path,
        n_qubits=args.n_qubits,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        k_random=args.k_random,
        num_sample=args.num_sample,
        noise=None,  # Can be configured via NoiseConfig if needed
        device=None,
        gate_blocks=args.gate_blocks,
        detach_base_noise=args.detach_base_noise,
        resume_checkpoint=args.resume,
    )
