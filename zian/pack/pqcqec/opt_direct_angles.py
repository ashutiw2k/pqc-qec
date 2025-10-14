#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct-angle optimization module extracted from newest_work_correct.
Provides a small module that exposes trainable angles and returns logits in a
compatible shape for the simulator.

Exports:
- DirectAnglesParamModule: optimizes a 1D vector of angles and exposes as logits
- train_direct: a simple training loop using fixed-interval PQC blocks
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from .simulator_core import (
    DEVICE,
    CircuitDataset,
    collate,
    build_base_cache_vectorized,
    NoiseConfig,
    simulate_loss,
)


class DirectAnglesParamModule(nn.Module):
    def __init__(self, Lp: int, tanh_bound: bool = True, angle_scale_init: float = 0.1, init_std: float = 0.01):
        super().__init__()
        # Raw unconstrained parameters; converted to angles in forward
        base = torch.zeros(Lp)
        if init_std and init_std > 0:
            base = base + torch.randn_like(base) * float(init_std)
        self.theta_raw = nn.Parameter(base)
        self.Lp = Lp
        self.tanh_bound = bool(tanh_bound)
        # simple scalar scale to avoid early saturation when using tanh bound
        self.register_buffer('angle_scale', torch.tensor(float(angle_scale_init)))

    def forward(self, batch):
        B = batch.base_g.size(0)
        # map raw -> angle
        if self.tanh_bound:
            ang = math.pi * torch.tanh(self.theta_raw * self.angle_scale)
        else:
            ang = self.theta_raw
        L = min(self.Lp, 1500)
        ang_padded = torch.zeros(1500, device=ang.device, dtype=ang.dtype)
        if L > 0:
            ang_padded[:L] = ang[:L]
        logits = ang_padded.view(1, 1500, 1).expand(B, -1, -1).contiguous()
        mask = torch.ones(B, 1500, dtype=torch.bool, device=logits.device)
        mask[:, :L] = False
        return logits, mask


def train_direct(
    data_path: str,
    gate_blocks: int = 50,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 5e-4,
    k_random: int = 32,
    num_sample: Optional[int] = None,
    noise: Optional[NoiseConfig] = None,
    device: Optional[torch.device] = None,
    val_ratio: float = 0.1,
    n_sample: Optional[int] = None,
    direct_angle_tanh: bool = True,
    angle_scale_init: float = 0.1,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    betas=(0.9, 0.99),
    angle_l2: float = 0.0,
    detach_base_noise: bool = True,
    init_std: float = 0.01,
):
    if device is None:
        device = DEVICE
    # Allow limiting dataset size via n_sample (preferred) or num_sample
    eff_num = n_sample if (n_sample is not None) else num_sample
    ds = CircuitDataset(data_path, num_sample=eff_num)
    if len(ds) == 0:
        raise RuntimeError("Empty dataset")
    init_cache, ref_cache, noise_schedules = build_base_cache_vectorized(ds, k_random=k_random, device=device, noise=noise)
    # Derive Lp using dataset-wide maxima to avoid under-parameterization on longer circuits
    max_Lb = 0
    max_n = 0
    for it in ds.items:
        max_Lb = max(max_Lb, len(it['base_gates']))
        max_n = max(max_n, int(it['n_qubits']))
    blocks_needed = math.ceil(max_Lb / max(1, gate_blocks)) if max_Lb > 0 else 1
    Lp = blocks_needed * max(1, max_n) * 3
    model = DirectAnglesParamModule(
        Lp,
        tanh_bound=direct_angle_tanh,
        angle_scale_init=angle_scale_init,
        init_std=init_std,
    ).to(device)

    # Train/Val split
    n_total = len(ds)
    n_val = max(1, int(round(n_total * val_ratio))) if n_total > 1 else 0
    n_train = n_total - n_val
    idx_all = list(range(n_total))
    import random as _r
    _r.shuffle(idx_all)
    ds_train = Subset(ds, idx_all[:n_train])
    ds_val = Subset(ds, idx_all[n_train:]) if n_val > 0 else None

    # DataLoaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate) if ds_val is not None else None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def evaluate():
        if val_loader is None:
            return float('nan')
        model.eval()
        total_v = 0.0
        nb = 0
        with torch.no_grad():
            for raw in val_loader:
                batch = raw.to(device)
                logits, _ = model(batch)
                loss = simulate_loss(
                    batch, logits, init_cache, ref_cache, noise_schedules,
                    mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise
                )
                if angle_l2 and angle_l2 > 0:
                    realized = math.pi * torch.tanh(model.theta_raw * model.angle_scale) if direct_angle_tanh else model.theta_raw
                    loss = loss + float(angle_l2) * realized.pow(2).mean()
                total_v += float(loss.detach())
                nb += 1
        model.train()
        return total_v / max(1, nb)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        iterator = train_loader
        if tqdm is not None:
            iterator = tqdm(train_loader, desc=f"Direct Train ep {ep}", unit="batch")
        for raw in iterator:
            batch = raw.to(device)
            logits, _ = model(batch)
            loss = simulate_loss(
                batch, logits, init_cache, ref_cache, noise_schedules,
                mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise
            )
            if angle_l2 and angle_l2 > 0:
                realized = math.pi * torch.tanh(model.theta_raw * model.angle_scale) if direct_angle_tanh else model.theta_raw
                loss = loss + float(angle_l2) * realized.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            opt.step()
            total += float(loss.detach())
            if tqdm is not None:
                iterator.set_postfix({"train_loss": f"{float(loss.detach()):.4f}"})
        avg_train = total / max(1, len(train_loader))
        avg_val = evaluate()
        print(f"[Direct] epoch {ep} train_loss={avg_train:.6f} val_loss={avg_val:.6f}")
    return model
