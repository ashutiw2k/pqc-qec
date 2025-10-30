#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervised fine-tuning of the AnglePredictor transformer using N+3 JSONL ground-truth angles.

- Loads an existing checkpoint trained by opt_transformer_simple_1022_start.py
- Reads angle_refine_*.jsonl (or any N+3 JSONL) where circuit_tokens = base_N + [rz:val, rx:val, rz:val]
- Trains the transformer to predict the final PQC block (rz, rx, rz) angles to match ground-truth
- Uses the same model architecture and I/O as opt_transformer_simple_1022_start.py.
- Saves the best model as tdf_[timestamp].pt in models/ by default.

Notes:
- Assumes all circuits are single-qubit and share the same base length N.
- We use ordered-seq path with full_context_one_step=True and T_vec=1; ctx.gate_blocks is set to base_len so the
  model predicts exactly one final PQC block appended after the base segment.
"""
from __future__ import annotations

import os, json, math, argparse, time
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

# Project imports: model, collate, dataset constants
from .angle_predictor_simple import AnglePredictor
from .simulator_core import (
    collate as proj_collate,
    build_base_cache_vectorized as proj_build_base_cache_vectorized,
    simulate_blocks_with_angles as proj_simulate_blocks_with_angles,
    NoiseConfig as ProjNoiseConfig,
)
from .dataset import MAX_BASE_LEN, MAX_PARAM, MAX_QUBITS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _wrap_angle_pi_t(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _parse_circuit_tokens(tokens: List[str]) -> Tuple[List[str], Tuple[float, float, float]]:
    if len(tokens) < 4:
        raise ValueError("Circuit must have at least N+3 tokens")
    base_len = len(tokens) - 3
    base_tokens = tokens[:base_len]
    t1, t2, t3 = tokens[base_len:]
    def _parse(tok: str, expected: str) -> float:
        if not tok.startswith(expected + ":"):
            raise ValueError(f"Expected token '{expected}:<angle>', got '{tok}'")
        try:
            return float(tok.split(":", 1)[1])
        except Exception:
            raise ValueError(f"Bad angle format in token: {tok}")
    rz1 = _parse(t1, 'rz'); rx = _parse(t2, 'rx'); rz2 = _parse(t3, 'rz')
    return list(base_tokens), (rz1, rx, rz2)


class NPlus3Dataset(Dataset):
    """Dataset reading N+3 JSONL and returning (item_dict, gt_angles[3])."""
    def __init__(self, path: str):
        self.items: List[dict] = []
        self.angles: List[Tuple[float, float, float]] = []
        self.base_len: Optional[int] = None
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                toks = obj.get('circuit_tokens')
                if not isinstance(toks, list):
                    raise ValueError("Each JSONL line must contain 'circuit_tokens': [...] list")
                base_tokens, (rz1, rx, rz2) = _parse_circuit_tokens(toks)
                if self.base_len is None:
                    self.base_len = len(base_tokens)
                elif self.base_len != len(base_tokens):
                    raise ValueError("All circuits must share the same base length N")
                self.items.append(dict(
                    idx=idx,
                    n_qubits=1,
                    base_gates=base_tokens,
                    base_q1=[0]*len(base_tokens),
                    base_q2=[-1]*len(base_tokens),
                    param_gates=[], param_qubits=[], after=[], param_angles_gt=[],
                ))
                self.angles.append((rz1, rx, rz2))
        if self.base_len is None:
            raise ValueError("No circuits found in input file")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        gt = torch.tensor(self.angles[i], dtype=torch.float32)
        return self.items[i], gt


def _collate_supervised(samples: List[Tuple[dict, torch.Tensor]]):
    item_dicts = [s[0] for s in samples]
    gt_angles = torch.stack([s[1] for s in samples], dim=0)  # [B,3]
    bat = proj_collate(item_dicts, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1)
    return bat, gt_angles


def _angle_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Angular L2 with periodic wrap: mean((wrap(pred-target))^2). pred/target shape [B,3]."""
    d = _wrap_angle_pi_t(pred - target)
    return (d * d).mean()


def _save_checkpoint(model: nn.Module, tag: str = "tdf") -> str:
    out_dir = os.path.join(os.getcwd(), "models")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.getcwd()
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{tag}_{ts}.pt")
    ckpt = {
        'state_dict': model.state_dict(),
        'timestamp': ts,
    }
    torch.save(ckpt, path)
    print(f"[Save] Checkpoint saved to: {path}")
    return path


def _load_model_from_ckpt(ckpt_path: str, device: torch.device) -> AnglePredictor:
    """Load AnglePredictor from checkpoint saved by opt_transformer_simple_1022_start.py (state_dict in 'state_dict')."""
    data = torch.load(ckpt_path, map_location=device)
    gate_blocks = None
    if isinstance(data, dict):
        sd = data.get('state_dict', None)
        if sd is None:
            sd = data
        gate_blocks = data.get('gate_blocks', None)
    else:
        sd = data
    if gate_blocks is None:
        # Fallback default; will override via ctx during forward to use base_len
        gate_blocks = 50
    model = AnglePredictor(gate_blocks=int(gate_blocks)).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[Load] state_dict loaded with missing={len(missing)} unexpected={len(unexpected)}")
    return model


def train_supervised(
    input_path: str,
    ckpt_path: str,
    batch_size: int = 256,
    epochs: int = 5,
    lr: float = 5e-5,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    val_frac: float = 0.1,
    num_workers: int = 0,
):
    if device is None:
        device = DEVICE
    ds_all = NPlus3Dataset(input_path)
    base_len = int(ds_all.base_len)
    print(f"[Load] samples={len(ds_all)} base_len={base_len}")

    # Split train/val
    n_total = len(ds_all)
    n_val = max(1, int(round(n_total * val_frac))) if n_total > 1 else 0
    n_train = n_total - n_val
    indices = list(range(n_total))
    import random as _r
    _r.shuffle(indices)
    ds_train = Subset(ds_all, indices[:n_train])
    ds_val = Subset(ds_all, indices[n_train:]) if n_val > 0 else None

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate_supervised)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_supervised) if ds_val else None

    # Load model
    model = _load_model_from_ckpt(ckpt_path, device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _forward_and_loss(bat, gt):
        bat = bat.to(device)
        gt = gt.to(device)
        B = bat.base_g.size(0)
        # Predict exactly one final block using ordered-seq path
        counts = torch.zeros(B, 1, dtype=torch.float32, device=device)
        T_vec = torch.ones(B, dtype=torch.long, device=device)
        ctx = {'batch': bat, 'gate_blocks': base_len, 'full_context_one_step': True}
        y = model(counts, T_vec, ctx)  # [B,3,1]
        if y.dim() == 3 and y.size(2) == 1:
            y = y[:, :, 0]
        elif y.dim() == 3 and y.size(1) >= 3:
            y = y[:, 0:3, 0]
        elif y.dim() == 2 and y.size(1) >= 3:
            y = y[:, 0:3]
        else:
            raise RuntimeError(f"Unexpected model output shape: {tuple(y.shape)}")
        return _angle_l2_loss(y, gt)

    # Train for fixed epochs; we'll save only the final checkpoint later
    best_val = float('inf')
    for ep in range(1, epochs + 1):
        model.train(); total = 0.0; nb = 0
        for bat, gt in train_loader:
            loss = _forward_and_loss(bat, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()); nb += 1
        train_loss = total / max(1, nb)
        # Val
        val_loss = float('nan')
        if val_loader is not None:
            model.eval(); total_v = 0.0; nb_v = 0
            with torch.no_grad():
                for bat, gt in val_loader:
                    l = _forward_and_loss(bat, gt)
                    total_v += float(l); nb_v += 1
            val_loss = total_v / max(1, nb_v)
            model.train()
        print(f"[TSF] epoch {ep} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        # Track best metric for logging, but do not save per-epoch
        if val_loader is not None:
            score = val_loss
        else:
            score = train_loss
        if score < best_val:
            best_val = score

    # Save only final checkpoint
    final_path = _save_checkpoint(model, tag="tdf")
    print(f"[Done] Final model saved at: {final_path}; best_score={best_val:.6f}")

    # After training: synthetic enum-all evaluation on base_len using default noise and K=100
    try:
        fid = evaluate_synthetic_enum_all(model, base_len=base_len, batch_size=batch_size, k_random=100, device=device)
        print(f"[Eval] synthetic enum-all avg_fidelity={fid:.6f} (N={base_len}, K=100, default noise)")
    except Exception as e:
        print(f"[Eval] Synthetic evaluation skipped due to error: {e}")
    return final_path


def _generate_all_items(base_len: int) -> List[dict]:
    import itertools
    items: List[dict] = []
    idx = 0
    for seq in itertools.product(("h", "x", "z"), repeat=base_len):
        base_tokens = list(seq)
        items.append(dict(
            idx=idx,
            n_qubits=1,
            base_gates=base_tokens,
            base_q1=[0]*len(base_tokens),
            base_q2=[-1]*len(base_tokens),
            param_gates=[], param_qubits=[], after=[], param_angles_gt=[],
        ))
        idx += 1
    return items


@torch.no_grad()
def evaluate_synthetic_enum_all(model: AnglePredictor, *, base_len: int, batch_size: int, k_random: int, device: torch.device) -> float:
    """Enumerate all 1q base circuits of length N, predict one final PQC block, and compute avg fidelity.
    Uses default noise (pi/10, deltas 0) via ProjNoiseConfig().
    """
    model.eval()
    items = _generate_all_items(base_len)
    if len(items) == 0:
        return float('nan')
    # Build caches on full set with default noise
    class _DS(torch.utils.data.Dataset):
        def __init__(self, it): self.items = it
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]
    ds = _DS(items)
    noise = ProjNoiseConfig(use_noise=True)
    init_cache, ref_cache, noise_sched = proj_build_base_cache_vectorized(ds, k_random=int(k_random), device=device, noise=noise)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda xs: proj_collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1))
    total = 0.0; count = 0
    for bat in loader:
        bat = bat.to(device)
        B = bat.base_g.size(0)
        counts = torch.zeros(B, 1, dtype=torch.float32, device=device)
        T_vec = torch.ones(B, dtype=torch.long, device=device)
        ctx = {'batch': bat, 'gate_blocks': base_len, 'full_context_one_step': True}
        y = model(counts, T_vec, ctx)  # [B,3,1]
        if y.dim() == 3 and y.size(2) == 1:
            y3 = y[:, :, 0]
        else:
            y3 = y[:, 0:3].reshape(B, 3)
        angles_blk = y3.view(B, 1, 1, 3)
        loss = proj_simulate_blocks_with_angles(
            bat, angles_blk, init_cache, ref_cache, noise_sched,
            gate_blocks=int(base_len), device=device, detach_base_noise=True
        )
        fid = float(1.0 - loss.item())
        total += fid * B
        count += B
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser(description="Supervised fine-tuning from angle_refine JSONL against ground-truth angles.")
    ap.add_argument('--input', type=str, required=True, help='Path to N+3 JSONL (e.g., angle_refine_*.jsonl)')
    ap.add_argument('--ckpt', type=str, required=True, help='Path to pretrained checkpoint from opt_transformer_simple_1022_start.py')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')
    ap.add_argument('--val-frac', type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu or (not torch.cuda.is_available()) else 'cuda')
    train_supervised(
        input_path=args.input,
        ckpt_path=args.ckpt,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        val_frac=float(args.val_frac),
    )


if __name__ == '__main__':
    main()
