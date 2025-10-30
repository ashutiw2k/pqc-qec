#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimize PQC angles for previously refined N+3 circuits.

Workflow:
1) Read JSONL produced by load_start_checkpoint (each line: {"circuit_tokens": [...]})
   where tokens = N base gates (h/x/z) + [rz:val, rx:val, rz:val].
2) Evaluate average fidelity over K random initial states using project simulator.
3) Optimize the three PQC angles per circuit (independently per circuit) to further
   improve average fidelity on the same cached initial states and noise schedules.
4) Optionally save updated N+3 circuits JSONL with optimized angles.

Usage example:
  python -m pqcqec.optimize_refined_circuits \
    --input A:/wings/pqc-qec/zian/pack/models/augmented/aug_angle_..._len10_enumall_100K_Nplus3.jsonl \
    --batch-size 256 --k-random 100 --epochs 5 --lr 0.05 --save-out A:/wings/out_opt.jsonl

Notes:
- Assumes all circuits are 1-qubit and of equal base length N.
- Uses the project's simulator_core for exact parity with training/eval.
"""
from __future__ import annotations

import os, json, math, argparse
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader

# Project simulator imports
from .simulator_core import (
    collate as proj_collate,
    build_base_cache_vectorized as proj_build_base_cache_vectorized,
    NoiseConfig as ProjNoiseConfig,
    simulate_blocks_with_angles as proj_simulate_blocks_with_angles,
)
from .dataset import MAX_BASE_LEN, MAX_PARAM, MAX_QUBITS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.complex64


def _parse_circuit_tokens(tokens: List[str]) -> Tuple[List[str], Tuple[float, float, float]]:
    """Split N+3 tokens into base gates and PQC angles (rz, rx, rz).
    Expects last three tokens to be formatted as 'rz:val', 'rx:val', 'rz:val'.
    """
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
    return base_tokens, (rz1, rx, rz2)


def _wrap_angle_pi(a: torch.Tensor) -> torch.Tensor:
    # Wrap to (-pi, pi]
    return (a + math.pi) % (2 * math.pi) - math.pi


class _ItemsDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[dict]):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i: int):
        return self.items[i]


def load_nplus3_jsonl(path: str) -> Tuple[List[dict], torch.Tensor, int]:
    """Read JSONL of N+3 circuits and produce project-style base items plus angles tensor.
    Returns (items, angles_init, base_len).
    - items: list of dicts suitable for proj_collate (base only; n_qubits=1)
    - angles_init: tensor [B, 1, 1, 3] with (rz1, rx, rz2)
    - base_len: inferred N
    """
    items: List[dict] = []
    angles: List[Tuple[float,float,float]] = []
    base_len: Optional[int] = None
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
            if base_len is None:
                base_len = len(base_tokens)
            elif base_len != len(base_tokens):
                raise ValueError("All circuits must share the same base length N")
            items.append(dict(
                idx=idx,
                n_qubits=1,
                base_gates=list(base_tokens),
                base_q1=[0]*len(base_tokens),
                base_q2=[-1]*len(base_tokens),
                param_gates=[], param_qubits=[], after=[], param_angles_gt=[],
            ))
            angles.append((rz1, rx, rz2))
    if base_len is None:
        raise ValueError("No circuits found in input file")
    ang = torch.tensor(angles, dtype=torch.float32).view(len(angles), 1, 1, 3)  # [B,1,1,3]
    return items, ang, base_len


def evaluate_avg_fidelity(items: List[dict], angles_blk_all: torch.Tensor, *, k_random: int, device: torch.device, use_noise: bool, batch_size: int) -> float:
    """Evaluate average fidelity for given items and per-item angles block.
    angles_blk_all: [B,1,1,3]
    """
    ds = _ItemsDataset(items)
    noise = ProjNoiseConfig(use_noise=bool(use_noise))
    init_cache, ref_cache, noise_sched = proj_build_base_cache_vectorized(ds, k_random=k_random, device=device, noise=noise)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda xs: proj_collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1))
    # idx mapping -> row
    idx2row = {it['idx']: i for i, it in enumerate(items)}

    total_fid_w = 0.0
    total_B = 0
    # Ensure angle tensor is on same device as index tensor for index_select
    angles_blk_all_dev = angles_blk_all.to(device)
    with torch.no_grad():
        for b in dl:
            b = b.to(device)
            rows = torch.tensor([idx2row[int(i.item())] for i in b.idx], device=device, dtype=torch.long)
            angles_blk = angles_blk_all_dev.index_select(0, rows)  # [B,1,1,3]
            loss = proj_simulate_blocks_with_angles(
                b, angles_blk, init_cache, ref_cache, noise_sched,
                gate_blocks=int(b.base_len[0].item()), device=device, detach_base_noise=True
            )
            fid = float(1.0 - loss.item())
            total_fid_w += fid * int(b.base_g.size(0))
            total_B += int(b.base_g.size(0))
    return total_fid_w / max(1, total_B)


def optimize_angles(items: List[dict], angles_init: torch.Tensor, *, k_random: int, device: torch.device, use_noise: bool, batch_size: int, epochs: int, lr: float, weight_decay: float = 0.0, clamp: bool = True) -> torch.Tensor:
    """Optimize per-item angles to maximize fidelity (minimize 1 - F).
    angles_init: [B,1,1,3] (float32)
    Returns optimized angles tensor with same shape.
    """
    ds = _ItemsDataset(items)
    noise = ProjNoiseConfig(use_noise=bool(use_noise))
    init_cache, ref_cache, noise_sched = proj_build_base_cache_vectorized(ds, k_random=k_random, device=device, noise=noise)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda xs: proj_collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1))
    num_items = len(items)
    # Parameterize as [B,3]
    angles_param = torch.nn.Parameter(angles_init.view(num_items, 3).to(device))
    opt = torch.optim.Adam([angles_param], lr=lr, weight_decay=weight_decay)
    idx2row = {it['idx']: i for i, it in enumerate(items)}

    for ep in range(max(1, int(epochs))):
        running = 0.0
        count = 0
        for b in dl:
            b = b.to(device)
            rows = torch.tensor([idx2row[int(i.item())] for i in b.idx], device=device)
            batch_ang = angles_param.index_select(0, rows).view(-1, 1, 1, 3)  # [B,1,1,3]
            loss = proj_simulate_blocks_with_angles(
                b, batch_ang, init_cache, ref_cache, noise_sched,
                gate_blocks=int(b.base_len[0].item()), device=device, detach_base_noise=True
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if clamp:
                with torch.no_grad():
                    angles_param[:] = _wrap_angle_pi(angles_param)
            running += float(loss.detach().item()) * int(b.base_g.size(0))
            count += int(b.base_g.size(0))
        avg_loss = running / max(1, count)
        print(f"[Optimize] epoch={ep+1}/{epochs} avg_loss={avg_loss:.6f} avg_fid={1-avg_loss:.6f}")
    return angles_param.detach().view(num_items, 1, 1, 3).to('cpu')


def save_updated_jsonl(path_in: str, path_out: str, angles_all: torch.Tensor):
    """Overwrite angles in input JSONL and write to path_out.
    angles_all: [B,1,1,3] (cpu tensor)
    """
    # Read all first
    lines = []
    with open(path_in, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(line.rstrip("\n"))
    assert len(lines) == angles_all.size(0), "Angle count mismatch with input lines"

    def _fmt(tok: str, val: float) -> str:
        return f"{tok}:{val:.6f}"

    with open(path_out, 'w', encoding='utf-8') as f:
        for i, line in enumerate(lines):
            obj = json.loads(line)
            toks = obj['circuit_tokens']
            base_len = len(toks) - 3
            rz1, rx, rz2 = [float(x) for x in angles_all[i, 0, 0, :].tolist()]
            toks_out = list(toks[:base_len]) + [_fmt('rz', rz1), _fmt('rx', rx), _fmt('rz', rz2)]
            obj['circuit_tokens'] = toks_out
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[Save] Updated circuits written: {path_out}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate and optimize PQC angles for N+3 circuits JSONL.")
    ap.add_argument('--input', type=str, required=True, help='Path to N+3 circuits JSONL (from load_start_checkpoint)')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--k-random', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=0, help='Number of optimization epochs (0 = no optimization)')
    ap.add_argument('--lr', type=float, default=0.05, help='Optimizer learning rate')
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--cpu', action='store_true', help='Force CPU')
    ap.add_argument('--no-noise', action='store_true', help='Disable noise during evaluation/optimization')
    ap.add_argument('--save-out', type=str, default=None, help='Path to save updated N+3 JSONL with optimized angles')
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu or (not torch.cuda.is_available()) else 'cuda')

    items, ang_init, base_len = load_nplus3_jsonl(args.input)
    print(f"[Load] items={len(items)} base_len={base_len}")

    # Evaluate initial
    fid0 = evaluate_avg_fidelity(items, ang_init, k_random=int(args.k_random), device=device, use_noise=(not args.no_noise), batch_size=int(args.batch_size))
    print(f"[Eval] initial avg_fidelity={fid0:.6f}")

    ang_opt = ang_init
    if int(args.epochs) > 0:
        ang_opt = optimize_angles(
            items, ang_init, k_random=int(args.k_random), device=device,
            use_noise=(not args.no_noise), batch_size=int(args.batch_size),
            epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay)
        )
        # Re-evaluate after optimization and report
        fid1 = evaluate_avg_fidelity(
            items, ang_opt, k_random=int(args.k_random), device=device,
            use_noise=(not args.no_noise), batch_size=int(args.batch_size)
        )
        print(f"[Eval] optimized avg_fidelity={fid1:.6f}")

        # Save optimized angles: use --save-out if provided, else timestamped default next to input
        out_path = args.save_out
        if not out_path:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            input_dir = os.path.dirname(args.input) or '.'
            out_path = os.path.join(input_dir, f'angle_refine_{ts}.jsonl')
        if os.path.dirname(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_updated_jsonl(args.input, out_path, ang_opt)
    else:
        # Preserve existing behavior: if no optimization but user requested output, save initial angles
        if args.save_out:
            if os.path.dirname(args.save_out):
                os.makedirs(os.path.dirname(args.save_out), exist_ok=True)
            save_updated_jsonl(args.input, args.save_out, ang_opt)


if __name__ == '__main__':
    main()
