"""
Utility to load a model checkpoint saved by opt_transformer_simple_1022_start and
optionally run a quick synthetic evaluation to verify it works.

Usage examples:
  - Load and print metadata only:
      python -m pqcqec.load_start_checkpoint --ckpt A:/wings/models/angle_predictor_1022_start_subcircuits_gb5_20251022_120000.pt

  - Load and run a tiny synthetic eval (1q circuits):
      python -m pqcqec.load_start_checkpoint --ckpt <path> --eval-synthetic --synthetic-base-len 5

Notes:
    - This script mirrors the fused-kernel environment setup used by
        opt_transformer_simple_1022_start, but does not modify that file.
    - Modified: for synthetic eval, always enumerate ALL H/X/Z base circuits of length N,
        predict one PQC block (rz, rx, rz), evaluate on N+3 circuits, and save N+3 circuits only
        (no additional random tail of N gates).
"""
from __future__ import annotations

import os
import argparse
import platform as _platform
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn

# Enable fused base+noise kernel by default (unless user already set env)
if os.environ.get('PQC_USE_FUSED') is None and os.environ.get('TKFS_USE_FUSED_BASE_NOISE') is None:
    os.environ['PQC_USE_FUSED'] = '1'
if ((_platform.system().lower().startswith('win')) and os.environ.get('PQC_FORCE_INLINE') is None):
    os.environ['PQC_FORCE_INLINE'] = '1'

from .angle_predictor_simple import AnglePredictor  # model definition
from .dataset import MAX_BASE_LEN, MAX_PARAM, MAX_QUBITS
from .simulator_core import (
    collate, simulate_loss, build_base_cache_vectorized, NoiseConfig,
    get_fused_status, ensure_fused_compiled,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _print_fused_status_once():
    try:
        fs0 = get_fused_status()
        if fs0.get('enabled', False) and not fs0.get('available', False):
            ensure_fused_compiled()
        fs = get_fused_status()
        print(f"[Fused] enabled={fs['enabled']} available={fs['available']} attempted={fs['attempted']} used_calls={fs['used_calls']} reason={fs['reason']}")
    except Exception:
        pass


def _load_checkpoint(ckpt_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj
    # Raw state_dict case
    return {'state_dict': obj}


def load_model_from_ckpt(ckpt_path: str, device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict[str, object]]:
    """Load AnglePredictor from checkpoint saved by opt_transformer_simple_1022_start.

    Returns (model, meta) where meta contains gate_blocks and extra fields.
    """
    if device is None:
        device = DEVICE
    ckpt = _load_checkpoint(ckpt_path)
    sd = ckpt['state_dict']
    gate_blocks = int(ckpt.get('gate_blocks', 5))
    use_quaternion_head = bool(ckpt.get('use_quaternion_head', False))
    model = AnglePredictor(gate_blocks=gate_blocks, use_quaternion_head=use_quaternion_head)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    meta: Dict[str, object] = {
        'gate_blocks': gate_blocks,
        'use_quaternion_head': use_quaternion_head,
        'use_ordered_seq': getattr(model, 'use_ordered_seq', None),
        'hid_dim': ckpt.get('hid_dim', None),
        'prev_k': ckpt.get('prev_k', None),
        'timestamp': ckpt.get('timestamp', None),
    }
    return model, meta


# Minimal synthetic generator (1-qubit circuits)
def _generate_synthetic_single_qubit_items(
    num_samples: int = 32,
    base_len: int = 5,
    gates_vocab: Tuple[str, ...] = ("h", "x", "z"),
) -> List[dict]:
    """Legacy random sampler (kept for completeness; not used in new flow)."""
    import random
    items: List[dict] = []
    for i in range(num_samples):
        base_gates = [random.choice(gates_vocab) for _ in range(base_len)]
        items.append(dict(
            idx=i,
            n_qubits=1,
            base_gates=base_gates,
            base_q1=[0]*base_len,
            base_q2=[-1]*base_len,
            param_gates=[], param_qubits=[], after=[], param_angles_gt=[],
        ))
    return items

def _generate_synthetic_single_qubit_all_items(
    base_len: int,
    gates_vocab: Tuple[str, ...] = ("h", "x", "z"),
) -> List[dict]:
    import itertools
    items: List[dict] = []
    idx = 0
    for seq in itertools.product(gates_vocab, repeat=base_len):
        base_gates = list(seq)
        items.append(dict(
            idx=idx,
            n_qubits=1,
            base_gates=base_gates,
            base_q1=[0]*base_len,
            base_q2=[-1]*base_len,
            param_gates=[], param_qubits=[], after=[], param_angles_gt=[],
        ))
        idx += 1
    return items


def _counts_from_batch(model: AnglePredictor, b, target_q: Optional[torch.Tensor] = None):
    # helper to get counts/T_vec in the same way as trainer
    if target_q is None:
        target_q = torch.zeros(b.base_g.size(0), dtype=torch.long, device=b.base_g.device)
    return model._counts_from_batch(b, target_q)


def eval_on_items(
    model: AnglePredictor,
    gate_blocks: int,
    items: List[dict],
    k_random: int = 32,
    device: Optional[torch.device] = None,
) -> float:
    """Evaluate average fidelity on provided 1q items list (blocks mode)."""
    if device is None:
        device = next(model.parameters()).device
    from torch.utils.data import DataLoader
    bs = min(64, max(1, len(items)))
    dl = DataLoader(items, batch_size=bs, shuffle=False,
                    collate_fn=lambda xs: collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1))

    class _Tmp:
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            return self.items[i]

    init_cache, ref_cache, noise_sched = build_base_cache_vectorized(_Tmp(items), k_random=k_random, device=device, noise=None)

    total = 0.0
    count = 0
    with torch.no_grad():
        for b in dl:
            b = b.to(device)
            tq = torch.zeros(b.base_g.size(0), dtype=torch.long, device=device)
            counts, T_vec = _counts_from_batch(model, b, tq)
            ctx = {'batch': b, 'gate_blocks': gate_blocks}
            logits = model(counts, T_vec, ctx)
            loss = simulate_loss(b, logits, init_cache, ref_cache, noise_sched,
                                 mode='blocks', gate_blocks=gate_blocks, detach_base_noise=True)
            fid = float(1.0 - loss.detach().item())
            total += fid * b.base_g.size(0)
            count += b.base_g.size(0)
    return total / max(1, count)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load a start-model checkpoint and optionally run a quick synthetic eval.")
    p.add_argument('--ckpt', type=str, required=True, help='Path to the saved checkpoint file (.pt)')
    p.add_argument('--device', type=str, default=None, help='cpu or cuda (auto if omitted)')
    p.add_argument('--eval-synthetic', action='store_true', help='Run synthetic 1q evaluation after loading')
    # Synthetic options: now always enumerate ALL H/X/Z of length N
    p.add_argument('--synthetic-1q5', action='store_true', help='[Deprecated] Previously selected 1q mode; ignored (always 1q).')
    p.add_argument('--synthetic-base-len', type=int, default=None, help='Base length N for synthetic 1q eval (defaults to gate_blocks)')
    p.add_argument('--k-random', type=int, default=32, help='K random initial states for synthetic eval cache')
    # Tail options removed; output now saves only N+3 circuits
    return p


def predict_and_report_per_circuit():
    # Removed per user request: per-circuit reporting has been deleted.
    raise NotImplementedError("Per-circuit report has been removed; use batch synthetic eval only.")


@torch.no_grad()
def predict_angles_for_items(
    model: AnglePredictor,
    items: List[dict],
    base_len: int,
    device: Optional[torch.device] = None,
) -> List[Tuple[float, float, float]]:
    """Run batched inference to get (rz, rx, rz) angles per item for the first PQC block.

    Returns a list of length len(items), each element is a 3-tuple of floats.
    """
    if device is None:
        device = next(model.parameters()).device
    from torch.utils.data import DataLoader
    dl = DataLoader(items, batch_size=min(256, max(1, len(items))), shuffle=False,
                    collate_fn=lambda xs: collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1))
    preds: List[Tuple[float, float, float]] = []
    for b in dl:
        b = b.to(device)
        tq = torch.zeros(b.base_g.size(0), dtype=torch.long, device=device)
        counts, T_vec = model._counts_from_batch(b, tq)
        ctx = {'batch': b, 'gate_blocks': base_len}
        logits = model(counts, T_vec, ctx)
        ang = logits[:, 0:3, 0].detach().cpu()
        for i in range(ang.size(0)):
            a = ang[i]
            preds.append((float(a[0].item()), float(a[1].item()), float(a[2].item())))
    return preds


def save_augmented_dataset_jsonl(
    path: str,
    items: List[dict],
    angles: List[Tuple[float, float, float]],
    base_len: int,
):
    """Save augmented dataset as JSONL: each line = base N gates + one PQC block (rz,rx,rz).

    No tails are appended; output circuits have length N+3.
    Tokens are simple strings, parameterized gates encoded as e.g. "rz:0.123456".
    """
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    total_lines = 0
    with open(path, 'w', encoding='utf-8') as f:
        for i, it in enumerate(items):
            base_gates = list(it['base_gates'])
            rz, rx, rz2 = angles[i]
            pqc_tokens = [
                f"rz:{rz:.6f}",
                f"rx:{rx:.6f}",
                f"rz:{rz2:.6f}",
            ]
            circuit_tokens = base_gates + pqc_tokens
            f.write(json.dumps({"circuit_tokens": circuit_tokens}, ensure_ascii=False) + "\n")
            total_lines += 1
    print(f"[Save] Augmented dataset written: path={path} items={len(items)} total_lines={total_lines} (each length N+3)")


def main():
    args = _build_argparser().parse_args()
    # Device selection
    if args.device is None:
        device = DEVICE
    else:
        device = torch.device(args.device)
    _print_fused_status_once()

    model, meta = load_model_from_ckpt(args.ckpt, device=device)
    print("[Load] Checkpoint loaded.")
    print(f"[Meta] gate_blocks={meta['gate_blocks']} use_quaternion_head={meta['use_quaternion_head']} use_ordered_seq={meta['use_ordered_seq']} hid_dim={meta['hid_dim']} prev_k={meta['prev_k']} timestamp={meta['timestamp']}")

    # Decide whether to run synthetic eval: if explicitly requested or any synthetic knobs set
    synthetic_knob = args.eval_synthetic or (args.synthetic_base_len is not None)
    if args.eval_synthetic or synthetic_knob:
        # base_len default to gate_blocks from ckpt
        base_len = int(args.synthetic_base_len) if args.synthetic_base_len is not None else int(meta['gate_blocks'])
        # Build base items: ALWAYS enumerate all H/X/Z sequences of length N
        items = _generate_synthetic_single_qubit_all_items(base_len=base_len)
        mode_str = f"enum-all({3**base_len})"
        gate_blocks = int(meta['gate_blocks']) if int(meta['gate_blocks']) > 0 else base_len
        # To ensure exactly one PQC block at the end, set gate_blocks == base_len
        gate_blocks = base_len
        avg_fid = eval_on_items(model, gate_blocks=gate_blocks, items=items, k_random=int(args.k_random), device=device)
        print(f"[Eval][Synthetic] mode={mode_str} base_len={base_len} gate_blocks={gate_blocks} K={args.k_random} => avg_fidelity={avg_fid:.6f}")
        # After eval: save augmented dataset (base+PQC only, length N+3)
        angles = predict_angles_for_items(model, items, base_len=base_len, device=device)
        # Derive a sensible default output path from ckpt and settings
        ckpt_dir = os.path.dirname(args.ckpt)
        ckpt_stem = os.path.basename(args.ckpt).rsplit('.', 1)[0]
        mode_tag = 'enumall'
        out_dir = os.path.join(ckpt_dir, 'augmented')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"aug_{ckpt_stem}_len{base_len}_{mode_tag}_{int(args.k_random)}K_Nplus3.jsonl")
        save_augmented_dataset_jsonl(
            path=out_path,
            items=items,
            angles=angles,
            base_len=base_len,
        )
        # Free the first transformer from memory (no longer needed after saving augmented data)
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == '__main__':
    main()
