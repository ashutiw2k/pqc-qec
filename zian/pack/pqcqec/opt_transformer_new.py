from typing import List, Tuple, Optional, Dict
import argparse
import os
import platform as _platform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

"""
Enable fused base+noise kernel by default from this entry point, unless the
user has explicitly set environment variables. This must be done BEFORE we
import simulator_core, because that module reads env toggles at import time.

Env precedence (already supported by simulator_core):
- TKFS_USE_FUSED_BASE_NOISE or PQC_USE_FUSED -> enable/disable fused path
- PQC_FORCE_INLINE (Windows only) -> allow inline CUDA build attempt when toolchain is present
"""
if os.environ.get('PQC_USE_FUSED') is None and os.environ.get('TKFS_USE_FUSED_BASE_NOISE') is None:
    os.environ['PQC_USE_FUSED'] = '1'
# On Windows, default to allowing inline compile attempt (harmless when no toolchain)
if ((_platform.system().lower().startswith('win')) and os.environ.get('PQC_FORCE_INLINE') is None):
    os.environ['PQC_FORCE_INLINE'] = '1'

# Re-import shared constants and helpers from this package
# Use unified simulator_core utilities (dataset, collate, simulator)
from .simulator_core import (
    CircuitDataset, Batch, collate,
    simulate_loss, simulate_blocks_with_angles, build_base_cache_vectorized, NoiseConfig,
    get_fused_status, ensure_fused_compiled,
)
from .dataset import MAX_BASE_LEN, MAX_PARAM, MAX_QUBITS
from .precision import get_amp_settings, make_grad_scaler
from .angle_predictor_simple import AnglePredictor, HID_DIM, PREV_K
from .subcircuits import SubcircuitDataset  # type: ignore

# Model hyperparameters (defaults match project-wide settings)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ = MAX_BASE_LEN + MAX_PARAM  # cap for positional embeddings over base+param

# Optional: torch.utils Subset type
from torch.utils.data import Subset
import random
import itertools

# --- Saving utility: always save trained model checkpoint ---
def _save_trained_model(model: nn.Module, gate_blocks: int, tag: str = "angle_predictor_1022_start") -> str:
    """Save the trained model unconditionally to models/ with a timestamped filename.

    Returns the absolute path to the saved checkpoint.
    """
    import time as _time
    out_dir = os.path.join(os.getcwd(), "models")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        # fallback to current working directory if models/ cannot be created
        out_dir = os.getcwd()
    ts = _time.strftime("%Y%m%d_%H%M%S")
    filename = f"{tag}_gb{int(gate_blocks)}_{ts}.pt"
    path = os.path.join(out_dir, filename)
    try:
        ckpt = {
            'state_dict': model.state_dict(),
            'gate_blocks': int(gate_blocks),
            'use_ordered_seq': getattr(model, 'use_ordered_seq', None),
            'use_quaternion_head': getattr(model, 'use_quaternion_head', None),
            'hid_dim': int(HID_DIM),
            'prev_k': int(PREV_K),
            'timestamp': ts,
        }
        torch.save(ckpt, path)
    except Exception as _e:
        # last resort: try saving state_dict only
        try:
            torch.save(model.state_dict(), path)
        except Exception:
            raise _e
    print(f"[Save] Model checkpoint saved to: {path}")
    return path

def generate_synthetic_single_qubit_items(
    num_samples: int = 32,
    base_len: int = 5,
    gate_blocks: int = 5,
    gates_vocab: Tuple[str, ...] = ("h", "x", "z"),
    seed: Optional[int] = 0,
) -> List[dict]:
    """Generate a tiny synthetic dataset for the fixed-grid shared-PQC trainer.

    Each sample:
      - n_qubits = 1
      - exactly `base_len` base 1q gates chosen from gates_vocab on qubit 0
      - a single PQC block is intended right after the base segment

    Notes:
      - With `gate_blocks == base_len`, the simulator will insert exactly one
        RZ-RX-RZ block after the 5 base gates (blocks_needed == 1).
      - Param fields (param_gates/after/angles) are left empty because the
        training loop uses the 'blocks' mode (fixed-grid) and ignores them.
      - We set an explicit 'idx' field so caches (idx2row) align with items.
    """
    if seed is not None:
        random.seed(int(seed))
    items: List[dict] = []
    for i in range(num_samples):
        base_gates = [random.choice(gates_vocab) for _ in range(base_len)]
        item = dict(
            idx=i,
            n_qubits=1,
            base_gates=base_gates,
            base_q1=[0] * base_len,
            base_q2=[-1] * base_len,
            # param_* fields unused in 'blocks' mode, keep empty for clarity
            param_gates=[],
            param_qubits=[],
            after=[],
            param_angles_gt=[],
        )
        items.append(item)
    return items

class SyntheticSingleQubitDataset(torch.utils.data.Dataset):
    """A minimal Dataset wrapper around synthetic 1q items.

    Use with build_base_cache_vectorized and collate; each sample is already a
    single-qubit circuit, so no SubcircuitDataset expansion is needed.
    """
    def __init__(self, num_samples: int = 32, base_len: int = 5, gate_blocks: int = 5,
                 gates_vocab: Tuple[str, ...] = ("h","x","z"), seed: Optional[int] = 0):
        self.items = generate_synthetic_single_qubit_items(num_samples, base_len, gate_blocks, gates_vocab, seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

def make_synthetic_loader(num_samples: int = 32,
                          base_len: int = 5,
                          gate_blocks: int = 5,
                          batch_size: int = 32,
                          seed: Optional[int] = 0) -> Tuple[DataLoader, int]:
    """Create a DataLoader over synthetic 1q samples and return (loader, gate_blocks).

    The collate packs directly into a Batch with max_qubits=1. Use this loader
    in place of a real dataset to quickly exercise the training/eval loops.
    """
    ds = SyntheticSingleQubitDataset(num_samples=num_samples, base_len=base_len,
                                     gate_blocks=gate_blocks, seed=seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda xs: collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1),
    )
    return loader, int(gate_blocks)

def generate_synthetic_single_qubit_all_items(
    base_len: int,
    gates_vocab: Tuple[str, ...] = ("h", "x", "z"),
) -> List[dict]:
    """Enumerate ALL 1-qubit circuits of length base_len using gates_vocab.

    Returns a list of items matching collate fields, one per sequence, with
    n_qubits=1 and a single PQC block intended at the end (handled by blocks mode).
    Dataset size is len(gates_vocab)**base_len.
    """
    items: List[dict] = []
    idx = 0
    for seq in itertools.product(gates_vocab, repeat=base_len):
        base_gates = list(seq)
        item = dict(
            idx=idx,
            n_qubits=1,
            base_gates=base_gates,
            base_q1=[0] * base_len,
            base_q2=[-1] * base_len,
            param_gates=[],
            param_qubits=[],
            after=[],
            param_angles_gt=[],
        )
        items.append(item)
        idx += 1
    return items

def make_collate_per_qubit(gate_blocks: int):
    def _fn(samples: List[Tuple[dict, int, int]]):
        sample_dicts = [s[0] for s in samples]
        orig_idx = torch.tensor([s[1] for s in samples], dtype=torch.long)
        target_q = torch.tensor([s[2] for s in samples], dtype=torch.long)
        bat = collate(sample_dicts, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=MAX_QUBITS)
        # Precompute counts per block for each line
        B = bat.base_g.size(0)
        counts_list: List[torch.Tensor] = []
        T_list: List[int] = []
        for i in range(B):
            Lb = int(bat.base_len[i].item())
            T = (Lb + gate_blocks - 1) // max(1, gate_blocks)
            q = int(target_q[i].item())
            q1 = bat.base_q1[i, :Lb]
            q2 = bat.base_q2[i, :Lb]
            touch = (q1 == q) | (q2 == q)
            per_block = []
            for t in range(T):
                s = t * gate_blocks; e = min(Lb, (t + 1) * gate_blocks)
                per_block.append(int(touch[s:e].sum().item()))
            counts_list.append(torch.tensor(per_block, dtype=torch.float32))
            T_list.append(T)
        maxT = max(T_list) if T_list else 0
        if maxT == 0:
            counts = torch.zeros(B, 0)
        else:
            counts = torch.zeros(B, maxT, dtype=torch.float32)
            for i, c in enumerate(counts_list):
                counts[i, :c.numel()] = c
        T_vec = torch.tensor(T_list, dtype=torch.long)
        return bat, orig_idx, target_q, counts, T_vec
    return _fn


## AnglePredictor class moved to .angle_predictor_simple; imported above.


def train_transformer(data_path: str,
                      batch_size: int = 32,
                      epochs: int = 10,
                      lr: float = 5e-4,
                      k_random: int = 32,
                      num_sample: Optional[int] = None,
                      noise: Optional[NoiseConfig] = None,
                      device: Optional[torch.device] = None,
                      gate_blocks: int = 50,
                      detach_base_noise: bool = True,
                      calibrate_centers: int = 0):
    if device is None:
        device = DEVICE
    # One-time fused status + proactive compile attempt
    try:
        fs0 = get_fused_status()
        if fs0.get('enabled', False) and not fs0.get('available', False):
            ensure_fused_compiled()
        fs = get_fused_status()
        print(f"[Fused] enabled={fs['enabled']} available={fs['available']} attempted={fs['attempted']} used_calls={fs['used_calls']} reason={fs['reason']}")
    except Exception:
        pass
    # Print noise configuration for visibility
    try:
        if noise is None:
            print("[Noise] None (defaults inside simulator may apply)")
        else:
            print(f"[Noise] use_noise={getattr(noise,'use_noise', None)}; "
                  f"x_rad={getattr(noise,'noise_x_rad', None):.6f}, z_rad={getattr(noise,'noise_z_rad', None):.6f}; "
                  f"delta_x={getattr(noise,'noise_delta_x', None):.6f}, delta_z={getattr(noise,'noise_delta_z', None):.6f}")
    except Exception:
        pass
    ds_full = CircuitDataset(data_path, num_sample=num_sample)
    # Flatten per-qubit lines by duplicating circuits per target qubit is not implemented here.
    # Use the full dataset directly with a per-circuit collate.
    ds = ds_full
    if len(ds) == 0:
        raise RuntimeError("Empty dataset")
    # Build caches on full dataset (simulation runs on full circuits)
    init_cache, ref_cache, noise_schedules = build_base_cache_vectorized(ds_full, k_random=k_random, device=device, noise=noise)

    # Train/Val split (simple)
    n_total = len(ds)
    n_val = max(1, int(round(n_total * 0.1))) if n_total > 1 else 0
    n_train = n_total - n_val
    idx_all = list(range(n_total))
    import random as _r
    _r.shuffle(idx_all)
    ds_train = Subset(ds, idx_all[:n_train])
    ds_val = Subset(ds, idx_all[n_train:]) if n_val > 0 else None

    # Summary: how many original circuits and per-qubit lines go to train/val
    try:
        M = len(ds_full)
        total_lines = len(ds)
        train_lines = len(ds_train)
        val_lines = len(ds_val) if ds_val is not None else 0
        print(f"[PerQubit] originals={M}, lines_total={total_lines}, train_lines={train_lines}, val_lines={val_lines}")
    except Exception:
        pass

    # Print training/test sample counts before training starts
    try:
        train_count = len(ds_train)
        test_count = len(ds_val) if ds_val is not None else 0
        print(f"[Data] training samples={train_count} test samples={test_count}")
    except Exception:
        pass

    collate_fn = make_collate_per_qubit(gate_blocks)
    # Important: group by circuit to ensure each simulation sees a full set of n_qubits lines
    # We'll use a moderate batch size but collate_fn returns mixed circuits; we will regroup inside the loop.
    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if ds_val else None

    model = AnglePredictor(gate_blocks=gate_blocks).to(device)
    # AMP/Scaler setup
    amp_enabled, amp_dtype = get_amp_settings()
    scaler = make_grad_scaler(amp_enabled, amp_dtype)
    # Optional: center calibration using a few batches from the original circuit dataset
    if calibrate_centers and calibrate_centers > 0:
        cal_loader = DataLoader(ds_full, batch_size=batch_size, shuffle=True,
                                collate_fn=lambda xs: collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=MAX_QUBITS))
        try:
            model.calibrate_centers(cal_loader, device=device, gate_blocks=gate_blocks, max_batches=calibrate_centers)
            print(f"[Calibrate] Centers estimated over {calibrate_centers} batch(es); center subtraction enabled.")
        except Exception as e:
            print(f"[Calibrate] Skipped due to error: {e}")
    # Optimizer + optional LR scheduler for faster convergence
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    # Lightweight epoch scheduler: linear warmup then cosine decay (transformer path)
    try:
        import math as _m
        warmup_ep = int(os.environ.get('PQC_WARMUP_EPOCHS', '5'))
        min_lr_ratio = float(os.environ.get('PQC_MIN_LR_RATIO', '0.1'))
        def _lr_lambda(ep_idx: int):
            if epochs <= 0:
                return 1.0
            if ep_idx < warmup_ep and warmup_ep > 0:
                return max(1e-3, float(ep_idx + 1) / float(max(1, warmup_ep)))
            t = max(0, ep_idx - warmup_ep)
            T = max(1, epochs - warmup_ep)
            cos_inner = _m.pi * float(t) / float(T)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + _m.cos(cos_inner))
        _sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    except Exception:
        _sched = None

    def evaluate():
        if val_loader is None:
            return float('nan')
        model.eval(); total = 0.0; nb = 0
        with torch.no_grad():
            for raw in val_loader:
                batch, orig_idx, tq, counts, T_vec = raw
                batch = batch.to(device); orig_idx = orig_idx.to(device); tq = tq.to(device)
                counts = counts.to(device); T_vec = T_vec.to(device)
                # model forward under AMP (simulator will run in FP32 separately)
                ctx = {'batch': batch, 'gate_blocks': gate_blocks}
                if amp_enabled and amp_dtype is not None and torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        per_logits = model(counts, T_vec, ctx)
                else:
                    per_logits = model(counts, T_vec, ctx)
                # simulator in FP32
                if amp_enabled and torch.cuda.is_available():
                    with torch.amp.autocast('cuda', enabled=False):
                        loss = simulate_loss(
                            batch, per_logits.float(), init_cache, ref_cache, noise_schedules,
                            mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise
                        )
                else:
                    loss = simulate_loss(
                        batch, per_logits, init_cache, ref_cache, noise_schedules,
                        mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise
                    )
                total += float(1.0 - loss.detach()); nb += 1
        model.train()
        return total / max(1, nb)

    for ep in range(1, epochs + 1):
        model.train(); total = 0.0
        for raw in loader:
            batch, orig_idx, tq, counts, T_vec = raw
            batch = batch.to(device); orig_idx = orig_idx.to(device); tq = tq.to(device)
            counts = counts.to(device); T_vec = T_vec.to(device)
            # model forward under AMP (simulator will run in FP32 separately)
            ctx = {'batch': batch, 'gate_blocks': gate_blocks}
            if amp_enabled and amp_dtype is not None and torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    per_logits = model(counts, T_vec, ctx)
            else:
                per_logits = model(counts, T_vec, ctx)
            # Optional similarity diagnostics removed for cleanliness

            # Print the first PQC block's rz, rx, rz angles for all lines in this batch
            try:
                with torch.no_grad():
                    block0 = per_logits[:, 0:3, 0]  # [B,3]
                    triplets = " ".join([f"[{a:.3f},{b:.3f},{c:.3f}]" for a, b, c in block0.tolist()])
                    print(f"[Block0Angles] {triplets}")
            except Exception:
                pass
            # Direct per-batch loss using current logits layout
            loss = simulate_loss(
                batch, per_logits, init_cache, ref_cache, noise_schedules,
                mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise
            )
            # Ensure simulator runs in FP32 (disable autocast region)
            if amp_enabled and torch.cuda.is_available():
                with torch.amp.autocast('cuda', enabled=False):
                    loss = simulate_loss(
                        batch, per_logits.float(), init_cache, ref_cache, noise_schedules,
                        mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise
                    )
            else:
                loss = simulate_loss(
                    batch, per_logits, init_cache, ref_cache, noise_schedules,
                    mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise
                )
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                if scaler is not None:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                print("[Warn] Non-finite loss encountered in train_transformer step; skipping optimizer step.")
            total += float(loss.detach())
        avg = total / max(1, len(loader))
        val_fid = evaluate()
        # Step scheduler and report LR
        try:
            if _sched is not None:
                _sched.step()
                cur_lr = opt.param_groups[0]['lr']
                print(f"[Transformer] epoch {ep} lr={cur_lr:.6e} loss={avg:.6f} val_fid={val_fid:.6f}")
            else:
                print(f"[Transformer] epoch {ep} loss={avg:.6f} val_fid={val_fid:.6f}")
        except Exception:
            print(f"[Transformer] epoch {ep} loss={avg:.6f} val_fid={val_fid:.6f}")
    # Always save the trained model before returning
    try:
        _save_trained_model(model, gate_blocks, tag="angle_predictor_1022_start_transformer")
    except Exception as e:
        print(f"[Save] Failed to save model checkpoint: {e}")
    return model


def train_subcircuits(data_path: str,
                      batch_size: int = 32,
                      epochs: int = 1,
                      lr: float = 5e-4,
                      k_random: int = 32,
                      num_sample: Optional[int] = None,
                      noise: Optional[NoiseConfig] = None,
                      device: Optional[torch.device] = None,
                      gate_blocks: int = 50,
                      detach_base_noise: bool = True,
                      use_synthetic_1q5: bool = False,
                      synthetic_num_samples: Optional[int] = None,
                      synthetic_base_len: Optional[int] = None,
                      synthetic_train_count: Optional[int] = None,
                      synthetic_train_frac: Optional[float] = None,
                      synthetic_split_seed: int = 0,
                      synthetic_enumerate_all: bool = False,
                      post_eval_chain: bool = False,
                      post_eval_count: int = 50,
                      post_eval_k: int = 100,
                      sub_val_count: Optional[int] = None,
                      calibrate_centers: int = 0,
                      use_quaternion_head: bool = False,
                      mix_l2: float = 0.0,
                      mix_warmup_epochs: int = 0,
                      mix_warmup_alpha: float = 0.7,
                      mix_warmup_beta: float = 0.3,
                      mix_warmup_gamma: float = 0.3,
                      hist_dropout: float = 0.0,
                      hist_scale_min: float = 1.0,
                      hist_scale_max: float = 1.0,
                      noise_boost: float = 1.0,
                      noise_boost_epochs: int = 0,
                      history_freeze_prob: float = 0.0,
                      prev_noise_std: float = 0.0,
                      history_freeze_epochs: Optional[int] = None,
                      aux_angle_weight: float = 0.0,
                      aux_angle_blocks: int = 0):
    if device is None:
        device = DEVICE
    # Minimal logging mode: keep epoch summaries only
    VERBOSE = False
    # One-time fused status + proactive compile attempt
    try:
        fs0 = get_fused_status()
        if fs0.get('enabled', False) and not fs0.get('available', False):
            ensure_fused_compiled()
        fs = get_fused_status()
        print(f"[Fused] enabled={fs['enabled']} available={fs['available']} attempted={fs['attempted']} used_calls={fs['used_calls']} reason={fs['reason']}")
    except Exception:
        pass
    import time as _time
    # Print noise configuration once at start of training (verbose only)
    if VERBOSE:
        try:
            if noise is None:
                print("[Noise] None (defaults inside simulator may apply)")
            else:
                print(f"[Noise] use_noise={getattr(noise,'use_noise', None)}; "
                      f"x_rad={getattr(noise,'noise_x_rad', None):.6f}, z_rad={getattr(noise,'noise_z_rad', None):.6f}; "
                      f"delta_x={getattr(noise,'noise_delta_x', None):.6f}, delta_z={getattr(noise,'noise_delta_z', None):.6f}")
        except Exception:
            pass
    # dataset: either synthetic 1q len-5 for a single PQC block, or real dataset
    SYN_ENV = str(os.environ.get('PQC_SYNTH_1Q5', '0')).strip().lower() in ('1','true','yes','y','on')
    use_syn = bool(use_synthetic_1q5 or SYN_ENV)
    if use_syn:
        # Force single-qubit, base_len = N (configurable), and a single PQC block after base.
        # To ensure exactly one PQC block, set gate_blocks >= base_len; we choose gate_blocks = base_len.
        base_len_cfg = int(os.environ.get('PQC_SYNTH_BASE_LEN', str(synthetic_base_len if synthetic_base_len is not None else 5)))
        enum_env = str(os.environ.get('PQC_SYNTH_ENUM_ALL', '0')).strip().lower() in ('1','true','yes','y','on')
        use_enum = bool(synthetic_enumerate_all or enum_env)
        base_len_cfg = max(1, min(base_len_cfg, MAX_BASE_LEN))
        gate_blocks = base_len_cfg
        if use_enum:
            syn_items = generate_synthetic_single_qubit_all_items(
                base_len=base_len_cfg,
                gates_vocab=("h","x","z"),
            )
        else:
            num_samples_cfg = int(os.environ.get('PQC_SYNTH_NUM_SAMPLES', str(synthetic_num_samples if synthetic_num_samples is not None else 10000)))
            syn_items = generate_synthetic_single_qubit_items(
                num_samples=num_samples_cfg,
                base_len=base_len_cfg,
                gate_blocks=gate_blocks,
                gates_vocab=("h","x","z"),
                seed=0,
            )
        class _SynWrap(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items
            def __len__(self):
                return len(self.items)
            def __getitem__(self, i):
                return self.items[i]
        base = _SynWrap(syn_items)
        # For synthetic 1q lines, we don't need SubcircuitDataset expansion
        sub = base
        if VERBOSE:
            mode_str = "enum-all" if use_enum else "random"
            print(f"[Synthetic-1q] mode={mode_str} samples={len(syn_items)} base_len={base_len_cfg} gate_blocks={gate_blocks} (one PQC post-base)")
    else:
        base = CircuitDataset(data_path, num_sample=num_sample,
                              max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=MAX_QUBITS)
        sub = SubcircuitDataset(base)
    if len(sub) == 0:
        raise RuntimeError("No subcircuits constructed from dataset")

    # Build caches for 1-qubit subcircuits
    init_cache, ref_cache, noise_schedules = build_base_cache_vectorized(sub, k_random=k_random, device=device, noise=noise)

    # Split into train/val
    N = len(sub)
    indices = list(range(N))
    import random as _r
    _r.shuffle(indices)
    from torch.utils.data import Subset
    if use_syn:
        # Synthetic mode: optionally perform a complementary train/eval split
        # via explicit train count/frac (CLI or env), else fallback to small val subset.
        try:
            if synthetic_train_count is None:
                _env_cnt = os.environ.get('PQC_SYNTH_TRAIN_COUNT', None)
                synthetic_train_count = int(_env_cnt) if _env_cnt not in (None, '') else None
        except Exception:
            pass
        try:
            if synthetic_train_frac is None:
                _env_frac = os.environ.get('PQC_SYNTH_TRAIN_FRAC', None)
                synthetic_train_frac = float(_env_frac) if _env_frac not in (None, '') else None
        except Exception:
            pass
        try:
            _env_seed = os.environ.get('PQC_SYNTH_SPLIT_SEED', None)
            if _env_seed not in (None, ''):
                synthetic_split_seed = int(_env_seed)
        except Exception:
            pass

        do_holdout = (synthetic_train_count is not None) or (synthetic_train_frac is not None)
        if do_holdout and N > 0:
            if synthetic_train_count is not None:
                train_cnt = int(max(1, min(int(synthetic_train_count), max(1, N - 1))))
            else:
                frac = float(synthetic_train_frac) if synthetic_train_frac is not None else 0.8
                frac = max(0.0, min(1.0, frac))
                train_cnt = int(round(N * frac))
                if N > 1:
                    train_cnt = max(1, min(train_cnt, N - 1))
                else:
                    train_cnt = 1
            _rng = random.Random(int(synthetic_split_seed))
            perm = list(range(N))
            _rng.shuffle(perm)
            train_idx = perm[:train_cnt]
            val_idx = perm[train_cnt:]
            ds_train = Subset(sub, train_idx)
            ds_val = Subset(sub, val_idx) if len(val_idx) > 0 else None
            print(f"[Synthetic Split] N={N} train={len(train_idx)} val={len(val_idx)} seed={synthetic_split_seed}")
            if VERBOSE and synthetic_enumerate_all:
                print("[Synthetic Split] source=ENUM-ALL (complementary split)")
        else:
            # Fallback: use ALL for train; val is a small random subset from the same pool
            if sub_val_count is not None:
                val_cnt = min(int(sub_val_count), N)
            else:
                # default: ~5% capped at 1000, at least 1 if N>0
                val_cnt = min(max(1, N // 20), 1000) if N > 0 else 0
            val_idx = _r.sample(indices, val_cnt) if val_cnt > 0 else []
            ds_train = sub
            ds_val = Subset(sub, val_idx) if val_idx else None
            if VERBOSE:
                print(f"[Synthetic Split] train=ALL({N}) val={len(val_idx)} (subset of train)")
    else:
        if sub_val_count is not None:
            val_cnt = min(int(sub_val_count), N)
        else:
            val_cnt = max(1, N // 10) if N > 1 else 0
        val_idx = indices[:val_cnt]
        train_idx = indices[val_cnt:]
        ds_train = Subset(sub, train_idx) if train_idx else None
        ds_val = Subset(sub, val_idx) if val_idx else None

    if VERBOSE:
        print(f"[Subcircuits] total={N}, train={len(train_idx)}, val={len(val_idx)}")

    # Always print train/test sample counts for subcircuits before training starts
    try:
        train_count = len(ds_train) if ds_train is not None else 0
        val_count = len(ds_val) if ds_val is not None else 0
        print(f"[Subcircuits][Data] training samples={train_count} test samples={val_count}")
    except Exception:
        pass

    collate_fn = lambda samples: collate(samples, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1)
    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) if ds_train else None
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if ds_val else None

    model = AnglePredictor(gate_blocks=gate_blocks, use_quaternion_head=use_quaternion_head).to(device)
    # AMP/Scaler setup
    amp_enabled, amp_dtype = get_amp_settings()
    scaler = make_grad_scaler(amp_enabled, amp_dtype)
    # configure train-time knobs
    model.hist_drop_p = float(hist_dropout)
    model.hist_scale_min = float(hist_scale_min)
    model.hist_scale_max = float(hist_scale_max)
    model.noise_boost = float(noise_boost)
    model.noise_boost_epochs = int(noise_boost_epochs)
    # AR stabilization knobs
    model.p_history_freeze = float(history_freeze_prob)
    model.prev_noise_std = float(prev_noise_std)
    model.history_freeze_epochs = int(mix_warmup_epochs if history_freeze_epochs is None else history_freeze_epochs)
    # Optional: center calibration using subcircuits (matches training domain)
    if calibrate_centers and calibrate_centers > 0:
        cal_base = sub
        from torch.utils.data import DataLoader as _DL
        cal_loader = _DL(cal_base, batch_size=batch_size, shuffle=True,
                         collate_fn=lambda xs: collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1))
        try:
            model.calibrate_centers(cal_loader, device=device, gate_blocks=gate_blocks, max_batches=calibrate_centers)
            if VERBOSE:
                print(f"[Calibrate] Centers estimated over {calibrate_centers} batch(es); center subtraction enabled.")
        except Exception as e:
            if VERBOSE:
                print(f"[Calibrate] Skipped due to error: {e}")
    # Optimizer with parameter groups: give heads/mix a higher LR to speed early learning
    head_lr_factor = float(os.environ.get('PQC_LR_HEAD_FACTOR', '3.0'))
    base_params = []
    fast_params = []
    fast_names = {"head", "head_q", "mix_alpha", "mix_beta", "mix_gamma"}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        root = name.split('.')[0]
        if (root in fast_names) or name.startswith('head.') or name.startswith('head_q.'):
            fast_params.append(p)
        else:
            base_params.append(p)
    if len(fast_params) == 0:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    else:
        opt = torch.optim.AdamW([
            {"params": base_params, "lr": lr},
            {"params": fast_params, "lr": lr * head_lr_factor},
        ], betas=(0.9, 0.99))

    # LR scheduler: linear warmup then cosine decay (epoch-level)
    import math as _m
    warmup_ep = int(os.environ.get('PQC_WARMUP_EPOCHS', '5'))
    min_lr_ratio = float(os.environ.get('PQC_MIN_LR_RATIO', '0.1'))
    def _lr_lambda(ep_idx: int):
        if epochs <= 0:
            return 1.0
        if ep_idx < warmup_ep and warmup_ep > 0:
            return max(1e-3, float(ep_idx + 1) / float(max(1, warmup_ep)))
        t = max(0, ep_idx - warmup_ep)
        T = max(1, epochs - warmup_ep)
        cos_inner = _m.pi * float(t) / float(T)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + _m.cos(cos_inner))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    # Optional curriculum on max blocks per sample to reduce difficulty early
    cur_enable = str(os.environ.get('PQC_CURRICULUM', '0')).strip().lower() in ('1','true','yes','y','on')
    cur_start = int(os.environ.get('PQC_CURR_START_BLOCKS', '2'))
    cur_step = int(os.environ.get('PQC_CURR_STEP', '1'))
    cur_every = int(os.environ.get('PQC_CURR_EVERY', '1'))  # epochs per increment
    cur_cap = int(os.environ.get('PQC_CURR_MAX_BLOCKS', '-1'))  # -1 means no explicit cap

    # one-time prints for examples
    prints = {"train_once": False, "eval_once": False}

    # Diagnostics controls
    diag_enable = str(os.environ.get('PQC_DIAG', '0')).strip().lower() in ('1','true','yes','y','on')
    diag_eps = float(os.environ.get('PQC_DIAG_EPS', '1e-3'))

    # --- Physics-aware context: cumulative quaternion up to each block ---
    def _quat_mul(a: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                  b: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        aw, ax, ay, az = a; bw, bx, by, bz = b
        return (
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        )

    def _rx(angle: torch.Tensor):
        c = torch.cos(angle/2); s = torch.sin(angle/2)
        z = torch.zeros_like(c)
        return (c, s, z, z)

    def _rz(angle: torch.Tensor):
        c = torch.cos(angle/2); s = torch.sin(angle/2)
        z = torch.zeros_like(c)
        return (c, z, z, s)

    def _gate_quat(gcode: int, device: torch.device):
        # 0=H, 1=X, 2=Z; approximate H as Rz(pi/2) Rx(pi/2)
        if gcode == 1:
            ang = torch.tensor(torch.pi, device=device)
            return _rx(ang)
        elif gcode == 2:
            ang = torch.tensor(torch.pi, device=device)
            return _rz(ang)
        else:
            a = torch.tensor(torch.pi/2, device=device)
            return _quat_mul(_rz(a), _rx(a))

    def _compute_context_quat(b: Batch, T_vec: torch.Tensor) -> torch.Tensor:
        Bsz = b.base_g.size(0)
        maxT = int(T_vec.max().item() if T_vec.numel() > 0 else 0)
        ctx = torch.zeros(Bsz, maxT, 4, device=b.base_g.device)
        if maxT == 0:
            return ctx
        for i in range(Bsz):
            Lb = int(b.base_len[i].item())
            if Lb <= 0:
                continue
            row = noise_schedules['idx2row'][int(b.idx[i].item())]
            rx_q1_full = noise_schedules['rx_q1'][row, :Lb]
            rz_q1_full = noise_schedules['rz_q1'][row, :Lb]
            T = int(T_vec[i].item())
            # cumulative across prefix; to keep it simple and robust, recompute up to e each time
            for t in range(T):
                s = t * gate_blocks; e = min(Lb, (t + 1) * gate_blocks)
                if s >= e:
                    continue
                qw = torch.tensor(1.0, device=b.base_g.device); qx = torch.tensor(0.0, device=b.base_g.device)
                qy = torch.tensor(0.0, device=b.base_g.device); qz = torch.tensor(0.0, device=b.base_g.device)
                for p in range(e):
                    g = int(b.base_g[i, p].item())
                    if g < 0:
                        continue
                    qg = _gate_quat(g, b.base_g.device)
                    angx = rx_q1_full[p] if p < rx_q1_full.numel() else torch.tensor(0.0, device=b.base_g.device)
                    angz = rz_q1_full[p] if p < rz_q1_full.numel() else torch.tensor(0.0, device=b.base_g.device)
                    qn = _quat_mul(_rx(angx), _rz(angz))
                    q_step = _quat_mul(qn, qg)
                    qw, qx, qy, qz = _quat_mul((qw, qx, qy, qz), q_step)
                norm = torch.sqrt(qw*qw + qx*qx + qy*qy + qz*qz + 1e-8)
                ctx[i, t, 0] = qw / norm
                ctx[i, t, 1] = qx / norm
                ctx[i, t, 2] = qy / norm
                ctx[i, t, 3] = qz / norm
        return ctx

    def _print_angle_examples(logits: torch.Tensor, T_vec: torch.Tensor, tag: str):
        try:
            with torch.no_grad():
                B = logits.size(0)
                show = min(6, B)
                msgs = []
                for i in range(show):
                    Ti = int(T_vec[i].item()) if T_vec.numel() > i else 0
                    if Ti <= 0:
                        msgs.append("[]"); continue
                    v0 = logits[i, 0:3, 0].tolist()
                    msgs.append(f"[{v0[0]:.3f},{v0[1]:.3f},{v0[2]:.3f}]")
                print(f"[Subcircuits][{tag}][Block0Angles] {' '.join(msgs)}")
        except Exception:
            pass

    def _slice_batch(b: Batch, idx: torch.Tensor) -> Batch:
        # Build a new Batch selecting rows by index tensor
        return Batch(
            base_g=b.base_g.index_select(0, idx),
            base_q1=b.base_q1.index_select(0, idx),
            base_q2=b.base_q2.index_select(0, idx),
            param_g=b.param_g.index_select(0, idx),
            param_q=b.param_q.index_select(0, idx),
            param_after=b.param_after.index_select(0, idx),
            param_angles_gt=b.param_angles_gt.index_select(0, idx),
            base_len=b.base_len.index_select(0, idx),
            param_len=b.param_len.index_select(0, idx),
            n_qubits=b.n_qubits.index_select(0, idx),
            idx=b.idx.index_select(0, idx),
        )

    def _grouped_blocks_loss(b: Batch, logits: torch.Tensor) -> torch.Tensor:
        """Return differentiable mean loss across length-groups.
        We must group by base_len to satisfy simulator assumptions, but keep the
        result as a torch scalar that retains gradients to the logits.
        """
        device0 = logits.device
        uniq_L = torch.unique(b.base_len)
        # Weighted by number of samples in each group to approximate per-sample mean
        num_sum = torch.tensor(0.0, device=device0)
        wloss_sum = torch.tensor(0.0, device=device0)
        for L in uniq_L.tolist():
            mask = (b.base_len == L)
            sel = mask.nonzero(as_tuple=False).squeeze(-1)
            if sel.numel() == 0:
                continue
            b_sub = _slice_batch(b, sel)
            l_sub = logits.index_select(0, sel)
            lval = simulate_loss(b_sub, l_sub, init_cache, ref_cache, noise_schedules,
                                 mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise)
            n = torch.tensor(float(sel.numel()), device=device0)
            wloss_sum = wloss_sum + lval * n
            num_sum = num_sum + n
        if float(num_sum.item()) <= 0.0:
            return torch.as_tensor(float('nan'), device=device0)
        return wloss_sum / num_sum

    # Controls for validation cost
    eval_every = int(os.environ.get('PQC_EVAL_EVERY', '1'))
    eval_max_batches = int(os.environ.get('PQC_EVAL_MAX_BATCHES', '-1'))  # -1 = all batches

    def eval_avg_fid() -> float:
        if val_loader is None:
            return float('nan')
        model.eval(); total = 0.0; count = 0
        with torch.no_grad():
            for bi, b in enumerate(val_loader):
                if eval_max_batches > 0 and bi >= eval_max_batches:
                    break
                b = b.to(device)
                # target qubit is always 0 in subcircuits
                tq = torch.zeros(b.base_g.size(0), dtype=torch.long, device=device)
                counts, T_vec = model._counts_from_batch(b, tq)
                # If ordered-seq mode, we don't build histograms or prefixes
                Bsz = b.base_g.size(0)
                maxT = counts.size(1)
                ctx = {'batch': b, 'gate_blocks': gate_blocks}
                logits = model(counts, T_vec, ctx)
                if VERBOSE and (not prints["eval_once"]):
                    _print_angle_examples(logits, T_vec, tag="Eval")
                    prints["eval_once"] = True
                # Compute fidelity grouped by base_len
                uniq_L = torch.unique(b.base_len)
                batch_sum = 0.0
                batch_cnt = 0
                for L in uniq_L.tolist():
                    mask = (b.base_len == L)
                    sel = mask.nonzero(as_tuple=False).squeeze(-1)
                    if sel.numel() == 0:
                        continue
                    b_sub = type(b)(
                        b.base_g.index_select(0, sel), b.base_q1.index_select(0, sel), b.base_q2.index_select(0, sel),
                        b.param_g.index_select(0, sel), b.param_q.index_select(0, sel), b.param_after.index_select(0, sel), b.param_angles_gt.index_select(0, sel),
                        b.base_len.index_select(0, sel), b.param_len.index_select(0, sel), b.n_qubits.index_select(0, sel), b.idx.index_select(0, sel)
                    )
                    l_sub = logits.index_select(0, sel)
                    lval = simulate_loss(b_sub, l_sub, init_cache, ref_cache, noise_schedules,
                                         mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise)
                    fid = float(1.0 - lval.item())
                    n = sel.numel()
                    batch_sum += fid * n
                    batch_cnt += n
                if batch_cnt > 0:
                    total += batch_sum
                    count += batch_cnt
        model.train()
        return total / max(1, count)

    # quick train (optional) then report validation average fidelity
    for ep in range(1, epochs + 1):
        model.set_epoch(ep)
        if mix_warmup_epochs > 0 and ep <= mix_warmup_epochs:
            # prefer 3-way override if provided
            model.set_mix_override3(mix_warmup_alpha, mix_warmup_beta, mix_warmup_gamma)
        else:
            model.clear_mix_override()
        if loader is not None:
            model.train(); total = 0.0; nb = 0
            # epoch timing accumulators (cumulative)
            fwd_t = 0.0; sim_t = 0.0; bwd_t = 0.0; opt_t = 0.0; prep_t = 0.0; data_t = 0.0
            # last-batch timings (per-iteration)
            fwd_last = 0.0; sim_last = 0.0; bwd_last = 0.0; opt_last = 0.0; prep_last = 0.0; data_last = 0.0; iter_last = 0.0; other_last = 0.0
            # optional CUDA synchronize for accurate timing (can add overhead)
            timing_sync = str(os.environ.get('TKFS_TIMING_SYNC', '0')).strip().lower() in ('1','true','yes','y','on')
            def _sync():
                try:
                    if timing_sync and torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
            # optional tqdm progress bar per epoch
            use_tqdm = str(os.environ.get('TKFS_TQDM', '0')).strip().lower() in ('1','true','yes','y','on')
            pbar = None
            try:
                if use_tqdm:
                    from tqdm import tqdm  # type: ignore
                    pbar = tqdm(loader, total=len(loader), desc=f"Epoch {ep}/{epochs}", dynamic_ncols=True, leave=False)
                    it = enumerate(pbar)
                else:
                    it = enumerate(loader)
            except Exception:
                it = enumerate(loader)
                pbar = None
            # track time gap between iterations to estimate DataLoader/host overhead
            prev_end = _time.perf_counter()
            for bi, b in it:
                # measure data loading interval since previous batch finished
                t_loop_start = _time.perf_counter()
                data_last = t_loop_start - prev_end
                data_t += data_last
                b = b.to(device)
                tq = torch.zeros(b.base_g.size(0), dtype=torch.long, device=device)
                # preparation timing: counts/T_vec, hist3, base_prefix, base_sum, ctx
                t_prep0 = _time.perf_counter()
                counts, T_vec = model._counts_from_batch(b, tq)
                # Apply curriculum by clamping max blocks this epoch if enabled
                if cur_enable:
                    # determine current allowed max blocks
                    steps_inc = max(0, (ep - 1) // max(1, cur_every))
                    allow = cur_start + steps_inc * cur_step
                    if cur_cap > 0:
                        allow = min(allow, cur_cap)
                    # clamp T_vec and slice feature tensors later
                    T_vec = torch.minimum(T_vec, torch.tensor(allow, device=T_vec.device, dtype=T_vec.dtype))
                Bsz = b.base_g.size(0)
                # effective max steps for this batch
                maxT = int(T_vec.max().item() if T_vec.numel() > 0 else 0)
                # truncate counts to reduce compute when curriculum active
                if counts.size(1) > maxT:
                    counts = counts[:, :maxT]
                # Build context: always use ordered-seq path; skip legacy feature prep entirely
                ctx = {'batch': b, 'gate_blocks': gate_blocks}
                t_prep1 = _time.perf_counter()
                prep_last = t_prep1 - t_prep0 + (t_prep0 - t_loop_start)  # include device copy + counts
                prep_t += prep_last
                t0 = _time.perf_counter()
                _sync()
                # model forward under AMP
                if amp_enabled and amp_dtype is not None and torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits = model(counts, T_vec, ctx)
                else:
                    logits = model(counts, T_vec, ctx)
                _sync()
                t1 = _time.perf_counter()
                fwd_last = (t1 - t0); fwd_t += fwd_last

                # Diagnostics on first batch each epoch (lightweight)
                if diag_enable and bi == 0:
                    try:
                        with torch.no_grad():
                            # Angle stats at block 0
                            if logits.size(1) >= 3:
                                blk0 = logits[:, 0:3, 0]
                                m = blk0.mean(dim=0); s = blk0.std(dim=0)
                                print(f"[Diag][Angles@blk0] mean=({m[0]:.3f},{m[1]:.3f},{m[2]:.3f}) std=({s[0]:.3f},{s[1]:.3f},{s[2]:.3f})")
                            # T coverage
                            mean_T = float(T_vec.float().mean().item()) if T_vec.numel()>0 else 0.0
                            max_T = int(T_vec.max().item()) if T_vec.numel()>0 else 0
                            print(f"[Diag][T] mean={mean_T:.2f} max={max_T}")
                        # Finite-difference loss sensitivity on sample 0 and first angle
                        if b.base_g.size(0) > 0 and logits.size(1) >= 1:
                            sel = torch.tensor([0], dtype=torch.long, device=device)
                            b1 = _slice_batch(b, sel)
                            l0 = simulate_loss(b1, logits.index_select(0, sel), init_cache, ref_cache, noise_schedules,
                                               mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise)
                            pert = logits.index_select(0, sel).clone()
                            pert[0, 0, 0] = pert[0, 0, 0] + diag_eps
                            l1 = simulate_loss(b1, pert, init_cache, ref_cache, noise_schedules,
                                               mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise)
                            dL = (float(l1.detach()) - float(l0.detach())) / diag_eps
                            print(f"[Diag][FD] dL/dy(blk0.rz1)≈{dL:.6f} (eps={diag_eps})")
                    except Exception as _e:
                        print(f"[Diag] skipped due to: {_e}")
                # Diagnostics on the first batch of each epoch (disabled in minimal mode)
                if VERBOSE and bi == 0:
                    try:
                        Bsz_dbg = logits.size(0)
                        # Angle dispersion at block 0
                        if Bsz_dbg > 1 and logits.size(1) >= 3:
                            blk0 = logits[:, 0:3, 0]
                            stds = blk0.std(dim=0)
                            print(f"[Diag][Train][Blk0 Angle STD] rz1={stds[0]:.4f}, rx={stds[1]:.4f}, rz2={stds[2]:.4f}")
                        # Valid step coverage
                        maxT_dbg = int(T_vec.max().item() if T_vec.numel()>0 else 0)
                        mean_T = float(T_vec.float().mean().item()) if T_vec.numel()>0 else 0.0
                        print(f"[Diag][Train][StepCoverage] mean_T={mean_T:.2f}, max_T={maxT_dbg}")
                        # Legacy-path diagnostics removed (always using ordered-seq path now)
                    except Exception:
                        pass
                if VERBOSE and (not prints["train_once"]):
                    _print_angle_examples(logits, T_vec, tag="Train")
                    prints["train_once"] = True
                # Compute base loss grouped by base_len (differentiable)
                t2 = _time.perf_counter(); _sync()
                # simulator (loss) in FP32 regardless of AMP
                if amp_enabled and torch.cuda.is_available():
                    with torch.amp.autocast('cuda', enabled=False):
                        base_loss = _grouped_blocks_loss(b, logits.float())
                else:
                    base_loss = _grouped_blocks_loss(b, logits)
                _sync(); t3 = _time.perf_counter()
                sim_last = (t3 - t2); sim_t += sim_last
                # Optional auxiliary MSE on early blocks against zero target (or ground-truth if available)
                aux_loss = 0.0
                if aux_angle_weight > 0.0 and aux_angle_blocks > 0:
                    # we don't have ground-truth per-block angles here; encourage non-zero diversity via small L2 away from 0
                    # to nudge encoder to produce variance, we can push angles slightly away from uniform zero
                    Bcur = logits.size(0)
                    take = min(aux_angle_blocks * 3, logits.size(1))
                    if take > 0:
                        ang = logits[:, :take, 0]
                        aux_loss = aux_angle_weight * (ang.pow(2).mean())
                # mix regularization: encourage |alpha-beta| small & keep magnitudes bounded (include gamma)
                reg = 0.0
                if mix_l2 and mix_l2 > 0.0:
                    a = model.mix_alpha; bmx = model.mix_beta; gmx = model.mix_gamma
                    reg = mix_l2 * ((a - bmx)**2 + 0.01 * (a*a + bmx*bmx + gmx*gmx))
                loss = base_loss + (reg if isinstance(reg, torch.Tensor) else torch.as_tensor(reg, device=device, dtype=base_loss.dtype))
                if isinstance(aux_loss, torch.Tensor):
                    loss = loss + aux_loss
                opt.zero_grad(set_to_none=True)
                # Guard against NaN/Inf losses
                if not torch.isfinite(loss):
                    print("[Warn] Non-finite loss encountered; skipping backward/step this batch.")
                    continue
                t4 = _time.perf_counter(); _sync()
                # Retain logits grad for flow check when diagnostics enabled
                if diag_enable and bi == 0:
                    try:
                        logits.retain_grad()
                    except Exception:
                        pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                _sync(); t5 = _time.perf_counter()
                bwd_last = (t5 - t4); bwd_t += bwd_last
                if diag_enable and bi == 0:
                    try:
                        # Gradient flow snapshot
                        lg = float(logits.grad.norm().item()) if (hasattr(logits, 'grad') and logits.grad is not None) else float('nan')
                        def pgn(m):
                            s = 0.0
                            for p in m.parameters(recurse=True):
                                if p.grad is not None:
                                    s += float(p.grad.detach().norm().item())
                            return s
                        g_head = pgn(model.head)
                        # report the correct encoder grads depending on mode
                        g_enc = pgn(model.seq_encoder) if getattr(model, 'use_ordered_seq', False) else pgn(model.encoder)
                        g_hist = pgn(model.hist_pair_mlp)
                        g_in = pgn(model.in_proj)
                        a_g = float(model.mix_alpha.grad.abs().item()) if model.mix_alpha.grad is not None else 0.0
                        b_g = float(model.mix_beta.grad.abs().item()) if model.mix_beta.grad is not None else 0.0
                        g_g = float(model.mix_gamma.grad.abs().item()) if model.mix_gamma.grad is not None else 0.0
                        print(f"[Diag][GradFlow] logits={lg:.6f} head={g_head:.3f} encoder={g_enc:.3f} hist_mlp={g_hist:.3f} in_proj={g_in:.3f} mixα={a_g:.3e} mixβ={b_g:.3e} mixγ={g_g:.3e}")
                    except Exception:
                        pass
                # Print gradient norms (first batch only) — disabled in minimal mode
                if VERBOSE and bi == 0:
                    try:
                        def grad_norm(module):
                            total = 0.0
                            for p in module.parameters(recurse=True):
                                if p.grad is not None:
                                    g = p.grad.detach()
                                    total += float(g.norm().item())
                            return total
                        mix_alpha_g = float(model.mix_alpha.grad.abs().item()) if model.mix_alpha.grad is not None else 0.0
                        mix_beta_g  = float(model.mix_beta.grad.abs().item()) if model.mix_beta.grad is not None else 0.0
                        mix_gamma_g = float(model.mix_gamma.grad.abs().item()) if model.mix_gamma.grad is not None else 0.0
                        print(f"[Diag][Grad] in_proj={grad_norm(model.in_proj):.3f} encoder={grad_norm(model.encoder):.3f} head={grad_norm(model.head):.3f} head_q={grad_norm(model.head_q):.3f} hist_pair_mlp={grad_norm(model.hist_pair_mlp):.3f} noise_proj={grad_norm(model.noise_proj):.3f} mixα={mix_alpha_g:.6f} mixβ={mix_beta_g:.6f} mixγ={mix_gamma_g:.6f}")
                    except Exception:
                        pass
                # If any gradient is non-finite, skip the step to avoid corrupting weights
                grads_ok = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_ok = False
                        break
                if not grads_ok:
                    print("[Warn] Non-finite gradients encountered; zeroing grads and skipping optimizer step.")
                    opt.zero_grad(set_to_none=True)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    t6 = _time.perf_counter(); _sync()
                    if scaler is not None:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    _sync(); t7 = _time.perf_counter()
                    opt_last = (t7 - t6); opt_t += opt_last
                total += float(loss.detach()); nb += 1
                # update tqdm postfix with running metrics
                if pbar is not None:
                    try:
                        # estimate total iteration time and unaccounted time
                        t_iter_end = _time.perf_counter()
                        iter_last = t_iter_end - t_loop_start
                        accounted = prep_last + fwd_last + sim_last + bwd_last + opt_last
                        other_last = max(0.0, iter_last - accounted)
                        # batch complexity indicators
                        mean_T = float(T_vec.float().mean().item()) if T_vec.numel() > 0 else 0.0
                        max_T = int(T_vec.max().item()) if T_vec.numel() > 0 else 0
                        # show both per-batch (suffix _b) and cumulative totals (suffix _t)
                        pbar.set_postfix({
                            "loss": f"{float(loss.detach()):.4f}",
                            "fwd_b": f"{fwd_last:.2f}s", "sim_b": f"{sim_last:.2f}s", "bwd_b": f"{bwd_last:.2f}s",
                            "opt_b": f"{opt_last:.2f}s", "prep_b": f"{prep_last:.2f}s", "data_b": f"{data_last:.2f}s",
                            "oth_b": f"{other_last:.2f}s", "T": f"{mean_T:.1f}/{max_T}"
                        })
                    except Exception:
                        pass
                # end-of-iteration timestamp for next data gap measurement
                prev_end = _time.perf_counter()
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass
            tr_loss = total / max(1, nb)
        else:
            tr_loss = float('nan')
        # Step scheduler for next epoch and measure validation time
        try:
            sched.step()
            cur_lr = opt.param_groups[0]['lr']
        except Exception:
            cur_lr = opt.param_groups[0]['lr']
        # Measure validation time (can be throttled by eval_every / eval_max_batches)
        if (eval_every <= 1) or (ep % eval_every == 0):
            t_val0 = _time.perf_counter()
            val_fid = eval_avg_fid()
            t_val1 = _time.perf_counter()
            val_time = t_val1 - t_val0
        else:
            val_fid = float('nan')
            val_time = 0.0
        # Basic epoch summary + timings
        print(f"[Subcircuits] epoch {ep} lr={cur_lr:.6e} train_loss={tr_loss:.6f} val_avg_fid={val_fid:.6f}")
        print(f"[Time][Epoch {ep}] fwd={fwd_t:.3f}s sim={sim_t:.3f}s bwd={bwd_t:.3f}s opt={opt_t:.3f}s prep={prep_t:.3f}s data_gap={data_t:.3f}s val={val_time:.3f}s")
        # fused status snapshot at epoch end (helps confirm whether fused path was used)
        try:
            fs = get_fused_status()
            print(f"[Fused][Epoch {ep}] enabled={fs['enabled']} available={fs['available']} used_calls={fs['used_calls']} reason={fs['reason']}")
        except Exception:
            pass
        # Print fresh examples at epoch end: prefer validation samples; fall back to train
        try:
            model.eval()
            with torch.no_grad():
                if VERBOSE and (val_loader is not None):
                    vb = next(iter(val_loader))
                    vb = vb.to(device)
                    tq = torch.zeros(vb.base_g.size(0), dtype=torch.long, device=device)
                    vcounts, vT = model._counts_from_batch(vb, tq)
                    ctx = {'batch': vb, 'gate_blocks': gate_blocks}
                    vlogits = model(vcounts, vT, ctx)
                    _print_angle_examples(vlogits, vT, tag=f"Epoch{ep}-Eval")
                    # Fidelity distribution on this val batch (up to 16 samples)
                    try:
                        Bv = vb.base_g.size(0)
                        take = min(16, Bv)
                        fids = []
                        for i in range(take):
                            b1 = type(vb)(
                                vb.base_g[i:i+1], vb.base_q1[i:i+1], vb.base_q2[i:i+1],
                                vb.param_g[i:i+1], vb.param_q[i:i+1], vb.param_after[i:i+1], vb.param_angles_gt[i:i+1],
                                vb.base_len[i:i+1], vb.param_len[i:i+1], vb.n_qubits[i:i+1], vb.idx[i:i+1]
                            )
                            li = vlogits[i:i+1]
                            l = simulate_loss(b1, li, init_cache, ref_cache, noise_schedules,
                                              mode='blocks', gate_blocks=gate_blocks, detach_base_noise=detach_base_noise)
                            fids.append(float(1.0 - l.detach().item()))
                        if fids:
                            import numpy as _np
                            arr = _np.array(fids)
                            print(f"[Diag][Val][Fid] mean={arr.mean():.4f} std={arr.std():.4f} p10={_np.percentile(arr,10):.4f} p90={_np.percentile(arr,90):.4f}")
                    except Exception:
                        pass
                    # Also print first few blocks of sample 0 for within-sample variation
                    try:
                        Bsz = int(vb.base_g.size(0))
                        maxT = int(vcounts.size(1))
                        if Bsz > 0 and maxT > 0:
                            S = min(4, int(vT[0].item()) if vT.numel() > 0 else 0)
                            blocks = []
                            for t in range(S):
                                v = vlogits[0, t*3:(t+1)*3, 0].tolist()
                                blocks.append(f"[{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}]")
                            print(f"[Subcircuits][Epoch{ep}-Eval][Sample0-First{S}Blocks] {' '.join(blocks)}")
                        # Compute cosine sims of pre/post embeddings among first three samples
                        import torch.nn.functional as F
                        if (not model.use_ordered_seq) and Bsz >= 3:
                            # compute step embeddings with history-aware context
                            maxS = int(vT.max().item() if vT.numel() > 0 else 0)
                            pre_steps, post_steps = model.get_step_embeddings(vcounts, vT, max_steps=maxS, extra_feats=None, hist_ctx=ctx)
                            def cos(a,b):
                                return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
                            # choose a step index and ensure we compare only samples that have that step
                            step_idx = 24
                            S = pre_steps.size(1)
                            if S > 0:
                                use_idx = step_idx if step_idx < S else (S - 1)
                                # pick first three indices with T >= use_idx+1
                                need = use_idx + 1
                                ok = (vT >= need).nonzero(as_tuple=False).squeeze(-1).tolist()
                                if len(ok) >= 3:
                                    i0, i1, i2 = ok[0], ok[1], ok[2]
                                    p1, p2, p3 = pre_steps[i0, use_idx, :], pre_steps[i1, use_idx, :], pre_steps[i2, use_idx, :]
                                    e1, e2, e3 = post_steps[i0, use_idx, :], post_steps[i1, use_idx, :], post_steps[i2, use_idx, :]
                                    tag = f"t{use_idx+1}"
                                    print(f"[PreEmbSim-Eval@{tag}] 1vs2={cos(p1,p2):.4f}, 1vs3={cos(p1,p3):.4f}, 2vs3={cos(p2,p3):.4f}")
                                    print(f"[EmbSim-Eval@{tag}]     1vs2={cos(e1,e2):.4f}, 1vs3={cos(e1,e3):.4f}, 2vs3={cos(e2,e3):.4f}")
                    except Exception:
                        pass
                elif VERBOSE and (loader is not None):
                    tb = next(iter(loader))
                    tb = tb.to(device)
                    tq = torch.zeros(tb.base_g.size(0), dtype=torch.long, device=device)
                    tcounts, tT = model._counts_from_batch(tb, tq)
                    ctx = {'batch': tb, 'gate_blocks': gate_blocks}
                    tlogits = model(tcounts, tT, ctx)
                    _print_angle_examples(tlogits, tT, tag=f"Epoch{ep}-Train")
                    try:
                        Bsz = int(tb.base_g.size(0))
                        maxT = int(tcounts.size(1))
                        if Bsz > 0 and maxT > 0:
                            S = min(4, int(tT[0].item()) if tT.numel() > 0 else 0)
                            blocks = []
                            for t in range(S):
                                v = tlogits[0, t*3:(t+1)*3, 0].tolist()
                                blocks.append(f"[{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}]")
                            print(f"[Subcircuits][Epoch{ep}-Train][Sample0-First{S}Blocks] {' '.join(blocks)}")
                        import torch.nn.functional as F
                        if Bsz >= 3:
                            maxS = int(tT.max().item() if tT.numel() > 0 else 0)
                            pre_steps, post_steps = model.get_step_embeddings(tcounts, tT, max_steps=maxS, extra_feats=None, hist_ctx=ctx)
                            def cos(a,b):
                                return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
                            step_idx = 24
                            S = pre_steps.size(1)
                            if S > 0:
                                use_idx = step_idx if step_idx < S else (S - 1)
                                need = use_idx + 1
                                ok = (tT >= need).nonzero(as_tuple=False).squeeze(-1).tolist()
                                if len(ok) >= 3:
                                    i0, i1, i2 = ok[0], ok[1], ok[2]
                                    p1, p2, p3 = pre_steps[i0, use_idx, :], pre_steps[i1, use_idx, :], pre_steps[i2, use_idx, :]
                                    e1, e2, e3 = post_steps[i0, use_idx, :], post_steps[i1, use_idx, :], post_steps[i2, use_idx, :]
                                    tag = f"t{use_idx+1}"
                                    print(f"[PreEmbSim-Train@{tag}] 1vs2={cos(p1,p2):.4f}, 1vs3={cos(p1,p3):.4f}, 2vs3={cos(p2,p3):.4f}")
                                    print(f"[EmbSim-Train@{tag}]    1vs2={cos(e1,e2):.4f}, 1vs3={cos(e1,e3):.4f}, 2vs3={cos(e2,e3):.4f}")
                    except Exception:
                        pass
        except Exception:
            pass
    # final report
    final_fid = eval_avg_fid()
    print(f"[Subcircuits] Final average fidelity on validation: {final_fid:.6f}")
    # Always save the trained model before optional post-eval and return
    try:
        _save_trained_model(model, gate_blocks, tag="angle_predictor_1022_start_subcircuits")
    except Exception as e:
        print(f"[Save] Failed to save model checkpoint: {e}")
    # Optional: post-training chained evaluation
    if post_eval_chain:
        try:
            model.eval()
            with torch.no_grad():
                # Determine base length N and block interval for 1q circuits
                if use_syn:
                    N = gate_blocks  # already equal to synthetic_base_len
                else:
                    # fallback: use current gate_blocks as the per-circuit base length
                    N = int(gate_blocks)
                C = int(max(1, post_eval_count))
                # 1) generate C random 1q circuits (no PQC angles; blocks mode will insert)
                items = generate_synthetic_single_qubit_items(num_samples=C, base_len=N, gate_blocks=N, gates_vocab=("h","x","z"), seed=0)
                # 2) predict angles per circuit using trained model
                from torch.utils.data import DataLoader as _DL
                dl = _DL(items, batch_size=C, shuffle=False,
                         collate_fn=lambda xs: collate(xs, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1))
                b: Batch = next(iter(dl))
                b = b.to(device)
                tq = torch.zeros(b.base_g.size(0), dtype=torch.long, device=device)
                counts, T_vec = model._counts_from_batch(b, tq)
                ctx = {'batch': b, 'gate_blocks': N}
                logits = model(counts, T_vec, ctx)  # [B, 3, 1]
                ang_each = logits[:, 0:3, 0]      # [C,3]
                # 3) concatenate the C circuits (base) into one long circuit, and build angles_blk with C blocks
                concat_gates: List[str] = []
                for it in items:
                    concat_gates.extend(it['base_gates'])
                one = dict(
                    idx=0,
                    n_qubits=1,
                    base_gates=concat_gates,
                    base_q1=[0] * len(concat_gates),
                    base_q2=[-1] * len(concat_gates),
                    param_gates=[],
                    param_qubits=[],
                    after=[],
                    param_angles_gt=[],
                )
                # Normalize idx to 0 to avoid any idx2row mismatches
                one['idx'] = 0
                ds_tmp = type('TmpDS', (), {'items':[one], '__len__': lambda self: 1, '__getitem__': lambda self, i: self.items[i]})()
                # Build caches with K=post_eval_k initial states
                init_cache2, ref_cache2, noise_sched2 = build_base_cache_vectorized(ds_tmp, k_random=post_eval_k, device=device, noise=noise)
                # Build batch for the concatenated circuit
                bcat = collate([one], max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1).to(device)
                # Force idx match with cache rows
                try:
                    bcat.idx = torch.zeros_like(bcat.idx)
                except Exception:
                    pass
                # Angles per block for chained evaluation: [1, C, 1, 3]
                angles_blk = ang_each.unsqueeze(0).unsqueeze(2)  # [1,C,1,3]
                loss_chain = simulate_blocks_with_angles(bcat, angles_blk, init_cache2, ref_cache2, noise_sched2, gate_blocks=N, device=device, detach_base_noise=True)
                fid_chain = float(1.0 - loss_chain.detach().item())
                # Also compute theoretical estimates from individual circuit fidelities
                f_list: List[float] = []
                for i in range(C):
                    one_i = dict(items[i])
                    # Normalize idx to 0 for single-sample cache
                    one_i['idx'] = 0
                    ds1 = type('TmpDS1', (), {'items':[one_i], '__len__': lambda self: 1, '__getitem__': lambda self, j: self.items[j]})()
                    init1, ref1, noise1 = build_base_cache_vectorized(ds1, k_random=post_eval_k, device=device, noise=noise)
                    b1 = collate([one_i], max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1).to(device)
                    try:
                        b1.idx = torch.zeros_like(b1.idx)
                    except Exception:
                        pass
                    ang1 = ang_each[i].view(1,1,1,3)
                    loss1 = simulate_blocks_with_angles(b1, ang1, init1, ref1, noise1, gate_blocks=N, device=device, detach_base_noise=True)
                    f1 = float(1.0 - loss1.detach().item())
                    f_list.append(f1)
                import math as _math
                # Estimates
                f_mean = sum(f_list)/len(f_list)
                f_prod = float(_math.prod(f_list)) if hasattr(_math, 'prod') else float(torch.tensor(f_list).prod().item())
                sum_inf = sum(1.0 - f for f in f_list)
                exp_est = float(_math.exp(-sum_inf))
                dep_est_factor = 1.0
                for f in f_list:
                    dep_est_factor *= max(0.0, 2.0*f - 1.0)
                dep_est = 0.5 + 0.5 * dep_est_factor
                print(f"[PostEval][Chain] N={N} count={C} K={post_eval_k} => avg_fidelity={fid_chain:.6f}")
                print(f"[PostEval][Est] mean_f={f_mean:.6f} prod_f={f_prod:.6f} exp(-sum(1-f))={exp_est:.6f} dep_comp_est={dep_est:.6f}")
        except Exception as e:
            print(f"[PostEval] Skipped due to error: {e}")
    return model, final_fid


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train angle predictor (simple) with optional synthetic 1q5 dataset.")
    p.add_argument("--data-path", type=str, default="", help="Path to JSON dataset folder/file (ignored in synthetic mode)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--k-random", type=int, default=32)
    p.add_argument("--gate-blocks", type=int, default=5, help="Blocks size (forced to 5 when --synthetic-1q5)")
    p.add_argument("--detach-base-noise", action="store_true", default=True)
    p.add_argument("--no-detach-base-noise", dest="detach_base_noise", action="store_false")
    p.add_argument("--num-sample", type=int, default=None)
    p.add_argument("--sub-val-count", type=int, default=None)
    p.add_argument("--calibrate-centers", type=int, default=0, help="Calibrate centers using N batches before training")
    p.add_argument("--synthetic-1q5", action="store_true", help="Use synthetic dataset: 1 qubit, N base gates (default 5), one PQC block at end")
    p.add_argument("--synthetic-num-samples", type=int, default=None, help="Number of synthetic samples to generate when --synthetic-1q5")
    p.add_argument("--synthetic-base-len", type=int, default=None, help="Base sequence length N for synthetic mode (default 5)")
    p.add_argument("--synthetic-train-count", type=int, default=None, help="Synthetic mode: number of samples to use for training (remainder used for eval)")
    p.add_argument("--synthetic-train-frac", type=float, default=None, help="Synthetic mode: fraction (0-1] of samples to use for training (remainder used for eval)")
    p.add_argument("--synthetic-split-seed", type=int, default=0, help="Seed for deterministic synthetic train/eval split")
    p.add_argument("--synthetic-enum-all", action="store_true", help="Enumerate ALL H/X/Z sequences of length N (ignores --synthetic-num-samples). ENV: PQC_SYNTH_ENUM_ALL=1")
    # Post-training chained evaluation: generate C circuits, predict PQC, chain and simulate with K init states
    p.add_argument("--post-eval-chain", action="store_true", help="After training, run chained evaluation over random 1q circuits and report average fidelity")
    p.add_argument("--post-eval-count", type=int, default=50, help="Number of circuits to chain in post-training evaluation")
    p.add_argument("--post-eval-k", type=int, default=100, help="Number of initial states K for chained evaluation")
    return p


if __name__ == "__main__":
    # Run via: python -m pqcqec.opt_transformer_simple --synthetic-1q5 --epochs 100 --batch-size 64
    args = _build_argparser().parse_args()
    # Delegate to train_subcircuits; device picked automatically
    train_subcircuits(
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        k_random=args.k_random,
        num_sample=args.num_sample,
        noise=None,
        device=None,
        gate_blocks=args.gate_blocks,
        detach_base_noise=args.detach_base_noise,
        sub_val_count=args.sub_val_count,
        calibrate_centers=args.calibrate_centers,
        use_quaternion_head=False,
        mix_l2=0.0,
        mix_warmup_epochs=0,
        mix_warmup_alpha=0.7,
        mix_warmup_beta=0.3,
        mix_warmup_gamma=0.3,
        hist_dropout=0.0,
        hist_scale_min=1.0,
        hist_scale_max=1.0,
        noise_boost=1.0,
        noise_boost_epochs=0,
        history_freeze_prob=0.0,
        prev_noise_std=0.0,
        history_freeze_epochs=None,
        aux_angle_weight=0.0,
        aux_angle_blocks=0,
        use_synthetic_1q5=args.synthetic_1q5,
        synthetic_num_samples=args.synthetic_num_samples,
        synthetic_base_len=args.synthetic_base_len,
        synthetic_train_count=args.synthetic_train_count,
        synthetic_train_frac=args.synthetic_train_frac,
        synthetic_split_seed=args.synthetic_split_seed,
        synthetic_enumerate_all=args.synthetic_enum_all,
        post_eval_chain=args.post_eval_chain,
        post_eval_count=args.post_eval_count,
        post_eval_k=args.post_eval_k,
    )
