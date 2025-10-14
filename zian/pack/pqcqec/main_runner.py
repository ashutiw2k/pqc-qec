#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point to run either direct-angle optimization or transformer training.
This file does NOT modify newest_work_correct.py; it reuses extracted modules.

Usage (programmatic):
- from pqcqec.main_runner import run_main
- run_main(mode='direct'|'transformer', data_path='...', ...)

Usage (CLI):
python -m pqcqec.main_runner --mode direct --data <path> [--epochs 100] [...]
"""
from __future__ import annotations

import argparse
import math
from typing import Optional

import torch
from .precision import configure_precision

from .simulator_core import NoiseConfig, DEVICE
from .opt_direct_angles import train_direct
from .opt_transformer import train_transformer, train_subcircuits, AnglePredictor, MAX_BASE_LEN, MAX_PARAM
from .simulator_core import collate, BASE_GATES


def run_main(mode: str,
             data_path: str,
             epochs: int = 10,
             batch_size: int = 32,
             lr: float = 5e-4,
             k_random: int = 32,
             num_sample: Optional[int] = None,
             n_sample: Optional[int] = None,
             gate_blocks: int = 50,
             use_noise: bool = True,
            #  noise_x_rad: float = math.pi/10,
            #  noise_z_rad: float = math.pi/10,
             noise_x_rad: float = 0.0,
             noise_z_rad: float = 0.0,
             noise_delta_x: float = 0.0,
             noise_delta_z: float = 0.0,
             sub_val_count: Optional[int] = None,
             device: Optional[torch.device] = None,
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
             history_freeze_epochs: int = 0,
             aux_angle_weight: float = 0.0,
             aux_angle_blocks: int = 0):
    if device is None:
        device = DEVICE
    # One-time precision/Kernel setup (TF32, Flash SDP); AMP is handled inside training loops
    try:
        configure_precision(verbose=True)
    except Exception:
        pass
    noise = NoiseConfig(use_noise=use_noise,
                        noise_x_rad=noise_x_rad,
                        noise_z_rad=noise_z_rad,
                        noise_delta_x=noise_delta_x,
                        noise_delta_z=noise_delta_z)
    if mode.lower() in ("direct", "angles", "blocks"):
        return train_direct(data_path=data_path,
                            gate_blocks=gate_blocks,
                            batch_size=batch_size,
                            epochs=epochs,
                            lr=lr,
                            k_random=k_random,
                            num_sample=num_sample,
                            n_sample=n_sample,
                            noise=noise,
                            device=device)
    elif mode.lower() in ("test",):
        # Generate 5 random 1-qubit circuits with exactly 5 base 1q gates (h/x/z),
        # compute their embeddings (head-input at the last available step), and print pairwise cosine.
        import random
        gates_vocab = ['h','x','z']
        items = []
        for i in range(5):
            gs = [random.choice(gates_vocab) for _ in range(5)]
            items.append(dict(
                idx=i,
                base_gates=gs,
                base_q1=[0]*5,
                base_q2=[-1]*5,
                param_gates=[],
                param_qubits=[],
                after=[],
                param_angles_gt=[],
                n_qubits=1,
            ))
        b = collate(items, max_base_len=MAX_BASE_LEN, max_param=MAX_PARAM, max_qubits=1)
        b = b.to(device)
        model = AnglePredictor(gate_blocks=gate_blocks).to(device)
        # counts and T per line (target qubit = 0)
        tq = torch.zeros(b.base_g.size(0), dtype=torch.long, device=device)
        counts, T_vec = model._counts_from_batch(b, tq)
        Bsz = b.base_g.size(0)
        maxT = counts.size(1)
        # per-block [h,x,z] histogram
        hist3 = torch.zeros(Bsz, maxT, 3, device=device)
        for i in range(Bsz):
            Lb = int(b.base_len[i].item())
            T = int(T_vec[i].item())
            for t in range(T):
                s = t * gate_blocks; e = min(Lb, (t + 1) * gate_blocks)
                if s >= e:
                    continue
                seg = b.base_g[i, s:e]
                # base_g codes: 0=h,1=x,2=z
                for gcode in (0,1,2):
                    hist3[i, t, gcode] = (seg == gcode).sum().float()
        # build base_sum using non-decomposable token+position pairing
        base_sum = torch.zeros(Bsz, model.head.in_features, device=device)  # will overwrite per-sample
        base_sum = torch.zeros(Bsz, model.pos_emb.embedding_dim, device=device)
        base_len_vec = b.base_len.to(device)
        for i in range(Bsz):
            Lb_i = int(b.base_len[i].item())
            if Lb_i <= 0:
                continue
            acc = torch.zeros(model.pos_emb.embedding_dim, device=device)
            for p in range(Lb_i):
                g = int(b.base_g[i, p].item())
                if g < 0:
                    continue
                tok_id = 1 if g == 0 else (2 if g == 1 else (3 if g == 2 else 0))
                if tok_id == 0:
                    continue
                pos_idx = min(p, MAX_BASE_LEN + MAX_PARAM - 1)
                t_emb = model.hist_token_emb(torch.tensor([tok_id], device=device)).squeeze(0)
                p_emb = model.hist_pos_emb(torch.tensor([pos_idx], device=device)).squeeze(0)
                acc = acc + model.hist_pair_mlp(torch.cat([t_emb, p_emb], dim=-1))
            base_sum[i] = acc
        ctx = dict(extra_feats=hist3, hist_base_sum=base_sum, base_len=base_len_vec)
        # step embeddings (history-aware); use last available step
        maxS = int(T_vec.max().item() if T_vec.numel() > 0 else 0)
        pre_steps, post_steps = model.get_step_embeddings(counts, T_vec, max_steps=maxS, extra_feats=hist3, hist_ctx=ctx)
        if pre_steps.size(1) == 0:
            print("[Test] No valid steps; base_len likely zero.")
            return None
        use_idx = pre_steps.size(1) - 1
        emb = post_steps[:, use_idx, :]  # [5, HID]
        # print circuits and embeddings summary
        def vec_str(v: torch.Tensor, k: int = 8):
            vlist = v.tolist()
            head = ", ".join(f"{x:.4f}" for x in vlist[:k])
            return f"[{head}{', ...' if len(vlist)>k else ''}]"
        print("[Test] Random 1-qubit circuits (5 gates each):")
        for i, it in enumerate(items):
            print(f"  #{i}: {' '.join(it['base_gates'])}")
        print(f"[Test] Embedding dim={emb.size(1)}; last-step head-input vectors:")
        for i in range(emb.size(0)):
            nv = float(torch.norm(emb[i]).item())
            print(f"  emb[{i}] norm={nv:.4f} vec={vec_str(emb[i])}")
        # pairwise cosine similarities
        import torch.nn.functional as F
        print("[Test] Pairwise cosine (post@last-step):")
        for i in range(emb.size(0)):
            for j in range(i+1, emb.size(0)):
                c = float(F.cosine_similarity(emb[i].unsqueeze(0), emb[j].unsqueeze(0)).item())
                print(f"  {i} vs {j}: {c:.4f}")
        # centered post embeddings to mitigate common-mode offsets
        emb_c = emb - emb.mean(dim=0, keepdim=True)
        print("[Test] Pairwise cosine (post@last-step, centered):")
        for i in range(emb_c.size(0)):
            for j in range(i+1, emb_c.size(0)):
                c = float(F.cosine_similarity(emb_c[i].unsqueeze(0), emb_c[j].unsqueeze(0)).item())
                print(f"  {i} vs {j}: {c:.4f}")
        # pre-encoder last-step vectors
        pre_last = pre_steps[:, use_idx, :]
        print("[Test] Pairwise cosine (pre@last-step):")
        for i in range(pre_last.size(0)):
            for j in range(i+1, pre_last.size(0)):
                c = float(F.cosine_similarity(pre_last[i].unsqueeze(0), pre_last[j].unsqueeze(0)).item())
                print(f"  {i} vs {j}: {c:.4f}")
        # base_sum only
        print("[Test] Pairwise cosine (base_sum only):")
        for i in range(base_sum.size(0)):
            for j in range(i+1, base_sum.size(0)):
                c = float(F.cosine_similarity(base_sum[i].unsqueeze(0), base_sum[j].unsqueeze(0)).item())
                print(f"  {i} vs {j}: {c:.4f}")
        return None
    elif mode.lower() in ("transformer", "tfm"):
        return train_transformer(data_path=data_path,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 lr=lr,
                                 k_random=k_random,
                                 num_sample=(n_sample if n_sample is not None else num_sample),
                                 noise=noise,
                                 device=device,
                                 gate_blocks=gate_blocks,
                                 detach_base_noise=True,
                                 calibrate_centers=calibrate_centers)
    elif mode.lower() in ("subcircuits", "sub"):
        return train_subcircuits(data_path=data_path,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 lr=lr,
                                 k_random=k_random,
                                 num_sample=(n_sample if n_sample is not None else num_sample),
                                 noise=noise,
                                 device=device,
                                 gate_blocks=gate_blocks,
                                 detach_base_noise=True,
                                  sub_val_count=sub_val_count,
                                  calibrate_centers=calibrate_centers,
                                  use_quaternion_head=use_quaternion_head,
                                  mix_l2=mix_l2,
                                  mix_warmup_epochs=mix_warmup_epochs,
                                  mix_warmup_alpha=mix_warmup_alpha,
                                  mix_warmup_beta=mix_warmup_beta,
                                  mix_warmup_gamma=mix_warmup_gamma,
                                  hist_dropout=hist_dropout,
                                  hist_scale_min=hist_scale_min,
                                  hist_scale_max=hist_scale_max,
                                  noise_boost=noise_boost,
                                  noise_boost_epochs=noise_boost_epochs,
                                  history_freeze_prob=history_freeze_prob,
                                  prev_noise_std=prev_noise_std,
                                  history_freeze_epochs=history_freeze_epochs,
                                  aux_angle_weight=aux_angle_weight,
                                  aux_angle_blocks=aux_angle_blocks)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='transformer', choices=['transformer', 'direct', 'subcircuits', 'test'])
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--k-random', type=int, default=32)
    p.add_argument('--num-sample', type=int, default=None)
    p.add_argument('--n-sample', type=int, default=None, help='Limit number of samples to load (overrides --num-sample)')
    p.add_argument('--gate-blocks', type=int, default=50)
    p.add_argument('--use-noise', action='store_true')
    p.add_argument('--noise-x-rad', type=float, default=math.pi/10)
    p.add_argument('--noise-z-rad', type=float, default=math.pi/10)
    p.add_argument('--noise-delta-x', type=float, default=0.0)
    p.add_argument('--noise-delta-z', type=float, default=0.0)
    p.add_argument('--sub-val-count', type=int, default=None, help='Number of subcircuits to use for validation (if provided)')
    p.add_argument('--calibrate-centers', type=int, default=0, help='If >0, run N mini-batches of center calibration before training')
    p.add_argument('--use-quaternion-head', action='store_true', help='Predict quaternion then convert to angles (ZXZ)')
    # training knobs
    p.add_argument('--mix-l2', type=float, default=0.0, help='L2 regularization for mix weights (alpha,beta)')
    p.add_argument('--mix-warmup-epochs', type=int, default=0, help='Use fixed mix weights during first N epochs')
    p.add_argument('--mix-warmup-alpha', type=float, default=0.7, help='Warmup enc weight')
    p.add_argument('--mix-warmup-beta', type=float, default=0.3, help='Warmup hist weight')
    p.add_argument('--mix-warmup-gamma', type=float, default=0.3, help='Warmup noise weight for head mixing')
    p.add_argument('--hist-dropout', type=float, default=0.0, help='Dropout prob on hist_sum before head')
    p.add_argument('--hist-scale-min', type=float, default=1.0, help='Min random scale for hist_sum (train)')
    p.add_argument('--hist-scale-max', type=float, default=1.0, help='Max random scale for hist_sum (train)')
    p.add_argument('--noise-boost', type=float, default=1.0, help='Scale noise_proj output during first N epochs')
    p.add_argument('--noise-boost-epochs', type=int, default=0, help='Number of epochs to apply noise boost')
    p.add_argument('--history-freeze-prob', type=float, default=0.0, help='Prob to freeze AR history update (teacher forcing-like) early on')
    p.add_argument('--prev-noise-std', type=float, default=0.0, help='Std of Gaussian noise added to prev angles fed to history (early epochs)')
    p.add_argument('--history-freeze-epochs', type=int, default=0, help='Apply history freeze/noise for first N epochs (default: warmup epochs)')
    p.add_argument('--aux-angle-weight', type=float, default=0.0, help='Auxiliary angle L2 weight on first K blocks (0=off)')
    p.add_argument('--aux-angle-blocks', type=int, default=0, help='Number of early blocks for auxiliary angle loss')
    return p


def main_cli():
    args = _build_argparser().parse_args()
    run_main(mode=args.mode,
             data_path=args.data,
             epochs=args.epochs,
             batch_size=args.batch_size,
             lr=args.lr,
             k_random=args.k_random,
             num_sample=args.num_sample,
             n_sample=args.n_sample,
             gate_blocks=args.gate_blocks,
             use_noise=bool(args.use_noise),
             noise_x_rad=float(args.noise_x_rad),
             noise_z_rad=float(args.noise_z_rad),
             noise_delta_x=float(args.noise_delta_x),
             noise_delta_z=float(args.noise_delta_z),
             sub_val_count=(int(args.sub_val_count) if args.sub_val_count is not None else None),
             calibrate_centers=int(args.calibrate_centers) if hasattr(args, 'calibrate_centers') else 0,
             use_quaternion_head=bool(args.use_quaternion_head),
             mix_l2=float(args.mix_l2),
             mix_warmup_epochs=int(args.mix_warmup_epochs),
             mix_warmup_alpha=float(args.mix_warmup_alpha),
             mix_warmup_beta=float(args.mix_warmup_beta),
             mix_warmup_gamma=float(args.mix_warmup_gamma),
             hist_dropout=float(args.hist_dropout),
             hist_scale_min=float(args.hist_scale_min),
             hist_scale_max=float(args.hist_scale_max),
             noise_boost=float(args.noise_boost),
             noise_boost_epochs=int(args.noise_boost_epochs),
             history_freeze_prob=float(args.history_freeze_prob),
             prev_noise_std=float(args.prev_noise_std),
             history_freeze_epochs=int(args.history_freeze_epochs),
             aux_angle_weight=float(args.aux_angle_weight),
             aux_angle_blocks=int(args.aux_angle_blocks))


if __name__ == '__main__':
    main_cli()
