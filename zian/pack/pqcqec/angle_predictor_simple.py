from typing import List, Tuple, Optional, Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Package imports for constants and types
from .dataset import MAX_BASE_LEN, MAX_PARAM, MAX_QUBITS
from .simulator_core import Batch

# Model hyperparameters (kept here to keep model self-contained)
HID_DIM = 768
N_LAYERS = 8
N_HEADS = 12
FF_DIM = HID_DIM * 4
DROP = 0.1
PREV_K = 1  # sliding window length for previous angles
MAX_SEQ = MAX_BASE_LEN + MAX_PARAM  # cap for positional embeddings over base+param


class AnglePredictor(nn.Module):
    """Autoregressive per-qubit predictor.

    Default path: ordered-seq encoder over tokens = [CLS] + base-gate tokens + prior PQC tokens
    with their angles embedded via sin/cos MLP. Fallback path: per-block counts + encoder +
    base-history residual mixing.
    """
    def __init__(self, gate_blocks: int, use_quaternion_head: bool = False):
        super().__init__()
        self.gate_blocks = int(gate_blocks)
        import math as _math
        self.max_blocks = _math.ceil(MAX_BASE_LEN / max(1, self.gate_blocks))
        # Per-step features (legacy path): [count_t, cum_t, t_index, prev_angles_window_flat(K*3)]
        feat_dim = 3 + 3 * PREV_K
        self.in_proj = nn.Sequential(
            nn.Linear(feat_dim, HID_DIM), nn.GELU(), nn.Dropout(DROP)
        )
        self.in_proj_ln = nn.LayerNorm(HID_DIM)
        # Optional per-block extras
        self.extra_proj = nn.Sequential(
            nn.Linear(3, HID_DIM), nn.GELU(), nn.Dropout(DROP)
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(2, HID_DIM), nn.GELU(), nn.Dropout(DROP)
        )
        self.phys_proj = nn.Sequential(
            nn.Linear(4, HID_DIM), nn.GELU(), nn.Dropout(DROP)
        )
        self.pos_emb = nn.Embedding(self.max_blocks, HID_DIM)
        enc_layer = nn.TransformerEncoderLayer(HID_DIM, N_HEADS, FF_DIM, DROP, batch_first=True, norm_first=True)
        try:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        # Heads
        self.head = nn.Linear(HID_DIM, 6)
        self.head_q = nn.Linear(HID_DIM, 4)
        self.use_quaternion_head = bool(use_quaternion_head)
        with torch.no_grad():
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            if self.head.bias.numel() >= 6:
                b = self.head.bias.view(3, 2)
                b[:, 0] = 1.0  # x=1, y=0 -> theta=0
                b[:, 1] = 0.0
            nn.init.zeros_(self.head_q.weight)
            nn.init.zeros_(self.head_q.bias)
            if self.head_q.bias.numel() >= 4:
                self.head_q.bias.data[0] = 1.0
        # History encoding and centers
        self.hist_vocab_size = 7
        self.hist_token_emb = nn.Embedding(self.hist_vocab_size, HID_DIM)
        self.hist_pos_emb = nn.Embedding(MAX_SEQ, HID_DIM)
        self.hist_value_proj = nn.Sequential(nn.Linear(1, HID_DIM), nn.GELU(), nn.Dropout(DROP))
        self.hist_pair_mlp = nn.Sequential(
            nn.Linear(2 * HID_DIM, HID_DIM), nn.GELU(), nn.Dropout(DROP)
        )
        self.register_buffer('enc_center', torch.zeros(HID_DIM))
        self.register_buffer('hist_center', torch.zeros(HID_DIM))
        self.use_centers: bool = False
        self.hist_pair_ln = nn.LayerNorm(HID_DIM)
        self.head_in_ln = nn.LayerNorm(HID_DIM)
        self.mix_alpha = nn.Parameter(torch.tensor(1.0))
        self.mix_beta = nn.Parameter(torch.tensor(1.0))
        self.mix_gamma = nn.Parameter(torch.tensor(1.0))
        self.noise_head_ln = nn.LayerNorm(HID_DIM)
        # Train-time knobs
        self.hist_drop_p = 0.0
        self.hist_scale_min = 1.0
        self.hist_scale_max = 1.0
        self.noise_boost = 1.0
        self.noise_boost_epochs = 0
        self._cur_epoch = 0
        self._mix_override: Optional[Tuple[float, float]] = None
        self._mix_override3: Optional[Tuple[float, float, float]] = None
        # AR stabilization
        self.p_history_freeze = 0.0
        self.prev_noise_std = 0.0
        self.history_freeze_epochs = 0
        # Ordered-seq switch (default on)
        self.use_ordered_seq = str(os.environ.get('PQC_ORDERED_SEQ', '1')).strip().lower() in ('1','true','yes','y','on')
        # Tokens
        self.TOK_PAD = 0
        self.TOK_CLS = 1
        self.TOK_H   = 2
        self.TOK_X   = 3
        self.TOK_Z   = 4
        self.TOK_UNK = 5
        self.TOK_RZ1 = 10
        self.TOK_RX  = 11
        self.TOK_RZ2 = 12
        vocab = 16
        self.seq_token_emb = nn.Embedding(vocab, HID_DIM)
        self.seq_pos_emb = nn.Embedding(MAX_SEQ, HID_DIM)
        self.ang_mlp = nn.Sequential(
            nn.Linear(2, HID_DIM), nn.GELU(), nn.Dropout(DROP), nn.Linear(HID_DIM, HID_DIM)
        )
        enc_layer2 = nn.TransformerEncoderLayer(HID_DIM, max(4, N_HEADS//2), FF_DIM, DROP, batch_first=True, norm_first=True)
        try:
            self.seq_encoder = nn.TransformerEncoder(enc_layer2, num_layers=max(2, N_LAYERS//2), enable_nested_tensor=False)
        except TypeError:
            self.seq_encoder = nn.TransformerEncoder(enc_layer2, num_layers=max(2, N_LAYERS//2))

    def set_epoch(self, ep: int):
        self._cur_epoch = int(ep)

    def set_mix_override(self, alpha: float, beta: float):
        self._mix_override = (float(alpha), float(beta))

    def clear_mix_override(self):
        self._mix_override = None
        self._mix_override3 = None

    def set_mix_override3(self, alpha: float, beta: float, gamma: float):
        self._mix_override3 = (float(alpha), float(beta), float(gamma))

    @staticmethod
    def _angles_from_head_logits(vec6: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        B = vec6.size(0)
        v = vec6.view(B, 3, 2)
        x = v[:, :, 0]
        y = v[:, :, 1]
        r = torch.sqrt(x * x + y * y + eps * eps)
        ux = x / r
        uy = y / r
        theta = torch.atan2(uy, ux)
        return theta, r

    @staticmethod
    def _angles_from_quaternion(q4: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        B = q4.size(0)
        w, x, y, z = q4[:, 0], q4[:, 1], q4[:, 2], q4[:, 3]
        norm = torch.sqrt(w*w + x*x + y*y + z*z + eps)
        w = w / norm; x = x / norm; y = y / norm; z = z / norm
        R11 = 1 - 2*(y*y + z*z)
        R12 = 2*(x*y - w*z)
        R13 = 2*(x*z + w*y)
        R21 = 2*(x*y + w*z)
        R22 = 1 - 2*(x*x + z*z)
        R23 = 2*(y*z - w*x)
        R31 = 2*(x*z - w*y)
        R32 = 2*(y*z + w*x)
        R33 = 1 - 2*(x*x + y*y)
        sinb = torch.sqrt(torch.clamp(R13 * R13 + R23 * R23, min=eps))
        beta = torch.atan2(sinb, torch.clamp(R33, -1.0, 1.0))
        alpha = torch.atan2(R13, -R23)
        gamma = torch.atan2(R31, R32)
        small = (sinb.abs() < 1e-3)
        if small.any():
            alpha_alt = torch.atan2(R21, R11)
            alpha = torch.where(small, alpha_alt, alpha)
            gamma = torch.where(small, torch.zeros_like(gamma), gamma)
        def wrap(a):
            return (a + torch.pi) % (2*torch.pi) - torch.pi
        alpha = wrap(alpha); beta = wrap(beta); gamma = wrap(gamma)
        return torch.stack([alpha, beta, gamma], dim=1)

    def _counts_from_batch(self, b, target_qubit: torch.Tensor):
        B = b.base_g.size(0)
        device = b.base_g.device
        counts_list = []; T_list = []
        for i in range(B):
            Lb = int(b.base_len[i].item())
            T = (Lb + self.gate_blocks - 1) // max(1, self.gate_blocks)
            q = int(target_qubit[i].item())
            base_q1 = b.base_q1[i, :Lb]
            base_q2 = b.base_q2[i, :Lb]
            touch = (base_q1 == q) | (base_q2 == q)
            per_block = []
            for t in range(T):
                s = t * self.gate_blocks; e = min(Lb, (t + 1) * self.gate_blocks)
                per_block.append(int(touch[s:e].sum().item()))
            counts_list.append(torch.tensor(per_block, dtype=torch.float32, device=device))
            T_list.append(T)
        maxT = max(T_list) if T_list else 0
        if maxT == 0:
            return torch.zeros(B, 0, device=device), torch.tensor(T_list, dtype=torch.long, device=device)
        counts = torch.zeros(B, maxT, device=device)
        for i, c in enumerate(counts_list):
            counts[i, :c.numel()] = c
        return counts, torch.tensor(T_list, dtype=torch.long, device=device)

    def get_line_embeddings(self, counts: torch.Tensor, T_vec: torch.Tensor) -> torch.Tensor:
        device = counts.device
        B, maxT = counts.shape
        if maxT == 0:
            return torch.zeros(B, HID_DIM, device=device)
        cum = counts.cumsum(dim=1)
        idx_seq = torch.arange(maxT, device=device).unsqueeze(0).expand(B, -1).float()
        prev_zero = torch.zeros(B, maxT, 3 * PREV_K, device=device)
        feats = torch.cat([
            counts.unsqueeze(-1),
            cum.unsqueeze(-1),
            idx_seq.unsqueeze(-1),
            prev_zero,
        ], dim=-1)
        x = self.in_proj(feats)
        x = self.in_proj_ln(x)
        pos = self.pos_emb(torch.arange(maxT, device=device)).unsqueeze(0)
        x = x + pos
        attn_mask = torch.triu(torch.ones((maxT, maxT), dtype=torch.bool, device=device), diagonal=1)
        key_pad = (torch.arange(maxT, device=device).unsqueeze(0) >= T_vec.unsqueeze(1))
        h = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_pad)
        idx_last = (T_vec - 1).clamp_min(0).to(device)
        idx_last = idx_last.view(B, 1, 1).expand(B, 1, HID_DIM)
        emb = h.gather(dim=1, index=idx_last).squeeze(1)
        return emb

    def get_line_pre_embeddings(self, counts: torch.Tensor, T_vec: torch.Tensor) -> torch.Tensor:
        device = counts.device
        B, maxT = counts.shape
        if maxT == 0:
            return torch.zeros(B, HID_DIM, device=device)
        cum = counts.cumsum(dim=1)
        idx_seq = torch.arange(maxT, device=device).unsqueeze(0).expand(B, -1).float()
        prev_zero = torch.zeros(B, maxT, 3 * PREV_K, device=device)
        feats = torch.cat([
            counts.unsqueeze(-1),
            cum.unsqueeze(-1),
            idx_seq.unsqueeze(-1),
            prev_zero,
        ], dim=-1)
        x0 = self.in_proj(feats)
        idx_last = (T_vec - 1).clamp_min(0).to(device)
        idx_last = idx_last.view(B, 1, 1).expand(B, 1, HID_DIM)
        emb = x0.gather(dim=1, index=idx_last).squeeze(1)
        return emb

    def get_step_embeddings(self,
                            counts: torch.Tensor,
                            T_vec: torch.Tensor,
                            max_steps: Optional[int] = None,
                            extra_feats: Optional[torch.Tensor] = None,
                            hist_ctx: Optional[Dict] = None):
        device = counts.device
        B, maxT = counts.shape
        if maxT == 0:
            z = torch.zeros(B, 0, HID_DIM, device=device)
            return z, z
        S = min(maxT, max_steps) if max_steps is not None else maxT
        cum = counts.cumsum(dim=1)
        idx_seq = torch.arange(maxT, device=device).unsqueeze(0).expand(B, -1).float()
        prev_buf = torch.zeros(B, PREV_K, 3, device=device)
        prev_seq = torch.zeros(B, maxT, 3 * PREV_K, device=device)
        pre_steps = torch.zeros(B, S, HID_DIM, device=device)
        post_steps = torch.zeros(B, S, HID_DIM, device=device)
        base_sum = torch.zeros(B, HID_DIM, device=device)
        base_len_vec = torch.zeros(B, dtype=torch.long, device=device)
        if isinstance(hist_ctx, dict):
            base_sum = hist_ctx.get('hist_base_sum', base_sum)
            base_len_vec = hist_ctx.get('base_len', base_len_vec)
        pqc_sum = torch.zeros(B, HID_DIM, device=device)
        for t in range(S):
            L = t + 1
            prev_seq[:, t, :] = prev_buf.reshape(B, 3 * PREV_K)
            feats_base = torch.cat([
                counts[:, :L].unsqueeze(-1),
                cum[:, :L].unsqueeze(-1),
                idx_seq[:, :L].unsqueeze(-1),
                prev_seq[:, :L, :],
            ], dim=-1)
            x = self.in_proj(feats_base)
            x = self.in_proj_ln(x)
            if extra_feats is not None:
                x = x + self.extra_proj(extra_feats[:, :L, :])
            if isinstance(hist_ctx, dict) and hist_ctx.get('noise_feats', None) is not None:
                noise_feats = hist_ctx['noise_feats']
                nproj = self.noise_proj(noise_feats[:, :L, :])
                if self.training and (self._cur_epoch < self.noise_boost_epochs) and (self.noise_boost != 1.0):
                    nproj = nproj * self.noise_boost
                x = x + nproj
            if isinstance(hist_ctx, dict) and hist_ctx.get('context_quat', None) is not None:
                cq = hist_ctx['context_quat']
                x = x + self.phys_proj(cq[:, :L, :])
            pre_vec = x[:, -1, :]
            pos = self.pos_emb(torch.arange(L, device=device)).unsqueeze(0)
            x = x + pos
            attn_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=1)
            key_pad = (torch.arange(L, device=device).unsqueeze(0) >= T_vec.unsqueeze(1))
            h = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_pad)
            base_prefix = None
            if isinstance(hist_ctx, dict):
                base_prefix = hist_ctx.get('base_prefix', None)
            if base_prefix is not None:
                raw_hist = base_prefix[:, L-1, :] + pqc_sum
            else:
                raw_hist = base_sum + pqc_sum
            raw_enc = h[:, -1, :]
            if self.use_centers:
                raw_hist = raw_hist - self.hist_center.view(1, -1)
                raw_enc = raw_enc - self.enc_center.view(1, -1)
            hist_sum = self.hist_pair_ln(raw_hist)
            if self.training:
                Bcur = hist_sum.size(0)
                if self.hist_drop_p > 0.0:
                    keep = (torch.rand(Bcur, device=device) > self.hist_drop_p).float().unsqueeze(1)
                else:
                    keep = 1.0
                if self.hist_scale_max > self.hist_scale_min:
                    scale = torch.empty(Bcur, device=device).uniform_(self.hist_scale_min, self.hist_scale_max).unsqueeze(1)
                else:
                    scale = 1.0
                hist_sum = hist_sum * keep * scale
            enc_last = self.head_in_ln(raw_enc)
            n_last = torch.zeros_like(enc_last)
            if isinstance(hist_ctx, dict) and hist_ctx.get('noise_feats', None) is not None:
                noise_feats = hist_ctx['noise_feats']
                nproj_full = self.noise_proj(noise_feats[:, :L, :])
                if self.training and (self._cur_epoch < self.noise_boost_epochs) and (self.noise_boost != 1.0):
                    nproj_full = nproj_full * self.noise_boost
                n_last = self.noise_head_ln(nproj_full[:, -1, :])
            if self._mix_override3 is not None and self.training:
                a, b, g = self._mix_override3
                h_last = a * enc_last + b * hist_sum + g * n_last
            elif self._mix_override is not None and self.training:
                a, b = self._mix_override
                h_last = a * enc_last + b * hist_sum
            else:
                h_last = self.mix_alpha * enc_last + self.mix_beta * hist_sum + self.mix_gamma * n_last
            valid = (t < T_vec)
            if valid.any():
                pre_steps[valid, t, :] = pre_vec[valid]
                post_steps[valid, t, :] = h_last[valid]
            if self.use_quaternion_head:
                q4 = self.head_q(h_last)
                theta = self._angles_from_quaternion(q4)
                y = theta
            else:
                vec6 = self.head(h_last)
                theta, _ = self._angles_from_head_logits(vec6)
                y = theta
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if y.dtype != prev_seq.dtype:
                y = y.to(prev_seq.dtype)
            y = y.clamp(min=-torch.pi + 1e-6, max=torch.pi - 1e-6)
            prev_buf = torch.roll(prev_buf, shifts=-1, dims=1)
            y_for_hist = y
            upd_mask = valid
            if self.training and (self._cur_epoch <= self.history_freeze_epochs):
                Bcur = y.size(0)
                freeze_mask = torch.zeros(Bcur, dtype=torch.bool, device=device)
                if self.p_history_freeze > 0.0:
                    freeze_mask = (torch.rand(Bcur, device=device) < self.p_history_freeze)
                if self.prev_noise_std > 0.0:
                    y_for_hist = y + torch.randn_like(y) * self.prev_noise_std
                upd_mask = valid & (~freeze_mask)
            if upd_mask.any():
                prev_buf[upd_mask, -1, :] = y_for_hist[upd_mask]
            if upd_mask.any():
                pos0 = base_len_vec[upd_mask] + (t * 3)
                for k, tok_id in enumerate((4, 5, 6)):
                    pos_k = (pos0 + k).clamp_max(MAX_SEQ - 1)
                    tok_emb = self.hist_token_emb(torch.full((upd_mask.sum().item(),), tok_id, dtype=torch.long, device=device))
                    pos_emb = self.hist_pos_emb(pos_k)
                    ang_val = y_for_hist[upd_mask, k].unsqueeze(1)
                    val_emb = self.hist_value_proj(ang_val)
                    pqc_sum[upd_mask] = pqc_sum[upd_mask] + (tok_emb + pos_emb + val_emb)
        return pre_steps, post_steps

    def _forward_ordered_seq(self, counts: torch.Tensor, T_vec: torch.Tensor, ctx: Dict[str, any]) -> torch.Tensor:
        assert isinstance(ctx, dict) and ('batch' in ctx), "ctx['batch'] required for ordered-seq mode"
        b: Batch = ctx['batch']
        device = counts.device if isinstance(counts, torch.Tensor) and counts.numel() > 0 else (T_vec.device)
        B = int(T_vec.size(0))
        maxT = int(T_vec.max().item()) if T_vec.numel() > 0 else 0
        if maxT == 0:
            return torch.zeros(B, 0, 1, device=device)
        base_g = b.base_g.to(device)
        base_len = b.base_len.to(device)
        # Optional: per-sample fixed angles for block 0, to force the encoder to see true PQC0
        fixed_b0 = None
        if isinstance(ctx, dict):
            fx = ctx.get('fixed_block0_angles', None)
            if isinstance(fx, torch.Tensor):
                # expect shape [B, 3]
                if fx.device != device:
                    fx = fx.to(device)
                fixed_b0 = fx
        # Only-last-N-base mode: hide first N base gates and PQC0 tokens from the encoder.
        only_last_base = bool(ctx.get('only_last_base', False)) if isinstance(ctx, dict) else False
        # Full-context one-step: still expose full base + PQC0(fixed) + remaining base to encoder,
        # but only predict a single block (the final PQC block). Expected with T_vec == 1.
        full_context_one_step = bool(ctx.get('full_context_one_step', False)) if isinstance(ctx, dict) else False
        def _tok_from_gate_code(code: torch.Tensor) -> torch.Tensor:
            out = torch.full_like(code, self.TOK_UNK)
            out = torch.where(code == 0, torch.tensor(self.TOK_H, device=device, dtype=out.dtype), out)
            out = torch.where(code == 1, torch.tensor(self.TOK_X, device=device, dtype=out.dtype), out)
            out = torch.where(code == 2, torch.tensor(self.TOK_Z, device=device, dtype=out.dtype), out)
            return out
        gb = int(ctx.get('gate_blocks', self.gate_blocks)) if isinstance(ctx, dict) else self.gate_blocks
        def _boundary(_i: int, k: int, Lb_i: int) -> int:
            return min((k + 1) * gb, Lb_i)
        logits = torch.zeros(B, 3 * maxT, 1, device=device)
        prev_angles: list[list[torch.Tensor]] = [[] for _ in range(B)]
        for t in range(maxT):
            valid_b = (T_vec > t)
            if not valid_b.any():
                break
            seq_tok: list[torch.Tensor] = []
            seq_is_pqc: list[torch.Tensor] = []
            seq_ang: list[torch.Tensor] = []
            maxS = 1
            for i in range(B):
                if not bool(valid_b[i].item()):
                    ti = torch.tensor([self.TOK_CLS], device=device, dtype=torch.long)
                    pi = torch.tensor([0], device=device, dtype=torch.long)
                    ai = torch.zeros(1, device=device)
                    seq_tok.append(ti); seq_is_pqc.append(pi); seq_ang.append(ai)
                    continue
                Lb = int(base_len[i].item())
                if only_last_base:
                    # Expose only the last N=gb base gates to the encoder; no PQC tokens.
                    s0 = max(0, Lb - gb)
                    tokens_i: list[int] = [self.TOK_CLS]
                    is_pqc_i: list[int] = [0]
                    ang_i: list[float] = [0.0]
                    if Lb > s0:
                        seg = base_g[i, s0:Lb]
                        toks = _tok_from_gate_code(seg)
                        tokens_i.extend([int(x.item()) for x in toks])
                        is_pqc_i.extend([0] * toks.numel())
                        ang_i.extend([0.0] * toks.numel())
                    ti = torch.tensor(tokens_i, device=device, dtype=torch.long)
                    pi = torch.tensor(is_pqc_i, device=device, dtype=torch.long)
                    ai = torch.tensor(ang_i, device=device, dtype=torch.float32)
                    seq_tok.append(ti); seq_is_pqc.append(pi); seq_ang.append(ai)
                    if ti.numel() > maxS:
                        maxS = ti.numel()
                    continue
                # If we're asked to do full-context but only one-step prediction (T=1),
                # construct the full sandwich sequence: [CLS] + base[:N] + PQC0(fixed) + base[N:]
                if full_context_one_step and (maxT == 1):
                    tokens_i: list[int] = [self.TOK_CLS]
                    is_pqc_i: list[int] = [0]
                    ang_i: list[float] = [0.0]
                    # First N=gb base gates
                    if Lb > 0:
                        e0 = min(gb, Lb)
                        if e0 > 0:
                            seg0 = base_g[i, 0:e0]
                            toks0 = _tok_from_gate_code(seg0)
                            tokens_i.extend([int(x.item()) for x in toks0])
                            is_pqc_i.extend([0] * toks0.numel())
                            ang_i.extend([0.0] * toks0.numel())
                    # PQC0 tokens with fixed angles if provided
                    vals = [0.0, 0.0, 0.0]
                    if (fixed_b0 is not None) and (fixed_b0.size(0) > i):
                        fb = fixed_b0[i]
                        vals = [float(fb[0].item()), float(fb[1].item()), float(fb[2].item())]
                    tokens_i.extend([self.TOK_RZ1, self.TOK_RX, self.TOK_RZ2])
                    is_pqc_i.extend([1, 1, 1])
                    ang_i.extend(vals)
                    # Remaining base gates after N
                    if Lb > gb:
                        seg1 = base_g[i, gb:Lb]
                        toks1 = _tok_from_gate_code(seg1)
                        tokens_i.extend([int(x.item()) for x in toks1])
                        is_pqc_i.extend([0] * toks1.numel())
                        ang_i.extend([0.0] * toks1.numel())
                    ti = torch.tensor(tokens_i, device=device, dtype=torch.long)
                    pi = torch.tensor(is_pqc_i, device=device, dtype=torch.long)
                    ai = torch.tensor(ang_i, device=device, dtype=torch.float32)
                    seq_tok.append(ti); seq_is_pqc.append(pi); seq_ang.append(ai)
                    if ti.numel() > maxS:
                        maxS = ti.numel()
                    continue
                At = _boundary(i, t, Lb)
                tokens_i: list[int] = [self.TOK_CLS]
                is_pqc_i: list[int] = [0]
                ang_i: list[float] = [0.0]
                s = 0
                for k in range(t):
                    e = _boundary(i, k, Lb)
                    if e > s and s < Lb:
                        seg = base_g[i, s:min(e, Lb)]
                        toks = _tok_from_gate_code(seg)
                        tokens_i.extend([int(x.item()) for x in toks])
                        is_pqc_i.extend([0] * toks.numel())
                        ang_i.extend([0.0] * toks.numel())
                    # PQC block k angles for sequence context:
                    # Prefer dataset-provided fixed angles for block 0 if available.
                    if (fixed_b0 is not None) and (k == 0):
                        fb = fixed_b0[i]
                        vals = [float(fb[0].item()), float(fb[1].item()), float(fb[2].item())]
                    elif k < len(prev_angles[i]):
                        yz = prev_angles[i][k]
                        vals = [float(yz[0].item()), float(yz[1].item()), float(yz[2].item())]
                    else:
                        vals = [0.0, 0.0, 0.0]
                    tokens_i.extend([self.TOK_RZ1, self.TOK_RX, self.TOK_RZ2])
                    is_pqc_i.extend([1, 1, 1])
                    ang_i.extend(vals)
                    s = e
                if At > s and s < Lb:
                    seg = base_g[i, s:min(At, Lb)]
                    toks = _tok_from_gate_code(seg)
                    tokens_i.extend([int(x.item()) for x in toks])
                    is_pqc_i.extend([0] * toks.numel())
                    ang_i.extend([0.0] * toks.numel())
                ti = torch.tensor(tokens_i, device=device, dtype=torch.long)
                pi = torch.tensor(is_pqc_i, device=device, dtype=torch.long)
                ai = torch.tensor(ang_i, device=device, dtype=torch.float32)
                seq_tok.append(ti); seq_is_pqc.append(pi); seq_ang.append(ai)
                if ti.numel() > maxS:
                    maxS = ti.numel()
            tok_mat = torch.full((B, maxS), self.TOK_PAD, device=device, dtype=torch.long)
            is_pqc_mat = torch.zeros(B, maxS, device=device, dtype=torch.long)
            ang_mat = torch.zeros(B, maxS, device=device)
            key_pad = torch.ones(B, maxS, device=device, dtype=torch.bool)
            for i in range(B):
                ti = seq_tok[i]; pi = seq_is_pqc[i]; ai = seq_ang[i]
                S = ti.numel()
                tok_mat[i, :S] = ti
                is_pqc_mat[i, :S] = pi
                ang_mat[i, :S] = ai
                key_pad[i, :S] = False
            pos_idx = torch.arange(maxS, device=device)
            emb = self.seq_token_emb(tok_mat) + self.seq_pos_emb(pos_idx).unsqueeze(0)
            if (is_pqc_mat.sum() > 0):
                a = ang_mat
                ac = torch.stack([torch.sin(a), torch.cos(a)], dim=-1)
                ang_emb = self.ang_mlp(ac)
                emb = emb + ang_emb * (is_pqc_mat.float().unsqueeze(-1))
            h = self.seq_encoder(emb, src_key_padding_mask=key_pad)
            h_t = h[:, 0, :]
            if self.use_centers:
                h_t = h_t - self.enc_center.view(1, -1)
            h_t = self.head_in_ln(h_t)
            if self.use_quaternion_head:
                q4 = self.head_q(h_t)
                theta = self._angles_from_quaternion(q4)
                y = theta
            else:
                vec6 = self.head(h_t)
                theta, _r = self._angles_from_head_logits(vec6)
                y = (theta + torch.pi) % (2 * torch.pi) - torch.pi
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-torch.pi + 1e-6, max=torch.pi - 1e-6)
            y_for_hist = y.clone()
            if self.training and (self._cur_epoch <= getattr(self, 'history_freeze_epochs', 0)):
                Bcur = y.size(0)
                freeze_mask = torch.zeros(Bcur, dtype=torch.bool, device=device)
                if getattr(self, 'p_history_freeze', 0.0) > 0.0:
                    freeze_mask = (torch.rand(Bcur, device=device) < float(self.p_history_freeze))
                if freeze_mask.any():
                    y_for_hist[freeze_mask] = 0.0
                pns = float(getattr(self, 'prev_noise_std', 0.0))
                if pns > 0.0:
                    upd_mask = (~freeze_mask) & valid_b
                    if upd_mask.any():
                        y_for_hist[upd_mask] = y_for_hist[upd_mask] + torch.randn_like(y_for_hist[upd_mask]) * pns
            valid_b_idx = valid_b.nonzero(as_tuple=False).squeeze(-1)
            if valid_b_idx.numel() > 0:
                logits[valid_b, t*3:(t+1)*3, 0] = y[valid_b].to(logits.dtype)
            for i in range(B):
                if bool(valid_b[i].item()):
                    prev_angles[i].append(y_for_hist[i, :3])
        return logits

    def forward(self, *args):
        extra_feats = None
        hist_ctx = {}
        ctx = None
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            counts, T_vec = args[0], args[1]
            if len(args) >= 3:
                if isinstance(args[2], torch.Tensor):
                    extra_feats = args[2]
                elif isinstance(args[2], dict):
                    ctx = args[2]
                    extra_feats = ctx.get('extra_feats', None)
                    hist_ctx = ctx
        else:
            b, target_qubit = args
            counts, T_vec = self._counts_from_batch(b, target_qubit)
            ctx = {'batch': b}
        if self.use_ordered_seq and isinstance(ctx, dict) and ('batch' in ctx):
            return self._forward_ordered_seq(counts, T_vec, ctx)
        device = counts.device
        B, maxT = counts.size(0), counts.size(1)
        if maxT == 0:
            return torch.zeros(B, 0, 1, device=device)
        cum = counts.cumsum(dim=1)
        idx_seq = torch.arange(maxT, device=device).unsqueeze(0).expand(B, -1).float()
        prev_buf = torch.zeros(B, PREV_K, 3, device=device)
        prev_seq = torch.zeros(B, maxT, 3 * PREV_K, device=device)
        Y = torch.zeros(B, maxT, 3, device=device)
        def causal_mask(L: int, device0):
            return torch.triu(torch.ones((L, L), dtype=torch.bool, device=device0), diagonal=1)
        base_sum = hist_ctx.get('hist_base_sum', None)
        base_len_vec = hist_ctx.get('base_len', None)
        if base_sum is None or base_len_vec is None:
            base_sum = torch.zeros(B, HID_DIM, device=device)
            base_len_vec = torch.zeros(B, dtype=torch.long, device=device)
        pqc_sum = torch.zeros(B, HID_DIM, device=device)
        for t in range(maxT):
            L = t + 1
            prev_seq[:, t, :] = prev_buf.reshape(B, 3 * PREV_K)
            feats_base = torch.cat([
                counts[:, :L].unsqueeze(-1),
                cum[:, :L].unsqueeze(-1),
                idx_seq[:, :L].unsqueeze(-1),
                prev_seq[:, :L, :],
            ], dim=-1)
            x = self.in_proj(feats_base)
            x = self.in_proj_ln(x)
            if extra_feats is not None:
                x = x + self.extra_proj(extra_feats[:, :L, :])
            noise_feats = hist_ctx.get('noise_feats', None)
            if noise_feats is not None:
                nproj = self.noise_proj(noise_feats[:, :L, :])
                if self.training and (self._cur_epoch < self.noise_boost_epochs) and (self.noise_boost != 1.0):
                    nproj = nproj * self.noise_boost
                x = x + nproj
            pos = self.pos_emb(torch.arange(L, device=device)).unsqueeze(0)
            x = x + pos
            mask = causal_mask(L, device)
            pad = (torch.arange(L, device=device).unsqueeze(0) >= T_vec.unsqueeze(1))
            h = self.encoder(x, mask=mask, src_key_padding_mask=pad)
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            base_prefix = hist_ctx.get('base_prefix', None)
            if base_prefix is not None:
                raw_hist = base_prefix[:, L-1, :] + pqc_sum
            else:
                raw_hist = base_sum + pqc_sum
            raw_enc = h[:, -1, :]
            if self.use_centers:
                raw_hist = raw_hist - self.hist_center.view(1, -1)
                raw_enc = raw_enc - self.enc_center.view(1, -1)
            hist_sum = self.hist_pair_ln(raw_hist)
            hist_sum = torch.nan_to_num(hist_sum, nan=0.0, posinf=0.0, neginf=0.0)
            if self.training:
                Bcur = hist_sum.size(0)
                if self.hist_drop_p > 0.0:
                    keep = (torch.rand(Bcur, device=device) > self.hist_drop_p).float().unsqueeze(1)
                else:
                    keep = 1.0
                if self.hist_scale_max > self.hist_scale_min:
                    scale = torch.empty(Bcur, device=device).uniform_(self.hist_scale_min, self.hist_scale_max).unsqueeze(1)
                else:
                    scale = 1.0
                hist_sum = hist_sum * keep * scale
            enc_last = self.head_in_ln(raw_enc)
            enc_last = torch.nan_to_num(enc_last, nan=0.0, posinf=0.0, neginf=0.0)
            n_last = torch.zeros_like(enc_last)
            if noise_feats is not None:
                nproj_full = self.noise_proj(noise_feats[:, :L, :])
                if self.training and (self._cur_epoch < self.noise_boost_epochs) and (self.noise_boost != 1.0):
                    nproj_full = nproj_full * self.noise_boost
                n_last = self.noise_head_ln(nproj_full[:, -1, :])
                n_last = torch.nan_to_num(n_last, nan=0.0, posinf=0.0, neginf=0.0)
            if self._mix_override3 is not None and self.training:
                a, b, g = self._mix_override3
                h_last = a * enc_last + b * hist_sum + g * n_last
            elif self._mix_override is not None and self.training:
                a, b = self._mix_override
                h_last = a * enc_last + b * hist_sum
            else:
                h_last = self.mix_alpha * enc_last + self.mix_beta * hist_sum + self.mix_gamma * n_last
            if self.use_quaternion_head:
                q4 = self.head_q(h_last)
                theta = self._angles_from_quaternion(q4)
                y = theta
            else:
                vec6 = self.head(h_last)
                theta, radii = self._angles_from_head_logits(vec6)
                y = (theta + torch.pi) % (2 * torch.pi) - torch.pi
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            y = y.clamp(min=-torch.pi + 1e-6, max=torch.pi - 1e-6)
            valid = (t < T_vec)
            if valid.any():
                if y.dtype != Y.dtype:
                    y = y.to(Y.dtype)
                Y[valid, t, :] = y[valid]
            prev_buf = torch.roll(prev_buf, shifts=-1, dims=1)
            y_for_hist = y
            upd_mask = valid
            if self.training and (self._cur_epoch <= self.history_freeze_epochs):
                Bcur = y.size(0)
                freeze_mask = torch.zeros(Bcur, dtype=torch.bool, device=device)
                if self.p_history_freeze > 0.0:
                    freeze_mask = (torch.rand(Bcur, device=device) < self.p_history_freeze)
                if self.prev_noise_std > 0.0:
                    y_for_hist = y + torch.randn_like(y) * self.prev_noise_std
                upd_mask = valid & (~freeze_mask)
            if upd_mask.any():
                prev_buf[upd_mask, -1, :] = y_for_hist[upd_mask]
            if upd_mask.any():
                pos0 = base_len_vec[upd_mask] + (t * 3)
                for k, tok_id in enumerate((4, 5, 6)):
                    pos_k = (pos0 + k).clamp_max(MAX_SEQ - 1)
                    tok_emb = self.hist_token_emb(torch.full((upd_mask.sum().item(),), tok_id, dtype=torch.long, device=device))
                    pos_emb = self.hist_pos_emb(pos_k)
                    ang_val = y_for_hist[upd_mask, k].unsqueeze(1)
                    val_emb = self.hist_value_proj(ang_val)
                    pqc_sum[upd_mask] = pqc_sum[upd_mask] + (tok_emb + pos_emb + val_emb)
        return Y.reshape(B, maxT * 3, 1).contiguous()

    @torch.no_grad()
    def calibrate_centers(self, loader: DataLoader, device: Optional[torch.device] = None, gate_blocks: Optional[int] = None, max_batches: int = 10):
        if device is None:
            device = next(self.parameters()).device
        enc_acc = torch.zeros(HID_DIM, device=device)
        hist_acc = torch.zeros(HID_DIM, device=device)
        nvec = 0
        self.eval()
        seen = 0
        for b in loader:
            b = b.to(device)
            if gate_blocks is None:
                raise ValueError("gate_blocks required for center calibration")
            tq = torch.zeros(b.base_g.size(0), dtype=torch.long, device=device)
            counts, T_vec = self._counts_from_batch(b, tq)
            Bsz = b.base_g.size(0)
            maxT = counts.size(1)
            hist3 = torch.zeros(Bsz, maxT, 3, device=device)
            for i in range(Bsz):
                Lb = int(b.base_len[i].item()); T = int(T_vec[i].item())
                for t in range(T):
                    s = t * gate_blocks; e = min(Lb, (t + 1) * gate_blocks)
                    if s >= e: continue
                    seg = b.base_g[i, s:e]
                    for gcode in (0,1,2):
                        hist3[i, t, gcode] = (seg == gcode).sum().float()
            base_sum = torch.zeros(Bsz, HID_DIM, device=device)
            base_len_vec = b.base_len.to(device)
            for i in range(Bsz):
                Lb_i = int(b.base_len[i].item())
                if Lb_i <= 0: continue
                acc = torch.zeros(HID_DIM, device=device)
                for p in range(Lb_i):
                    g = int(b.base_g[i, p].item());
                    if g < 0: continue
                    tok_id = 1 if g == 0 else (2 if g == 1 else (3 if g == 2 else 0))
                    if tok_id == 0: continue
                    pos_idx = min(p, MAX_SEQ - 1)
                    t_emb = self.hist_token_emb(torch.tensor([tok_id], device=device)).squeeze(0)
                    p_emb = self.hist_pos_emb(torch.tensor([pos_idx], device=device)).squeeze(0)
                    acc = acc + self.hist_pair_mlp(torch.cat([t_emb, p_emb], dim=-1))
                base_sum[i] = acc
            self.train(False)
            S = int(T_vec.max().item() if T_vec.numel() > 0 else 0)
            if S <= 0:
                continue
            pre_steps, post_steps = self.get_step_embeddings(counts, T_vec, max_steps=S, extra_feats=None, hist_ctx={'hist_base_sum': base_sum, 'base_len': base_len_vec})
            enc_last_raw = post_steps[:, -1, :]
            raw_hist = base_sum
            enc_acc += enc_last_raw.sum(dim=0)
            hist_acc += raw_hist.sum(dim=0)
            nvec += Bsz
            seen += 1
            if seen >= max_batches:
                break
        if nvec > 0:
            self.enc_center.copy_(enc_acc / nvec)
            self.hist_center.copy_(hist_acc / nvec)
            self.use_centers = True
        self.train(True)
