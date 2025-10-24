from typing import List, Tuple, Optional, Dict
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Re-import shared constants and helpers from this package
# Use unified simulator_core utilities (dataset, collate, simulator)
from .simulator_core import (
    CircuitDataset, Batch, collate,
    build_base_cache_vectorized, NoiseConfig,
    get_fused_status, ensure_fused_compiled,
)
from .dataset import MAX_BASE_LEN, MAX_PARAM, MAX_QUBITS
from .precision import get_amp_settings, make_grad_scaler
# Import ZZ-ring specific simulator
from .simulator_lelzz import simulate_loss_lelzz_blocks

# Model hyperparameters (defaults match project-wide settings)
HID_DIM = 768
N_LAYERS = 8
N_HEADS = 12
FF_DIM = HID_DIM * 4
DROP = 0.1
PREV_K = 1  # sliding window length for previous angles
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ = MAX_BASE_LEN + MAX_PARAM  # cap for positional embeddings over base+param

# Optional: torch.utils Subset type
from torch.utils.data import Subset
import random
import itertools

def generate_synthetic_single_qubit_items(
    num_samples: int = 32,
    base_len: int = 5,
    gate_blocks: int = 10,
    gates_vocab: Tuple[str, ...] = ("h", "x", "z", "cx", "cz"),
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


class AnglePredictorLELZZ(nn.Module):
    """Autoregressive predictor for ZZ-ring PQC architecture.

    Predicts 7*n_qubits angles per block:
    - 3*n_qubits: pre-local RZ-RX-RZ per qubit
    - n_qubits: ZZ-ring angles (one per adjacent pair)
    - 3*n_qubits: post-local RZ-RX-RZ per qubit

    Unlike the per-qubit subcircuit predictor, this operates on full multi-qubit circuits.
    """
    def __init__(self, gate_blocks: int, n_qubits: int = 2, use_quaternion_head: bool = False):
        super().__init__()
        self.gate_blocks = int(gate_blocks)
        self.n_qubits = int(n_qubits)  # Fixed number of qubits
        self.angles_per_block = 7 * self.n_qubits  # 7 angles per qubit per block
        import math as _math
        self.max_blocks = _math.ceil(MAX_BASE_LEN / max(1, self.gate_blocks))
        # Per-step features: [count_t, cum_t, t_index, prev_angles_window_flat(K*7*n_qubits)]
        feat_dim = 3 + self.angles_per_block * PREV_K
        self.in_proj = nn.Sequential(
            nn.Linear(feat_dim, HID_DIM), nn.GELU(), nn.Dropout(DROP)
        )
        self.in_proj_ln = nn.LayerNorm(HID_DIM)
        # Optional extra per-block features (e.g., gate-type histogram [h,x,z]): project and add
        self.extra_proj = nn.Sequential(
            nn.Linear(3, HID_DIM), nn.GELU(), nn.Dropout(DROP)
        )
        # Optional per-block noise features (e.g., mean rx/rz on target qubit): project and add
        self.noise_proj = nn.Sequential(
            nn.Linear(2, HID_DIM), nn.GELU(), nn.Dropout(DROP)
        )
        # Optional physics-aware features: per-block cumulative rotation as quaternion (w,x,y,z)
        self.phys_proj = nn.Sequential(
            nn.Linear(4, HID_DIM), nn.GELU(), nn.Dropout(DROP)
        )
        self.pos_emb = nn.Embedding(self.max_blocks, HID_DIM)
        enc_layer = nn.TransformerEncoderLayer(HID_DIM, N_HEADS, FF_DIM, DROP, batch_first=True, norm_first=True)
        # Disable nested tensor to avoid PyTorch warning when norm_first=True; fall back if not supported
        try:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        # Output head predicts 7*n_qubits angles via S¹ representation: 2D (x,y) per angle
        self.head = nn.Linear(HID_DIM, 2 * self.angles_per_block)
        # Quaternion head not used for multi-qubit ZZ-ring (too many dimensions)
        self.use_quaternion_head = False
        # initialize to predict identity rotation (all angles 0)
        with torch.no_grad():
            # S¹ head -> zeros, x=1 (angle=0)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            if self.head.bias.numel() >= 2 * self.angles_per_block:
                b = self.head.bias.view(self.angles_per_block, 2)
                b[:, 0] = 1.0  # x=1
                b[:, 1] = 0.0  # y=0 -> theta=0

        self._angle_reg_lambda = 0.0  # optional radius regularization weight (disabled by default)
        self._last_radius_penalty = None

        # History encoding (base + past PQC tokens) via additive position-aware sum
        # Vocab: 0=PAD, 1=H, 2=X, 3=Z, 4=RZ_PQC, 5=RX_PQC, 6=RZ2_PQC
        self.hist_vocab_size = 7
        self.hist_token_emb = nn.Embedding(self.hist_vocab_size, HID_DIM)
        self.hist_pos_emb = nn.Embedding(MAX_SEQ, HID_DIM)
        self.hist_value_proj = nn.Sequential(nn.Linear(1, HID_DIM), nn.GELU(), nn.Dropout(DROP))
        # Non-decomposable token+position pairing to avoid sum(E_tok)+sum(E_pos) collapse
        self.hist_pair_mlp = nn.Sequential(
            nn.Linear(2 * HID_DIM, HID_DIM), nn.GELU(), nn.Dropout(DROP)
        )
        # Centers for calibration (to remove cross-sample common bias before LN/mix)
        self.register_buffer('enc_center', torch.zeros(HID_DIM))
        self.register_buffer('hist_center', torch.zeros(HID_DIM))
        self.use_centers: bool = False
        # Normalization and mixing to mitigate common-mode bias
        self.hist_pair_ln = nn.LayerNorm(HID_DIM)
        self.head_in_ln = nn.LayerNorm(HID_DIM)
        self.mix_alpha = nn.Parameter(torch.tensor(1.0))  # encoder contribution
        self.mix_beta = nn.Parameter(torch.tensor(1.0))   # history contribution
        # Optional third path: noise contribution mixed at the head to guarantee gradient flow
        self.mix_gamma = nn.Parameter(torch.tensor(1.0))  # noise contribution
        # LayerNorm for noise head input (reuse head_in_ln if desired)
        self.noise_head_ln = nn.LayerNorm(HID_DIM)
        # Train-time controls
        self.hist_drop_p = 0.0
        self.hist_scale_min = 1.0
        self.hist_scale_max = 1.0
        self.noise_boost = 1.0
        self.noise_boost_epochs = 0
        self._cur_epoch = 0
        self._mix_override: Optional[Tuple[float, float]] = None
        self._mix_override3: Optional[Tuple[float, float, float]] = None
        # AR stabilization controls
        self.p_history_freeze = 0.0  # prob to skip feeding current y into history during early epochs
        self.prev_noise_std = 0.0    # std of Gaussian noise added to angles before feeding to history
        self.history_freeze_epochs = 0 # apply history freeze/noise only for these first epochs

        # Ordered history-sequence mode (default enabled). When on, forward() ignores
        # legacy counts/hist features and instead builds a variable-length token
        # sequence per step consisting of base gates and previously-inserted PQC
        # tokens (with their predicted angles), preserving strict order.
        self.use_ordered_seq = str(os.environ.get('PQC_ORDERED_SEQ', '1')).strip().lower() in ('1','true','yes','y','on')
        # Token ids for ordered sequence
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
        # Angle embedding for PQC tokens: sin/cos(angle) -> HID
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
        """Override encoder/history/noise mixing during training warmup."""
        self._mix_override3 = (float(alpha), float(beta), float(gamma))

    @staticmethod
    def _angles_from_head_logits(vec6: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map head outputs to angles via S¹: for each pair (x,y):
        u = (x,y) / sqrt(x^2 + y^2 + eps^2); theta = atan2(u_y, u_x).
        Returns (theta[B,3], radii[B,3]).
        """
        B = vec6.size(0)
        v = vec6.view(B, 3, 2)  # [B,3,2]
        x = v[:, :, 0]
        y = v[:, :, 1]
        r = torch.sqrt(x * x + y * y + eps * eps)
        ux = x / r
        uy = y / r
        theta = torch.atan2(uy, ux)  # [-pi, pi]
        return theta, r

    @staticmethod
    def _angles_from_quaternion(q4: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Map raw quaternion to ZXZ Euler angles (rz1, rx, rz2) in [-pi, pi].
        q4: [B,4] ordered as (w, x, y, z). All torch ops, differentiable.
        Steps:
        - normalize quaternion
        - compute rotation matrix R
        - extract ZXZ: beta = arccos(R[2,2]); alpha = atan2(R[0,2], -R[1,2]); gamma = atan2(R[2,0], R[2,1])
        - gimbal lock when sin(beta)~0: set gamma=0 and alpha = atan2(R[1,0], R[0,0])
        """
        B = q4.size(0)
        w, x, y, z = q4[:, 0], q4[:, 1], q4[:, 2], q4[:, 3]
        norm = torch.sqrt(w*w + x*x + y*y + z*z + eps)
        w = w / norm; x = x / norm; y = y / norm; z = z / norm
        # rotation matrix entries
        R11 = 1 - 2*(y*y + z*z)
        R12 = 2*(x*y - w*z)
        R13 = 2*(x*z + w*y)
        R21 = 2*(x*y + w*z)
        R22 = 1 - 2*(x*x + z*z)
        R23 = 2*(y*z - w*x)
        R31 = 2*(x*z - w*y)
        R32 = 2*(y*z + w*x)
        R33 = 1 - 2*(x*x + y*y)
        # ZXZ extraction
        # Use a stable formulation for beta: beta = atan2(sinb, cosb)
        # where cosb = R33 and sinb = sqrt(R13^2 + R23^2)
        sinb = torch.sqrt(torch.clamp(R13 * R13 + R23 * R23, min=eps))
        beta = torch.atan2(sinb, torch.clamp(R33, -1.0, 1.0))
        # normal case
        alpha = torch.atan2(R13, -R23)
        gamma = torch.atan2(R31, R32)
        # gimbal lock handling: where sinb small, use alternative
        # Use a slightly larger threshold to avoid atan2(0,0) instability near identity
        small = (sinb.abs() < 1e-3)
        if small.any():
            # when beta ~ 0: R ~ Rz(alpha+gamma)
            alpha_alt = torch.atan2(R21, R11)
            alpha = torch.where(small, alpha_alt, alpha)
            gamma = torch.where(small, torch.zeros_like(gamma), gamma)
        # wrap to [-pi, pi]
        def wrap(a):
            return (a + torch.pi) % (2*torch.pi) - torch.pi
        alpha = wrap(alpha); beta = wrap(beta); gamma = wrap(gamma)
        return torch.stack([alpha, beta, gamma], dim=1)

    def _counts_from_batch(self, b, target_qubit: torch.Tensor):
        # compute counts per block for each sample in batch for target_qubit
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
        """Return one embedding per per-qubit line: the encoder hidden at the last valid block.
        Uses only embedder features (prev angles set to zeros) and a causal mask.
        Shape: counts [B,maxT], T_vec [B] -> embeddings [B,HID_DIM].
        """
        device = counts.device
        B, maxT = counts.shape
        if maxT == 0:
            return torch.zeros(B, HID_DIM, device=device)
        hist3 = torch.zeros(B, maxT, 3, device=device)
        noise_feats = None
        base_prefix = torch.zeros(B, maxT, HID_DIM, device=device)
        cum = counts.cumsum(dim=1)
        idx_seq = torch.arange(maxT, device=device).unsqueeze(0).expand(B, -1).float()
        prev_zero = torch.zeros(B, maxT, 3 * PREV_K, device=device)
        feats = torch.cat([
            counts.unsqueeze(-1),      # count_t
            cum.unsqueeze(-1),         # cum_t (inclusive)
            idx_seq.unsqueeze(-1),     # raw t index
            prev_zero,                 # zero history window (debug-only)
        ], dim=-1)
        x = self.in_proj(feats)
        x = self.in_proj_ln(x)
        pos = self.pos_emb(torch.arange(maxT, device=device)).unsqueeze(0)
        x = x + pos
        # causal mask and key padding mask (boolean to match src_key_padding_mask dtype)
        attn_mask = torch.triu(torch.ones((maxT, maxT), dtype=torch.bool, device=device), diagonal=1)
        key_pad = (torch.arange(maxT, device=device).unsqueeze(0) >= T_vec.unsqueeze(1))  # [B,L]
        h = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_pad)
        # gather last valid hidden per sample
        idx_last = (T_vec - 1).clamp_min(0).to(device)
        idx_last = idx_last.view(B, 1, 1).expand(B, 1, HID_DIM)
        emb = h.gather(dim=1, index=idx_last).squeeze(1)  # [B,H]
        return emb

    def get_line_pre_embeddings(self, counts: torch.Tensor, T_vec: torch.Tensor) -> torch.Tensor:
        """Return per-line embeddings BEFORE encoder attention: in_proj(feats) at last valid block.
        This shows how similar inputs are prior to attention mixing.
        Shape: [B,HID_DIM].
        """
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
        x0 = self.in_proj(feats)  # [B,L,H]
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
        """Return per-step embeddings for each line using an AR loop mirroring forward.
        Returns two tensors:
          - pre_steps: [B, S, HID_DIM]  in-proj embedding at last position (before encoder), per step
          - post_steps: [B, S, HID_DIM] embedding fed into the head per step (i.e., encoder last state + history summary)

        Notes:
          - When hist_ctx is provided, we add base_sum and running pqc_sum just like forward, and use that vector for head.
          - When hist_ctx is None, base_sum and base_len default to zeros and pqc_sum is not updated with positions (still AR on outputs).
          - Steps S = min(maxT, max_steps) if max_steps else maxT. For steps >= T_i for a given sample, vectors are zeros.
        """
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

        # history context (optional)
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
            # physics-aware quaternion features (if provided)
            if isinstance(hist_ctx, dict) and hist_ctx.get('context_quat', None) is not None:
                cq = hist_ctx['context_quat']
                x = x + self.phys_proj(cq[:, :L, :])
            pre_vec = x[:, -1, :]
            pos = self.pos_emb(torch.arange(L, device=device)).unsqueeze(0)
            x = x + pos
            attn_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=1)
            key_pad = (torch.arange(L, device=device).unsqueeze(0) >= T_vec.unsqueeze(1))
            h = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_pad)
            # align with forward: add history residual to the last position before head
            # prefer per-step base prefix if provided
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
            # train-time: apply hist_sum dropout or random scaling
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
            # noise head vector at the last position to ensure direct gradient for noise path
            n_last = torch.zeros_like(enc_last)
            if isinstance(hist_ctx, dict) and hist_ctx.get('noise_feats', None) is not None:
                noise_feats = hist_ctx['noise_feats']
                nproj_full = self.noise_proj(noise_feats[:, :L, :])
                if self.training and (self._cur_epoch < self.noise_boost_epochs) and (self.noise_boost != 1.0):
                    nproj_full = nproj_full * self.noise_boost
                n_last = self.noise_head_ln(nproj_full[:, -1, :])
            # mixing with optional 3-way override
            if self._mix_override3 is not None and self.training:
                a, b, g = self._mix_override3
                h_last = a * enc_last + b * hist_sum + g * n_last
            elif self._mix_override is not None and self.training:
                a, b = self._mix_override
                h_last = a * enc_last + b * hist_sum
            else:
                h_last = self.mix_alpha * enc_last + self.mix_beta * hist_sum + self.mix_gamma * n_last
            # use a proper boolean mask per sample for step t
            valid = (t < T_vec)
            if valid.any():
                pre_steps[valid, t, :] = pre_vec[valid]
                post_steps[valid, t, :] = h_last[valid]
            # advance AR with current prediction (two head options)
            if self.use_quaternion_head:
                q4 = self.head_q(h_last)  # [B,4]
                theta = self._angles_from_quaternion(q4)
                y = theta
            else:
                vec6 = self.head(h_last)
                theta, radii = self._angles_from_head_logits(vec6)
                y = theta  # use angles directly
            # sanitize
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            # ensure dtype matches buffers under AMP (prev_seq/post tensors are float32 by construction)
            if y.dtype != prev_seq.dtype:
                y = y.to(prev_seq.dtype)
            y = y.clamp(min=-torch.pi + 1e-6, max=torch.pi - 1e-6)
            # advance AR with optional history freeze/noise (diagnostic path mirrors forward)
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
            # update pqc_sum like forward if we have base_len; mask by updated samples only
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
        """Encode full prior gate timeline (base + prior PQC tokens) per step and predict angles.
        Uses CLS token representation at each step. Angles for prior PQC tokens are the model's
        own previous predictions (zeros at t=0). Order is preserved via positional embeddings
        and self-attention over the constructed sequence.
        """
        assert isinstance(ctx, dict) and ('batch' in ctx), "ctx['batch'] required for ordered-seq mode"
        b: Batch = ctx['batch']
        device = counts.device if isinstance(counts, torch.Tensor) and counts.numel() > 0 else (T_vec.device)
        B = int(T_vec.size(0))
        maxT = int(T_vec.max().item()) if T_vec.numel() > 0 else 0
        if maxT == 0:
            return torch.zeros(B, 0, 1, device=device)

        base_g = b.base_g.to(device)
        base_len = b.base_len.to(device)

        # helper: map base gate code to token id
        def _tok_from_gate_code(code: torch.Tensor) -> torch.Tensor:
            out = torch.full_like(code, self.TOK_UNK)
            out = torch.where(code == 0, torch.tensor(self.TOK_H, device=device, dtype=out.dtype), out)
            out = torch.where(code == 1, torch.tensor(self.TOK_X, device=device, dtype=out.dtype), out)
            out = torch.where(code == 2, torch.tensor(self.TOK_Z, device=device, dtype=out.dtype), out)
            return out

        # Use fixed-interval block boundaries to match simulator exactly
        gb = int(ctx.get('gate_blocks', self.gate_blocks)) if isinstance(ctx, dict) else self.gate_blocks
        def _boundary(_i: int, k: int, Lb_i: int) -> int:
            return min((k + 1) * gb, Lb_i)

        logits = torch.zeros(B, 3 * maxT, 1, device=device)
        prev_angles: list[list[torch.Tensor]] = [[] for _ in range(B)]

        for t in range(maxT):
            valid_b = (T_vec > t)
            if not valid_b.any():
                break
            # build sequences per sample
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
                # boundary for current step and previous steps using fixed grid
                At = _boundary(i, t, Lb)
                tokens_i: list[int] = [self.TOK_CLS]
                is_pqc_i: list[int] = [0]
                ang_i: list[float] = [0.0]
                s = 0
                # iterate previous blocks
                for k in range(t):
                    e = _boundary(i, k, Lb)
                    if e > s and s < Lb:
                        seg = base_g[i, s:min(e, Lb)]
                        toks = _tok_from_gate_code(seg)
                        tokens_i.extend([int(x.item()) for x in toks])
                        is_pqc_i.extend([0] * toks.numel())
                        ang_i.extend([0.0] * toks.numel())
                    # insert PQC k angles
                    if k < len(prev_angles[i]):
                        yz = prev_angles[i][k]
                        vals = [float(yz[0].item()), float(yz[1].item()), float(yz[2].item())]
                    else:
                        vals = [0.0, 0.0, 0.0]
                    tokens_i.extend([self.TOK_RZ1, self.TOK_RX, self.TOK_RZ2])
                    is_pqc_i.extend([1, 1, 1])
                    ang_i.extend(vals)
                    s = e
                # final base segment up to At
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
            # pad and encode
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
            # sanitize and clamp
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-torch.pi + 1e-6, max=torch.pi - 1e-6)
            # teacher-forcing: optionally freeze history update and/or add small noise early on
            y_for_hist = y.clone()
            if self.training and (self._cur_epoch <= getattr(self, 'history_freeze_epochs', 0)):
                Bcur = y.size(0)
                freeze_mask = torch.zeros(Bcur, dtype=torch.bool, device=device)
                if getattr(self, 'p_history_freeze', 0.0) > 0.0:
                    freeze_mask = (torch.rand(Bcur, device=device) < float(self.p_history_freeze))
                # when frozen, do not feed prediction into history (use zeros)
                if freeze_mask.any():
                    y_for_hist[freeze_mask] = 0.0
                # add small Gaussian noise to non-frozen updates if requested
                pns = float(getattr(self, 'prev_noise_std', 0.0))
                if pns > 0.0:
                    upd_mask = (~freeze_mask) & valid_b
                    if upd_mask.any():
                        y_for_hist[upd_mask] = y_for_hist[upd_mask] + torch.randn_like(y_for_hist[upd_mask]) * pns
            valid_b_idx = valid_b.nonzero(as_tuple=False).squeeze(-1)
            if valid_b_idx.numel() > 0:
                # ensure dtype matches destination (AMP may produce bf16 for y)
                logits[valid_b, t*3:(t+1)*3, 0] = y[valid_b].to(logits.dtype)
            # update prev angles
            for i in range(B):
                if bool(valid_b[i].item()):
                    prev_angles[i].append(y_for_hist[i, :3])
        return logits

    def forward(self, *args):
        # supports: (counts, T_vec[, extra_feats or ctx]) or (batch, target_qubit)
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
        # Ordered history-sequence path (preferred): ignore counts/extra_feats if enabled
        if self.use_ordered_seq and isinstance(ctx, dict) and ('batch' in ctx):
            return self._forward_ordered_seq(counts, T_vec, ctx)
        device = counts.device
        B, maxT = counts.size(0), counts.size(1)
        if maxT == 0:
            return torch.zeros(B, 0, 1, device=device)
        hist3 = torch.zeros(B, maxT, 3, device=device)
        noise_feats = None
        base_prefix = torch.zeros(B, maxT, HID_DIM, device=device)
        # Prepare features: counts, cumulative counts, raw index, and a sliding window of prev angles
        cum = counts.cumsum(dim=1)
        idx_seq = torch.arange(maxT, device=device).unsqueeze(0).expand(B, -1).float()
        # sliding window buffer for previous K outputs
        prev_buf = torch.zeros(B, PREV_K, 3, device=device)
        # store per-position used window
        prev_seq = torch.zeros(B, maxT, 3 * PREV_K, device=device)
        # outputs buffer
        Y = torch.zeros(B, maxT, 3, device=device)
        # causal mask [L,L]
        def causal_mask(L: int, device):
            # return boolean mask (True = masked) to match src_key_padding_mask dtype
            return torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=1)
        # History additive summary: base_sum (required) + running pqc_sum
        base_sum = hist_ctx.get('hist_base_sum', None)
        base_len_vec = hist_ctx.get('base_len', None)
        if base_sum is None or base_len_vec is None:
            # fall back to zeros if not provided
            base_sum = torch.zeros(B, HID_DIM, device=device)
            base_len_vec = torch.zeros(B, dtype=torch.long, device=device)
        pqc_sum = torch.zeros(B, HID_DIM, device=device)

        for t in range(maxT):
            L = t + 1
            # write current window into position t (zero when t==0, grows with t)
            prev_seq[:, t, :] = prev_buf.reshape(B, 3 * PREV_K)
            feats_base = torch.cat([
                counts[:, :L].unsqueeze(-1),         # count_t
                cum[:, :L].unsqueeze(-1),            # cum_t
                idx_seq[:, :L].unsqueeze(-1),        # raw t index
                prev_seq[:, :L, :],                  # flattened prev window
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
            # physics-aware quaternion features intentionally not used (noise-only info is excluded from inputs)
            # add history summary to last position representation via residual after encoder
            pos = self.pos_emb(torch.arange(L, device=device)).unsqueeze(0)
            x = x + pos
            mask = causal_mask(L, device)
            # per-sample padding: positions >= T_i are invalid
            pad = (torch.arange(L, device=device).unsqueeze(0) >= T_vec.unsqueeze(1))  # [B,L]
            h = self.encoder(x, mask=mask, src_key_padding_mask=pad)
            # Sanitize encoder output to avoid NaN propagation
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            # History summary at step t uses base_prefix[:,t,:] + pqc_sum (angles up to t-1), if available
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
            # noise path disabled when no noise_feats provided in ctx
            n_last = torch.zeros_like(enc_last)
            if noise_feats is not None:
                nproj_full = self.noise_proj(noise_feats[:, :L, :])
                if self.training and (self._cur_epoch < self.noise_boost_epochs) and (self.noise_boost != 1.0):
                    nproj_full = nproj_full * self.noise_boost
                n_last = self.noise_head_ln(nproj_full[:, -1, :])
                n_last = torch.nan_to_num(n_last, nan=0.0, posinf=0.0, neginf=0.0)
            # mixing with optional 3-way override
            if self._mix_override3 is not None and self.training:
                a, b, g = self._mix_override3
                h_last = a * enc_last + b * hist_sum + g * n_last
            elif self._mix_override is not None and self.training:
                a, b = self._mix_override
                h_last = a * enc_last + b * hist_sum
            else:
                h_last = self.mix_alpha * enc_last + self.mix_beta * hist_sum + self.mix_gamma * n_last
            if self.use_quaternion_head:
                q4 = self.head_q(h_last)  # [B,4]
                theta = self._angles_from_quaternion(q4)
                y = theta
            else:
                vec6 = self.head(h_last)  # [B,6]
                theta, radii = self._angles_from_head_logits(vec6)
                # keep angles in [-pi, pi]
                y = (theta + torch.pi) % (2 * torch.pi) - torch.pi  # [B,3]
            # sanitize
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            y = y.clamp(min=-torch.pi + 1e-6, max=torch.pi - 1e-6)
            # write output only for valid samples at step t
            valid = (t < T_vec)
            if valid.any():
                if y.dtype != Y.dtype:
                    y = y.to(Y.dtype)
                Y[valid, t, :] = y[valid]
            # update sliding window for next step: shift-left and append current y
            prev_buf = torch.roll(prev_buf, shifts=-1, dims=1)
            # Scheduled history freeze/noise during early epochs to break self-conditioning
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
            # update history pqc_sum only for samples we updated
            if upd_mask.any():
                pos0 = base_len_vec[upd_mask] + (t * 3)  # first pqc token position of current block
                for k, tok_id in enumerate((4, 5, 6)):
                    pos_k = (pos0 + k).clamp_max(MAX_SEQ - 1)
                    tok_emb = self.hist_token_emb(torch.full((upd_mask.sum().item(),), tok_id, dtype=torch.long, device=device))
                    pos_emb = self.hist_pos_emb(pos_k)
                    ang_val = y_for_hist[upd_mask, k].unsqueeze(1)  # [B_upd,1]
                    val_emb = self.hist_value_proj(ang_val)
                    pqc_sum[upd_mask] = pqc_sum[upd_mask] + (tok_emb + pos_emb + val_emb)
        # zero out positions >= T_i (already ensured) and reshape
        return Y.reshape(B, maxT * 3, 1).contiguous()

    @torch.no_grad()
    def calibrate_centers(self, loader: DataLoader, device: Optional[torch.device] = None, gate_blocks: Optional[int] = None, max_batches: int = 10):
        """Estimate enc_center and hist_center using a few mini-batches, then enable use_centers.
        Only uses encoder last state and base_sum at t=last per sample. No training happens here.
        """
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
            # build extra_feats and base_sum like train_subcircuits
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
            # one forward pass: use ordered-seq encoder CLS at last valid step to compute centers
            self.train(False)
            S = int(T_vec.max().item() if T_vec.numel() > 0 else 0)
            if S <= 0:
                continue
            # Build ordered-seq context up to last step and run through seq_encoder
            ctx = {'batch': b, 'gate_blocks': gate_blocks}
            # Reuse forward to produce per-step logits and, via internal path, encoder inputs; but to get enc last, approximate by using get_step_embeddings
            # Fallback: estimate enc_last_raw as post_steps' encoder contribution before mixing by disabling centers and using head_in_ln inverse is not trivial,
            # so we approximate encoder contribution by post_steps (it blends enc and hist). For center removal, this is acceptable.
            pre_steps, post_steps = self.get_step_embeddings(counts, T_vec, max_steps=S, extra_feats=None, hist_ctx={'hist_base_sum': base_sum, 'base_len': base_len_vec})
            enc_last_raw = post_steps[:, -1, :]
            raw_hist = base_sum  # pqc_sum ~ 0 at calibration
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
                if amp_enabled and amp_dtype is not None and torch.cuda.is_available():
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        per_logits = model(counts, T_vec)
                else:
                    per_logits = model(counts, T_vec)
                # simulator in FP32
                if amp_enabled and torch.cuda.is_available():
                    with torch.cuda.amp.autocast(enabled=False):
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
            if amp_enabled and amp_dtype is not None and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    per_logits = model(counts, T_vec)
            else:
                per_logits = model(counts, T_vec)
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
                with torch.cuda.amp.autocast(enabled=False):
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
                      synthetic_enumerate_all: bool = False,
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
        # Synthetic mode: use ALL data for training; validation is a small random subset sampled from training data.
        if sub_val_count is not None:
            val_cnt = min(int(sub_val_count), N)
        else:
            # default: ~5% capped at 1000, at least 1 if N>0
            val_cnt = min(max(1, N // 20), 1000) if N > 0 else 0
        val_idx = _r.sample(indices, val_cnt) if val_cnt > 0 else []
        ds_train = sub  # full dataset
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
        if use_syn:
            print(f"[Subcircuits][Data] training samples={N} test samples={len(val_idx)} (synthetic subset)")
        else:
            print(f"[Subcircuits][Data] training samples={len(train_idx)} test samples={len(val_idx)}")
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
                # Build context: if ordered-seq is enabled, pass the batch and skip legacy feature prep
                if model.use_ordered_seq:
                    ctx = {'batch': b, 'gate_blocks': gate_blocks}
                else:
                    # Vectorized prep (legacy path)
                    # 1) per-block gate-type histogram [h,x,z]
                    hist3 = torch.zeros(Bsz, maxT, 3, device=device)
                    for i in range(Bsz):
                        Lb = int(b.base_len[i].item())
                        Ti = int(T_vec[i].item())
                        if Lb <= 0 or Ti <= 0:
                            continue
                        pos = torch.arange(Lb, device=device)
                        block_idx = torch.div(pos, gate_blocks, rounding_mode='floor').clamp_max(Ti - 1)  # [Lb]
                        gates = b.base_g[i, :Lb]  # [Lb]
                        # count for gate codes 0/1/2
                        for gcode in (0, 1, 2):
                            mask = (gates == gcode)
                            if mask.any():
                                hist3[i, :Ti, gcode].scatter_add_(0, block_idx[mask], torch.ones(mask.sum(), device=device, dtype=torch.float32))
                    # 2) base_prefix and base_sum via batched MLP + prefix-sum per sample
                    base_prefix = torch.zeros(Bsz, maxT, HID_DIM, device=device)
                    base_sum = torch.zeros(Bsz, HID_DIM, device=device)
                    base_len_vec = b.base_len.to(device)
                    tok_w = model.hist_token_emb.weight  # [vocab,H]
                    pos_w = model.hist_pos_emb.weight    # [MAX_SEQ,H]
                    for i in range(Bsz):
                        Lb = int(b.base_len[i].item())
                        Ti = int(T_vec[i].item())
                        if Lb <= 0 or Ti <= 0:
                            continue
                        pos = torch.arange(Lb, device=device)
                        gates = b.base_g[i, :Lb]
                        # map gate to tok_id: 0->1,1->2,2->3 else 0
                        tok_id = torch.full((Lb,), 0, dtype=torch.long, device=device)
                        tok_id = torch.where(gates == 0, torch.tensor(1, device=device), tok_id)
                        tok_id = torch.where(gates == 1, torch.tensor(2, device=device), tok_id)
                        tok_id = torch.where(gates == 2, torch.tensor(3, device=device), tok_id)
                        valid = tok_id > 0
                        if not valid.any():
                            continue
                        t_embs = tok_w.index_select(0, tok_id[valid])            # [N,H]
                        p_idx = pos[valid].clamp_max(MAX_SEQ - 1)
                        p_embs = pos_w.index_select(0, p_idx)                    # [N,H]
                        pairs = torch.cat([t_embs, p_embs], dim=-1)              # [N,2H]
                        vecs = model.hist_pair_mlp(pairs)                        # [N,H]
                        v_full = torch.zeros(Lb, HID_DIM, device=device)
                        v_full[valid] = vecs
                        cum = v_full.cumsum(dim=0)                                # [Lb,H]
                        # base_sum: sum over all positions
                        base_sum[i] = cum[-1]
                        # base_prefix at block ends
                        ends = (torch.arange(Ti, device=device) + 1) * gate_blocks - 1
                        ends = ends.clamp_max(Lb - 1)
                        base_prefix[i, :Ti, :] = cum.index_select(0, ends)
                    # Strictly exclude noise/physics from encoder inputs
                    ctx = dict(extra_feats=hist3, hist_base_sum=base_sum, base_len=base_len_vec, noise_feats=None, base_prefix=base_prefix)
                t_prep1 = _time.perf_counter()
                prep_last = t_prep1 - t_prep0 + (t_prep0 - t_loop_start)  # include device copy + counts
                prep_t += prep_last
                t0 = _time.perf_counter()
                _sync()
                # model forward under AMP
                if amp_enabled and amp_dtype is not None and torch.cuda.is_available():
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
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
                        # Embedding variance and cosine collapse check (legacy path only)
                        if not model.use_ordered_seq:
                            import torch.nn.functional as F
                            S = min(maxT_dbg, 3)
                            if S > 0:
                                pre_s, post_s = model.get_step_embeddings(counts, T_vec, max_steps=S, extra_feats=hist3, hist_ctx=ctx)
                                for step_name, mat in (("t1", post_s[:,0,:]), (f"t{S}", post_s[:,S-1,:])):
                                    if mat.size(0) >= 2:
                                        norms = mat.norm(dim=1)
                                        mean_n = float(norms.mean().item()); std_n = float(norms.std().item())
                                        # sample up to 20 pairwise cosines
                                        idx = torch.arange(mat.size(0), device=device)
                                        pairs = list(zip(idx[:-1].tolist(), idx[1:].tolist()))[:20]
                                        cos_vals = []
                                        for i,j in pairs:
                                            cos_vals.append(float(F.cosine_similarity(mat[i].unsqueeze(0), mat[j].unsqueeze(0)).item()))
                                        if cos_vals:
                                            mc = sum(cos_vals)/len(cos_vals)
                                            print(f"[Diag][Train][Emb@{step_name}] norm_mean={mean_n:.3f} norm_std={std_n:.3f} mean_pair_cos={mc:.3f}")
                        # Base/history stats
                        if not model.use_ordered_seq:
                            bs_norm = base_sum.norm(dim=1)
                            print(f"[Diag][Train][BaseSum] norm_mean={float(bs_norm.mean().item()):.3f} norm_std={float(bs_norm.std().item()):.3f}")
                        # Deeper: enc vs hist dominance at t=1 (noise path excluded)
                        if maxT_dbg >= 1:
                            L = 1
                            device0 = device
                            feats_base = torch.cat([
                                counts[:, :L].unsqueeze(-1),
                                counts.cumsum(dim=1)[:, :L].unsqueeze(-1),
                                torch.arange(counts.size(1), device=device0).unsqueeze(0).expand(Bsz_dbg, -1)[:, :L].float().unsqueeze(-1),
                                torch.zeros(Bsz_dbg, L, 3*PREV_K, device=device0)
                            ], dim=-1)
                            xdbg = model.in_proj(feats_base)
                            xdbg = xdbg + model.extra_proj(hist3[:, :L, :])
                            xdbg = xdbg + model.pos_emb(torch.arange(L, device=device0)).unsqueeze(0)
                            attn_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=device0), diagonal=1)
                            key_pad = (torch.arange(L, device=device0).unsqueeze(0) >= T_vec.unsqueeze(1))
                            hdbg = model.encoder(xdbg, mask=attn_mask, src_key_padding_mask=key_pad)
                            raw_enc = hdbg[:, -1, :]
                            raw_hist = base_sum
                            if model.use_centers:
                                raw_enc = raw_enc - model.enc_center.view(1, -1)
                                raw_hist = raw_hist - model.hist_center.view(1, -1)
                            enc_last = model.head_in_ln(raw_enc)
                            hist_sum = model.hist_pair_ln(raw_hist)
                            h_last = model.mix_alpha * enc_last + model.mix_beta * hist_sum
                            # contribution ratios
                            h_norm = h_last.norm(dim=1) + 1e-8
                            r_enc = (model.mix_alpha * enc_last).norm(dim=1) / h_norm
                            r_hist = (model.mix_beta * hist_sum).norm(dim=1) / h_norm
                            print(f"[Diag][Train][Mix@t1] enc_ratio_mean={float(r_enc.mean().item()):.3f} hist_ratio_mean={float(r_hist.mean().item()):.3f}")
                            # pairwise cosine within enc_last and hist_sum
                            if Bsz_dbg >= 2:
                                pairs = list(zip(range(min(10,Bsz_dbg-1)), range(1, min(11,Bsz_dbg))))
                                enc_cos = []
                                his_cos = []
                                for i,j in pairs:
                                    enc_cos.append(float(F.cosine_similarity(enc_last[i].unsqueeze(0), enc_last[j].unsqueeze(0)).item()))
                                    his_cos.append(float(F.cosine_similarity(hist_sum[i].unsqueeze(0), hist_sum[j].unsqueeze(0)).item()))
                                if enc_cos:
                                    print(f"[Diag][Train][PairCos@t1] enc_mean={sum(enc_cos)/len(enc_cos):.3f} hist_mean={sum(his_cos)/len(his_cos):.3f}")
                    except Exception:
                        pass
                if VERBOSE and (not prints["train_once"]):
                    _print_angle_examples(logits, T_vec, tag="Train")
                    prints["train_once"] = True
                # Compute base loss grouped by base_len (differentiable)
                t2 = _time.perf_counter(); _sync()
                # simulator (loss) in FP32 regardless of AMP
                if amp_enabled and torch.cuda.is_available():
                    with torch.cuda.amp.autocast(enabled=False):
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
    return model, final_fid


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train angle predictor (simple) with optional synthetic 1q5 dataset.")
    p.add_argument("--data-path", type=str, default="", help="Path to JSON dataset folder/file (ignored in synthetic mode)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
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
    p.add_argument("--synthetic-enum-all", action="store_true", help="Enumerate ALL H/X/Z sequences of length N (ignores --synthetic-num-samples). ENV: PQC_SYNTH_ENUM_ALL=1")
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
        synthetic_enumerate_all=args.synthetic_enum_all,
    )
