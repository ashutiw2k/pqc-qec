import math, random, os, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
from tqdm import tqdm
import os
import json
import time
from collections import defaultdict

"""Optimization Notes
Implemented in this file (relative to earlier versions):
1. Vectorized base circuit simulation path (simulate_base_only_vectorized) operating on [B,S,2^n] removing per-sample inner loop for base-only evolution; toggled via USE_VECTORIZED_BASE.
2. Unified handling for single initial state (S=1) and multi initial states to reuse the same vectorized base evolution when enabled.
3. Optional torch.compile integration (ENABLE_COMPILE env flag) for model graph to fuse kernels where possible.
4. Lightweight CUDA event based profiling (ENABLE_PROFILING env flag) printing encoder vs simulation timing per batch.

Deferred / Not yet implemented:
- Fully vectorized interleaved PQC application (current path still per-sample for parameterized gates and noise after base gates).
- Gate fusion: merging consecutive single-qubit rotations of same type on same qubit.
- Grouping gates by qubit to apply multiple distinct single-qubit operations in a single batched tensor op.
- Custom Triton / CUDA kernels for further kernel launch reduction.
- FlashAttention or sequence length reduction (gate grouping) in Transformer encoder.

Configuration (hard-coded defaults, toggle in code if needed):
    USE_VECTORIZED_BASE = True  -> master switch for base simulation path
    ENABLE_COMPILE      = True  -> always attempt torch.compile on the model
    ENABLE_PROFILING    = True  -> print per-batch timing (CUDA only)
"""

# ================== Simple Timing Utility ==================
ENABLE_PROFILING = True        # <— 方便直接跑。完全关闭计时请置 False
PROFILE_VERBOSE  = True        # 每个 epoch 打印阶段性统计；False 只在最后打印

TIME_STATS = defaultdict(lambda: {"total": 0.0, "count": 0})

def _sync_cuda():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

class SectionTimer:
    """Context manager for timing a code section with optional CUDA sync."""
    def __init__(self, name, sync_cuda=True):
        self.name = name
        self.sync_cuda = sync_cuda and ENABLE_PROFILING
        self.t0 = None

    def __enter__(self):
        if not ENABLE_PROFILING:
            return self
        if self.sync_cuda:
            _sync_cuda()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not ENABLE_PROFILING or self.t0 is None:
            return False
        if self.sync_cuda:
            _sync_cuda()
        dt = time.perf_counter() - self.t0
        s = TIME_STATS[self.name]
        s["total"] += dt
        s["count"] += 1
        return False

def _timed(name, sync_cuda=True):
    """Helper to use as: with _timed("train:encoder"): ..."""
    return SectionTimer(name, sync_cuda=sync_cuda)

def print_timing_report(header="=== Timing Report ===", top_k=None):
    if not ENABLE_PROFILING: 
        return
    print("\n" + header)
    rows = []
    for k, v in TIME_STATS.items():
        total = v["total"]
        cnt   = max(1, v["count"])
        rows.append((k, total, total/cnt, cnt))
    rows.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        rows = rows[:top_k]
    print(f"{'section':45} | {'total(s)':>10} | {'avg(s)':>9} | {'count':>6}")
    print("-"*78)
    for k, total, avg, cnt in rows:
        print(f"{k:45} | {total:10.4f} | {avg:9.6f} | {cnt:6d}")
    print("-"*78 + "\n")

# ---------------- Config / Data ----------------
#data_file_path = "A:/wings/transformers/data/5q_10g_5blk_quaternion_uncomp_rzrxrz_100k_data/5q_10g_5blk_data/good_fidelity"
data_file_path = "A:/wings/transformers/data/circuit_tokens_data/circuit_tokens/no_uncomp/processed_data"
train_num = 1000
NOISE_X_RAD_NUM = math.pi/100
NOISE_Z_RAD_NUM = math.pi/100
NOISE_DELTA_X_NUM = 0.05
NOISE_DELTA_Z_NUM = 0.05
gate_mapping = {"cx":0, "h":1, "x":2, "cz":3, "z":4, "<pad>":5}
NGATE = len(gate_mapping)
NQ1   = 10
NQ2   = NQ1 + 1    # padding for 2nd qubit

def read_all_json_files(folder_path):
    with _timed("io:read_json", sync_cuda=False):
        idx = 0
        json_data = []
        for filename in tqdm(os.listdir(folder_path), desc="Reading JSON files"):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    json_data.append(data)
                    idx += 1
                    if idx >= train_num:
                        break
        return json_data

# Base gates + PAD (extended to include 'cz' and 'z')
PARAM_GATE_TYPES = ['rz','rx']
param_gate_mapping = {g:i for i,g in enumerate(PARAM_GATE_TYPES)}

BASE_SET = {'cx','h','x','cz','z'}
PQC_SET  = {'rz','rx'}

# ---- Load ----
raw = read_all_json_files(data_file_path)
# Simply split here for quick test
raw = raw[:train_num]

# ---- Compute SRC_MAX_LEN and K_MAX, and build token lists ----
with _timed("tokenize:build_lists", sync_cuda=False):
    base_list = []
    full_with_pqc = []
    for item in raw:
        if 'base_circuit_tokens' not in item or 'pqc_circuit_tokens' not in item:
            continue
        base_list.append(item['base_circuit_tokens'])
        full_with_pqc.append(item['pqc_circuit_tokens'])

SRC_MAX_LEN = max(len(line) for line in base_list) if base_list else 0

def count_pqc(line):
    return sum(1 for g in line if g[0] in PQC_SET)

K_MAX = max(count_pqc(line) for line in full_with_pqc) if full_with_pqc else 0

# ---- Tokenizers ----
def tokenize_base_with_pad(lines, src_max_len):
    """Pad base gates to src_max_len with <pad> and placeholders."""
    with _timed("tokenize:base", sync_cuda=False):
        out=[]
        for line in lines:
            d={"gate_id":[],"qubit_1":[],"qubit_2":[]}
            for g in line:
                d["gate_id"].append(gate_mapping[g[0]])
                d["qubit_1"].append(g[1][0])
                if len(g[1]) == 1:
                    d["qubit_2"].append(-1)
                else:
                    d["qubit_2"].append(g[1][1])
            pad_needed = src_max_len - len(d["gate_id"])
            if pad_needed > 0:
                d["gate_id"].extend([gate_mapping["<pad>"]]*pad_needed)
                d["qubit_1"].extend([0]*pad_needed)
                d["qubit_2"].extend([-1]*pad_needed)
            out.append(d)
        return out

def extract_pqc_schedule(lines_full, base_set=BASE_SET, pqc_set=PQC_SET):
    """
    From full mixed circuit tokens, extract PQC schedule:
      - qubit[k], params[k], types[k], after_idx[k] (relative to base gates)
      - count (#PQC)
    """
    with _timed("tokenize:extract_schedule", sync_cuda=False):
        res=[]
        for line in lines_full:
            qubits=[]; params=[]; types=[]; after_idx=[]
            base_idx = -1  # -1 means before any base gate
            for g in line:
                name = g[0]
                if name in base_set:
                    base_idx += 1
                elif name in pqc_set:
                    q = g[1][0]
                    th = g[2][0] if len(g) > 2 and len(g[2]) > 0 else 0.0
                    qubits.append(q)
                    params.append(th)
                    types.append(name)
                    after_idx.append(base_idx)
            cnt = len(params)
            pad_n = K_MAX - cnt
            if pad_n > 0:
                qubits.extend([0]*pad_n)
                params.extend([0.0]*pad_n)
                types.extend(['rz']*pad_n)
                after_idx.extend([-999]*pad_n)
            res.append({
                "qubit":qubits, "params":params, "types":types,
                "after_idx":after_idx, "count":cnt
            })
        return res

tokenized_data_x = tokenize_base_with_pad(base_list, SRC_MAX_LEN)
extracted_sched  = extract_pqc_schedule(full_with_pqc)

# ---------------- Model ----------------
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0,max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self,x):
        return x + self.pe[:,:x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,n_layers,max_len):
        super().__init__()
        self.pos = PositionalEncoding(d_model,max_len)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model,n_heads,d_ff,batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self,x):
        x = self.pos(x)
        for l in self.layers: x = l(x)
        return self.norm(x)

class EncoderWithQuerySlotsPacked(nn.Module):
    def __init__(self,d_model=256,nhead=4,ff=1024,enc_layers=3,
                 max_src_len=40,tgt_len=60,dropout=0.1):
        super().__init__()
        self.K = tgt_len
        self.gate_emb = nn.Embedding(NGATE,d_model, padding_idx=gate_mapping["<pad>"])
        self.q1_emb   = nn.Embedding(NQ1,  d_model)
        self.q2_emb   = nn.Embedding(NQ2,  d_model, padding_idx=NQ2-1)
        self.proj = nn.Linear(d_model*3,d_model)
        self.encoder = TransformerEncoder(d_model,nhead,ff,enc_layers,max_src_len)
        self.block_emb = nn.Embedding(tgt_len,d_model)
        self.cross_attn = nn.MultiheadAttention(d_model,nhead,batch_first=True,dropout=dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model,ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff,d_model), nn.LayerNorm(d_model)
        )
        self.out = nn.Sequential(
            nn.Linear(d_model,d_model), nn.GELU(), nn.Linear(d_model,2)
        )
    def forward(self,A,B,C):
        if A.dim()==1:
            A=A.unsqueeze(0);B=B.unsqueeze(0);C=C.unsqueeze(0)
        with _timed("model:embeddings"):
            x = torch.cat([self.gate_emb(A), self.q1_emb(B), self.q2_emb(C)], dim=-1)
            x = self.proj(x)
        with _timed("model:encoder"):
            enc = self.encoder(x)
        idx = torch.arange(self.K, device=A.device).unsqueeze(0).expand(A.size(0),-1)
        with _timed("model:cross_attn_ff"):
            q = self.block_emb(idx)
            q2,_ = self.cross_attn(q, enc, enc)
            q2 = self.ff(q2)
        with _timed("model:head"):
            return self.out(q2)  # [B,K,2]

# ---------------- Dataset ----------------
class MyNumericSeq2SeqDataset(Dataset):
    """Dataset 增加样本索引, 便于使用缓存的 base ideal state。"""
    def __init__(self,X,sched):
        self.X=X; self.S=sched
        assert len(X)==len(sched)
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        X=self.X[i]; S=self.S[i]
        gate_id=torch.tensor(X["gate_id"],dtype=torch.long)
        q1=torch.tensor(X["qubit_1"],dtype=torch.long)
        q2=torch.tensor([q if q>=0 else NQ1 for q in X["qubit_2"]],dtype=torch.long)

        tgt_angles=torch.tensor(S["params"],dtype=torch.float32)
        param_qubits=torch.tensor(S["qubit"],dtype=torch.long)
        param_types =torch.tensor([param_gate_mapping[t] for t in S["types"]],dtype=torch.long)
        after_idx   =torch.tensor(S["after_idx"],dtype=torch.long)
        pqc_count   =torch.tensor(S["count"],dtype=torch.long)

        return {
            "idx": torch.tensor(i, dtype=torch.long),
            "A_ids":gate_id, "B_ids":q1, "C_ids":q2,
            "tgt_angles":tgt_angles,
            "param_qubits":param_qubits,
            "param_types":param_types,
            "after_idx":after_idx,
            "pqc_count":pqc_count
        }

def collate_fn_numeric(batch):
    return {k:torch.stack([b[k] for b in batch],0) for k in batch[0]}

# ---------------- Angle utils ----------------
def angles_to_sin_cos(a): return torch.stack([torch.sin(a),torch.cos(a)],dim=-1)
def sin_cos_to_angles(sc): return torch.atan2(sc[...,0], sc[...,1])

def circular_loss(pred_sc,tgt_angles, pqc_count):
    B, K, _ = pred_sc.shape
    pred = pred_sc / (pred_sc.norm(dim=-1,keepdim=True)+1e-8)
    tgt_sc= angles_to_sin_cos(tgt_angles)
    tgt  = tgt_sc / (tgt_sc.norm(dim=-1,keepdim=True)+1e-8)
    sim = (pred*tgt).sum(-1)
    mask = torch.arange(K, device=pqc_count.device).unsqueeze(0) < pqc_count.unsqueeze(1)
    loss = (1 - sim)*mask
    return loss.sum() / mask.sum().clamp_min(1)

# ---------------- Noise model (base gates only) ----------------
class TorchNoisyGates:
    def __init__(self,x_rad=math.pi/30,z_rad=math.pi/30,
                 delta_x=0.0,delta_z=0.0,seed=0):
        self.x_noise = x_rad
        self.z_noise = z_rad
        self.delta_x = delta_x * self.x_noise / 100.0
        self.delta_z = delta_z * self.z_noise / 100.0
        self.x_noise_max = self.x_noise + self.delta_x
        self.x_noise_min = self.x_noise - self.delta_x
        self.z_noise_max = self.z_noise + self.delta_z
        self.z_noise_min = self.z_noise - self.delta_z
        self.rng = random.Random(seed)
    def has_noise(self):
        return (abs(self.x_noise_max) > 0 or abs(self.z_noise_max) > 0)
    def sample_angles(self,n,device):
        x = (self.x_noise_min +
             (self.x_noise_max - self.x_noise_min)*torch.rand(n,device=device))
        z = (self.z_noise_min +
             (self.z_noise_max - self.z_noise_min)*torch.rand(n,device=device))
        return x,z

# ---------------- Quantum gates ----------------
def build_full_single(U, qubit, n_qubits, device):
    ops=[]
    I=torch.eye(2,dtype=torch.complex64,device=device)
    for q in range(n_qubits):
        ops.append(U if q==qubit else I)
    full=ops[0]
    for i in range(1,n_qubits):
        full = torch.kron(full, ops[i])
    return full

def build_cx(control,target,n_qubits,device):
    dim=1<<n_qubits
    I=torch.eye(dim,dtype=torch.complex64,device=device)
    proj_ops=[]
    for q in range(n_qubits):
        if q==control:
            proj_ops.append(torch.tensor([[0,0],[0,1]],dtype=torch.complex64,device=device))
        else:
            proj_ops.append(torch.eye(2,dtype=torch.complex64,device=device))
    P=proj_ops[0]
    for i in range(1,n_qubits):
        P=torch.kron(P,proj_ops[i])
    Xsmall=torch.tensor([[0,1],[1,0]],dtype=torch.complex64,device=device)
    x_ops=[]
    for q in range(n_qubits):
        x_ops.append(Xsmall if q==target else torch.eye(2,dtype=torch.complex64,device=device))
    Xfull=x_ops[0]
    for i in range(1,n_qubits):
        Xfull=torch.kron(Xfull,x_ops[i])
    U = I - P + Xfull @ P
    return U

def build_cz(control,target,n_qubits,device):
    dim = 1 << n_qubits
    I = torch.eye(dim, dtype=torch.complex64, device=device)
    proj_ops = []
    for q in range(n_qubits):
        if q == control:
            proj_ops.append(torch.tensor([[0,0],[0,1]], dtype=torch.complex64, device=device))
        else:
            proj_ops.append(torch.eye(2, dtype=torch.complex64, device=device))
    P = proj_ops[0]
    for i in range(1, n_qubits):
        P = torch.kron(P, proj_ops[i])
    Zsmall = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64, device=device)
    z_ops = []
    for q in range(n_qubits):
        z_ops.append(Zsmall if q == target else torch.eye(2, dtype=torch.complex64, device=device))
    Zfull = z_ops[0]
    for i in range(1, n_qubits):
        Zfull = torch.kron(Zfull, z_ops[i])
    U = I - P + Zfull @ P
    return U

def single_Rz(theta):
    e_m = torch.exp(-0.5j*theta)
    e_p = torch.exp( 0.5j*theta)
    U = torch.zeros(theta.size(0),2,2,dtype=torch.complex64,device=theta.device)
    U[:,0,0]=e_m; U[:,1,1]=e_p
    return U

def single_Rx(theta):
    c = torch.cos(theta/2); s=torch.sin(theta/2)
    U = torch.zeros(theta.size(0),2,2,dtype=torch.complex64,device=theta.device)
    U[:,0,0]=c; U[:,1,1]=c
    U[:,0,1]= -1j*s
    U[:,1,0]= -1j*s
    return U

# ---- Memory-efficient state updates (no full 2^n x 2^n matrices) ----
def _apply_single_qubit_gate_on_state(state: torch.Tensor, U2: torch.Tensor, qubit: int, n_qubits: int):
    dim = state.numel()
    outer = 1 << (n_qubits - qubit - 1)
    inner = 1 << qubit
    st = state.view(outer, 2, inner)
    new = torch.einsum('ab,obi->oai', U2, st)
    return new.reshape(dim)

def apply_rx_state(state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int):
    c = torch.cos(theta/2)
    s = torch.sin(theta/2)
    U = torch.stack([
        torch.stack([c, -1j*s]),
        torch.stack([-1j*s, c])
    ]).to(state.dtype).to(state.device)
    return _apply_single_qubit_gate_on_state(state, U, qubit, n_qubits)

def apply_rz_state(state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int):
    e_m = torch.exp(-0.5j*theta)
    e_p = torch.exp( 0.5j*theta)
    U = torch.stack([
        torch.stack([e_m, torch.zeros((), dtype=state.dtype, device=state.device)]) ,
        torch.stack([torch.zeros((), dtype=state.dtype, device=state.device), e_p])
    ])
    return _apply_single_qubit_gate_on_state(state, U, qubit, n_qubits)

_CONST_X = torch.tensor([[0,1],[1,0]], dtype=torch.complex64)
_CONST_H = (1/math.sqrt(2))*torch.tensor([[1,1],[1,-1]], dtype=torch.complex64)
_CONST_Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64)

# ---------------- Small gate per-device cache ----------------
_CONST_GATE_CACHE = {}
def _get_const_gates(device):
    cached = _CONST_GATE_CACHE.get(device)
    if cached is None:
        cached = (
            _CONST_X.to(device),
            _CONST_H.to(device),
            _CONST_Z.to(device)
        )
        _CONST_GATE_CACHE[device] = cached
    return cached  # (X, H, Z)

_CX_SWAP_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_CZ_MASK_CACHE: dict[tuple, torch.Tensor] = {}

def _get_cx_swap_tensors(device, n_qubits, control, target):
    key = (device, n_qubits, control, target)
    cached = _CX_SWAP_CACHE.get(key)
    if cached is not None:
        return cached
    dim = 1 << n_qubits
    idx = torch.arange(dim, device=device)
    mc = 1 << control
    mt = 1 << target
    ctrl_one = (idx & mc) != 0
    tgt_zero = (idx & mt) == 0
    mask = ctrl_one & tgt_zero
    mask_indices = idx[mask]
    partner_indices = mask_indices ^ mt
    _CX_SWAP_CACHE[key] = (mask_indices, partner_indices)
    return mask_indices, partner_indices

def _get_cz_mask(device, n_qubits, control, target):
    key = (device, n_qubits, control, target)
    cached = _CZ_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    dim = 1 << n_qubits
    idx = torch.arange(dim, device=device)
    mc = 1 << control
    mt = 1 << target
    mask = ((idx & mc) != 0) & ((idx & mt) != 0)
    mask_indices = idx[mask]
    _CZ_MASK_CACHE[key] = mask_indices
    return mask_indices

def apply_x_state(state: torch.Tensor, qubit: int, n_qubits: int):
    X,H,Z = _get_const_gates(state.device)
    return _apply_single_qubit_gate_on_state(state, X, qubit, n_qubits)

def apply_h_state(state: torch.Tensor, qubit: int, n_qubits: int):
    X,H,Z = _get_const_gates(state.device)
    return _apply_single_qubit_gate_on_state(state, H, qubit, n_qubits)

def apply_z_state(state: torch.Tensor, qubit: int, n_qubits: int):
    X,H,Z = _get_const_gates(state.device)
    return _apply_single_qubit_gate_on_state(state, Z, qubit, n_qubits)

def apply_cx_state(state: torch.Tensor, control: int, target: int, n_qubits: int):
    mask_indices, partner_indices = _get_cx_swap_tensors(state.device, n_qubits, control, target)
    if mask_indices.numel() == 0:
        return state
    out = state.clone()
    tmp = out[mask_indices].clone()
    out[mask_indices] = out[partner_indices]
    out[partner_indices] = tmp
    return out

def apply_cz_state(state: torch.Tensor, control: int, target: int, n_qubits: int):
    mask_indices = _get_cz_mask(state.device, n_qubits, control, target)
    if mask_indices.numel() == 0:
        return state
    out = state.clone()
    out[mask_indices] = -out[mask_indices]
    return out

def apply_base_gate_with_noise(st, gate_id, q1, q2, n_qubits, device, noise_model: TorchNoisyGates|None):
    if gate_id == gate_mapping["<pad>"]:
        return st
    acted=[]
    if gate_id == gate_mapping['h']:
        st = apply_h_state(st, q1, n_qubits); acted=[q1]
    elif gate_id == gate_mapping['x']:
        st = apply_x_state(st, q1, n_qubits); acted=[q1]
    elif gate_id == gate_mapping['z']:
        st = apply_z_state(st, q1, n_qubits); acted=[q1]
    elif gate_id == gate_mapping['cx'] and (q2 < n_qubits):
        st = apply_cx_state(st, q1, q2, n_qubits); acted=[q1,q2]
    elif gate_id == gate_mapping['cz'] and (q2 < n_qubits):
        st = apply_cz_state(st, q1, q2, n_qubits); acted=[q1,q2]
    if noise_model is not None and noise_model.has_noise() and acted:
        x_angles,z_angles = noise_model.sample_angles(len(acted), device)
        for i,qb in enumerate(acted):
            st = apply_rx_state(st, x_angles[i], qb, n_qubits)
            st = apply_rz_state(st, z_angles[i], qb, n_qubits)
    return st

# ---------------- Simulation ----------------
_STATE_BUFFER = {}
def _get_state_buffer(device, batch_size, n_qubits, dtype=torch.complex64):
    key = (device, batch_size, n_qubits, dtype)
    dim = 1 << n_qubits
    buf = _STATE_BUFFER.get(key)
    if buf is None or buf.shape != (batch_size, dim):
        buf = torch.zeros(batch_size, dim, dtype=dtype, device=device)
        _STATE_BUFFER[key] = buf
    else:
        buf.zero_()
    return buf

def simulate_base_only(base_gate_ids, base_q1, base_q2, n_qubits, noise_model: TorchNoisyGates|None, reuse_buffer: bool = False):
    with _timed("simulate:base_only"):
        device = base_gate_ids.device
        B,L = base_gate_ids.shape
        dim = 1<<n_qubits
        if reuse_buffer:
            states = _get_state_buffer(device, B, n_qubits, dtype=torch.complex64)
        else:
            states = torch.zeros(B,dim,dtype=torch.complex64,device=device)
        states[:,0] = 1+0j
        for i in range(L):
            g_col  = base_gate_ids[:,i]
            q1_col = base_q1[:,i]
            q2_col = base_q2[:,i]
            for b in range(B):
                states[b] = apply_base_gate_with_noise(
                    states[b],
                    int(g_col[b]),
                    int(q1_col[b]),
                    int(q2_col[b]),
                    n_qubits,
                    device,
                    noise_model=noise_model
                )
        states /= (states.norm(dim=-1,keepdim=True)+1e-12)
        return states

# ================= Unified Vectorized Simulation (Batch + Optional S) =================
def _reshape_for_qubit(states: torch.Tensor, qubit: int, n_qubits: int):
    dim = states.shape[-1]
    outer = 1 << (n_qubits - qubit - 1)
    inner = 1 << qubit
    return states.view(states.shape[0], states.shape[1], outer, 2, inner), outer, inner

def _apply_single_qubit_all(states: torch.Tensor, qubit: int, U2: torch.Tensor, n_qubits: int):
    # NOTE: helper kept for future; not used in main path
    v, outer, inner = _reshape_for_qubit(states, qubit, n_qubits)
    new = torch.einsum('ab,bso bi->bso ai', U2, v)  # (kept as placeholder)
    return new.reshape_as(states)

def _apply_param_rot_all(states: torch.Tensor, qubit: int, theta: torch.Tensor, gate_type: str, n_qubits: int):
    if theta.dim() == 1:
        theta_exp = theta.view(-1, 1)
    else:
        theta_exp = theta
    if gate_type == 'rz':
        e_m = torch.exp(-0.5j*theta_exp)
        e_p = torch.exp( 0.5j*theta_exp)
        U = torch.zeros(theta_exp.size(0), 2, 2, dtype=states.dtype, device=states.device)
        U[:,0,0] = e_m.squeeze(-1)
        U[:,1,1] = e_p.squeeze(-1)
    else:
        c = torch.cos(theta_exp/2.0)
        s = torch.sin(theta_exp/2.0)
        U = torch.zeros(theta_exp.size(0), 2, 2, dtype=states.dtype, device=states.device)
        U[:,0,0] = c.squeeze(-1)
        U[:,1,1] = c.squeeze(-1)
        val = -1j*s.squeeze(-1)
        U[:,0,1] = val
        U[:,1,0] = val
    B = states.shape[0]
    for b in range(B):
        states[b] = _apply_single_qubit_gate_on_state(states[b], U[b], qubit, n_qubits)
    return states

def simulate_base_only_vectorized(base_gate_ids: torch.Tensor,
                                  base_q1: torch.Tensor,
                                  base_q2: torch.Tensor,
                                  n_qubits: int,
                                  noise_model: TorchNoisyGates|None,
                                  initial_states: torch.Tensor|None=None) -> torch.Tensor:
    with _timed("simulate:base_only_vectorized"):
        device = base_gate_ids.device
        B, L = base_gate_ids.shape
        dim = 1 << n_qubits
        if initial_states is None:
            states = torch.zeros(B,1,dim, dtype=torch.complex64, device=device)
            states[:,:,0] = 1+0j
        else:
            states = initial_states.clone()
        for i in range(L):
            gid_col = base_gate_ids[:, i]
            q1_col = base_q1[:, i]
            q2_col = base_q2[:, i]
            if torch.all(gid_col == gate_mapping['<pad>']):
                continue
            for gname in ('h','x','z','cx','cz'):
                gcode = gate_mapping[gname]
                mask = (gid_col == gcode)
                if not torch.any(mask):
                    continue
                idx = mask.nonzero(as_tuple=False).squeeze(-1)
                if gname in ('h','x','z'):
                    X,H,Z = _get_const_gates(device)
                    U2 = {'h':H,'x':X,'z':Z}[gname]
                    for b in idx.tolist():
                        qb = int(q1_col[b])
                        states[b] = _apply_single_qubit_gate_on_state(states[b], U2, qb, n_qubits)
                        if noise_model is not None and noise_model.has_noise():
                            x_ang, z_ang = noise_model.sample_angles(1, device)
                            states[b] = apply_rx_state(states[b], x_ang[0], qb, n_qubits)
                            states[b] = apply_rz_state(states[b], z_ang[0], qb, n_qubits)
                else:
                    for b in idx.tolist():
                        c = int(q1_col[b]); t = int(q2_col[b])
                        if t >= n_qubits:
                            continue
                        if gname == 'cx':
                            states[b] = apply_cx_state(states[b], c, t, n_qubits)
                        else:
                            states[b] = apply_cz_state(states[b], c, t, n_qubits)
                        if noise_model is not None and noise_model.has_noise():
                            x_ang, z_ang = noise_model.sample_angles(2, device)
                            for qi, qb in enumerate([c,t]):
                                states[b] = apply_rx_state(states[b], x_ang[qi], qb, n_qubits)
                                states[b] = apply_rz_state(states[b], z_ang[qi], qb, n_qubits)
        states /= (states.norm(dim=-1, keepdim=True)+1e-12)
        return states

def simulate_interleaved_with_params(base_gate_ids, base_q1, base_q2,
                                     pqc_qubits, pqc_types, pqc_angles,
                                     pqc_after_idx, pqc_count,
                                     n_qubits, noise_model: TorchNoisyGates,
                                     reuse_buffer: bool = False,
                                     assume_sorted_after: bool = True):
    with _timed("simulate:interleaved_params"):
        device = base_gate_ids.device
        B,L = base_gate_ids.shape
        dim = 1<<n_qubits
        if reuse_buffer and (not torch.is_grad_enabled()):
            states = _get_state_buffer(device, B, n_qubits, dtype=torch.complex64)
        else:
            states = torch.zeros(B,dim,dtype=torch.complex64,device=device)
        states[:,0]=1+0j

        def apply_param(st, q, t_idx, theta):
            if t_idx == param_gate_mapping['rz']:
                return apply_rz_state(st, theta, int(q), n_qubits)
            else:
                return apply_rx_state(st, theta, int(q), n_qubits)

        grad_mode = torch.is_grad_enabled()
        if grad_mode:
            new_states = []
            for b in range(B):
                st = states[b].clone()
                Kb = int(pqc_count[b])
                if Kb == 0:
                    for i in range(L):
                        st = apply_base_gate_with_noise(
                            st,
                            int(base_gate_ids[b,i]),
                            int(base_q1[b,i]),
                            int(base_q2[b,i]),
                            n_qubits,
                            device,
                            noise_model=noise_model
                        )
                    new_states.append(st)
                    continue

                if not assume_sorted_after:
                    valid = pqc_after_idx[b,:Kb]
                    sort_idx = torch.argsort(valid)
                    pqc_after_sorted = valid[sort_idx]
                    pqc_qubits_sorted = pqc_qubits[b,:Kb][sort_idx]
                    pqc_types_sorted  = pqc_types[b,:Kb][sort_idx]
                    pqc_angles_sorted = pqc_angles[b,:Kb][sort_idx]
                else:
                    pqc_after_sorted = pqc_after_idx[b,:Kb]
                    pqc_qubits_sorted = pqc_qubits[b,:Kb]
                    pqc_types_sorted  = pqc_types[b,:Kb]
                    pqc_angles_sorted = pqc_angles[b,:Kb]

                ptr = 0
                while ptr < Kb and int(pqc_after_sorted[ptr]) == -1:
                    st = apply_param(st, int(pqc_qubits_sorted[ptr]), int(pqc_types_sorted[ptr]), pqc_angles_sorted[ptr])
                    ptr += 1
                for i in range(L):
                    st = apply_base_gate_with_noise(
                        st,
                        int(base_gate_ids[b,i]),
                        int(base_q1[b,i]),
                        int(base_q2[b,i]),
                        n_qubits,
                        device,
                        noise_model=noise_model
                    )
                    while ptr < Kb and int(pqc_after_sorted[ptr]) == i:
                        st = apply_param(st, int(pqc_qubits_sorted[ptr]), int(pqc_types_sorted[ptr]), pqc_angles_sorted[ptr])
                        ptr += 1
                new_states.append(st)
            states = torch.stack(new_states, 0)
            states = states / (states.norm(dim=-1,keepdim=True)+1e-12)
        else:
            for b in range(B):
                Kb = int(pqc_count[b])
                if Kb == 0:
                    for i in range(L):
                        states[b] = apply_base_gate_with_noise(
                            states[b],
                            int(base_gate_ids[b,i]),
                            int(base_q1[b,i]),
                            int(base_q2[b,i]),
                            n_qubits,
                            device,
                            noise_model=noise_model
                        )
                    continue

                if not assume_sorted_after:
                    valid = pqc_after_idx[b,:Kb]
                    sort_idx = torch.argsort(valid)
                    pqc_after_sorted = valid[sort_idx]
                    pqc_qubits_sorted = pqc_qubits[b,:Kb][sort_idx]
                    pqc_types_sorted  = pqc_types[b,:Kb][sort_idx]
                    pqc_angles_sorted = pqc_angles[b,:Kb][sort_idx]
                else:
                    pqc_after_sorted = pqc_after_idx[b,:Kb]
                    pqc_qubits_sorted = pqc_qubits[b,:Kb]
                    pqc_types_sorted  = pqc_types[b,:Kb]
                    pqc_angles_sorted = pqc_angles[b,:Kb]

                ptr = 0
                while ptr < Kb and int(pqc_after_sorted[ptr]) == -1:
                    states[b] = apply_param(states[b], int(pqc_qubits_sorted[ptr]), int(pqc_types_sorted[ptr]), pqc_angles_sorted[ptr])
                    ptr += 1
                for i in range(L):
                    states[b] = apply_base_gate_with_noise(
                        states[b],
                        int(base_gate_ids[b,i]),
                        int(base_q1[b,i]),
                        int(base_q2[b,i]),
                        n_qubits,
                        device,
                        noise_model=noise_model
                    )
                    while ptr < Kb and int(pqc_after_sorted[ptr]) == i:
                        states[b] = apply_param(states[b], int(pqc_qubits_sorted[ptr]), int(pqc_types_sorted[ptr]), pqc_angles_sorted[ptr])
                        ptr += 1
            states = states / (states.norm(dim=-1,keepdim=True)+1e-12)
        return states

# ---------------- Fidelity loss (JAX-aligned) ----------------
USE_VECTORIZED_BASE = True
ENABLE_COMPILE = False

def fidelity_loss_interleaved(pred_sincos,
                              base_gate_ids, base_q1, base_q2,
                              pqc_qubits, pqc_types, pqc_after_idx, pqc_count,
                              n_qubits,
                              noisy_model: TorchNoisyGates,
                              ideal_model: TorchNoisyGates,
                              tgt_angles=None,
                              psi_base_ideal_cached=None):
    with _timed("loss:fidelity_single"):
        pred_angles = sin_cos_to_angles(pred_sincos)  # [B,K_MAX]
        if psi_base_ideal_cached is not None:
            psi_base_ideal = psi_base_ideal_cached
        else:
            with torch.no_grad():
                if USE_VECTORIZED_BASE:
                    psi_base_ideal = simulate_base_only_vectorized(base_gate_ids, base_q1, base_q2,
                                                                   n_qubits, noise_model=ideal_model,
                                                                   initial_states=None)[:,0]
                else:
                    psi_base_ideal = simulate_base_only(base_gate_ids, base_q1, base_q2,
                                                        n_qubits, noise_model=ideal_model, reuse_buffer=True)
        psi_measured = simulate_interleaved_with_params(
            base_gate_ids, base_q1, base_q2,
            pqc_qubits, pqc_types, pred_angles,
            pqc_after_idx, pqc_count,
            n_qubits, noise_model=noisy_model
        )
        ov = torch.sum(torch.conj(psi_base_ideal)*psi_measured, dim=-1)
        F  = (ov.abs()**2)
        loss = (1 - F).mean()

        F_gt_mean = None
        if tgt_angles is not None:
            with torch.no_grad():
                psi_gt = simulate_interleaved_with_params(
                    base_gate_ids, base_q1, base_q2,
                    pqc_qubits, pqc_types, tgt_angles,
                    pqc_after_idx, pqc_count,
                    n_qubits, noise_model=noisy_model
                )
                ov_gt = torch.sum(torch.conj(psi_base_ideal)*psi_gt, dim=-1)
                F_gt_mean = (ov_gt.abs()**2).mean()
        return loss, F.mean(), F_gt_mean

# ================= Multi-initial-state extensions =================
def random_initial_states(batch_size: int, n_init_states: int, n_qubits: int, device, dtype=torch.complex64):
    dim = 1 << n_qubits
    real = torch.randn(batch_size, n_init_states, dim, device=device)
    imag = torch.randn(batch_size, n_init_states, dim, device=device)
    st = (real + 1j*imag).to(dtype)
    st /= (st.norm(dim=-1, keepdim=True) + 1e-12)
    return st

def precompute_fixed_multi_initial_states(dataset: Dataset, batch_size: int, n_qubits: int, n_init_states: int, device, ideal_model: TorchNoisyGates, seed: int = 0):
    with _timed("precompute:multi_init_all"):
        cpu_gen = torch.Generator(device='cpu').manual_seed(seed)
        dim = 1 << n_qubits
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_numeric)
        init_chunks = []
        base_chunks = []
        for batch in tqdm(loader, desc="PrecomputeFixedInitStates"):
            B = batch["A_ids"].shape[0]
            A = batch["A_ids"].to(device)
            Bq1 = batch["B_ids"].to(device)
            Bq2 = batch["C_ids"].to(device)
            with _timed("precompute:multi_init_sample", sync_cuda=False):
                real = torch.randn(B, n_init_states, dim, generator=cpu_gen)
                imag = torch.randn(B, n_init_states, dim, generator=cpu_gen)
                init_states = (real + 1j*imag).to(torch.complex64).to(device)
                init_states /= (init_states.norm(dim=-1, keepdim=True) + 1e-12)
            with _timed("precompute:multi_base_evolve"):
                base_evolved = simulate_base_only_multi(A, Bq1, Bq2, n_qubits, noise_model=ideal_model, initial_states=init_states)
            init_chunks.append(init_states.cpu())
            base_chunks.append(base_evolved.cpu())
        init_states_all = torch.cat(init_chunks, dim=0).to(device)
        base_ideal_all = torch.cat(base_chunks, dim=0).to(device)
        return init_states_all, base_ideal_all

def simulate_base_only_multi(base_gate_ids, base_q1, base_q2, n_qubits, noise_model: TorchNoisyGates|None, initial_states: torch.Tensor):
    with _timed("simulate:base_only_multi"):
        device = base_gate_ids.device
        B, L = base_gate_ids.shape
        B2, S, dim = initial_states.shape
        assert B == B2, "Batch size mismatch initial states vs gate ids"

        def apply_single_batch(states_s: torch.Tensor, U2: torch.Tensor, qubit: int):
            n = n_qubits
            outer = 1 << (n - qubit - 1)
            inner = 1 << qubit
            Slocal = states_s.shape[0]
            st = states_s.view(Slocal, outer, 2, inner)
            new = torch.einsum('ab,so bi->so ai', U2, st)
            return new.reshape(Slocal, -1)

        def apply_x_batch(states_s, q):
            X,H,Z = _get_const_gates(states_s.device)
            return apply_single_batch(states_s, X, q)
        def apply_h_batch(states_s, q):
            X,H,Z = _get_const_gates(states_s.device)
            return apply_single_batch(states_s, H, q)
        def apply_z_batch(states_s, q):
            X,H,Z = _get_const_gates(states_s.device)
            return apply_single_batch(states_s, Z, q)
        def apply_cx_batch(states_s, control, target):
            mask_indices, partner_indices = _get_cx_swap_tensors(states_s.device, n_qubits, control, target)
            if mask_indices.numel() == 0:
                return states_s
            out = states_s.clone()
            tmp = out[:, mask_indices].clone()
            out[:, mask_indices] = out[:, partner_indices]
            out[:, partner_indices] = tmp
            return out
        def apply_cz_batch(states_s, control, target):
            mask_indices = _get_cz_mask(states_s.device, n_qubits, control, target)
            if mask_indices.numel() == 0:
                return states_s
            out = states_s.clone()
            out[:, mask_indices] = -out[:, mask_indices]
            return out

        def apply_rx_batch(states_s, theta, q):
            c = torch.cos(theta/2)
            s = torch.sin(theta/2)
            U = torch.stack([
                torch.stack([c, -1j*s]),
                torch.stack([-1j*s, c])
            ]).to(states_s.dtype)
            return apply_single_batch(states_s, U, q)
        def apply_rz_batch(states_s, theta, q):
            e_m = torch.exp(-0.5j*theta)
            e_p = torch.exp( 0.5j*theta)
            zero = torch.zeros((), dtype=states_s.dtype, device=states_s.device)
            U = torch.stack([
                torch.stack([e_m, zero]),
                torch.stack([zero, e_p])
            ])
            return apply_single_batch(states_s, U, q)

        def apply_base_with_noise_batch(states_s, gate_id, q1, q2):
            if gate_id == gate_mapping['h']:
                states_s = apply_h_batch(states_s, q1); acted=[q1]
            elif gate_id == gate_mapping['x']:
                states_s = apply_x_batch(states_s, q1); acted=[q1]
            elif gate_id == gate_mapping['z']:
                states_s = apply_z_batch(states_s, q1); acted=[q1]
            elif gate_id == gate_mapping['cx'] and (q2 < n_qubits):
                states_s = apply_cx_batch(states_s, q1, q2); acted=[q1,q2]
            elif gate_id == gate_mapping['cz'] and (q2 < n_qubits):
                states_s = apply_cz_batch(states_s, q1, q2); acted=[q1,q2]
            else:
                return states_s
            if noise_model is not None and noise_model.has_noise() and acted:
                x_angles, z_angles = noise_model.sample_angles(len(acted), device=states_s.device)
                for i,qb in enumerate(acted):
                    states_s = apply_rx_batch(states_s, x_angles[i], qb)
                    states_s = apply_rz_batch(states_s, z_angles[i], qb)
            return states_s

        new_states = []
        for b in range(B):
            states_b = initial_states[b].clone()
            for i in range(L):
                gid = int(base_gate_ids[b,i])
                if gid == gate_mapping['<pad>']:
                    continue
                q1 = int(base_q1[b,i]); q2 = int(base_q2[b,i])
                states_b = apply_base_with_noise_batch(states_b, gid, q1, q2)
            states_b = states_b / (states_b.norm(dim=-1, keepdim=True)+1e-12)
            new_states.append(states_b)
        return torch.stack(new_states,0)

def simulate_interleaved_with_params_multi(base_gate_ids, base_q1, base_q2,
                                           pqc_qubits, pqc_types, pqc_angles,
                                           pqc_after_idx, pqc_count,
                                           n_qubits, noise_model: TorchNoisyGates,
                                           initial_states: torch.Tensor,
                                           assume_sorted_after: bool = True):
    with _timed("simulate:interleaved_params_multi"):
        device = base_gate_ids.device
        B, L = base_gate_ids.shape
        B2, S, dim = initial_states.shape
        assert B == B2, "Batch mismatch"

        def apply_single_batch(states_s: torch.Tensor, U2: torch.Tensor, qubit: int):
            outer = 1 << (n_qubits - qubit - 1)
            inner = 1 << qubit
            Slocal = states_s.shape[0]
            st = states_s.view(Slocal, outer, 2, inner)
            new = torch.einsum('ab,so bi->so ai', U2, st)
            return new.reshape(Slocal, -1)
        def apply_rx_batch(states_s, theta, q):
            c = torch.cos(theta/2); s = torch.sin(theta/2)
            U = torch.stack([torch.stack([c, -1j*s]), torch.stack([-1j*s, c])]).to(states_s.dtype)
            return apply_single_batch(states_s, U, q)
        def apply_rz_batch(states_s, theta, q):
            e_m = torch.exp(-0.5j*theta); e_p = torch.exp(0.5j*theta)
            zero = torch.zeros((), dtype=states_s.dtype, device=states_s.device)
            U = torch.stack([torch.stack([e_m, zero]), torch.stack([zero, e_p])])
            return apply_single_batch(states_s, U, q)
        def apply_h_batch(states_s, q):
            _,H,_ = _get_const_gates(states_s.device)
            return apply_single_batch(states_s, H, q)
        def apply_x_batch(states_s, q):
            X,_,_ = _get_const_gates(states_s.device)
            return apply_single_batch(states_s, X, q)
        def apply_z_batch(states_s, q):
            _,_,Z = _get_const_gates(states_s.device)
            return apply_single_batch(states_s, Z, q)
        def apply_cx_batch(states_s, control, target):
            mask_indices, partner_indices = _get_cx_swap_tensors(states_s.device, n_qubits, control, target)
            if mask_indices.numel()==0: return states_s
            out = states_s.clone()
            tmp = out[:, mask_indices].clone()
            out[:, mask_indices] = out[:, partner_indices]
            out[:, partner_indices] = tmp
            return out
        def apply_cz_batch(states_s, control, target):
            mask_indices = _get_cz_mask(states_s.device, n_qubits, control, target)
            if mask_indices.numel()==0: return states_s
            out = states_s.clone()
            out[:, mask_indices] = -out[:, mask_indices]
            return out
        def apply_base_gate_batch(states_s, gid, q1, q2):
            if gid == gate_mapping['h']:
                return apply_h_batch(states_s, q1), [q1]
            if gid == gate_mapping['x']:
                return apply_x_batch(states_s, q1), [q1]
            if gid == gate_mapping['z']:
                return apply_z_batch(states_s, q1), [q1]
            if gid == gate_mapping['cx'] and (q2 < n_qubits):
                return apply_cx_batch(states_s, q1, q2), [q1,q2]
            if gid == gate_mapping['cz'] and (q2 < n_qubits):
                return apply_cz_batch(states_s, q1, q2), [q1,q2]
            return states_s, []

        grad_mode = torch.is_grad_enabled()
        out_states = []
        for b in range(B):
            states_b = initial_states[b].clone() if grad_mode else initial_states[b]
            Kb = int(pqc_count[b])
            if Kb>0:
                if not assume_sorted_after:
                    valid = pqc_after_idx[b,:Kb]
                    sort_idx = torch.argsort(valid)
                    pqc_after_sorted = valid[sort_idx]
                    pqc_qubits_sorted = pqc_qubits[b,:Kb][sort_idx]
                    pqc_types_sorted  = pqc_types[b,:Kb][sort_idx]
                    pqc_angles_sorted = pqc_angles[b,:Kb][sort_idx]
                else:
                    pqc_after_sorted = pqc_after_idx[b,:Kb]
                    pqc_qubits_sorted = pqc_qubits[b,:Kb]
                    pqc_types_sorted  = pqc_types[b,:Kb]
                    pqc_angles_sorted = pqc_angles[b,:Kb]
            else:
                pqc_after_sorted = pqc_qubits_sorted = pqc_types_sorted = pqc_angles_sorted = None
            ptr = 0
            if Kb>0:
                while ptr < Kb and int(pqc_after_sorted[ptr]) == -1:
                    q = int(pqc_qubits_sorted[ptr]); t_idx = int(pqc_types_sorted[ptr]); theta = pqc_angles_sorted[ptr]
                    if t_idx == param_gate_mapping['rz']:
                        states_b = apply_rz_batch(states_b, theta, q)
                    else:
                        states_b = apply_rx_batch(states_b, theta, q)
                    ptr += 1
            for i in range(L):
                gid = int(base_gate_ids[b,i])
                if gid != gate_mapping['<pad>']:
                    q1 = int(base_q1[b,i]); q2 = int(base_q2[b,i])
                    states_b, acted = apply_base_gate_batch(states_b, gid, q1, q2)
                    if noise_model is not None and noise_model.has_noise() and acted:
                        x_angles, z_angles = noise_model.sample_angles(len(acted), device=states_b.device)
                        for ai,qb in enumerate(acted):
                            states_b = apply_rx_batch(states_b, x_angles[ai], qb)
                            states_b = apply_rz_batch(states_b, z_angles[ai], qb)
                while Kb>0 and ptr < Kb and int(pqc_after_sorted[ptr]) == i:
                    q = int(pqc_qubits_sorted[ptr]); t_idx = int(pqc_types_sorted[ptr]); theta = pqc_angles_sorted[ptr]
                    if t_idx == param_gate_mapping['rz']:
                        states_b = apply_rz_batch(states_b, theta, q)
                    else:
                        states_b = apply_rx_batch(states_b, theta, q)
                    ptr += 1
            states_b = states_b / (states_b.norm(dim=-1, keepdim=True)+1e-12)
            out_states.append(states_b)
        return torch.stack(out_states,0)

def fidelity_loss_interleaved_multi(pred_sincos,
                                    base_gate_ids, base_q1, base_q2,
                                    pqc_qubits, pqc_types, pqc_after_idx, pqc_count,
                                    n_qubits,
                                    noisy_model: TorchNoisyGates,
                                    ideal_model: TorchNoisyGates,
                                    initial_states: torch.Tensor,
                                    tgt_angles=None,
                                    precomputed_base_ideal: torch.Tensor | None = None):
    with _timed("loss:fidelity_multi"):
        pred_angles = sin_cos_to_angles(pred_sincos)  # [B,K_MAX]
        if precomputed_base_ideal is not None:
            psi_base_ideal = precomputed_base_ideal
        else:
            if USE_VECTORIZED_BASE:
                psi_base_ideal = simulate_base_only_vectorized(base_gate_ids, base_q1, base_q2,
                                                               n_qubits, noise_model=ideal_model,
                                                               initial_states=initial_states)
            else:
                psi_base_ideal = simulate_base_only_multi(base_gate_ids, base_q1, base_q2,
                                                          n_qubits, noise_model=ideal_model,
                                                          initial_states=initial_states)
        psi_measured = simulate_interleaved_with_params_multi(base_gate_ids, base_q1, base_q2,
                                                              pqc_qubits, pqc_types, pred_angles,
                                                              pqc_after_idx, pqc_count,
                                                              n_qubits, noise_model=noisy_model,
                                                              initial_states=initial_states)
        ov = torch.sum(torch.conj(psi_base_ideal) * psi_measured, dim=-1)  # [B,S]
        F = (ov.abs() ** 2)  # [B,S]
        F_sample = F.mean(dim=1)  # average over S per sample
        loss = (1 - F_sample).mean()
        F_gt_mean = None
        if tgt_angles is not None:
            with torch.no_grad():
                psi_gt = simulate_interleaved_with_params_multi(base_gate_ids, base_q1, base_q2,
                                                                pqc_qubits, pqc_types, tgt_angles,
                                                                pqc_after_idx, pqc_count,
                                                                n_qubits, noise_model=noisy_model,
                                                                initial_states=initial_states)
                ov_gt = torch.sum(torch.conj(psi_base_ideal) * psi_gt, dim=-1)
                F_gt = (ov_gt.abs() ** 2)  # [B,S]
                F_gt_mean = F_gt.mean()
        return loss, F.mean(), F_gt_mean

def angular_errors(pred_sincos, tgt_angles, pqc_count):
    B,K,_ = pred_sincos.shape
    tgt_sc = angles_to_sin_cos(tgt_angles)
    pred = pred_sincos / (pred_sincos.norm(dim=-1,keepdim=True)+1e-8)
    tgt  = tgt_sc      / (tgt_sc.norm(dim=-1,keepdim=True)+1e-8)
    sim = (pred*tgt).sum(-1)  # [B,K]
    mask = torch.arange(K, device=pqc_count.device).unsqueeze(0) < pqc_count.unsqueeze(1)
    ang = torch.acos(sim.clamp(-1,1))
    return (ang*mask).sum()/mask.sum().clamp_min(1)

# ---------------- Training / Evaluation ----------------
def train_one_epoch(model, loader, opt, device,
                    max_grad_norm=1.0, scaler=None, use_amp=True,
                    noisy_model: TorchNoisyGates|None=None,
                    ideal_model: TorchNoisyGates|None=None,
                    base_ideal_cache: torch.Tensor|None=None,
                    n_init_states: int = 1,
                    fixed_multi_init: torch.Tensor | None = None,
                    fixed_multi_base_ideal: torch.Tensor | None = None):
    model.train()
    tot=0; n=0
    for batch in tqdm(loader,desc="Training",leave=False):
        idx = batch["idx"].to(device)
        A = batch["A_ids"].to(device)
        B = batch["B_ids"].to(device)
        C = batch["C_ids"].to(device)
        tgt = batch["tgt_angles"].to(device)
        pq  = batch["param_qubits"].to(device)
        pt  = batch["param_types"].to(device)
        ai  = batch["after_idx"].to(device)
        kc  = batch["pqc_count"].to(device)
        opt.zero_grad(set_to_none=True)
        use_autocast = (scaler is not None) and use_amp and device.type=="cuda"
        with _timed("train:batch:forward"):
            with torch.cuda.amp.autocast(enabled=use_autocast):
                with _timed("train:encoder_forward"):
                    pred_sc = model(A,B,C)
                with _timed("train:simulate+loss"):
                    if n_init_states == 1:
                        cached = base_ideal_cache[idx] if base_ideal_cache is not None else None
                        loss, _, _ = fidelity_loss_interleaved(pred_sc, A,B,C, pq,pt,ai,kc, NQ1,
                                                               noisy_model=noisy_model,
                                                               ideal_model=ideal_model,
                                                               tgt_angles=None,
                                                               psi_base_ideal_cached=cached)
                    else:
                        assert fixed_multi_init is not None and fixed_multi_base_ideal is not None, "Fixed multi initial states not precomputed"
                        init_states = fixed_multi_init[idx]
                        base_ideal_states = fixed_multi_base_ideal[idx]
                        loss, _, _ = fidelity_loss_interleaved_multi(pred_sc, A,B,C, pq,pt,ai,kc, NQ1,
                                                                     noisy_model=noisy_model,
                                                                     ideal_model=ideal_model,
                                                                     initial_states=init_states,
                                                                     tgt_angles=None,
                                                                     precomputed_base_ideal=base_ideal_states)
        with _timed("train:batch:backward+step"):
            if scaler is not None and use_autocast:
                scaler.scale(loss).backward()
                if max_grad_norm:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
        tot += loss.item(); n += 1
    return tot/max(1,n)

@torch.no_grad()
def evaluate(model, loader, device, use_amp=True,
             noisy_model: TorchNoisyGates|None=None,
             ideal_model: TorchNoisyGates|None=None,
             base_ideal_cache: torch.Tensor|None=None,
             n_init_states: int = 1,
             fixed_multi_init: torch.Tensor | None = None,
             fixed_multi_base_ideal: torch.Tensor | None = None):
    model.eval()
    tot=0;n=0
    for batch in tqdm(loader,desc="Evaluating",leave=False):
        idx = batch["idx"].to(device)
        A = batch["A_ids"].to(device)
        B = batch["B_ids"].to(device)
        C = batch["C_ids"].to(device)
        tgt = batch["tgt_angles"].to(device)
        pq  = batch["param_qubits"].to(device)
        pt  = batch["param_types"].to(device)
        ai  = batch["after_idx"].to(device)
        kc  = batch["pqc_count"].to(device)
        use_autocast = (device.type=="cuda") and use_amp
        with _timed("eval:batch:forward+loss"):
            with torch.cuda.amp.autocast(enabled=use_autocast):
                pred_sc = model(A,B,C)
                if n_init_states == 1:
                    cached = base_ideal_cache[idx] if base_ideal_cache is not None else None
                    loss, Fm, Fgt = fidelity_loss_interleaved(pred_sc, A,B,C, pq,pt,ai,kc, NQ1,
                                                              noisy_model=noisy_model,
                                                              ideal_model=ideal_model,
                                                              tgt_angles=tgt,
                                                              psi_base_ideal_cached=cached)
                else:
                    assert fixed_multi_init is not None and fixed_multi_base_ideal is not None, "Fixed multi initial states not precomputed"
                    init_states = fixed_multi_init[idx]
                    base_ideal_states = fixed_multi_base_ideal[idx]
                    loss, Fm, Fgt = fidelity_loss_interleaved_multi(pred_sc, A,B,C, pq,pt,ai,kc, NQ1,
                                                                    noisy_model=noisy_model,
                                                                    ideal_model=ideal_model,
                                                                    initial_states=init_states,
                                                                    tgt_angles=tgt,
                                                                    precomputed_base_ideal=base_ideal_states)
        tot += loss.item(); n += 1
    if n==0: return float('nan')
    return tot/n

# ---------------- Final fidelity benchmark ----------------
@torch.no_grad()
def evaluate_fidelity_benchmark(model, loader, device,
                                noisy_model: TorchNoisyGates,
                                ideal_model: TorchNoisyGates,
                                base_ideal_cache: torch.Tensor|None=None,
                                n_init_states: int = 1,
                                fixed_multi_init: torch.Tensor | None = None,
                                fixed_multi_base_ideal: torch.Tensor | None = None):
    model.eval()
    Fm=[]; Fg=[]
    for batch in tqdm(loader,desc="FidBenchmark",leave=False):
        idx = batch["idx"].to(device)
        A = batch["A_ids"].to(device)
        B = batch["B_ids"].to(device)
        C = batch["C_ids"].to(device)
        tgt = batch["tgt_angles"].to(device)
        pq  = batch["param_qubits"].to(device)
        pt  = batch["param_types"].to(device)
        ai  = batch["after_idx"].to(device)
        kc  = batch["pqc_count"].to(device)

        with _timed("bench:batch"):
            pred_sc = model(A,B,C)
            if n_init_states == 1:
                pred_angles = sin_cos_to_angles(pred_sc)
                if base_ideal_cache is not None:
                    psi_base_ideal = base_ideal_cache[idx]
                else:
                    psi_base_ideal = simulate_base_only(A,B,C,NQ1,ideal_model)
                psi_pred = simulate_interleaved_with_params(A,B,C, pq,pt,pred_angles, ai,kc, NQ1, noisy_model, reuse_buffer=True)
                ov_pred = torch.sum(torch.conj(psi_base_ideal)*psi_pred, dim=-1)
                F_pred = (ov_pred.abs()**2)
                psi_gt = simulate_interleaved_with_params(A,B,C, pq,pt,tgt, ai,kc, NQ1, noisy_model, reuse_buffer=True)
                ov_gt = torch.sum(torch.conj(psi_base_ideal)*psi_gt, dim=-1)
                F_gt = (ov_gt.abs()**2)
                Fm.append(F_pred.mean().item())
                Fg.append(F_gt.mean().item())
            else:
                assert fixed_multi_init is not None and fixed_multi_base_ideal is not None, "Fixed multi initial states not precomputed"
                pred_angles = sin_cos_to_angles(pred_sc)
                init_states = fixed_multi_init[idx]
                psi_base_ideal = fixed_multi_base_ideal[idx]
                psi_pred = simulate_interleaved_with_params_multi(A,B,C, pq,pt,pred_angles, ai,kc, NQ1, noisy_model, init_states)
                ov_pred = torch.sum(torch.conj(psi_base_ideal) * psi_pred, dim=-1)  # [B,S]
                F_pred = (ov_pred.abs() ** 2).mean().item()
                psi_gt = simulate_interleaved_with_params_multi(A,B,C, pq,pt,tgt, ai,kc, NQ1, noisy_model, init_states)
                ov_gt = torch.sum(torch.conj(psi_base_ideal) * psi_gt, dim=-1)
                F_gt = (ov_gt.abs() ** 2).mean().item()
                Fm.append(F_pred); Fg.append(F_gt)
    if not Fm:
        return float('nan'), float('nan')
    return sum(Fm)/len(Fm), sum(Fg)/len(Fg)

# ---------------- Scheduler ----------------
def build_scheduler(optimizer, epochs, warmup_epochs=5, min_lr_ratio=0.1, schedule='cosine'):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep+1)/float(max(1,warmup_epochs))
        prog = (ep-warmup_epochs)/max(1,epochs-warmup_epochs)
        prog = min(max(prog,0.0),1.0)
        if schedule=='cosine':
            return min_lr_ratio + 0.5*(1-min_lr_ratio)*(1+math.cos(math.pi*prog))
        if schedule=='linear':
            return (1-prog)*(1-min_lr_ratio)+min_lr_ratio
        return 1.0
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

# ---------------- Main ----------------
def precompute_base_ideal_states(dataset: Dataset, batch_size: int, device, n_qubits: int, ideal_model: TorchNoisyGates):
    with _timed("precompute:base_ideal_all"):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_numeric)
        states_list=[]
        for batch in tqdm(loader, desc="PrecomputeBaseIdeal"):
            A = batch["A_ids"].to(device)
            B = batch["B_ids"].to(device)
            C = batch["C_ids"].to(device)
            with torch.no_grad():
                with _timed("precompute:base_ideal_batch"):
                    st = simulate_base_only(A,B,C,n_qubits, noise_model=ideal_model)
            states_list.append(st.cpu())
        base_states = torch.cat(states_list, dim=0)
        return base_states.to(device)

def main_training_loop(tokenized_data_x, extracted_sched,
                       batch_size=32, epochs=20, lr=1e-5, val_ratio=0.1,
                       use_amp=True, save_path="best_model.pt",
                       use_scheduler=False, warmup_epochs=5, min_lr_ratio=0.1,
                       schedule='cosine',
                       noise_x_rad=math.pi/30, noise_z_rad=math.pi/30,
                       noise_delta_x=0.0, noise_delta_z=0.0, seed=0,
                       run_final_fidelity=True,
                       cache_base_ideal: bool = True,
                       n_init_states: int = 1):
    global ENABLE_COMPILE
    with _timed("setup:data_splits", sync_cuda=False):
        full = MyNumericSeq2SeqDataset(tokenized_data_x, extracted_sched)
        n_total=len(full)
        n_val=max(1,int(n_total*val_ratio))
        n_train=n_total-n_val
        train_subset, val_subset = random_split(full,[n_train,n_val],
            generator=torch.Generator().manual_seed(42))
        train_dataset=Subset(full, train_subset.indices)
        val_dataset  =Subset(full, val_subset.indices)
        dl_kwargs = dict(collate_fn=collate_fn_numeric, pin_memory=torch.cuda.is_available())
        train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True, **dl_kwargs)
        val_loader  =DataLoader(val_dataset,batch_size=batch_size,shuffle=False, **dl_kwargs)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with _timed("setup:model"):
        model=EncoderWithQuerySlotsPacked(
            d_model=768,nhead=12,ff=4096,enc_layers=8,
            max_src_len=SRC_MAX_LEN,tgt_len=K_MAX,dropout=0.1).to(device)

    if ENABLE_COMPILE and hasattr(torch, 'compile'):
        is_windows = (os.name == 'nt')
        triton_ok = True
        try:
            import triton  # noqa: F401
        except Exception:
            triton_ok = False
        if is_windows and not triton_ok:
            print("[INFO] Detected Windows without Triton; skipping torch.compile.")
            ENABLE_COMPILE = False
        if ENABLE_COMPILE:
            with _timed("setup:torch.compile"):
                try:
                    model = torch.compile(model, mode="max-autotune")
                    print("[INFO] torch.compile enabled for model")
                    try:
                        with torch.no_grad():
                            fake_A = torch.zeros(1, SRC_MAX_LEN, dtype=torch.long, device=device)
                            fake_B = torch.zeros(1, SRC_MAX_LEN, dtype=torch.long, device=device)
                            fake_C = torch.zeros(1, SRC_MAX_LEN, dtype=torch.long, device=device)
                            _ = model(fake_A, fake_B, fake_C)
                    except Exception as fe:
                        print("[WARN] torch.compile forward dry-run failed, reverting to eager: ")
                        print(f"       {fe}")
                        model=EncoderWithQuerySlotsPacked(
                            d_model=768,nhead=12,ff=4096,enc_layers=8,
                            max_src_len=SRC_MAX_LEN,tgt_len=K_MAX,dropout=0.1).to(device)
                        ENABLE_COMPILE = False
                except Exception as e:
                    print("[WARN] torch.compile disabled (falling back to eager). Reason:")
                    print(f"       {e}")
                    ENABLE_COMPILE = False

    with _timed("setup:optim_sched"):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = build_scheduler(opt, epochs, warmup_epochs, min_lr_ratio, schedule) if use_scheduler else None
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type=="cuda")

    with _timed("setup:noise_models", sync_cuda=False):
        ideal_model = TorchNoisyGates(x_rad=0, z_rad=0, delta_x=0, delta_z=0, seed=seed)
        noisy_model = TorchNoisyGates(x_rad=noise_x_rad, z_rad=noise_z_rad,
                                      delta_x=noise_delta_x, delta_z=noise_delta_z, seed=seed+1)

    base_ideal_cache = None
    fixed_multi_init = None
    fixed_multi_base_ideal = None
    if cache_base_ideal and n_init_states == 1:
        print("Precomputing base ideal states (once)...")
        base_ideal_cache = precompute_base_ideal_states(full, batch_size=batch_size, device=device, n_qubits=NQ1, ideal_model=ideal_model)
    elif n_init_states > 1:
        print(f"Precomputing fixed {n_init_states} initial states and their base ideal evolutions...")
        fixed_multi_init, fixed_multi_base_ideal = precompute_fixed_multi_initial_states(full, batch_size=batch_size,
                                                                                        n_qubits=NQ1, n_init_states=n_init_states,
                                                                                        device=device, ideal_model=ideal_model, seed=seed+999)

    best_val=float('inf'); best_epoch=0
    for ep in range(1,epochs+1):
        with _timed("epoch:train"):
            tr_loss = train_one_epoch(model, train_loader, opt, device,
                                      scaler=scaler, use_amp=use_amp,
                                      noisy_model=noisy_model, ideal_model=ideal_model,
                                      base_ideal_cache=base_ideal_cache,
                                      n_init_states=n_init_states,
                                      fixed_multi_init=fixed_multi_init,
                                      fixed_multi_base_ideal=fixed_multi_base_ideal)
        with _timed("epoch:eval"):
            val_loss = evaluate(model, val_loader, device, use_amp=use_amp,
                                noisy_model=noisy_model, ideal_model=ideal_model,
                                base_ideal_cache=base_ideal_cache,
                                n_init_states=n_init_states,
                                fixed_multi_init=fixed_multi_init,
                                fixed_multi_base_ideal=fixed_multi_base_ideal)
        if scheduler:
            with _timed("epoch:scheduler_step", sync_cuda=False):
                scheduler.step()
        print(f"[Epoch {ep:02d}] train={tr_loss:.4f} val={val_loss:.4f}")
        if PROFILE_VERBOSE:
            print_timing_report(header=f"=== Timing (end of epoch {ep}) ===", top_k=20)
        if val_loss < best_val:
            best_val=val_loss; best_epoch=ep
            with _timed("epoch:checkpoint_io", sync_cuda=False):
                torch.save({"model":model.state_dict(),"val_loss":val_loss,"epoch":ep}, save_path)
            print(f"  -> saved best (val={val_loss:.4f})")

    print(f"Best val: {best_val:.4f} @ epoch {best_epoch}")

    if run_final_fidelity:
        with _timed("final:load_ckpt", sync_cuda=False):
            ckpt = torch.load(save_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print("Loaded best checkpoint for final fidelity benchmark.")
        with _timed("final:fid_benchmark"):
            full_loader = DataLoader(full, batch_size=batch_size, shuffle=False,
                                     collate_fn=collate_fn_numeric)
            F_model, F_gt = evaluate_fidelity_benchmark(model, full_loader, device,
                                                        noisy_model=noisy_model,
                                                        ideal_model=ideal_model,
                                                        base_ideal_cache=base_ideal_cache,
                                                        n_init_states=n_init_states,
                                                        fixed_multi_init=fixed_multi_init,
                                                        fixed_multi_base_ideal=fixed_multi_base_ideal)
            print(f"[Final Fidelity Benchmark] mean_F_model={F_model:.6f}  mean_F_ground_truth={F_gt:.6f}")

    # Print final global timing table
    print_timing_report(header="=== Timing Report (Final) ===")
    return model

if __name__ == "__main__":
    # Example: to use multiple random initial states per sample for robustness, set n_init_states>1
    # model = main_training_loop(tokenized_data_x, extracted_sched, n_init_states=8, ...)
    model = main_training_loop(tokenized_data_x, extracted_sched,
                               batch_size=1, epochs=3, lr=1e-5,
                               use_scheduler=False,
                               noise_x_rad=NOISE_X_RAD_NUM, noise_z_rad=NOISE_Z_RAD_NUM,
                               noise_delta_x=NOISE_DELTA_X_NUM, noise_delta_z=NOISE_DELTA_Z_NUM if 'NOISE_DELTA_Z_NUM' in globals() else 0.05,  # guard
                               seed=0,
                               run_final_fidelity=True, save_path="best_model.pt",
                               n_init_states=100)
