# filepath: a:\wings\transformers\main_unit_current_best_0908_fid_multi_pqc.py
import math, random, torch, torch.nn as nn
from utils import read_all_json_files
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm
import os
import json

# ---------------- Config / Data ----------------
data_file_path = "A:/wings/transformers/data/10q_100g_10blk_data/10q_100g_10blk_data/good_fidelity"
train_num = 2500
NOISE_X_RAD_NUM = math.pi/100
NOISE_Z_RAD_NUM = math.pi/100
NOISE_DELTA_X_NUM = 0.05
NOISE_DELTA_Z_NUM = 0.05

def read_all_json_files(folder_path):
    json_data = []
    for filename in tqdm(os.listdir(folder_path), desc="Reading JSON files"):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                json_data.append(data)
    return json_data


# Base gates + PAD (extended to include 'cz' and 'z')
# NOTE: Original indices for existing gates kept the same to preserve compatibility if retraining from scratch.
# Adding new gates appends new indices; existing checkpoints with old vocab size will NOT load directly.
gate_mapping = {"cx":0, "h":1, "x":2, "cz":3, "z":4, "<pad>":5}
NGATE = len(gate_mapping)
NQ1   = 10
NQ2   = NQ1 + 1    # padding for 2nd qubit

PARAM_GATE_TYPES = ['rz','rx']
param_gate_mapping = {g:i for i,g in enumerate(PARAM_GATE_TYPES)}

BASE_SET = {'cx','h','x','cz','z'}
PQC_SET  = {'rz','rx'}

# ---- Load ----
raw = read_all_json_files(data_file_path)
# Simply split here for quick test
raw = raw[:train_num]


# ---- Compute SRC_MAX_LEN and K_MAX, and build token lists ----
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
    out=[]
    for line in lines:
        d={"gate_id":[],"qubit_1":[],"qubit_2":[]}
        for g in line:
            # base lines contain only base gates
            d["gate_id"].append(gate_mapping[g[0]])
            d["qubit_1"].append(g[1][0])
            # pad second qubit if single-qubit gate
            if len(g[1]) == 1:
                d["qubit_2"].append(-1)
            else:
                d["qubit_2"].append(g[1][1])
        # pad to SRC_MAX_LEN
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
    res=[]
    for line in lines_full:
        qubits=[]; params=[]; types=[]; after_idx=[]
        base_idx = -1  # -1 means before any base gate
        for g in line:
            name = g[0]
            if name in base_set:
                base_idx += 1
            elif name in pqc_set:
                # g: [name, [q], [theta]]
                q = g[1][0]
                th = g[2][0] if len(g) > 2 and len(g[2]) > 0 else 0.0
                qubits.append(q)
                params.append(th)
                types.append(name)
                after_idx.append(base_idx)
        cnt = len(params)
        # pad to K_MAX
        pad_n = K_MAX - cnt
        if pad_n > 0:
            qubits.extend([0]*pad_n)
            params.extend([0.0]*pad_n)
            types.extend(['rz']*pad_n)
            after_idx.extend([-999]*pad_n)  # sentinel, ignored by count
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
        self.q1_emb   = nn.Embedding(NQ1,  d_model)  # no explicit pad idx
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
        x = torch.cat([self.gate_emb(A), self.q1_emb(B), self.q2_emb(C)], dim=-1)
        x = self.proj(x)
        enc = self.encoder(x)
        idx = torch.arange(self.K, device=A.device).unsqueeze(0).expand(A.size(0),-1)
        q = self.block_emb(idx)
        q2,_ = self.cross_attn(q, enc, enc)
        q2 = self.ff(q2)
        return self.out(q2)  # [B,K,2]

# ---------------- Dataset ----------------
class MyNumericSeq2SeqDataset(Dataset):
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
    # mask only valid K
    B, K, _ = pred_sc.shape
    pred = pred_sc / (pred_sc.norm(dim=-1,keepdim=True)+1e-8)
    tgt_sc= angles_to_sin_cos(tgt_angles)
    tgt  = tgt_sc / (tgt_sc.norm(dim=-1,keepdim=True)+1e-8)
    sim = (pred*tgt).sum(-1)  # [B,K]
    loss_per = 1 - sim
    mask = torch.arange(K, device=pqc_count.device).unsqueeze(0) < pqc_count.unsqueeze(1)
    loss = (loss_per*mask).sum() / mask.sum().clamp_min(1)
    return loss

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
    # Projector onto control qubit = 1
    proj_ops = []
    for q in range(n_qubits):
        if q == control:
            proj_ops.append(torch.tensor([[0,0],[0,1]], dtype=torch.complex64, device=device))
        else:
            proj_ops.append(torch.eye(2, dtype=torch.complex64, device=device))
    P = proj_ops[0]
    for i in range(1, n_qubits):
        P = torch.kron(P, proj_ops[i])
    # Z on target
    Zsmall = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64, device=device)
    z_ops = []
    for q in range(n_qubits):
        z_ops.append(Zsmall if q == target else torch.eye(2, dtype=torch.complex64, device=device))
    Zfull = z_ops[0]
    for i in range(1, n_qubits):
        Zfull = torch.kron(Zfull, z_ops[i])
    # Controlled-Z: (I - P) + Z P
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
    """Fast apply a 2x2 single-qubit unitary U2 to |state> on the given qubit using view+einsum.
    state: [2^n] complex64 vector, U2: [2,2] complex64.
    """
    dim = state.numel()
    outer = 1 << (n_qubits - qubit - 1)
    inner = 1 << qubit
    st = state.view(outer, 2, inner)
    # new[o, a, i] = sum_b U[a,b] * st[o, b, i]
    new = torch.einsum('ab,obi->oai', U2, st)
    return new.reshape(dim)

def apply_rx_state(state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int):
    # Build small 2x2 from theta (keeps grad) and use vectorized kernel
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

def apply_x_state(state: torch.Tensor, qubit: int, n_qubits: int):
    U = torch.tensor([[0,1],[1,0]], dtype=state.dtype, device=state.device)
    return _apply_single_qubit_gate_on_state(state, U, qubit, n_qubits)

def apply_h_state(state: torch.Tensor, qubit: int, n_qubits: int):
    s = 1/math.sqrt(2)
    U = torch.tensor([[s,s],[s,-s]], dtype=state.dtype, device=state.device)
    return _apply_single_qubit_gate_on_state(state, U, qubit, n_qubits)

def apply_z_state(state: torch.Tensor, qubit: int, n_qubits: int):
    U = torch.tensor([[1,0],[0,-1]], dtype=state.dtype, device=state.device)
    return _apply_single_qubit_gate_on_state(state, U, qubit, n_qubits)

def apply_cx_state(state: torch.Tensor, control: int, target: int, n_qubits: int):
    # Vectorized swap of target bit when control bit is 1
    src = state
    dim = src.numel()
    idx = torch.arange(dim, device=src.device)
    mc = 1 << control
    mt = 1 << target
    ctrl_one = (idx & mc) != 0
    tgt_zero = (idx & mt) == 0
    mask = ctrl_one & tgt_zero
    pair_idx = idx ^ mt
    out = src.clone()
    base_vals = src[mask]
    pair_vals = src[pair_idx[mask]]
    out[mask] = pair_vals
    out[pair_idx[mask]] = base_vals
    return out

def apply_cz_state(state: torch.Tensor, control: int, target: int, n_qubits: int):
    # Vectorized phase flip on |control=1, target=1>
    src = state
    dim = src.numel()
    idx = torch.arange(dim, device=src.device)
    mc = 1 << control
    mt = 1 << target
    mask = ((idx & mc) != 0) & ((idx & mt) != 0)
    out = src.clone()
    out[mask] = -src[mask]
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
def simulate_base_only(base_gate_ids, base_q1, base_q2, n_qubits, noise_model: TorchNoisyGates|None):
    device = base_gate_ids.device
    B,L = base_gate_ids.shape
    dim = 1<<n_qubits
    states = torch.zeros(B,dim,dtype=torch.complex64,device=device)
    states[:,0]=1+0j
    for i in range(L):
        g  = base_gate_ids[:,i]
        q1 = base_q1[:,i]
        q2 = base_q2[:,i]
        new=[]
        for b in range(B):
            st = states[b]
            st = apply_base_gate_with_noise(
                st,
                int(g[b]),
                int(q1[b]),
                int(q2[b]),
                n_qubits,
                device,
                noise_model=noise_model
            )
            new.append(st)
        states = torch.stack(new,0)
    states = states / (states.norm(dim=-1,keepdim=True)+1e-12)
    return states

def simulate_interleaved_with_params(base_gate_ids, base_q1, base_q2,
                                     pqc_qubits, pqc_types, pqc_angles,
                                     pqc_after_idx, pqc_count,
                                     n_qubits, noise_model: TorchNoisyGates):
    """
    Run base gates with noise, and interleave K_b PQC gates at their after_idx positions, in order k=0..K_b-1.
    """
    device = base_gate_ids.device
    B,L = base_gate_ids.shape
    dim = 1<<n_qubits
    states = torch.zeros(B,dim,dtype=torch.complex64,device=device)
    states[:,0]=1+0j

    def apply_param(st, q, t_idx, theta):
        if t_idx == param_gate_mapping['rz']:
            return apply_rz_state(st, theta, int(q), n_qubits)
        else:
            return apply_rx_state(st, theta, int(q), n_qubits)

    new_states=[]
    for b in range(B):
        st = states[b]
        Kb = int(pqc_count[b])
        # before first base gate (after_idx == -1)
        for k in range(Kb):
            if int(pqc_after_idx[b,k]) == -1:
                st = apply_param(st, int(pqc_qubits[b,k]), int(pqc_types[b,k]), pqc_angles[b,k])

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
            # interleave PQC scheduled after this base index
            for k in range(Kb):
                if int(pqc_after_idx[b,k]) == i:
                    st = apply_param(st, int(pqc_qubits[b,k]), int(pqc_types[b,k]), pqc_angles[b,k])
        new_states.append(st)
    states = torch.stack(new_states,0)
    states = states / (states.norm(dim=-1,keepdim=True)+1e-12)
    return states

# ---------------- Fidelity loss (JAX-aligned) ----------------
def fidelity_loss_interleaved(pred_sincos,
                              base_gate_ids, base_q1, base_q2,
                              pqc_qubits, pqc_types, pqc_after_idx, pqc_count,
                              n_qubits,
                              noisy_model: TorchNoisyGates,
                              ideal_model: TorchNoisyGates,
                              tgt_angles=None):
    """
    Compare:
      psi_base_ideal = Base(no-noise), no PQC
      psi_measured   = Base(noise) + interleaved PQC (pred params)
    Optionally returns fidelity vs GT as监控（当提供 tgt_angles 时）。
    """
    pred_angles = sin_cos_to_angles(pred_sincos)  # [B,K_MAX]
    with torch.no_grad():
        psi_base_ideal = simulate_base_only(base_gate_ids, base_q1, base_q2,
                                            n_qubits, noise_model=ideal_model)
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

def angular_errors(pred_sincos, tgt_angles, pqc_count):
    B,K,_ = pred_sincos.shape
    tgt_sc = angles_to_sin_cos(tgt_angles)
    pred = pred_sincos / (pred_sincos.norm(dim=-1,keepdim=True)+1e-8)
    tgt  = tgt_sc      / (tgt_sc.norm(dim=-1,keepdim=True)+1e-8)
    sim = (pred*tgt).sum(-1)  # [B,K]
    mask = torch.arange(K, device=pqc_count.device).unsqueeze(0) < pqc_count.unsqueeze(1)
    ang = torch.acos(sim.clamp(-1,1))
    # masked mean per-batch
    return (ang*mask).sum()/mask.sum().clamp_min(1)

# ---------------- Training / Evaluation ----------------
def train_one_epoch(model, loader, opt, device,
                    max_grad_norm=1.0, scaler=None, use_amp=True,
                    noisy_model: TorchNoisyGates|None=None,
                    ideal_model: TorchNoisyGates|None=None):
    model.train()
    tot=0; n=0
    for batch in tqdm(loader,desc="Training",leave=False):
        A=batch["A_ids"].to(device)
        B=batch["B_ids"].to(device)
        C=batch["C_ids"].to(device)
        tgt=batch["tgt_angles"].to(device)
        pq=batch["param_qubits"].to(device)
        pt=batch["param_types"].to(device)
        ai=batch["after_idx"].to(device)
        kc=batch["pqc_count"].to(device)
        opt.zero_grad(set_to_none=True)
        use_autocast = (scaler is not None) and use_amp and device.type=="cuda"
        with torch.cuda.amp.autocast(enabled=use_autocast):
            pred_sc = model(A,B,C)
            loss, _, _ = fidelity_loss_interleaved(pred_sc, A,B,C, pq,pt,ai,kc, NQ1,
                                                   noisy_model=noisy_model,
                                                   ideal_model=ideal_model,
                                                   tgt_angles=None)
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
        tot+=loss.item(); n+=1
    return tot/max(1,n)

@torch.no_grad()
def evaluate(model, loader, device, use_amp=True,
             noisy_model: TorchNoisyGates|None=None,
             ideal_model: TorchNoisyGates|None=None):
    model.eval()
    tot=0;n=0
    for batch in tqdm(loader,desc="Evaluating",leave=False):
        A=batch["A_ids"].to(device)
        B=batch["B_ids"].to(device)
        C=batch["C_ids"].to(device)
        tgt=batch["tgt_angles"].to(device)
        pq=batch["param_qubits"].to(device)
        pt=batch["param_types"].to(device)
        ai=batch["after_idx"].to(device)
        kc=batch["pqc_count"].to(device)
        use_autocast = (device.type=="cuda") and use_amp
        with torch.cuda.amp.autocast(enabled=use_autocast):
            pred_sc = model(A,B,C)
            loss, Fm, Fgt = fidelity_loss_interleaved(pred_sc, A,B,C, pq,pt,ai,kc, NQ1,
                                                      noisy_model=noisy_model,
                                                      ideal_model=ideal_model,
                                                      tgt_angles=tgt)
        tot+=loss.item(); n+=1
    if n==0: return float('nan')
    return tot/n

# ---------------- Final fidelity benchmark ----------------
@torch.no_grad()
def evaluate_fidelity_benchmark(model, loader, device,
                                noisy_model: TorchNoisyGates,
                                ideal_model: TorchNoisyGates):
    model.eval()
    Fm=[]; Fg=[]
    for batch in tqdm(loader,desc="FidBenchmark",leave=False):
        A=batch["A_ids"].to(device)
        B=batch["B_ids"].to(device)
        C=batch["C_ids"].to(device)
        tgt=batch["tgt_angles"].to(device)
        pq=batch["param_qubits"].to(device)
        pt=batch["param_types"].to(device)
        ai=batch["after_idx"].to(device)
        kc=batch["pqc_count"].to(device)

        pred_sc = model(A,B,C)
        pred_angles = sin_cos_to_angles(pred_sc)

        psi_base_ideal = simulate_base_only(A,B,C,NQ1,ideal_model)
        psi_base_noisy = simulate_base_only(A,B,C,NQ1,noisy_model)

        psi_pred = simulate_interleaved_with_params(A,B,C, pq,pt,pred_angles, ai,kc, NQ1, noisy_model)
        ov_pred = torch.sum(torch.conj(psi_base_ideal)*psi_pred, dim=-1)
        F_pred = (ov_pred.abs()**2)

        psi_gt = simulate_interleaved_with_params(A,B,C, pq,pt,tgt, ai,kc, NQ1, noisy_model)
        ov_gt = torch.sum(torch.conj(psi_base_ideal)*psi_gt, dim=-1)
        F_gt = (ov_gt.abs()**2)

        Fm.append(F_pred.mean().item())
        Fg.append(F_gt.mean().item())
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
def main_training_loop(tokenized_data_x, extracted_sched,
                       batch_size=32, epochs=20, lr=1e-5, val_ratio=0.1,
                       use_amp=True, save_path="best_model.pt",
                       use_scheduler=False, warmup_epochs=5, min_lr_ratio=0.1,
                       schedule='cosine',
                       noise_x_rad=math.pi/30, noise_z_rad=math.pi/30,
                       noise_delta_x=0.0, noise_delta_z=0.0, seed=0,
                       run_final_fidelity=True):
    full = MyNumericSeq2SeqDataset(tokenized_data_x, extracted_sched)
    n_total=len(full)
    n_val=max(1,int(n_total*val_ratio))
    n_train=n_total-n_val
    train_subset, val_subset = random_split(full,[n_train,n_val],
        generator=torch.Generator().manual_seed(42))
    train_dataset=Subset(full, train_subset.indices)
    val_dataset  =Subset(full, val_subset.indices)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,
                            collate_fn=collate_fn_numeric)
    val_loader  =DataLoader(val_dataset,batch_size=batch_size,shuffle=False,
                            collate_fn=collate_fn_numeric)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    model=EncoderWithQuerySlotsPacked(
        d_model=768,nhead=12,ff=4096,enc_layers=6,
        max_src_len=SRC_MAX_LEN,tgt_len=K_MAX,dropout=0.1).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = build_scheduler(opt, epochs, warmup_epochs, min_lr_ratio, schedule) if use_scheduler else None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type=="cuda")

    ideal_model = TorchNoisyGates(x_rad=0, z_rad=0, delta_x=0, delta_z=0, seed=seed)
    noisy_model = TorchNoisyGates(x_rad=noise_x_rad, z_rad=noise_z_rad,
                                  delta_x=noise_delta_x, delta_z=noise_delta_z, seed=seed+1)

    best_val=float('inf'); best_epoch=0
    for ep in range(1,epochs+1):
        tr_loss = train_one_epoch(model, train_loader, opt, device,
                                  scaler=scaler, use_amp=use_amp,
                                  noisy_model=noisy_model, ideal_model=ideal_model)
        val_loss = evaluate(model, val_loader, device, use_amp=use_amp,
                            noisy_model=noisy_model, ideal_model=ideal_model)
        if scheduler: scheduler.step()
        print(f"[Epoch {ep:02d}] train={tr_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val=val_loss; best_epoch=ep
            torch.save({"model":model.state_dict(),"val_loss":val_loss,"epoch":ep}, save_path)
            print(f"  -> saved best (val={val_loss:.4f})")
    print(f"Best val: {best_val:.4f} @ epoch {best_epoch}")

    if run_final_fidelity:
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("Loaded best checkpoint for final fidelity benchmark.")
        full_loader = DataLoader(full, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn_numeric)
        F_model, F_gt = evaluate_fidelity_benchmark(model, full_loader, device,
                                                    noisy_model=noisy_model,
                                                    ideal_model=ideal_model)
        print(f"[Final Fidelity Benchmark] mean_F_model={F_model:.6f}  mean_F_ground_truth={F_gt:.6f}")
    return model

if __name__ == "__main__":
    model = main_training_loop(tokenized_data_x, extracted_sched,
                               batch_size=1, epochs=3, lr=1e-5,
                               use_scheduler=False,
                               noise_x_rad=NOISE_X_RAD_NUM, noise_z_rad=NOISE_Z_RAD_NUM,
                               noise_delta_x=NOISE_DELTA_X_NUM, noise_delta_z=NOISE_DELTA_Z_NUM, seed=0,
                               run_final_fidelity=True, save_path="best_model.pt")
