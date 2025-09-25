#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal training script (vectorized training kernel; shared-PQC structure).
Implements:
- JSON/JSONL data loading -> Dataset / DataLoader
- Transformer predicting parameter gate angles (outputs sin, cos)
- Minimal statevector simulator: h/x/z/cx/cz + rz/rx
- Multi-initial fidelity loss over K_RANDOM random initial states: loss = 1 - mean(F)
- Optional auxiliary angle supervision (AUX_ANGLE_LOSS)
- Optional cosine scheduler
- Vectorized precompute (fast), AND vectorized training replay with shared PQC structure
- Eliminates .item() sync points; angles computed once; optional checkpoint per-step

Removed:
- legacy noise implementation (hash variant), heavy vectorization variants not needed here
"""

from __future__ import annotations
import os, json, math, random, time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= Settings =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.complex64
MAX_BASE_LEN=500; MAX_PARAM=75; MAX_QUBITS=5
EMB_DIM=256; NUM_LAYERS=4; NUM_HEADS=8; FF_DIM=EMB_DIM*4; DROP=0.1
K_RANDOM=100; BATCH_SIZE=32; EPOCHS=5; LR=1e-4; GRAD_CLIP=1.0
PRECOMPUTE_BASE=True
FAST_BASE_CACHE=True
FAST_NOISE_SCHEDULE=True
PACK_REF_STATES=True
VERBOSE_PRECOMP_TIMINGS=True
PARAM_CHECKPOINT=False  # 暂时关闭 checkpoint，避免与小状态多次原地修改导致版本冲突；后续稳定后可再开启
DATA_PATH='A:/wings/transformers/data/5q_500_2000g_no_uncomp_circuit_data/5q_500g_circuit_data_processed'; SEED=42
USE_SCHEDULER=False
AUX_ANGLE_LOSS=False; AUX_ANGLE_WEIGHT=0.05
PRINT_INTERVAL=50
DIFF_FIDELITY=True  # 训练主损可反传：基座+噪声 no_grad，参数门可微
USE_NOISE = True
NOISE_X_RAD = math.pi/100
NOISE_Z_RAD = math.pi/100
NOISE_DELTA_X = 0.05
NOISE_DELTA_Z = 0.05

random.seed(SEED); torch.manual_seed(SEED)

BASE_GATES={'h':0,'x':1,'z':2,'cx':3,'cz':4}
PARAM_GATES={'rz':0,'rx':1}
INV_BASE={v:k for k,v in BASE_GATES.items()}
INV_PARAM={v:k for k,v in PARAM_GATES.items()}
PAD_ID=-1

# ===================== Noise model =====================
def _build_noise_schedule(item:dict):
    Lb=len(item['base_gates'])
    if not USE_NOISE:
        zeros=[0.0]*Lb
        return dict(rx_q1=zeros, rz_q1=zeros, rx_q2=zeros, rz_q2=zeros)
    rx_q1=[]; rz_q1=[]; rx_q2=[]; rz_q2=[]
    for _ in range(Lb):
        rx1=(random.random()*2-1)*NOISE_X_RAD if random.random()<NOISE_DELTA_X else 0.0
        rz1=(random.random()*2-1)*NOISE_Z_RAD if random.random()<NOISE_DELTA_Z else 0.0
        rx2=(random.random()*2-1)*NOISE_X_RAD if random.random()<NOISE_DELTA_X else 0.0
        rz2=(random.random()*2-1)*NOISE_Z_RAD if random.random()<NOISE_DELTA_Z else 0.0
        rx_q1.append(rx1); rz_q1.append(rz1); rx_q2.append(rx2); rz_q2.append(rz2)
    return dict(rx_q1=rx_q1, rz_q1=rz_q1, rx_q2=rx_q2, rz_q2=rz_q2)

# ===== Optimized scalar-angle variants (support 0-dim tensors) =====
def _ensure_scalar_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device, dtype=torch.float32)

def _apply_rz_scalar(batch_states: torch.Tensor, q: int, angle, splits):
    angle = _ensure_scalar_tensor(angle, batch_states.device)
    if torch.all(angle==0): return
    i0, i1 = splits[q]
    em = torch.exp(-0.5j * angle)
    ep = torch.exp(0.5j * angle)
    batch_states[:, i0] *= em
    batch_states[:, i1] *= ep

def _apply_rx_scalar(batch_states: torch.Tensor, q: int, angle, splits):
    angle = _ensure_scalar_tensor(angle, batch_states.device)
    if torch.all(angle==0): return
    i0, i1 = splits[q]
    c = torch.cos(0.5 * angle)
    s = -1j * torch.sin(0.5 * angle)
    s0 = batch_states[:, i0]
    s1 = batch_states[:, i1]
    batch_states[:, i0] = c * s0 + s * s1
    batch_states[:, i1] = s * s0 + c * s1

# ================= Dataset =================
class CircuitDataset(Dataset):
    def __init__(self,path:str):
        self.items: list[dict] = []
        self._next_index = 0
        if not os.path.exists(path):
            print(f"[WARN] Data path does not exist: {path}")
            return
        def process_obj(o:dict):
            # New token format
            if 'base_circuit_tokens' in o and 'pqc_circuit_tokens' in o:
                base_tokens = o['base_circuit_tokens']
                pqc_tokens  = o['pqc_circuit_tokens']
                base_gates=[]; base_q1=[]; base_q2=[]
                for tok in base_tokens:
                    g=tok[0]; qs=tok[1]
                    if g not in BASE_GATES: continue
                    if len(qs)==1:
                        q1=qs[0]; q2=-1
                    elif len(qs)>=2:
                        q1,q2=qs[0],qs[1]
                    else:
                        continue
                    base_gates.append(g); base_q1.append(q1); base_q2.append(q2)
                param_gates=[]; param_qubits=[]; after_list=[]; param_angles=[]
                base_ptr=0; last_base_idx=-1
                def is_same_base(tok, idx):
                    if idx>=len(base_gates): return False
                    g=tok[0]; qs=tok[1]
                    if g!=base_gates[idx]: return False
                    bq1=base_q1[idx]; bq2=base_q2[idx]
                    if len(qs)==1: return qs[0]==bq1 and bq2==-1
                    if len(qs)>=2: return qs[0]==bq1 and qs[1]==bq2
                    return False
                for tok in pqc_tokens:
                    g=tok[0]; qs=tok[1]; params = tok[2] if len(tok)>2 else []
                    if is_same_base(tok, base_ptr):
                        last_base_idx=base_ptr; base_ptr+=1; continue
                    if g in PARAM_GATES:
                        q = qs[0] if qs else 0
                        ang = params[0] if params else 0.0
                        param_gates.append(g); param_qubits.append(q); after_list.append(last_base_idx); param_angles.append(ang)
                n_q=o.get('n_qubits')
                if n_q is None:
                    qs_all=[*base_q1,*[q for q in base_q2 if q>=0],*param_qubits]
                    n_q=(max(qs_all)+1) if qs_all else 1
                if len(base_gates)>MAX_BASE_LEN:
                    base_gates=base_gates[:MAX_BASE_LEN]; base_q1=base_q1[:MAX_BASE_LEN]; base_q2=base_q2[:MAX_BASE_LEN]
                if len(param_gates)>MAX_PARAM:
                    param_gates=param_gates[:MAX_PARAM]; param_qubits=param_qubits[:MAX_PARAM]; after_list=after_list[:MAX_PARAM]; param_angles=param_angles[:MAX_PARAM]
                self.items.append(dict(idx=self._next_index, base_gates=base_gates, base_q1=base_q1, base_q2=base_q2,
                                       param_gates=param_gates, param_qubits=param_qubits,
                                       after=after_list, param_angles_gt=param_angles, n_qubits=n_q))
                self._next_index += 1
                return
            # Old format
            base_g=o['base_gates']; bq=o['base_qubits']
            if len(bq)!=2: raise ValueError('base_qubits must be [q1_list, q2_list]')
            param_g=o.get('param_gates',[]); param_q=o.get('param_qubits',[])
            after=o.get('after',[-1]*len(param_g)); ang=o.get('pqc_angles_gt',[0.0]*len(param_g))
            if not (len(param_g)==len(param_q)==len(after)==len(ang)):
                raise ValueError('parameter list length mismatch')
            if len(base_g)>MAX_BASE_LEN:
                base_g=base_g[:MAX_BASE_LEN]; bq=[bq[0][:MAX_BASE_LEN], bq[1][:MAX_BASE_LEN]]
            if len(param_g)>MAX_PARAM:
                param_g=param_g[:MAX_PARAM]; param_q=param_q[:MAX_PARAM]; after=after[:MAX_PARAM]; ang=ang[:MAX_PARAM]
            n_q=o.get('n_qubits')
            if n_q is None:
                qs=[*bq[0],*bq[1],*param_q]; qs=[q for q in qs if q>=0]; n_q=(max(qs)+1) if qs else 1
            self.items.append(dict(idx=self._next_index, base_gates=base_g, base_q1=bq[0], base_q2=bq[1],
                                   param_gates=param_g, param_qubits=param_q,
                                   after=after, param_angles_gt=ang, n_qubits=n_q))
            self._next_index += 1

        if os.path.isdir(path):
            files=[f for f in os.listdir(path) if f.lower().endswith(('.json','.jsonl'))]
            files.sort()
            iterator = tqdm(files, desc='Reading data files', unit='file') if tqdm else files
            for fname in iterator:
                fp=os.path.join(path,fname)
                try:
                    with open(fp,'r',encoding='utf-8') as fh:
                        for line in fh:
                            if not line.strip(): continue
                            process_obj(json.loads(line))
                            break
                except Exception as e:
                    print(f"[WARN] Failed to read file {fp}: {e}")
            if tqdm:
                print(f"[INFO] Loaded samples: {len(self.items)}")
        else:
            with open(path,'r',encoding='utf-8') as f:
                lines=f.readlines()
            iterator=tqdm(lines, desc='Reading lines', unit='line') if tqdm else lines
            for line in iterator:
                if not line.strip(): continue
                process_obj(json.loads(line))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

@dataclass
class Batch:
    base_g:torch.Tensor; base_q1:torch.Tensor; base_q2:torch.Tensor
    param_g:torch.Tensor; param_q:torch.Tensor; param_after:torch.Tensor
    param_angles_gt:torch.Tensor; base_len:torch.Tensor; param_len:torch.Tensor; n_qubits:torch.Tensor; idx:torch.Tensor
    def to(self,device):
        for k,v in self.__dict__.items():
            if isinstance(v,torch.Tensor): setattr(self,k,v.to(device))
        return self

def _pad(seq,pad,L):
    seq=list(seq); return seq[:L]+[pad]*max(0,L-len(seq))

def collate(samples:List[dict]):
    bg=[]; bq1=[]; bq2=[]; pg=[]; pq=[]; pafter=[]; pang=[]; base_l=[]; param_l=[]; nqs=[]; idxs=[]
    for o in samples:
        g=[BASE_GATES[x] for x in o['base_gates']]; p=[PARAM_GATES[x] for x in o['param_gates']]
        bg.append(_pad(g,PAD_ID,MAX_BASE_LEN)); bq1.append(_pad(o['base_q1'],PAD_ID,MAX_BASE_LEN)); bq2.append(_pad(o['base_q2'],PAD_ID,MAX_BASE_LEN))
        pg.append(_pad(p,PAD_ID,MAX_PARAM)); pq.append(_pad(o['param_qubits'],PAD_ID,MAX_PARAM))
        pafter.append(_pad(o['after'],-999,MAX_PARAM)); pang.append(_pad(o['param_angles_gt'],0.0,MAX_PARAM))
        base_l.append(len(g)); param_l.append(len(p)); nqs.append(o['n_qubits']); idxs.append(o['idx'])
    to_long=lambda x: torch.tensor(x,dtype=torch.long)
    return Batch(to_long(bg),to_long(bq1),to_long(bq2),to_long(pg),to_long(pq),to_long(pafter),
                 torch.tensor(pang,dtype=torch.float32),to_long(base_l),to_long(param_l),to_long(nqs),to_long(idxs))

# ================= Model =================
class AnglePredictor(nn.Module):
    def __init__(self):
        super().__init__(); d=EMB_DIM
        self.base_emb=nn.Embedding(len(BASE_GATES)+1,d,padding_idx=len(BASE_GATES))
        self.qubit_emb=nn.Embedding(MAX_QUBITS+1,d); self.pos_emb=nn.Embedding(MAX_BASE_LEN,d)
        layer=nn.TransformerEncoderLayer(d,NUM_HEADS,FF_DIM,DROP,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,NUM_LAYERS)
        self.param_type_emb=nn.Embedding(len(PARAM_GATES)+1,d,padding_idx=len(PARAM_GATES))
        self.param_pos_emb=nn.Embedding(MAX_PARAM,d); self.query_proj=nn.Linear(d,d)
        self.attn=nn.MultiheadAttention(d,NUM_HEADS,dropout=DROP,batch_first=True); self.out=nn.Linear(d,2)
    def forward(self,b:Batch):
        ids=b.base_g.clone(); mask=(ids==PAD_ID); ids[mask]=len(BASE_GATES)
        x=self.base_emb(ids)+self.qubit_emb(torch.clamp(b.base_q1,0,MAX_QUBITS))+self.qubit_emb(torch.clamp(b.base_q2,0,MAX_QUBITS))
        pos=torch.arange(x.size(1),device=x.device); x=x+self.pos_emb(pos)[None]; x=self.encoder(x,src_key_padding_mask=mask)
        p=b.param_g.clone(); pmask=(p==PAD_ID); p[pmask]=len(PARAM_GATES)
        q=self.param_type_emb(p)+self.param_pos_emb(torch.arange(p.size(1),device=p.device))[None]; q=self.query_proj(q)
        attn,_=self.attn(q,x,x,key_padding_mask=mask)
        return self.out(attn), pmask

# ================= Minimal simulator =================
_SPLIT_CACHE: Dict[Tuple[int, torch.device], List[Tuple[torch.Tensor, torch.Tensor]]] = {}
def _split_indices(n,device):
    k=(n,device)
    if k in _SPLIT_CACHE: return _SPLIT_CACHE[k]
    dim=1<<n; ar=torch.arange(dim,device=device); out=[]
    for q in range(n):
        bit=(ar>>q)&1; out.append(((bit==0).nonzero(as_tuple=False).squeeze(-1),(bit==1).nonzero(as_tuple=False).squeeze(-1)))
    _SPLIT_CACHE[k]=out; return out

def _apply_const_1q(st,q,kind,splits):
    i0,i1=splits[q]; s0=st[...,i0]; s1=st[...,i1]
    if kind=='h': n0=(s0+s1)/math.sqrt(2); n1=(s0-s1)/math.sqrt(2)
    elif kind=='x': n0,n1=s1,s0
    elif kind=='z': n0,n1=s0,-s1
    else: raise ValueError(kind)
    st[...,i0]=n0; st[...,i1]=n1

def _apply_rz(st,q,a,splits):
    i0,i1=splits[q]; em=torch.exp(-0.5j*a).unsqueeze(-1); ep=torch.exp(0.5j*a).unsqueeze(-1); st[...,i0]*=em; st[...,i1]*=ep

def _apply_rx(st,q,a,splits):
    i0,i1=splits[q]; c=torch.cos(0.5*a).unsqueeze(-1); s=-1j*torch.sin(0.5*a).unsqueeze(-1); s0=st[...,i0]; s1=st[...,i1]; st[...,i0]=c*s0+s*s1; st[...,i1]=s*s0+c*s1

def _apply_cx(st,c,t): dim=st.size(-1); idx=torch.arange(dim,device=st.device); mc=1<<c; mt=1<<t; sel=((idx&mc)!=0)&((idx&mt)==0); i0=idx[sel]; i1=i0|mt; tmp=st[...,i0].clone(); st[...,i0]=st[...,i1]; st[...,i1]=tmp
def _apply_cz(st,q1,q2): dim=st.size(-1); idx=torch.arange(dim,device=st.device); mask=((idx&(1<<q1))!=0)&((idx&(1<<q2))!=0); st[...,idx[mask]]=-st[...,idx[mask]]

def sincos_to_angle(sc):
    # sc: [B, L, 2] or [L,2]
    sc = sc / (sc.norm(dim=-1,keepdim=True)+1e-8)
    return torch.atan2(sc[...,0], sc[...,1])

def angle_supervise_loss(pred,gt,mask):
    if gt is None: return torch.tensor(0.0,device=pred.device)
    valid=~mask
    if valid.sum()==0: return torch.tensor(0.0,device=pred.device)
    sc=pred/(pred.norm(dim=-1,keepdim=True)+1e-9); ang=torch.atan2(sc[...,0],sc[...,1]); diff=torch.angle(torch.exp(1j*(ang-gt))); return (diff[valid]**2).mean()

# ===================== Precompute (vectorized) =====================
def build_base_cache_vectorized(dataset: CircuitDataset):
    iterator = tqdm(dataset.items, desc='[fast] grouping', unit='sample') if tqdm else dataset.items
    groups: dict[int, list[dict]] = {}
    for it in iterator:
        groups.setdefault(it['n_qubits'], []).append(it)

    init_states_per_n: dict[int, torch.Tensor] = {}
    ref_states_per_idx: dict[int, torch.Tensor] | dict = {}
    noise_schedules: dict[int, dict] | dict = {}
    device = DEVICE
    ref_states_packed = None
    ref_idx2row = {}
    for n, items in groups.items():
        dim = 1 << n
        Bn = len(items)
        if Bn == 0: continue
        L_max = max(len(it['base_gates']) for it in items)
        # Build gate tensors on CPU then H2D once
        gate_ids_cpu = torch.full((Bn, L_max), PAD_ID, dtype=torch.long)
        q1_cpu      = torch.full((Bn, L_max), -1, dtype=torch.long)
        q2_cpu      = torch.full((Bn, L_max), -1, dtype=torch.long)
        sample_idx_list = []
        for bi, it in enumerate(items):
            sample_idx_list.append(it['idx'])
            Lb_i = len(it['base_gates'])
            if Lb_i == 0: continue
            gate_ids_row = [BASE_GATES[g] for g in it['base_gates']]
            gate_ids_cpu[bi, :Lb_i] = torch.tensor(gate_ids_row, dtype=torch.long)
            q1_cpu[bi, :Lb_i] = torch.tensor(it['base_q1'], dtype=torch.long)
            q2_cpu[bi, :Lb_i] = torch.tensor(it['base_q2'], dtype=torch.long)

        gate_ids = gate_ids_cpu.to(device, non_blocking=True)
        q1 = q1_cpu.to(device, non_blocking=True)
        q2 = q2_cpu.to(device, non_blocking=True)

        # shared init states for this n
        if n not in init_states_per_n:
            splits_tmp = _split_indices(n, device)
            states_init = []
            for _ in range(K_RANDOM):
                st = torch.zeros(dim, dtype=DTYPE, device=device); st[0] = 1+0j
                for qb in range(n):
                    r = random.random()
                    if r < 0.33: pass
                    elif r < 0.66: _apply_const_1q(st.unsqueeze(0), qb, 'x', splits_tmp)
                    else: _apply_const_1q(st.unsqueeze(0), qb, 'h', splits_tmp)
                states_init.append(st)
            init_states_per_n[n] = torch.stack(states_init, 0)  # [K, 2^n]

        states = init_states_per_n[n].unsqueeze(0).expand(Bn, -1, -1).clone()
        splits = _split_indices(n, device)
        idx_all = torch.arange(dim, device=device)
        cx_swap = {}
        cz_mask = {}
        for c in range(n):
            for t in range(n):
                if c == t: continue
                cb = 1 << c; tb = 1 << t
                sel = ((idx_all & cb) != 0) & ((idx_all & tb) == 0)
                i0 = idx_all[sel]; i1 = i0 | tb
                cx_swap[(c, t)] = (i0, i1)
                sel_cz = ((idx_all & cb) != 0) & ((idx_all & tb) != 0)
                cz_mask[(c, t)] = idx_all[sel_cz]

        with torch.no_grad():
            for t in range(L_max):
                g_t = gate_ids[:, t]
                if (g_t == PAD_ID).all(): break
                # 1q groups
                for gcode, gname in ((BASE_GATES['h'], 'h'), (BASE_GATES['x'], 'x'), (BASE_GATES['z'], 'z')):
                    mask = (g_t == gcode)
                    if not mask.any(): continue
                    qubits = q1[mask, t]
                    batches = mask.nonzero(as_tuple=False).squeeze(-1)
                    uq = qubits.unique()
                    for qb in uq.tolist():
                        sel = batches[(qubits == qb)]
                        if sel.numel() == 0: continue
                        i0, i1 = splits[qb]
                        states_sel = states.index_select(0, sel)
                        a = states_sel[:, :, i0]; b = states_sel[:, :, i1]
                        if gname == 'h':
                            new0 = (a + b) / math.sqrt(2); new1 = (a - b) / math.sqrt(2)
                        elif gname == 'x':
                            new0, new1 = b, a
                        else:
                            new0, new1 = a, -b
                        states_sel[:, :, i0] = new0
                        states_sel[:, :, i1] = new1
                        states[sel] = states_sel
                # 2q groups
                for gcode, gname in ((BASE_GATES['cx'], 'cx'), (BASE_GATES['cz'], 'cz')):
                    mask = (g_t == gcode)
                    if not mask.any(): continue
                    c_list = q1[mask, t]; t_list = q2[mask, t]
                    batches = mask.nonzero(as_tuple=False).squeeze(-1)
                    pairs = torch.stack([c_list, t_list], dim=1)
                    uniq_pairs, inv_idx = torch.unique(pairs, dim=0, return_inverse=True)
                    for pi, (c_val, t_val) in enumerate(uniq_pairs.tolist()):
                        sel = batches[inv_idx == pi]
                        if sel.numel() == 0: continue
                        if gname == 'cx':
                            i0, i1 = cx_swap[(c_val, t_val)]
                            states_sel = states.index_select(0, sel)
                            tmp = states_sel[:, :, i0].clone()
                            states_sel[:, :, i0] = states_sel[:, :, i1]
                            states_sel[:, :, i1] = tmp
                            states[sel] = states_sel
                        else:
                            m_idx = cz_mask[(c_val, t_val)]
                            states_sel = states.index_select(0, sel)
                            states_sel[:, :, m_idx] = -states_sel[:, :, m_idx]
                            states[sel] = states_sel

        # store reference
        if PACK_REF_STATES:
            if ref_states_packed is None:
                ref_states_packed = torch.empty(len(dataset.items), K_RANDOM, dim, dtype=DTYPE, device=device)
            for bi, sample_idx in enumerate(sample_idx_list):
                row = sample_idx
                ref_states_packed[row].copy_(states[bi])
                ref_idx2row[sample_idx] = row
        else:
            for bi, sample_idx in enumerate(sample_idx_list):
                ref_states_per_idx[sample_idx] = states[bi].clone()

    # tensor-mode noise schedules
    if FAST_NOISE_SCHEDULE:
        items_all = dataset.items
        idx_list = [it['idx'] for it in items_all]
        L_per_sample = [len(it['base_gates']) for it in items_all]
        L_max_global = max(L_per_sample) if L_per_sample else 0
        B_total = len(items_all)
        device = DEVICE
        q2_mat = torch.full((B_total, L_max_global), -1, dtype=torch.long, device=device)
        gate_mask = torch.zeros((B_total, L_max_global), dtype=torch.bool, device=device)
        for row, it in enumerate(items_all):
            Lb = len(it['base_gates'])
            gate_mask[row, :Lb] = True
            if Lb>0:
                q2_vals = torch.tensor(it['base_q2'], dtype=torch.long, device=device)
                q2_mat[row, :Lb] = q2_vals
        if USE_NOISE:
            rx_flag1 = (torch.rand(B_total, L_max_global, device=device) < NOISE_DELTA_X) & gate_mask
            rx_amp1  = (torch.rand(B_total, L_max_global, device=device)*2 - 1) * NOISE_X_RAD
            rx_q1 = torch.where(rx_flag1, rx_amp1, torch.zeros(1, device=device))
            rz_flag1 = (torch.rand(B_total, L_max_global, device=device) < NOISE_DELTA_Z) & gate_mask
            rz_amp1  = (torch.rand(B_total, L_max_global, device=device)*2 - 1) * NOISE_Z_RAD
            rz_q1 = torch.where(rz_flag1, rz_amp1, torch.zeros(1, device=device))
            valid_q2 = (q2_mat >= 0) & gate_mask
            rx_flag2 = (torch.rand(B_total, L_max_global, device=device) < NOISE_DELTA_X) & valid_q2
            rx_amp2  = (torch.rand(B_total, L_max_global, device=device)*2 - 1) * NOISE_X_RAD
            rx_q2 = torch.where(rx_flag2, rx_amp2, torch.zeros(1, device=device))
            rz_flag2 = (torch.rand(B_total, L_max_global, device=device) < NOISE_DELTA_Z) & valid_q2
            rz_amp2  = (torch.rand(B_total, L_max_global, device=device)*2 - 1) * NOISE_Z_RAD
            rz_q2 = torch.where(rz_flag2, rz_amp2, torch.zeros(1, device=device))
        else:
            rx_q1 = rz_q1 = rx_q2 = rz_q2 = torch.zeros(B_total, L_max_global, device=device)
        idx2row = {idx: i for i, idx in enumerate(idx_list)}
        noise_schedules = dict(tensor_mode=True, rx_q1=rx_q1, rz_q1=rz_q1, rx_q2=rx_q2, rz_q2=rz_q2,
                               idx2row=idx2row, L_max=L_max_global)

    if PACK_REF_STATES:
        ref_states_per_idx = dict(packed=True, tensor=ref_states_packed, idx2row=ref_idx2row)

    return init_states_per_n, ref_states_per_idx, noise_schedules

# ===================== Vectorized training kernel (shared PQC) =====================

def _apply_base_step_batched(states, gate_ids_step, q1_step, q2_step, splits, cx_swap, cz_mask):
    """Apply base gates at one step for a group: vectorized across samples sharing n."""
    # 1q: h/x/z
    for gcode, gname in ((BASE_GATES['h'],'h'), (BASE_GATES['x'],'x'), (BASE_GATES['z'],'z')):
        mask = (gate_ids_step == gcode)
        if not mask.any(): continue
        qubits = q1_step[mask]
        batches = mask.nonzero(as_tuple=False).squeeze(-1)
        uq = qubits.unique()
        for qb in uq.tolist():
            sel = batches[(qubits == qb)]
            if sel.numel()==0: continue
            i0, i1 = splits[qb]
            states_sel = states.index_select(0, sel)
            a = states_sel[:, :, i0]; b = states_sel[:, :, i1]
            if gname == 'h':
                new0 = (a + b)/math.sqrt(2); new1 = (a - b)/math.sqrt(2)
            elif gname == 'x':
                new0, new1 = b, a
            else:
                new0, new1 = a, -b
            states_sel[:, :, i0] = new0
            states_sel[:, :, i1] = new1
            states[sel] = states_sel
    # 2q: cx/cz
    for gcode, gname in ((BASE_GATES['cx'],'cx'), (BASE_GATES['cz'],'cz')):
        mask = (gate_ids_step == gcode)
        if not mask.any(): continue
        c_list = q1_step[mask]; t_list = q2_step[mask]
        batches = mask.nonzero(as_tuple=False).squeeze(-1)
        pairs = torch.stack([c_list, t_list], dim=1)
        uniq_pairs, inv_idx = torch.unique(pairs, dim=0, return_inverse=True)
        for pi, (c_val, t_val) in enumerate(uniq_pairs.tolist()):
            sel = batches[inv_idx == pi]
            if sel.numel()==0: continue
            if gname == 'cx':
                i0, i1 = cx_swap[(c_val, t_val)]
                states_sel = states.index_select(0, sel)
                tmp = states_sel[:, :, i0].clone()
                states_sel[:, :, i0] = states_sel[:, :, i1]
                states_sel[:, :, i1] = tmp
                states[sel] = states_sel
            else:
                m_idx = cz_mask[(c_val, t_val)]
                states_sel = states.index_select(0, sel)
                states_sel[:, :, m_idx] = -states_sel[:, :, m_idx]
                states[sel] = states_sel

def _apply_noise_step_batched(states, q1_step, q2_step, rx1, rz1, rx2, rz2, splits):
    """Apply sparse Rx/Rz noise after this base step, per-sample qubit; vectorized by grouping same qubit."""
    # qubit 1
    uq = q1_step.unique()
    for qb in uq.tolist():
        mask = (q1_step == qb)
        if not mask.any(): continue
        sel = mask.nonzero(as_tuple=False).squeeze(-1)
        states_sel = states.index_select(0, sel)
        # rz then rx, both per-sample different angles
        ang_rz = rz1[sel]  # [B_sel]
        if ang_rz.abs().sum() != 0:
            i0,i1 = splits[qb]
            em = torch.exp(-0.5j * ang_rz)[:, None, None]
            ep = torch.exp(0.5j  * ang_rz)[:, None, None]
            states_sel[:, :, i0] = states_sel[:, :, i0] * em
            states_sel[:, :, i1] = states_sel[:, :, i1] * ep
        ang_rx = rx1[sel]
        if ang_rx.abs().sum() != 0:
            i0,i1 = splits[qb]
            c = torch.cos(0.5*ang_rx)[:, None, None]
            s = -1j*torch.sin(0.5*ang_rx)[:, None, None]
            a = states_sel[:, :, i0]; b = states_sel[:, :, i1]
            states_sel[:, :, i0] = c*a + s*b
            states_sel[:, :, i1] = s*a + c*b
        states[sel] = states_sel
    # qubit 2 (only if present)
    valid_q2 = (q2_step >= 0)
    if valid_q2.any():
        q2_vals = q2_step[valid_q2]
        uq2 = q2_vals.unique()
        base_idx = valid_q2.nonzero(as_tuple=False).squeeze(-1)
        for qb in uq2.tolist():
            mask_local = (q2_vals == qb)
            sel = base_idx[mask_local]  # indices in batch
            states_sel = states.index_select(0, sel)
            ang_rz = rz2[sel]
            if ang_rz.abs().sum() != 0:
                i0,i1 = splits[qb]
                em = torch.exp(-0.5j * ang_rz)[:, None, None]
                ep = torch.exp(0.5j  * ang_rz)[:, None, None]
                states_sel[:, :, i0] = states_sel[:, :, i0] * em
                states_sel[:, :, i1] = states_sel[:, :, i1] * ep
            ang_rx = rx2[sel]
            if ang_rx.abs().sum() != 0:
                i0,i1 = splits[qb]
                c = torch.cos(0.5*ang_rx)[:, None, None]
                s = -1j*torch.sin(0.5*ang_rx)[:, None, None]
                a = states_sel[:, :, i0]; b = states_sel[:, :, i1]
                states_sel[:, :, i0] = c*a + s*b
                states_sel[:, :, i1] = s*a + c*b
            states[sel] = states_sel

def _apply_params_step_shared_structure(states, angles_all, t, param_pos, param_kind, param_qubit, splits):
    """Apply all param gates at step t (shared structure across samples). Keeps autograd for angles.
       angles_all: [B, Lp] (already atan2 of sin/cos logits)
    """
    # 为防止与外部引用共享版本号，先克隆一份（输出新的张量以便 autograd 不追踪原地修改冲突）
    states = states.clone()
    I_t = (param_pos == t).nonzero(as_tuple=False).squeeze(-1)
    if I_t.numel() == 0:
        return states
    # RZ group
    I_rz = I_t[(param_kind[I_t] == PARAM_GATES['rz'])]
    if I_rz.numel():
        q_rz = param_qubit[I_rz]                 # [Nr]
        uq, inv = torch.unique(q_rz, return_inverse=True)
        ang_rz = angles_all[:, I_rz]             # [B, Nr]
        for i, q in enumerate(uq.tolist()):
            sel = (inv == i)
            ang_q = ang_rz[:, sel].sum(dim=1)    # [B]
            i0, i1 = splits[q]
            em = torch.exp(-0.5j * ang_q)[:, None, None]
            ep = torch.exp(0.5j  * ang_q)[:, None, None]
            states[:, :, i0] = states[:, :, i0] * em
            states[:, :, i1] = states[:, :, i1] * ep
    # RX group
    I_rx = I_t[(param_kind[I_t] == PARAM_GATES['rx'])]
    if I_rx.numel():
        q_rx = param_qubit[I_rx]
        uq, inv = torch.unique(q_rx, return_inverse=True)
        ang_rx = angles_all[:, I_rx]             # [B, Nx]
        for i, q in enumerate(uq.tolist()):
            sel = (inv == i)
            ang_q = ang_rx[:, sel].sum(dim=1)    # [B]
            i0, i1 = splits[q]
            c = torch.cos(0.5*ang_q)[:, None, None]
            s = -1j*torch.sin(0.5*ang_q)[:, None, None]
            a = states[:, :, i0]; b = states[:, :, i1]
            states[:, :, i0] = c*a + s*b
            states[:, :, i1] = s*a + c*b
    return states

def simulate_loss_cached_vectorized_samepqc(batch: Batch, logits, init_cache, ref_cache, noise_schedules):
    """Vectorized replay assuming all samples share the same PQC structure (positions/types identical).
       - Groups by n_qubits
       - Base+noise executed in no_grad (not in graph)
       - Param gates per step applied batched with shared structure (angles keep grad)
    """
    assert isinstance(noise_schedules, dict) and noise_schedules.get('tensor_mode', False), \
        "Require FAST_NOISE_SCHEDULE tensor-mode noise schedules."

    B = batch.base_g.size(0)
    device = logits.device

    # angles once: [B, MAX_PARAM, 2] -> [B, Lp]
    # param_len assumed equal across batch due to shared structure; fall back to per-sample min
    Lp_list = batch.param_len.tolist()
    Lp = max(Lp_list) if len(set(Lp_list))==1 else min(Lp_list)
    angles_all = sincos_to_angle(logits[:, :Lp, :])  # [B, Lp]

    # shared PQC structure from the first sample
    param_pos  = batch.param_after[0, :Lp].to(device)
    param_kind = batch.param_g[0, :Lp].to(device)     # 0: rz, 1: rx
    param_qubit= batch.param_q[0, :Lp].to(device)

    # optional sanity (not断言，避免开销)
    # 可在开发时检查所有样本一致性

    # group by n_qubits inside this batch
    nvals = batch.n_qubits.tolist()
    groups: Dict[int, torch.Tensor] = {}
    for i, n in enumerate(nvals):
        groups.setdefault(n, []).append(i)
    losses = []

    for n, idx_list in groups.items():
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)  # [Bg]
        Bg = idx_tensor.numel()
        dim = 1 << n
        splits = _split_indices(n, device)
        # shared init clone (no H2D)
        states = init_cache[n].to(device).unsqueeze(0).expand(Bg, -1, -1).clone()  # [Bg, K, 2^n]
        # reference states
        if isinstance(ref_cache, dict) and ref_cache.get('packed', False):
            rows = torch.tensor([ref_cache['idx2row'][int(batch.idx[i])] for i in idx_list], device=device, dtype=torch.long)
            ref = ref_cache['tensor'].index_select(0, rows)  # [Bg, K, 2^n]
        else:
            # dict-of-tensors path
            ref = torch.stack([ref_cache[int(batch.idx[i])].to(device) for i in idx_list], dim=0)

        # per-group base tensors for this step
        # extract step tensors: [Bg, Lb_max]
        Lb_list = [int(batch.base_len[i]) for i in idx_list]
        Lb_max = max(Lb_list)
        gate_ids = batch.base_g.index_select(0, idx_tensor)[:, :Lb_max].to(device)   # [Bg, Lb]
        q1       = batch.base_q1.index_select(0, idx_tensor)[:, :Lb_max].to(device)
        q2       = batch.base_q2.index_select(0, idx_tensor)[:, :Lb_max].to(device)

        # build helpers for 2q gates
        idx_all = torch.arange(dim, device=device)
        cx_swap = {}
        cz_mask = {}
        for c in range(n):
            for t in range(n):
                if c == t: continue
                cb = 1 << c; tb = 1 << t
                sel = ((idx_all & cb) != 0) & ((idx_all & tb) == 0)
                i0 = idx_all[sel]; i1 = i0 | tb
                cx_swap[(c, t)] = (i0, i1)
                sel_cz = ((idx_all & cb) != 0) & ((idx_all & tb) != 0)
                cz_mask[(c, t)] = idx_all[sel_cz]

        # rows for noise schedule lookup
        noise_rows = torch.tensor([noise_schedules['idx2row'][int(batch.idx[i])] for i in idx_list],
                                  device=device, dtype=torch.long)

        # per-group angles slice
        angles_grp = angles_all.index_select(0, idx_tensor)  # [Bg, Lp]

        for t in range(Lb_max):
            g_t = gate_ids[:, t]
            all_pad = (g_t == PAD_ID).all()
            if all_pad: break

            # Base gates (no_grad)
            with torch.no_grad():
                q1_t = q1[:, t]; q2_t = q2[:, t]
                _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap, cz_mask)

                # Noise (tensor-mode), apply per sample/qubit
                rx1_t = noise_schedules['rx_q1'].index_select(0, noise_rows)[:, t]
                rz1_t = noise_schedules['rz_q1'].index_select(0, noise_rows)[:, t]
                rx2_t = noise_schedules['rx_q2'].index_select(0, noise_rows)[:, t]
                rz2_t = noise_schedules['rz_q2'].index_select(0, noise_rows)[:, t]
                if USE_NOISE:
                    _apply_noise_step_batched(states, q1_t, q2_t, rx1_t, rz1_t, rx2_t, rz2_t, splits)

            # Param gates (keep graph); shared structure means index set is same for all samples
            if DIFF_FIDELITY:
                # 不使用 checkpoint 先确保梯度正确，再做融合
                states = _apply_params_step_shared_structure(states, angles_grp, t, param_pos, param_kind, param_qubit, splits)
            # else: nothing (param gates也可以 no_grad，但那就不反传了)

        # fidelity for this group
        ov = (ref.conj() * states).sum(-1)   # [Bg, K]
        F = (ov.abs()**2).mean()
        losses.append(1 - F)

    return torch.stack(losses).mean()

# ===================== Scheduler & Train =====================
def build_scheduler(opt):
    if not USE_SCHEDULER: return None
    def lr_lambda(ep): prog=ep/max(1,EPOCHS); return 0.1+0.9*0.5*(1+math.cos(math.pi*prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def train():
    if not os.path.exists(DATA_PATH):
        print(f'[WARN] Data file not found: {DATA_PATH}')
        return
    ds=CircuitDataset(DATA_PATH)
    if len(ds)==0:
        print('[WARN] Empty dataset')
        return
    if PRECOMPUTE_BASE:
        print('[INFO] Precomputing random initial states and noiseless base references ...')
        t0=time.perf_counter()
        # 简化：仅保留 fast 版本（本脚本未实现普通 build_base_cache）
        init_cache, ref_cache, noise_schedules = build_base_cache_vectorized(ds)
        dt = time.perf_counter() - t0
        if isinstance(ref_cache, dict) and ref_cache.get('packed', False):
            print(f'[INFO] ref_cache packed tensor shape = {tuple(ref_cache["tensor"].shape)}')
        print(f'[INFO] Precompute done in {dt:.2f}s (FAST={FAST_BASE_CACHE})')
    else:
        init_cache = ref_cache = None
        noise_schedules = {}

    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate)
    model=AnglePredictor().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
    sch=build_scheduler(opt); step=0

    for ep in range(1,EPOCHS+1):
        model.train(); total=0.0
        epoch_iter = enumerate(loader, start=1)
        epoch_iter = tqdm(epoch_iter, total=len(loader), desc=f'Epoch {ep}', unit='batch')
        for bi, raw in epoch_iter:
            batch=raw.to(DEVICE)
            logits,mask=model(batch)  # logits: [B, MAX_PARAM, 2]
            if PRECOMPUTE_BASE:
                # 新的矢量化训练内核（共享 PQC 结构）
                main=simulate_loss_cached_vectorized_samepqc(batch, logits, init_cache, ref_cache, noise_schedules)
            else:
                # 非预计算路径（此处可保留旧实现或直接禁止）
                raise RuntimeError("Please enable PRECOMPUTE_BASE for the vectorized training kernel.")
            if AUX_ANGLE_LOSS:
                aux=angle_supervise_loss(logits,batch.param_angles_gt,mask)
            else:
                aux=torch.tensor(0.0,device=logits.device)
            loss=main + AUX_ANGLE_WEIGHT*aux
            opt.zero_grad(set_to_none=True); loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step(); total+=loss.item(); step+=1
            epoch_iter.set_postfix(loss=f'{loss.item():.4f}', main=f'{main.item():.4f}', aux=f'{aux.item():.4f}')
        if sch: sch.step()
        avg = total/len(loader)
        print(f'[Epoch {ep}] avg_loss={avg:.6f}')
    torch.save({'model':model.state_dict()},'minimal_model.pt'); print('Model saved -> minimal_model.pt')

if __name__=='__main__':
    train()
