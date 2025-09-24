#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal training script.
Includes:
1. JSON/JSONL data loading -> Dataset / DataLoader
2. Transformer predicting parameter gate angles (outputs sin, cos)
3. Minimal single-sample statevector simulator: h/x/z/cx/cz + rz/rx
4. Multi-initial fidelity loss over K_RANDOM random initial states: loss = 1 - mean(F)
5. Optional auxiliary angle supervision (AUX_ANGLE_LOSS)
6. Optional cosine scheduler

Removed: legacy noise implementation (hash variant), heavy vectorization, benchmarks, compile, triton, old multi-initial code variants.

Usage: python training.py  (DATA_PATH can point to a single jsonl file or a directory. If directory: read every *.json / *.jsonl file; assume one JSON object per file.)
Single-line JSON example:
{"base_gates":["h","cx","h"],"base_qubits":[[0,0,1],[-1,1,-1]],"param_gates":["rz","rx"],"param_qubits":[0,1],"after":[-1,1],"pqc_angles_gt":[0.2,-0.4],"n_qubits":2}
"""
from __future__ import annotations
import os, json, math, random
from dataclasses import dataclass
from typing import List
import torch
from tqdm import tqdm

from torch import nn
from torch.utils.data import Dataset, DataLoader

# ================= Settings =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.complex64
MAX_BASE_LEN=256; MAX_PARAM=64; MAX_QUBITS=12
EMB_DIM=256; NUM_LAYERS=4; NUM_HEADS=8; FF_DIM=EMB_DIM*4; DROP=0.1
K_RANDOM=100; BATCH_SIZE=8; EPOCHS=5; LR=1e-4; GRAD_CLIP=1.0
PRECOMPUTE_BASE=True  # Precompute shared random initial states + noiseless base terminal states (reference)
DATA_PATH='A:/wings/transformers/data/5q_500_2000g_no_uncomp_circuit_data/5q_1000g_circuit_data_processed'; SEED=42
USE_SCHEDULER=False
AUX_ANGLE_LOSS=True; AUX_ANGLE_WEIGHT=0.05
PRINT_INTERVAL=50
USE_NOISE = True          # Whether to enable noise: reference path is noiseless; noise only applied during noisy+PQC path
NOISE_X_RAD = math.pi/100 # Max Rx noise magnitude
NOISE_Z_RAD = math.pi/100 # Max Rz noise magnitude
NOISE_DELTA_X = 0.05      # Probability of adding an Rx perturbation
NOISE_DELTA_Z = 0.05      # Probability of adding an Rz perturbation
random.seed(SEED); torch.manual_seed(SEED)

BASE_GATES={'h':0,'x':1,'z':2,'cx':3,'cz':4}
PARAM_GATES={'rz':0,'rx':1}
INV_BASE={v:k for k,v in BASE_GATES.items()}
INV_PARAM={v:k for k,v in PARAM_GATES.items()}
PAD_ID=-1

# ===================== Noise model (current strategy) =====================
# Fidelity definition: F = | < ψ_ref(noiseless base) | ψ_noisy+PQC > |^2
# - reference: evolve only the noiseless base circuit
# - noisy+PQC: replay each base gate; after each gate apply (sparse) Rx/Rz noise from a fixed per-sample schedule
# - noise schedule is fixed per sample to avoid adding stochastic variance per iteration; randomness only from initial states.
# If PRECOMPUTE_BASE is False, schedules are lazily generated and cached.

## Removed deprecated hash-based noise implementation for clarity.

def _build_noise_schedule(item:dict):
    """Generate per-sample fixed noise schedule.
    Expected keys: base_gates, base_q1, base_q2.
    Returns dict with lists (length = number of base gates): rx_q1, rz_q1, rx_q2, rz_q2.
    If USE_NOISE is False -> all zeros.
    """
    Lb=len(item['base_gates'])
    if not USE_NOISE:
        zeros=[0.0]*Lb
        return dict(rx_q1=zeros, rz_q1=zeros, rx_q2=zeros, rz_q2=zeros)
    rx_q1=[]; rz_q1=[]; rx_q2=[]; rz_q2=[]
    # Python's global RNG seeded earlier -> reproducible schedules
    for t in range(Lb):
        q1=item['base_q1'][t]; q2=item['base_q2'][t]
        # qubit1
        rx1=(random.random()*2-1)*NOISE_X_RAD if random.random()<NOISE_DELTA_X else 0.0
        rz1=(random.random()*2-1)*NOISE_Z_RAD if random.random()<NOISE_DELTA_Z else 0.0
        if q2>=0:
            rx2=(random.random()*2-1)*NOISE_X_RAD if random.random()<NOISE_DELTA_X else 0.0
            rz2=(random.random()*2-1)*NOISE_Z_RAD if random.random()<NOISE_DELTA_Z else 0.0
        else:
            rx2=0.0; rz2=0.0
        rx_q1.append(rx1); rz_q1.append(rz1); rx_q2.append(rx2); rz_q2.append(rz2)
    return dict(rx_q1=rx_q1, rz_q1=rz_q1, rx_q2=rx_q2, rz_q2=rz_q2)

def _apply_noise_from_schedule(state:torch.Tensor, splits, q:int, rx:float, rz:float, k_random:int):
    if q<0: return
    if rz!=0.0:
        ang=torch.full((k_random,),rz,device=state.device)
        _apply_rz(state.unsqueeze(0),q,ang,splits)
    if rx!=0.0:
        ang=torch.full((k_random,),rx,device=state.device)
        _apply_rx(state.unsqueeze(0),q,ang,splits)

# ================= Dataset =================
class CircuitDataset(Dataset):
    def __init__(self,path:str):
        self.items: list[dict] = []
        self._next_index = 0  # Unique sample index for caching / schedules
        if not os.path.exists(path):
            print(f"[WARN] Data path does not exist: {path}")
            return

        def process_obj(o:dict):
            """Normalize schema variants into internal format."""
            # -------- New token format --------
            if 'base_circuit_tokens' in o and 'pqc_circuit_tokens' in o:
                base_tokens = o['base_circuit_tokens']
                pqc_tokens  = o['pqc_circuit_tokens']
                base_gates=[]; base_q1=[]; base_q2=[]
                for tok in base_tokens:
                    g=tok[0]; qs=tok[1]
                    if g not in BASE_GATES:
                        continue
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
            # -------- Old format --------
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
            # Read all .json / .jsonl files (non-recursive), assume one JSON object per file
            files=[f for f in os.listdir(path) if f.lower().endswith(('.json','.jsonl'))]
            files.sort()
            if not files:
                print(f"[WARN] No .json/.jsonl files found in directory {path}")
            iterator = files
            if tqdm:
                iterator = tqdm(files, desc='Reading data files', unit='file')
            for fname in iterator:
                fp=os.path.join(path,fname)
                try:
                    with open(fp,'r',encoding='utf-8') as fh:
                        for line in fh:
                            if not line.strip(): continue
                            process_obj(json.loads(line))
                            break # Only first line per file
                except Exception as e:
                    print(f"[WARN] Failed to read file {fp}: {e}")
            if tqdm:
                print(f"[INFO] Loaded samples: {len(self.items)}")
        else:
            # Single jsonl file compatibility
            with open(path,'r',encoding='utf-8') as f:
                lines=f.readlines()
            iterator=lines
            if tqdm:
                iterator=tqdm(lines, desc='Reading lines', unit='line')
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
_SPLIT_CACHE={}
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
def _apply_rz(st,q,a,splits): i0,i1=splits[q]; em=torch.exp(-0.5j*a).unsqueeze(-1); ep=torch.exp(0.5j*a).unsqueeze(-1); st[...,i0]*=em; st[...,i1]*=ep
def _apply_rx(st,q,a,splits): i0,i1=splits[q]; c=torch.cos(0.5*a).unsqueeze(-1); s=-1j*torch.sin(0.5*a).unsqueeze(-1); s0=st[...,i0]; s1=st[...,i1]; st[...,i0]=c*s0+s*s1; st[...,i1]=s*s0+c*s1
def _apply_cx(st,c,t): dim=st.size(-1); idx=torch.arange(dim,device=st.device); mc=1<<c; mt=1<<t; sel=((idx&mc)!=0)&((idx&mt)==0); i0=idx[sel]; i1=i0|mt; tmp=st[...,i0].clone(); st[...,i0]=st[...,i1]; st[...,i1]=tmp
def _apply_cz(st,q1,q2): dim=st.size(-1); idx=torch.arange(dim,device=st.device); mask=((idx&(1<<q1))!=0)&((idx&(1<<q2))!=0); st[...,idx[mask]]=-st[...,idx[mask]]

def sincos_to_angle(sc): sc=sc/(sc.norm(dim=-1,keepdim=True)+1e-8); return torch.atan2(sc[...,0],sc[...,1])
def angle_supervise_loss(pred,gt,mask):
    if gt is None: return torch.tensor(0.0,device=pred.device)
    valid=~mask
    if valid.sum()==0: return torch.tensor(0.0,device=pred.device)
    sc=pred/(pred.norm(dim=-1,keepdim=True)+1e-9); ang=torch.atan2(sc[...,0],sc[...,1]); diff=torch.angle(torch.exp(1j*(ang-gt))); return (diff[valid]**2).mean()

def simulate_loss(batch:Batch, logits, noise_schedules:dict):
    """Simulation when PRECOMPUTE_BASE=False.
    Reference path: noiseless base circuit.
    Param path: replay base gates interleaved with parameter gates; apply noise schedule after each base gate.
    noise_schedules: dict[idx -> schedule]; lazily generated if absent.
    """
    B=batch.base_g.size(0); losses=[]
    for b in range(B):
        n=int(batch.n_qubits[b]); splits=_split_indices(n,logits.device); dim=1<<n
    # Generate random initial states
        inits=[]
        for _ in range(K_RANDOM):
            st=torch.zeros(dim,dtype=DTYPE,device=logits.device); st[0]=1+0j
            for q in range(n):
                r=random.random()
                if r<0.33: pass
                elif r<0.66: _apply_const_1q(st.unsqueeze(0),q,'x',splits)
                else: _apply_const_1q(st.unsqueeze(0),q,'h',splits)
            inits.append(st)
        init=torch.stack(inits,0)
    # Reference (noiseless base)
        ref=init.clone(); Lb=int(batch.base_len[b])
        for t in range(Lb):
            gid=batch.base_g[b,t].item();
            if gid==PAD_ID: break
            name=INV_BASE[gid]; q1=batch.base_q1[b,t].item(); q2=batch.base_q2[b,t].item()
            if name in ('h','x','z'): _apply_const_1q(ref.unsqueeze(0),q1,name,splits)
            elif name=='cx': _apply_cx(ref.unsqueeze(0),q1,q2)
            elif name=='cz': _apply_cz(ref.unsqueeze(0),q1,q2)
        param=init.clone(); Lp=int(batch.param_len[b])
        p_after=batch.param_after[b]; p_g=batch.param_g[b]; p_q=batch.param_q[b]
        idx_sample=int(batch.idx[b])
        if idx_sample not in noise_schedules:
            # Build schedule on demand
            Lb_eff = Lb
            base_gates=[INV_BASE[batch.base_g[b,t].item()] for t in range(Lb_eff) if batch.base_g[b,t].item()!=PAD_ID]
            base_q1=[batch.base_q1[b,t].item() for t in range(Lb_eff) if batch.base_g[b,t].item()!=PAD_ID]
            base_q2=[batch.base_q2[b,t].item() for t in range(Lb_eff) if batch.base_g[b,t].item()!=PAD_ID]
            noise_schedules[idx_sample]=_build_noise_schedule(dict(base_gates=base_gates, base_q1=base_q1, base_q2=base_q2))
        sched=noise_schedules[idx_sample]
    # Replay base gates while inserting parameter gates
        for step in range(-1,Lb):
            for p in range(Lp):
                gid=p_g[p].item();
                if gid==PAD_ID: break
                if p_after[p].item()!=step: continue
                ang=sincos_to_angle(logits[b:b+1,p:p+1])[0,0]; q=p_q[p].item()
                if INV_PARAM[gid]=='rz': _apply_rz(param.unsqueeze(0),q,ang.repeat(K_RANDOM),splits)
                else: _apply_rx(param.unsqueeze(0),q,ang.repeat(K_RANDOM),splits)
            if step>=0:
                gid=batch.base_g[b,step].item();
                if gid==PAD_ID: break
                name=INV_BASE[gid]; q1=batch.base_q1[b,step].item(); q2=batch.base_q2[b,step].item()
                if name in ('h','x','z'): _apply_const_1q(param.unsqueeze(0),q1,name,splits)
                elif name=='cx': _apply_cx(param.unsqueeze(0),q1,q2)
                elif name=='cz': _apply_cz(param.unsqueeze(0),q1,q2)
                # Apply noise (schedule)
                if step < len(sched['rx_q1']):
                    _apply_noise_from_schedule(param,splits,q1,sched['rx_q1'][step],sched['rz_q1'][step],K_RANDOM)
                    if q2>=0:
                        _apply_noise_from_schedule(param,splits,q2,sched['rx_q2'][step],sched['rz_q2'][step],K_RANDOM)
        ov=(ref.conj()*param).sum(-1); F=(ov.abs()**2).mean(); losses.append(1-F)
    return torch.stack(losses).mean()

def build_base_cache(dataset: CircuitDataset):
    """Precompute caches:
    - init_states_per_n: shared K_RANDOM initial states per n_qubits
    - ref_states_per_idx: noiseless base terminal states
    - noise_schedules: fixed per-sample noise schedule (only used in param path)
    Returns (init_states_per_n, ref_states_per_idx, noise_schedules)
    """
    if tqdm:
        pbar = tqdm(dataset.items, desc='Precomputing base cache', unit='sample')
    else:
        pbar = dataset.items
    init_states_per_n: dict[int, torch.Tensor] = {}
    ref_states_per_idx: dict[int, torch.Tensor] = {}
    noise_schedules: dict[int, dict] = {}
    for item in pbar:
        n = item['n_qubits']; dim = 1 << n; splits = _split_indices(n, DEVICE)
        # Prepare shared initial states
        if n not in init_states_per_n:
            states=[]
            for _ in range(K_RANDOM):
                st=torch.zeros(dim,dtype=DTYPE,device=DEVICE); st[0]=1+0j
                for q in range(n):
                    r=random.random()
                    if r<0.33: pass
                    elif r<0.66: _apply_const_1q(st.unsqueeze(0),q,'x',splits)
                    else: _apply_const_1q(st.unsqueeze(0),q,'h',splits)
                states.append(st)
            init_states_per_n[n]=torch.stack(states,0)  # [K_RANDOM, dim]
        init_clone = init_states_per_n[n].clone()
        # Evolve noiseless base
        Lb = len(item['base_gates'])
        for t in range(Lb):
            name=item['base_gates'][t]; q1=item['base_q1'][t]; q2=item['base_q2'][t]
            if name in ('h','x','z'):
                _apply_const_1q(init_clone.unsqueeze(0),q1,name,splits)
            elif name=='cx':
                _apply_cx(init_clone.unsqueeze(0),q1,q2)
            elif name=='cz':
                _apply_cz(init_clone.unsqueeze(0),q1,q2)
        ref_states_per_idx[item['idx']] = init_clone.clone()  # reference
        # Generate per-sample noise schedule
        noise_schedules[item['idx']] = _build_noise_schedule(item)
    return init_states_per_n, ref_states_per_idx, noise_schedules

def simulate_loss_cached(batch:Batch, logits, init_cache:dict, ref_cache:dict, noise_schedules:dict):
    """Simulation with precomputed caches:
    - init_cache[n]: shared random initial states
    - ref_cache[idx]: noiseless base terminal state
    Still need to replay base gates (interleaved) for the noisy param path; reference reused.
    """
    losses=[]; B=batch.base_g.size(0)
    for b in range(B):
        idx=int(batch.idx[b]); n=int(batch.n_qubits[b]); device=logits.device
        splits=_split_indices(n,device)
        init = init_cache[n].to(device).clone()  # [K_RANDOM, 2^n]
        ref = ref_cache[idx].to(device)          # noiseless reference
        Lb=int(batch.base_len[b]); Lp=int(batch.param_len[b])
        p_after=batch.param_after[b]; p_g=batch.param_g[b]; p_q=batch.param_q[b]
        sched = noise_schedules.get(idx, None)
        # Group parameter gates by insertion point (avoid O(Lb*Lp) scanning)
        param_groups={}  # step -> list[param_index]
        for p in range(Lp):
            gid=p_g[p].item()
            if gid==PAD_ID: break
            step_key=p_after[p].item()
            param_groups.setdefault(step_key, []).append(p)
        # Compute all angles once
        if Lp>0:
            all_logits = logits[b,:Lp]
            angles_full = sincos_to_angle(all_logits)
        else:
            angles_full = None
        param_state=init
        for step in range(-1,Lb):
            if step in param_groups and angles_full is not None:
                for p in param_groups[step]:
                    gid=p_g[p].item(); q=p_q[p].item(); ang=angles_full[p]
                    if INV_PARAM[gid]=='rz': _apply_rz(param_state.unsqueeze(0),q,ang.repeat(K_RANDOM),splits)
                    else: _apply_rx(param_state.unsqueeze(0),q,ang.repeat(K_RANDOM),splits)
            if step>=0:
                gid=batch.base_g[b,step].item();
                if gid==PAD_ID: break
                name=INV_BASE[gid]; q1=batch.base_q1[b,step].item(); q2=batch.base_q2[b,step].item()
                if name in ('h','x','z'): _apply_const_1q(param_state.unsqueeze(0),q1,name,splits)
                elif name=='cx': _apply_cx(param_state.unsqueeze(0),q1,q2)
                elif name=='cz': _apply_cz(param_state.unsqueeze(0),q1,q2)
                if sched is not None and step < len(sched['rx_q1']):
                    _apply_noise_from_schedule(param_state,splits,q1,sched['rx_q1'][step],sched['rz_q1'][step],K_RANDOM)
                    if q2>=0:
                        _apply_noise_from_schedule(param_state,splits,q2,sched['rx_q2'][step],sched['rz_q2'][step],K_RANDOM)
        ov=(ref.conj()*param_state).sum(-1)
        F=(ov.abs()**2).mean()
        losses.append(1-F)
    return torch.stack(losses).mean()

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
        init_cache, ref_cache, noise_schedules = build_base_cache(ds)
        print(f'[INFO] Precompute done: n_qubits groups = {len(init_cache)}, reference samples = {len(ref_cache)}, noise schedules = {len(noise_schedules)}')
    else:
        init_cache=ref_cache=None; noise_schedules={}
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate)
    model=AnglePredictor().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01); sch=build_scheduler(opt); step=0
    for ep in range(1,EPOCHS+1):
        model.train(); total=0.0
        epoch_iter = enumerate(loader, start=1)
        if tqdm:
                epoch_iter = tqdm(epoch_iter, total=len(loader), desc=f'Epoch {ep}', unit='batch')
        for bi, raw in epoch_iter:
            batch=raw.to(DEVICE)
            logits,mask=model(batch)
            if PRECOMPUTE_BASE:
                main=simulate_loss_cached(batch,logits,init_cache,ref_cache,noise_schedules)
            else:
                main=simulate_loss(batch,logits,noise_schedules)
            if AUX_ANGLE_LOSS:
                aux=angle_supervise_loss(logits,batch.param_angles_gt,mask)
            else:
                aux=torch.tensor(0.0,device=logits.device)
            loss=main + AUX_ANGLE_WEIGHT*aux
            opt.zero_grad(); loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step(); total+=loss.item(); step+=1
            if tqdm:
                epoch_iter.set_postfix(loss=f'{loss.item():.4f}', main=f'{main.item():.4f}', aux=f'{aux.item():.4f}')
            elif step % PRINT_INTERVAL==0:
                print(f'[Ep {ep} Step {step}] loss={loss.item():.6f} main={main.item():.6f} aux={aux.item():.6f}')
        if sch:
            sch.step()
        avg = total/len(loader)
        if tqdm:
            print(f'[Epoch {ep}] avg_loss={avg:.6f}')
        else:
            print(f'[Epoch {ep}] avg_loss={avg:.6f}')
    torch.save({'model':model.state_dict()},'minimal_model.pt'); print('Model saved -> minimal_model.pt')

if __name__=='__main__':
    train()
