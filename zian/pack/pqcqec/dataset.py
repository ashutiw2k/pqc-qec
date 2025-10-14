#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal dataset loader (no CLI, no training logic).

Provides:
- Constants (gate vocab + limits) matching the original PyTorch training script.
- CircuitDataset: parses directory or single json/jsonl file (new + legacy format).
- collate: converts list[dict] -> Batch tensors with padding.
- Helper function load_circuit_dataset(path, num_sample=None) returning CircuitDataset.

Intended usage (example):
    from pqcqec.pytorch_jax_port_dataset import load_circuit_dataset, collate, Batch
    ds = load_circuit_dataset(DATA_PATH, num_sample=128)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
"""
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
try:  # tqdm optional
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

# ========= Constants (copy from training script) =========
MAX_BASE_LEN = 500
MAX_PARAM    = 750
MAX_QUBITS   = 5
PAD_ID       = -1
BASE_GATES   = {'h':0,'x':1,'z':2,'cx':3,'cz':4}
PARAM_GATES  = {'rz':0,'rx':1}
INV_BASE     = {v:k for k,v in BASE_GATES.items()}
INV_PARAM    = {v:k for k,v in PARAM_GATES.items()}

# ========= Dataset =========
class CircuitDataset(Dataset):
    """Dataset replicating original parsing semantics (new + legacy format)."""
    def __init__(self, path: str, num_sample: Optional[int] = None):
        self.items: list[dict] = []
        self._next_index = 0
        if not os.path.exists(path):
            print(f"[WARN] Data path does not exist: {path}")
            return
        # Normalize num_sample
        if num_sample is not None:
            try:
                num_sample = int(num_sample)
                if num_sample <= 0:
                    num_sample = None
            except Exception:
                num_sample = None
        self._num_limit = num_sample
        self._num_sample_applied = False

        class _EarlyStop(Exception):
            pass

        def process_obj(o: dict):
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
                if self._num_limit is not None and self._next_index >= self._num_limit:
                    self._num_sample_applied = True
                    raise _EarlyStop
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
            if self._num_limit is not None and self._next_index >= self._num_limit:
                self._num_sample_applied = True
                raise _EarlyStop

        if os.path.isdir(path):
            files=[f for f in os.listdir(path) if f.lower().endswith(('.json','.jsonl'))]
            files.sort()
            iterator = tqdm(files, desc='Reading data files', unit='file') if tqdm else files
            try:
                for fname in iterator:
                    fp=os.path.join(path,fname)
                    try:
                        with open(fp,'r',encoding='utf-8') as fh:
                            for line in fh:
                                if not line.strip(): continue
                                process_obj(json.loads(line))
                                break
                    except _EarlyStop:
                        break
                    except Exception as e:
                        print(f"[WARN] Failed to read file {fp}: {e}")
            except _EarlyStop:
                pass
            if tqdm:
                print(f"[INFO] Loaded samples: {len(self.items)}")
        else:
            with open(path,'r',encoding='utf-8') as f:
                lines=f.readlines()
            iterator=tqdm(lines, desc='Reading lines', unit='line') if tqdm else lines
            try:
                for line in iterator:
                    if not line.strip(): continue
                    process_obj(json.loads(line))
            except _EarlyStop:
                pass
        if self._num_sample_applied:
            print(f"[INFO] NUM_SAMPLE={self._num_limit} applied during load; dataset length limited to {len(self.items)}")

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

@dataclass
class Batch:
    base_g:torch.Tensor; base_q1:torch.Tensor; base_q2:torch.Tensor
    param_g:torch.Tensor; param_q:torch.Tensor; param_after:torch.Tensor
    param_angles_gt:torch.Tensor; base_len:torch.Tensor; param_len:torch.Tensor; n_qubits:torch.Tensor; idx:torch.Tensor
    def to(self, device: torch.device):
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self

# Helper
_def_pad = lambda seq, pad, L: list(seq)[:L] + [pad]*max(0, L-len(seq))

def collate(samples: List[dict]) -> Batch:
    bg=[]; bq1=[]; bq2=[]; pg=[]; pq=[]; pafter=[]; pang=[]; base_l=[]; param_l=[]; nqs=[]; idxs=[]
    for o in samples:
        g=[BASE_GATES[x] for x in o['base_gates']]; p=[PARAM_GATES[x] for x in o['param_gates']]
        bg.append(_def_pad(g,PAD_ID,MAX_BASE_LEN))
        bq1.append(_def_pad(o['base_q1'],PAD_ID,MAX_BASE_LEN))
        bq2.append(_def_pad(o['base_q2'],PAD_ID,MAX_BASE_LEN))
        pg.append(_def_pad(p,PAD_ID,MAX_PARAM))
        pq.append(_def_pad(o['param_qubits'],PAD_ID,MAX_PARAM))
        pafter.append(_def_pad(o['after'],-999,MAX_PARAM))
        pang.append(_def_pad(o['param_angles_gt'],0.0,MAX_PARAM))
        base_l.append(len(g)); param_l.append(len(p)); nqs.append(o['n_qubits']); idxs.append(o['idx'])
    to_long=lambda x: torch.tensor(x,dtype=torch.long)
    return Batch(
        to_long(bg),to_long(bq1),to_long(bq2),to_long(pg),to_long(pq),to_long(pafter),
        torch.tensor(pang,dtype=torch.float32),to_long(base_l),to_long(param_l),to_long(nqs),to_long(idxs)
    )

def load_circuit_dataset(path: str, num_sample: Optional[int] = None) -> CircuitDataset:
    """Convenience wrapper.

    Args:
        path: directory or json/jsonl file.
        num_sample: optional early limit.
    Returns:
        CircuitDataset
    """
    return CircuitDataset(path, num_sample=num_sample)

__all__ = [
    'MAX_BASE_LEN','MAX_PARAM','MAX_QUBITS','PAD_ID','BASE_GATES','PARAM_GATES',
    'INV_BASE','INV_PARAM','CircuitDataset','Batch','collate','load_circuit_dataset'
]


def main():  # simple local smoke test
    data_path = r'A:/wings/transformers/data/5q_500g_single_qubit_gate_uncomp_circuit_data_processed'
    print(f'[TEST] Loading dataset from: {data_path}')
    ds = load_circuit_dataset(data_path, num_sample=8)  # load a few samples for speed
    print(f'[TEST] Loaded samples: {len(ds)}')
    if len(ds) == 0:
        print('[TEST] No data loaded (path missing or empty).')
        return
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    print('[TEST] First batch shapes:')
    print('  base_g:', batch.base_g.shape, 'param_g:', batch.param_g.shape)
    print('  base_len:', batch.base_len.tolist())
    print('  param_len:', batch.param_len.tolist())
    print('  n_qubits:', batch.n_qubits.tolist())
    # basic sanity
    assert (batch.base_len <= MAX_BASE_LEN).all()
    assert (batch.param_len <= MAX_PARAM).all()
    print('[TEST] Sanity checks passed.')


if __name__ == '__main__':
    main()
