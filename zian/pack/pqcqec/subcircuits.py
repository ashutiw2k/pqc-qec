#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SubcircuitDataset: expands each multi-qubit circuit into 1-qubit subcircuits.
We reuse Batch from simulator_core collate, but here we produce per-subcircuit items
compatible with the same collate: base_gates stays 1q only; param tokens remain as-is.

This is a minimal implementation sufficient for train_subcircuits.
"""
from __future__ import annotations
from typing import List, Dict, Optional

from dataclasses import dataclass

from .simulator_core import CircuitDataset, BASE_GATES, PARAM_GATES

class SubcircuitDataset:
    def __init__(self, base: CircuitDataset):
        self.items: List[Dict] = []
        sub_idx = 0  # unique id per subcircuit sample for cache mapping
        # Build one 1-qubit item per original circuit per qubit
        for it in base.items:
            n_q = int(it.get('n_qubits', 1))
            base_gates: List[str] = list(it['base_gates'])
            base_q1: List[int] = list(it['base_q1'])
            base_q2: List[int] = list(it['base_q2'])
            param_gates: List[str] = list(it['param_gates'])
            param_qubits: List[int] = list(it['param_qubits'])
            after: List[int] = list(it['after'])
            param_angles: List[float] = list(it.get('param_angles_gt', [0.0]*len(param_gates)))
            for q in range(n_q):
                # Filter base gates that touch this qubit and are 1q gates
                b_g = []
                b_q1 = []
                b_q2 = []
                for g, q1, q2 in zip(base_gates, base_q1, base_q2):
                    if g not in BASE_GATES:
                        continue
                    if q2 != -1:
                        # skip 2q gates for 1q subcircuit
                        continue
                    if q1 == q:
                        b_g.append(g)
                        b_q1.append(0)  # remap to single qubit index 0
                        b_q2.append(-1)
                # Filter param gates on this qubit
                p_g = []
                p_q = []
                p_after = []
                p_ang = []
                # We remap after indices to the last seen base index within the filtered list
                # Build a map from original base index to subcircuit base index
                orig_to_new = {}
                new_idx = 0
                for i,(g, q1, q2) in enumerate(zip(base_gates, base_q1, base_q2)):
                    if g not in BASE_GATES:
                        continue
                    if q2 != -1:
                        continue
                    if q1 == q:
                        orig_to_new[i] = new_idx
                        new_idx += 1
                for g, q1, a, ang in zip(param_gates, param_qubits, after, param_angles):
                    if g not in PARAM_GATES:
                        continue
                    if q1 != q:
                        continue
                    p_g.append(g)
                    p_q.append(0)
                    # map 'after' to new base index if possible, else -1
                    p_after.append(orig_to_new.get(a, -1))
                    p_ang.append(ang)
                self.items.append(dict(
                    idx=sub_idx,
                    base_gates=b_g,
                    base_q1=b_q1,
                    base_q2=b_q2,
                    param_gates=p_g,
                    param_qubits=p_q,
                    after=p_after,
                    param_angles_gt=p_ang,
                    n_qubits=1,
                ))
                sub_idx += 1
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]
