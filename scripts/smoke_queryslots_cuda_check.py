#!/usr/bin/env python3
"""
Lightweight smoke check for ChatGPTTransformerQuerySlotsCUDA notebook setup:
- Verifies dataset/config paths
- Builds tokenizer/dataset and a DataLoader
- Prints a single batch shapes

Note: Does not require JAX/Pennylane and does not execute CUDA model.
Run: python scripts/smoke_queryslots_cuda_check.py
"""
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


def main() -> None:
    nb_dir = Path(__file__).resolve().parents[1] / 'testnotebooks' / 'transformers'
    data_path = (nb_dir / '../../nogit/quaternion/no_uncomp_data/rzrxrz/3q_4g_4blk_data').resolve()
    cfg_path = data_path / 'config.json'
    good_dir = data_path / 'good_fidelity'
    assert data_path.exists(), f"DATA_PATH missing: {data_path}"
    assert cfg_path.exists(), f"CONFIG missing: {cfg_path}"
    assert good_dir.exists(), f"good_fidelity dir missing: {good_dir}"

    with cfg_path.open('r') as f:
        cfg = json.load(f)

    num_qubits = cfg.get('qubits', [3])[0]
    print('Loaded CONFIG:', cfg)
    print(f'num_qubits={num_qubits}')

    # Minimal tokenizer mirroring QuantumCircuitTokenizer behavior for encode()
    from itertools import permutations, combinations

    class QuantumCircuitTokenizer:
        def __init__(self, max_qubits: int):
            self.max_qubits = max_qubits
            self.gates_config = {
                'special_tokens': ['[PAD]', '[UNK]'],
                'gates_1q': ['x', 'z', 'h', 'rx', 'rz'],
                'gates_2q': {
                    'cx': {'ordered': True},
                    'cz': {'ordered': False},
                },
            }
            self.token2id, self.id2token = self._build_vocab()
            self.pad_token_id = self.token2id.get(('[PAD]', ()), 0)

        def _build_vocab(self):
            token2id, id2token = {}, {}
            nid = 0
            for sp in self.gates_config['special_tokens']:
                token = (sp, ())
                token2id[token] = nid; id2token[nid] = token; nid += 1
            for g in self.gates_config['gates_1q']:
                for q in range(self.max_qubits):
                    token = (g, (q,))
                    token2id[token] = nid; id2token[nid] = token; nid += 1
            for g, props in self.gates_config['gates_2q'].items():
                pairs = permutations(range(self.max_qubits), 2) if props['ordered'] else combinations(range(self.max_qubits), 2)
                for qs in pairs:
                    token = (g, tuple(qs))
                    token2id[token] = nid; id2token[nid] = token; nid += 1
            return token2id, id2token

        @property
        def vocab_size(self) -> int:
            return len(self.token2id)

        def encode(self, circuit: Sequence[Tuple[str, Sequence[int], Any]], pad_to_len: int | None = None) -> Dict[str, torch.Tensor]:
            ids: List[int] = []
            gate_properties = self.gates_config['gates_2q']
            for gate, qlist, _ in circuit:
                if any(q >= self.max_qubits for q in qlist):
                    raise ValueError(f"qubit index in {qlist} exceeds max_qubits {self.max_qubits}")
                q_tuple = tuple(qlist)
                if len(q_tuple) == 2 and not gate_properties.get(gate, {}).get('ordered', True):
                    q_tuple = tuple(sorted(q_tuple))
                token = (gate, q_tuple)
                ids.append(self.token2id.get(token, 1))
            if pad_to_len is not None:
                ids = ids[:pad_to_len] + [self.pad_token_id] * max(0, pad_to_len - len(ids))
            input_ids = torch.tensor(ids, dtype=torch.long)
            attn_mask = (input_ids != self.pad_token_id).long()
            return {'input_ids': input_ids, 'attention_mask': attn_mask}

    class QuantumCircuitDataset(Dataset):
        def __init__(self, circuits_data: List[Sequence[Tuple[str, Sequence[int], Sequence[float]]]], tokenizer: QuantumCircuitTokenizer):
            self.circuits_data = circuits_data
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.circuits_data)
        def __getitem__(self, idx):
            circuit, pqc_params = self.circuits_data[idx]
            enc = self.tokenizer.encode(circuit)
            return enc, torch.tensor(pqc_params)

    # Load a couple of JSON samples to form dataset entries
    samples = []
    for i, fn in enumerate(sorted(good_dir.glob('*.json'))[:4]):
        with fn.open('r') as f:
            d = json.load(f)
        samples.append((d['base_circuit_tokens'], d['pqc_params']))

    tok = QuantumCircuitTokenizer(max_qubits=num_qubits)
    ds = QuantumCircuitDataset(samples, tok)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(dl))
    enc, params = batch
    print('Batch input_ids shape:', enc['input_ids'].shape)
    print('Batch attention_mask shape:', enc['attention_mask'].shape)
    print('Batch params shape:', params.shape)
    print('Vocab size:', tok.vocab_size)

    print('Smoke check OK')


if __name__ == '__main__':
    main()

