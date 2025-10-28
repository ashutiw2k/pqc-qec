"""
PyTorch utilities for transformer training.

Includes:
- Dataset class for loading circuits from JSON
- DataLoader utilities
- Batching and collation functions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional


class CircuitDatasetPyTorch(Dataset):
    """
    PyTorch Dataset for quantum circuits loaded from JSON/JSONL files.
    
    Each circuit contains:
    - base_gates: List of gate names
    - base_q1, base_q2: Qubit indices
    - n_qubits: Number of qubits
    - idx: Unique circuit index
    """
    
    def __init__(
        self,
        data_path: str,
        n_qubits: int,
        max_circuits: Optional[int] = None
    ):
        """
        Initialize dataset by loading circuits from JSON file(s).
        
        Args:
            data_path: Path to JSON/JSONL file or directory
            n_qubits: Filter circuits to this qubit count only
            max_circuits: Maximum number of circuits to load (None = all)
        """
        self.n_qubits = n_qubits
        self.circuits = []
        
        # Load circuits
        if os.path.isdir(data_path):
            # Load all JSON/JSONL files in directory
            files = sorted([
                f for f in os.listdir(data_path)
                if f.endswith('.json') or f.endswith('.jsonl')
            ])
            for fname in files:
                fpath = os.path.join(data_path, fname)
                self._load_file(fpath, max_circuits)
                if max_circuits and len(self.circuits) >= max_circuits:
                    break
        else:
            # Load single file
            self._load_file(data_path, max_circuits)
        
        # Filter by n_qubits
        self.circuits = [c for c in self.circuits if c['n_qubits'] == n_qubits]
        
        if len(self.circuits) == 0:
            raise ValueError(f"No circuits found with n_qubits={n_qubits}")
        
        print(f"Loaded {len(self.circuits)} circuits with {n_qubits} qubits")
    
    def _load_file(self, fpath: str, max_circuits: Optional[int]):
        """Load circuits from a single JSON/JSONL file."""
        with open(fpath, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    circuit = self._parse_circuit(data)
                    if circuit:
                        self.circuits.append(circuit)
                        if max_circuits and len(self.circuits) >= max_circuits:
                            break
                except json.JSONDecodeError:
                    continue
    
    def _parse_circuit(self, data: Dict) -> Optional[Dict]:
        """Parse circuit from JSON data."""
        # Handle token format
        if 'base_circuit_tokens' in data:
            base_tokens = data['base_circuit_tokens']
            base_gates = []
            base_q1 = []
            base_q2 = []
            
            for tok in base_tokens:
                gate = tok[0]
                qubits = tok[1]
                
                # Skip parameterized gates
                if gate not in ['h', 'x', 'z', 'cx', 'cz']:
                    continue
                
                if len(qubits) == 1:
                    q1 = qubits[0]
                    q2 = -1
                elif len(qubits) >= 2:
                    q1, q2 = qubits[0], qubits[1]
                else:
                    continue
                
                base_gates.append(gate)
                base_q1.append(q1)
                base_q2.append(q2)
            
            # Infer n_qubits
            n_q = data.get('n_qubits')
            if n_q is None:
                all_qs = base_q1 + [q for q in base_q2 if q >= 0]
                n_q = max(all_qs) + 1 if all_qs else 1
        
        # Handle legacy format
        elif 'base_gates' in data:
            base_gates = data['base_gates']
            base_qubits = data.get('base_qubits', [[], []])
            base_q1 = base_qubits[0] if len(base_qubits) > 0 else []
            base_q2 = base_qubits[1] if len(base_qubits) > 1 else []
            
            n_q = data.get('n_qubits')
            if n_q is None:
                all_qs = base_q1 + [q for q in base_q2 if q >= 0]
                n_q = max(all_qs) + 1 if all_qs else 1
        else:
            return None
        
        if len(base_gates) == 0:
            return None
        
        return {
            'base_gates': base_gates,
            'base_q1': base_q1,
            'base_q2': base_q2,
            'n_qubits': n_q,
            'idx': len(self.circuits)
        }
    
    def __len__(self):
        return len(self.circuits)
    
    def __getitem__(self, idx):
        return self.circuits[idx]


def create_circuit_ops_from_data(circuit: Dict) -> List[Tuple]:
    """
    Convert circuit data to operation list format.
    
    Args:
        circuit: Circuit dictionary with base_gates, base_q1, base_q2
    
    Returns:
        circuit_ops: List of (gate_name, qubits, params) tuples
    """
    ops = []
    for i, gate in enumerate(circuit['base_gates']):
        q1 = circuit['base_q1'][i]
        q2 = circuit['base_q2'][i] if i < len(circuit['base_q2']) else -1
        
        if q2 >= 0:
            qubits = [q1, q2]
        else:
            qubits = [q1]
        
        ops.append((gate, qubits, []))
    
    return ops


def generate_random_initial_states(
    n_qubits: int,
    k_random: int,
    device: torch.device,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate K random computational basis states.
    
    Args:
        n_qubits: Number of qubits
        k_random: Number of random states
        device: torch device
        seed: Random seed (optional)
    
    Returns:
        states: [K, 2^n] random basis states
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    dim = 2 ** n_qubits
    states = []
    
    for _ in range(k_random):
        # Random computational basis state
        idx = np.random.randint(0, dim)
        state = torch.zeros(dim, dtype=torch.complex64, device=device)
        state[idx] = 1.0 + 0.0j
        states.append(state)
    
    return torch.stack(states)


def generate_fixed_noise(
    num_gates: int,
    noise_x_rad: float = np.pi / 100,
    noise_z_rad: float = np.pi / 100,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate fixed noise arrays for a circuit.
    
    Args:
        num_gates: Number of gates in circuit
        noise_x_rad: X-noise magnitude (radians)
        noise_z_rad: Z-noise magnitude (radians)
        seed: Random seed for reproducibility
    
    Returns:
        x_noise: [num_gates] X-rotation noise
        z_noise: [num_gates] Z-rotation noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_noise = np.random.uniform(-noise_x_rad, noise_x_rad, (num_gates,))
    z_noise = np.random.uniform(-noise_z_rad, noise_z_rad, (num_gates,))
    
    return torch.tensor(x_noise, dtype=torch.float32), torch.tensor(z_noise, dtype=torch.float32)


def pad_gate_sequence(
    gate_ids: torch.Tensor,
    wire1s: torch.Tensor,
    wire2s: torch.Tensor,
    max_len: int,
    pad_value: int = -1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Pad gate sequence to max_len.
    
    Args:
        gate_ids: [L] gate identifiers
        wire1s: [L] first wire indices
        wire2s: [L] second wire indices
        max_len: target length
        pad_value: padding value
    
    Returns:
        gate_ids_padded: [max_len]
        wire1s_padded: [max_len]
        wire2s_padded: [max_len]
        actual_len: original length before padding
    """
    actual_len = len(gate_ids)
    
    if actual_len >= max_len:
        return gate_ids[:max_len], wire1s[:max_len], wire2s[:max_len], max_len
    
    pad_len = max_len - actual_len
    
    gate_ids_padded = torch.cat([
        gate_ids,
        torch.full((pad_len,), pad_value, dtype=gate_ids.dtype, device=gate_ids.device)
    ])
    
    wire1s_padded = torch.cat([
        wire1s,
        torch.full((pad_len,), pad_value, dtype=wire1s.dtype, device=wire1s.device)
    ])
    
    wire2s_padded = torch.cat([
        wire2s,
        torch.full((pad_len,), pad_value, dtype=wire2s.dtype, device=wire2s.device)
    ])
    
    return gate_ids_padded, wire1s_padded, wire2s_padded, actual_len


def collate_circuit_batch(
    batch: List[Dict],
    gate_blocks: int,
    device: torch.device
) -> Dict:
    """
    Collate a batch of circuits for training.
    
    Args:
        batch: List of circuit dictionaries
        gate_blocks: Maximum gates per block
        device: torch device
    
    Returns:
        batch_dict: Dictionary with batched tensors
    """
    # Just return the list of circuits - processing happens per-circuit during training
    return batch
