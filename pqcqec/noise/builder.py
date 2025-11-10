import numpy as np
from typing import List, Tuple, Dict, Optional


def add_rotation_noise_to_base_ops(base_ops, noise: Dict[str, np.ndarray]) -> List[Tuple]:
    """Add noise operations to a list of base operations according to a noise model."""
    noisy_ops = []
    x_noise = noise.get('x_noise', np.zeros(len(base_ops)))
    z_noise = noise.get('z_noise', np.zeros(len(base_ops)))
    for i, op in enumerate(base_ops):
        noisy_ops.append(op)
        gate, qubits, params = op
        for q in qubits:
            if x_noise[q] > 0:
                noisy_ops.append(('rx', [q], [x_noise[i]]))  # Add X error
            if z_noise[q] > 0:
                noisy_ops.append(('rz', [q], [z_noise[i]]))  # Add Z error

    return noisy_ops


def apply_gate_sequence_noise(
    base_ops: List[Tuple],
    noise: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
    seed: Optional[int] = None
) -> List[Tuple]:
    """
    Apply coherent noise by modifying gate sequences based on transformation rules.
    
    Instead of adding rotation errors, this modifies consecutive gate pairs on the same 
    qubit to simulate coherent errors in gate implementation.
    
    Args:
        base_ops: List of operations as (gate, qubits, params) tuples
        noise: Dict mapping (gate1, gate2) → (gate1, modified_gate2)
            If None, uses default rules: HH→HX, XX→XZ, ZZ→ZH
        seed: Random seed for determining which pairs to modify (future extension)
    
    Returns:
        Modified list of operations with coherent gate errors applied
        
    Example:
        >>> ops = [('h', [0], []), ('h', [0], []), ('x', [1], [])]
        >>> noisy = apply_gate_sequence_noise(ops)
        >>> # Result: [('h', [0], []), ('x', [0], []), ('x', [1], [])]
        >>> # The HH pair on qubit 0 was transformed to HX
    """
    if noise is None:
        # Default transformation rules
        noise = {
            ('h', 'h'): ('h', 'x'),   # HH → HX
            ('x', 'x'): ('x', 'z'),   # XX → XZ
            ('z', 'z'): ('z', 'h'),   # ZZ → ZH
        }
    
    # Future extension: use seed for probabilistic transformations
    # if seed is not None:
    #     rng = np.random.RandomState(seed)
    
    # Normalize gate names to lowercase for matching
    normalized_rules = {
        (g1.lower(), g2.lower()): (r1.lower(), r2.lower())
        for (g1, g2), (r1, r2) in noise.items()
    }
    
    # Track the last gate applied to each qubit (using ORIGINAL gates for pair detection)
    last_gate_per_qubit: Dict[int, Tuple[int, str]] = {}  # qubit → (index, gate_name)
    
    # Create output list (will be modified in place)
    noisy_ops = list(base_ops)
    
    # Scan through operations looking for matching pairs
    for idx, op in enumerate(base_ops):
        gate, qubits, params = op
        gate_lower = gate.lower()
        
        # Only consider single-qubit gates for now
        if len(qubits) != 1:
            # Update tracking for multi-qubit gates but don't transform
            for q in qubits:
                last_gate_per_qubit[q] = (idx, gate_lower)
            continue
        
        qubit = qubits[0]
        
        # Check if we've seen a gate on this qubit before
        if qubit in last_gate_per_qubit:
            prev_idx, prev_gate = last_gate_per_qubit[qubit]
            gate_pair = (prev_gate, gate_lower)
            
            # Check if this pair matches a transformation rule
            if gate_pair in normalized_rules:
                # Apply transformation: modify current gate
                _, new_gate = normalized_rules[gate_pair]
                
                # Replace the current gate with the transformed version
                noisy_ops[idx] = (new_gate, qubits, params)
                
                # print(f"  Noise applied: {prev_gate.upper()}{gate_lower.upper()} → "
                #       f"{prev_gate.upper()}{new_gate.upper()} on qubit {qubit} "
                #       f"(gates {prev_idx}, {idx})")
        
        # Update tracking with ORIGINAL gate (not modified) to avoid cascading transformations
        last_gate_per_qubit[qubit] = (idx, gate_lower)
    
    return noisy_ops


def apply_gate_sequence_noise_probabilistic(
    base_ops: List[Tuple],
    transformation_rules: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
    error_probability: float = 1.0,
    seed: Optional[int] = None
) -> List[Tuple]:
    """
    Apply coherent noise probabilistically - only transform some matching pairs.
    
    This variant adds stochasticity by randomly selecting which matching gate pairs
    to transform based on error_probability.
    
    Args:
        base_ops: List of operations as (gate, qubits, params) tuples
        transformation_rules: Dict mapping (gate1, gate2) → (gate1, modified_gate2)
            If None, uses default rules: HH→HX, XX→XZ, ZZ→ZH
        error_probability: Probability [0, 1] that a matching pair will be transformed
        seed: Random seed for reproducibility
    
    Returns:
        Modified list of operations with probabilistic coherent errors
    """
    if transformation_rules is None:
        transformation_rules = {
            ('h', 'h'): ('h', 'x'),
            ('x', 'x'): ('x', 'z'),
            ('z', 'z'): ('z', 'h'),
        }
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    normalized_rules = {
        (g1.lower(), g2.lower()): (r1.lower(), r2.lower())
        for (g1, g2), (r1, r2) in transformation_rules.items()
    }
    
    last_gate_per_qubit: Dict[int, Tuple[int, str]] = {}
    noisy_ops = list(base_ops)
    
    for idx, op in enumerate(base_ops):
        gate, qubits, params = op
        gate_lower = gate.lower()
        
        if len(qubits) != 1:
            for q in qubits:
                last_gate_per_qubit[q] = (idx, gate_lower)
            continue
        
        qubit = qubits[0]
        
        if qubit in last_gate_per_qubit:
            prev_idx, prev_gate = last_gate_per_qubit[qubit]
            gate_pair = (prev_gate, gate_lower)
            
            if gate_pair in normalized_rules:
                # Probabilistically decide whether to apply transformation
                if rng.random() < error_probability:
                    _, new_gate = normalized_rules[gate_pair]
                    noisy_ops[idx] = (new_gate, qubits, params)
                    
                    print(f"  Coherent error: {prev_gate.upper()}{gate_lower.upper()} → "
                          f"{prev_gate.upper()}{new_gate.upper()} on qubit {qubit} "
                          f"(gates {prev_idx}→{idx})")
        
        # Use ORIGINAL gate (not modified) to avoid cascading transformations
        last_gate_per_qubit[qubit] = (idx, gate_lower)
    
    return noisy_ops


