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
    noise: Optional[Dict[Tuple[str, str], 
                        Tuple[str, str] | 
                        Tuple[Tuple[str, List], Tuple[str, List]] |
                        List[Tuple[str, List]]]] = None
) -> List[Tuple]:
    """
    Apply coherent noise by modifying consecutive gate pairs based on transformation rules.
    
    Uses non-overlapping transformations: once a gate is part of a transformed pair,
    it cannot be part of another transformation. Supports variable-length replacements.
    
    Args:
        base_ops: List of operations as (gate, qubits, params) tuples
        noise: Dict mapping (gate1, gate2) → replacement
            Replacement can be:
            - Simple 2-tuple: ('h', 'x')  # 2→2, inherit params
            - Extended 2-tuple: (('rx', [0.5]), ('rz', [0.3]))  # 2→2, custom params
            - List (variable length): [('h', []), ('z', []), ('x', [0.1])]  # 2→N
            
            If None, uses default rules: HH→HX, XX→XZ, ZZ→ZH
            
            Parameter inheritance for list form:
            - First element inherits params from gate1
            - Last element inherits params from gate2
            - Middle elements use explicit params or [] if None
    
    Returns:
        Modified list of operations with coherent gate errors applied.
        May have different length than input if variable-length rules are used.
        
    Note:
        For multi-qubit gates, qubit lists must match exactly in order.
        E.g., [0,1] will not match [1,0].
        
    Example:
        >>> # Simple 2→2 transformation
        >>> ops = [('h', [0], []), ('h', [0], []), ('x', [1], [])]
        >>> noisy = apply_gate_sequence_noise(ops)
        >>> # Result: [('h', [0], []), ('x', [0], []), ('x', [1], [])]
        
        >>> # Variable-length transformation 2→3
        >>> ops = [('h', [0], []), ('h', [0], [])]
        >>> noisy = apply_gate_sequence_noise(ops, {('h','h'): [('h',[]), ('z',[]), ('x',[])]})
        >>> # Result: [('h', [0], []), ('z', [0], []), ('x', [0], [])]
    """
    if not base_ops:
        return []
    
    if noise is None:
        noise = {
            ('h', 'h'): ('h', 'x'),   # HH → HX
            ('x', 'x'): ('x', 'z'),   # XX → XZ
            ('z', 'z'): ('z', 'h'),   # ZZ → ZH
        }
    
    # Normalize rules: convert all forms to list of (gate, params) tuples
    normalized_rules = {}
    for (g1, g2), replacement in noise.items():
        key = (g1.lower(), g2.lower())
        
        if isinstance(replacement, list):
            # List form: [('gate1', params1), ('gate2', params2), ...]
            normalized_rules[key] = [(g.lower(), p) for g, p in replacement]
        elif isinstance(replacement[0], tuple):
            # Extended 2-tuple form: (('gate1', params1), ('gate2', params2))
            (r1, p1), (r2, p2) = replacement
            normalized_rules[key] = [(r1.lower(), p1), (r2.lower(), p2)]
        else:
            # Simple 2-tuple form: ('gate1', 'gate2')
            r1, r2 = replacement
            normalized_rules[key] = [(r1.lower(), None), (r2.lower(), None)]
    
    # Pre-compute lowercase gate names to avoid repeated .lower() calls in loop
    gate_names_lower = [op[0].lower() for op in base_ops]
    
    # Use streaming builder to construct output (always creates a new list)
    # This ensures base_ops is never modified or returned directly
    noisy_ops = []
    i = 0
    
    while i < len(base_ops):
        # Check if this position can start a pair transformation
        if i < len(base_ops) - 1:
            op1 = base_ops[i]
            op2 = base_ops[i + 1]
            
            gate1, qubits1, params1 = op1
            gate2, qubits2, params2 = op2
            
            # Check if gates operate on the same qubits and match a rule
            if tuple(qubits1) == tuple(qubits2):
                gate_pair = (gate_names_lower[i], gate_names_lower[i + 1])
                
                if gate_pair in normalized_rules:
                    replacement_seq = normalized_rules[gate_pair]
                    
                    # Apply replacement sequence
                    for idx, (new_gate, new_params) in enumerate(replacement_seq):
                        # Parameter inheritance logic:
                        # - First element (idx==0): inherit from gate1 if params is None
                        # - Last element (idx==len-1): inherit from gate2 if params is None
                        # - Middle elements: use explicit params or []
                        if new_params is None:
                            if idx == 0:
                                final_params = params1
                            elif idx == len(replacement_seq) - 1:
                                final_params = params2
                            else:
                                final_params = []
                        else:
                            final_params = new_params
                        
                        noisy_ops.append((new_gate, qubits1, final_params))
                    
                    # Skip both original gates (they were replaced)
                    i += 2
                    continue
        
        # No transformation: copy original gate
        noisy_ops.append(base_ops[i])
        i += 1
    
    return noisy_ops


def apply_gate_sequence_noise_probabilistic(
    base_ops: List[Tuple],
    transformation_rules: Optional[Dict[Tuple[str, str], 
                                       Tuple[str, str] | 
                                       Tuple[Tuple[str, List], Tuple[str, List]] |
                                       List[Tuple[str, List]]]] = None,
    error_probability: float = 1.0,
    seed: Optional[int] = None
) -> List[Tuple]:
    """
    Apply coherent noise probabilistically - only transform some matching pairs.
    
    Uses non-overlapping transformations: once a gate is part of a transformed pair,
    it cannot be part of another transformation. Supports variable-length replacements.
    
    Args:
        base_ops: List of operations as (gate, qubits, params) tuples
        transformation_rules: Dict mapping (gate1, gate2) → replacement
            Replacement can be:
            - Simple 2-tuple: ('h', 'x')
            - Extended 2-tuple: (('rx', [0.5]), ('rz', [0.3]))
            - List (variable length): [('h', []), ('z', []), ('x', [0.1])]
            If None, uses default rules: HH→HX, XX→XZ, ZZ→ZH
        error_probability: Probability [0, 1] that a matching pair will be transformed
        seed: Random seed for reproducibility
    
    Returns:
        Modified list of operations with probabilistic coherent errors.
        May have different length than input if variable-length rules are used.
    """
    if not base_ops:
        return []
    
    if transformation_rules is None:
        transformation_rules = {
            ('h', 'h'): ('h', 'x'),
            ('x', 'x'): ('x', 'z'),
            ('z', 'z'): ('z', 'h'),
        }
    
    rng = np.random.RandomState(seed)
    
    # Normalize rules: convert all forms to list of (gate, params) tuples
    normalized_rules = {}
    for (g1, g2), replacement in transformation_rules.items():
        key = (g1.lower(), g2.lower())
        
        if isinstance(replacement, list):
            # List form: [('gate1', params1), ('gate2', params2), ...]
            normalized_rules[key] = [(g.lower(), p) for g, p in replacement]
        elif isinstance(replacement[0], tuple):
            # Extended 2-tuple form: (('gate1', params1), ('gate2', params2))
            (r1, p1), (r2, p2) = replacement
            normalized_rules[key] = [(r1.lower(), p1), (r2.lower(), p2)]
        else:
            # Simple 2-tuple form: ('gate1', 'gate2')
            r1, r2 = replacement
            normalized_rules[key] = [(r1.lower(), None), (r2.lower(), None)]
    
    # Pre-compute lowercase gate names to avoid repeated .lower() calls in loop
    gate_names_lower = [op[0].lower() for op in base_ops]
    
    # Use streaming builder to construct output
    noisy_ops = []
    any_transformation = False
    i = 0
    
    while i < len(base_ops):
        # Check if this position can start a pair transformation
        if i < len(base_ops) - 1:
            op1 = base_ops[i]
            op2 = base_ops[i + 1]
            
            gate1, qubits1, params1 = op1
            gate2, qubits2, params2 = op2
            
            # Check if gates operate on the same qubits and match a rule
            if tuple(qubits1) == tuple(qubits2):
                gate_pair = (gate_names_lower[i], gate_names_lower[i + 1])
                
                if gate_pair in normalized_rules:
                    # Probabilistically apply transformation
                    if rng.random() < error_probability:
                        any_transformation = True
                        replacement_seq = normalized_rules[gate_pair]
                        
                        # Apply replacement sequence
                        for idx, (new_gate, new_params) in enumerate(replacement_seq):
                            # Parameter inheritance logic
                            if new_params is None:
                                if idx == 0:
                                    final_params = params1
                                elif idx == len(replacement_seq) - 1:
                                    final_params = params2
                                else:
                                    final_params = []
                            else:
                                final_params = new_params
                            
                            noisy_ops.append((new_gate, qubits1, final_params))
                        
                        # Format qubit display (single or multi-qubit)
                        qubit_str = f"qubit {qubits1[0]}" if len(qubits1) == 1 else f"qubits {qubits1}"
                        replacement_str = ''.join(g.upper() for g, _ in replacement_seq)
                        print(f"  Coherent error: {gate1.upper()}{gate2.upper()} → "
                              f"{replacement_str} on {qubit_str} (gates {i}→{i+1})")
                        
                        # Skip both original gates (they were replaced)
                        i += 2
                        continue
        
        # No transformation: copy original gate
        noisy_ops.append(base_ops[i])
        i += 1
    
    # Return original if no transformations occurred
    return noisy_ops if any_transformation else base_ops


