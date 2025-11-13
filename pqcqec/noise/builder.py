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
    noise: Optional[Dict[Tuple[str, ...], 
                        Tuple[str, str] | 
                        Tuple[Tuple[str, List], Tuple[str, List]] |
                        List[Tuple[str, List]]]] = None
) -> List[Tuple]:
    """
    Apply coherent noise by modifying consecutive gate sequences based on transformation rules.
    
    Uses non-overlapping transformations: once gates are part of a transformed sequence,
    they cannot be part of another transformation. Supports variable-length input patterns
    and variable-length output replacements.
    
    Args:
        base_ops: List of operations as (gate, qubits, params) tuples
        noise: Dict mapping gate_pattern → replacement
            Pattern key can be any length tuple: ('h', 'h'), ('h', 'h', 'h'), ('x', 'y', 'z'), etc.
            
            Replacement can be:
            - Simple tuple: ('h', 'x')  # For 2-gate patterns, 2→2, inherit params
            - Extended tuple: (('rx', [0.5]), ('rz', [0.3]))  # 2→2, custom params
            - List (any length): [('h', []), ('z', []), ('x', [0.1])]  # N→M transformation
            
            If None, uses default rules: HH→HX, XX→XZ, ZZ→ZH
            
            Parameter inheritance for list form:
            - First element inherits params from first gate in pattern
            - Last element inherits params from last gate in pattern
            - Middle elements use explicit params or [] if None
    
    Returns:
        Modified list of operations with coherent gate errors applied.
        May have different length than input if variable-length rules are used.
        Always returns a new list (never modifies base_ops).
        
    Note:
        - Patterns match consecutive gates on the SAME qubits only
        - Multi-qubit gates: qubit lists must match exactly in order ([0,1] ≠ [1,0])
        - Greedy longest-match: longer patterns take priority over shorter ones
        - Non-overlapping: once gates are consumed, they can't match again
        
    Example:
        >>> # 2-gate pattern → 2 gates
        >>> ops = [('h', [0], []), ('h', [0], []), ('x', [1], [])]
        >>> noisy = apply_gate_sequence_noise(ops, {('h','h'): ('h', 'x')})
        >>> # Result: [('h', [0], []), ('x', [0], []), ('x', [1], [])]
        
        >>> # 2-gate pattern → 3 gates
        >>> ops = [('h', [0], []), ('h', [0], [])]
        >>> noisy = apply_gate_sequence_noise(ops, {('h','h'): [('h',[]), ('z',[]), ('x',[])]})
        >>> # Result: [('h', [0], []), ('z', [0], []), ('x', [0], [])]
        
        >>> # 3-gate pattern → 1 gate
        >>> ops = [('h', [0], []), ('h', [0], []), ('h', [0], [])]
        >>> noisy = apply_gate_sequence_noise(ops, {('h','h','h'): [('z',[])]})
        >>> # Result: [('z', [0], [])]
        
        >>> # Greedy matching: longest pattern wins
        >>> ops = [('x', [0], []), ('x', [0], []), ('x', [0], [])]
        >>> noisy = apply_gate_sequence_noise(ops, {
        ...     ('x','x'): [('y',[])],      # 2→1
        ...     ('x','x','x'): [('z',[])]   # 3→1 (this wins)
        ... })
        >>> # Result: [('z', [0], [])]
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
    # Also organize by pattern length for efficient lookup
    normalized_rules = {}
    pattern_lengths = set()
    
    for gate_pattern, replacement in noise.items():
        # Ensure gate_pattern is a tuple
        if not isinstance(gate_pattern, tuple):
            raise ValueError(f"Pattern key must be a tuple, got {type(gate_pattern)}: {gate_pattern}")
        
        # Convert pattern to lowercase
        key = tuple(g.lower() for g in gate_pattern)
        pattern_lengths.add(len(key))
        
        # Normalize replacement to list of (gate, params) tuples
        if isinstance(replacement, list):
            # List form: [('gate1', params1), ('gate2', params2), ...]
            normalized_rules[key] = [(g.lower(), p) for g, p in replacement]
        elif len(gate_pattern) == 2 and len(replacement) == 2:
            # Could be simple 2-tuple or extended 2-tuple, check first element
            if isinstance(replacement[0], tuple):
                # Extended 2-tuple form: (('gate1', params1), ('gate2', params2))
                (r1, p1), (r2, p2) = replacement
                normalized_rules[key] = [(r1.lower(), p1), (r2.lower(), p2)]
            else:
                # Simple 2-tuple form: ('gate1', 'gate2')
                r1, r2 = replacement
                normalized_rules[key] = [(r1.lower(), None), (r2.lower(), None)]
        else:
            raise ValueError(
                f"Invalid replacement format for pattern {gate_pattern}: {replacement}. "
                f"Expected list of (gate, params) tuples."
            )
    
    # Sort pattern lengths from longest to shortest (greedy longest match)
    sorted_pattern_lengths = sorted(pattern_lengths, reverse=True)
    
    # Pre-compute lowercase gate names to avoid repeated .lower() calls in loop
    gate_names_lower = [op[0].lower() for op in base_ops]
    
    # Use streaming builder to construct output (always creates a new list)
    # This ensures base_ops is never modified or returned directly
    noisy_ops = []
    i = 0
    
    while i < len(base_ops):
        matched = False
        
        # Try patterns from longest to shortest (greedy matching)
        for pattern_length in sorted_pattern_lengths:
            # Check if we have enough gates left
            if i + pattern_length > len(base_ops):
                continue
            
            # Extract the gate sequence pattern
            gate_pattern = tuple(gate_names_lower[i:i+pattern_length])
            
            # Check if this pattern is in our rules
            if gate_pattern in normalized_rules:
                # Verify all gates in the pattern operate on the same qubits
                first_qubits = tuple(base_ops[i][1])
                qubits_match = all(
                    tuple(base_ops[i+j][1]) == first_qubits
                    for j in range(pattern_length)
                )
                
                if qubits_match:
                    # Extract params from first and last gates in pattern
                    first_gate_params = base_ops[i][2]
                    last_gate_params = base_ops[i + pattern_length - 1][2]
                    
                    # Apply the replacement sequence
                    replacement_seq = normalized_rules[gate_pattern]
                    
                    for idx, (new_gate, new_params) in enumerate(replacement_seq):
                        # Parameter inheritance logic:
                        # - First element (idx==0): inherit from first gate if params is None
                        # - Last element (idx==len-1): inherit from last gate if params is None
                        # - Middle elements: use explicit params or []
                        if new_params is None:
                            if idx == 0:
                                final_params = first_gate_params
                            elif idx == len(replacement_seq) - 1:
                                final_params = last_gate_params
                            else:
                                final_params = []
                        else:
                            final_params = new_params
                        
                        noisy_ops.append((new_gate, first_qubits, final_params))
                    
                    # Skip all gates in the matched pattern
                    i += pattern_length
                    matched = True
                    break  # Break out of pattern_length loop
        
        # No transformation: copy original gate
        if not matched:
            noisy_ops.append(base_ops[i])
            i += 1
    
    return noisy_ops


def apply_gate_sequence_noise_probabilistic(
    base_ops: List[Tuple],
    transformation_rules: Optional[Dict[Tuple[str, ...], 
                                       Tuple[str, str] | 
                                       Tuple[Tuple[str, List], Tuple[str, List]] |
                                       List[Tuple[str, List]]]] = None,
    error_probability: float = 1.0,
    seed: Optional[int] = None
) -> List[Tuple]:
    """
    Apply coherent noise probabilistically - only transform some matching sequences.
    
    Uses non-overlapping transformations: once gates are part of a transformed sequence,
    they cannot be part of another transformation. Supports variable-length input patterns
    and variable-length output replacements.
    
    Args:
        base_ops: List of operations as (gate, qubits, params) tuples
        transformation_rules: Dict mapping gate_pattern → replacement
            Pattern key can be any length tuple: ('h', 'h'), ('h', 'h', 'h'), etc.
            Replacement can be:
            - Simple tuple: ('h', 'x')
            - Extended tuple: (('rx', [0.5]), ('rz', [0.3]))
            - List (any length): [('h', []), ('z', []), ('x', [0.1])]
            If None, uses default rules: HH→HX, XX→XZ, ZZ→ZH
        error_probability: Probability [0, 1] that a matching pattern will be transformed
        seed: Random seed for reproducibility
    
    Returns:
        Modified list of operations with probabilistic coherent errors.
        May have different length than input if variable-length rules are used.
        Always returns a new list.
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
    # Also organize by pattern length for efficient lookup
    normalized_rules = {}
    pattern_lengths = set()
    
    for gate_pattern, replacement in transformation_rules.items():
        # Ensure gate_pattern is a tuple
        if not isinstance(gate_pattern, tuple):
            raise ValueError(f"Pattern key must be a tuple, got {type(gate_pattern)}: {gate_pattern}")
        
        # Convert pattern to lowercase
        key = tuple(g.lower() for g in gate_pattern)
        pattern_lengths.add(len(key))
        
        # Normalize replacement to list of (gate, params) tuples
        if isinstance(replacement, list):
            # List form: [('gate1', params1), ('gate2', params2), ...]
            normalized_rules[key] = [(g.lower(), p) for g, p in replacement]
        elif len(gate_pattern) == 2 and len(replacement) == 2:
            # Could be simple 2-tuple or extended 2-tuple, check first element
            if isinstance(replacement[0], tuple):
                # Extended 2-tuple form: (('gate1', params1), ('gate2', params2))
                (r1, p1), (r2, p2) = replacement
                normalized_rules[key] = [(r1.lower(), p1), (r2.lower(), p2)]
            else:
                # Simple 2-tuple form: ('gate1', 'gate2')
                r1, r2 = replacement
                normalized_rules[key] = [(r1.lower(), None), (r2.lower(), None)]
        else:
            raise ValueError(
                f"Invalid replacement format for pattern {gate_pattern}: {replacement}. "
                f"Expected list of (gate, params) tuples."
            )
    
    # Sort pattern lengths from longest to shortest (greedy longest match)
    sorted_pattern_lengths = sorted(pattern_lengths, reverse=True)
    
    # Pre-compute lowercase gate names to avoid repeated .lower() calls in loop
    gate_names_lower = [op[0].lower() for op in base_ops]
    
    # Use streaming builder to construct output
    noisy_ops = []
    i = 0
    
    while i < len(base_ops):
        matched = False
        
        # Try patterns from longest to shortest (greedy matching)
        for pattern_length in sorted_pattern_lengths:
            # Check if we have enough gates left
            if i + pattern_length > len(base_ops):
                continue
            
            # Extract the gate sequence pattern
            gate_pattern = tuple(gate_names_lower[i:i+pattern_length])
            
            # Check if this pattern is in our rules
            if gate_pattern in normalized_rules:
                # Verify all gates in the pattern operate on the same qubits
                first_qubits = tuple(base_ops[i][1])
                qubits_match = all(
                    tuple(base_ops[i+j][1]) == first_qubits
                    for j in range(pattern_length)
                )
                
                if qubits_match:
                    # Probabilistically apply transformation
                    if rng.random() < error_probability:
                        
                        # Extract params from first and last gates in pattern
                        first_gate_params = base_ops[i][2]
                        last_gate_params = base_ops[i + pattern_length - 1][2]
                        
                        # Apply the replacement sequence
                        replacement_seq = normalized_rules[gate_pattern]
                        
                        for idx, (new_gate, new_params) in enumerate(replacement_seq):
                            # Parameter inheritance logic
                            if new_params is None:
                                if idx == 0:
                                    final_params = first_gate_params
                                elif idx == len(replacement_seq) - 1:
                                    final_params = last_gate_params
                                else:
                                    final_params = []
                            else:
                                final_params = new_params
                            
                            noisy_ops.append((new_gate, first_qubits, final_params))
                        
                        # Format pattern display
                        pattern_str = ''.join(g.upper() for g in gate_pattern)
                        replacement_str = ''.join(g.upper() for g, _ in replacement_seq)
                        qubit_str = f"qubit {first_qubits[0]}" if len(first_qubits) == 1 else f"qubits {first_qubits}"
                        print(f"  Coherent error: {pattern_str} → {replacement_str} on {qubit_str} (gates {i}→{i+pattern_length-1})")
                        
                        # Skip all gates in the matched pattern
                        i += pattern_length
                        matched = True
                        break  # Break out of pattern_length loop
        
        # No transformation: copy original gate
        if not matched:
            noisy_ops.append(base_ops[i])
            i += 1
    
    # Always return a new list (for consistency with deterministic version)
    return noisy_ops


