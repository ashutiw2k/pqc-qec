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
    noise: Optional[Dict[Tuple[str, str], Tuple[str, str] | Tuple[Tuple[str, List], Tuple[str, List]]]] = None
) -> List[Tuple]:
    """
    Apply coherent noise by modifying consecutive gate pairs based on transformation rules.
    
    Uses non-overlapping transformations: once a gate is part of a transformed pair,
    it cannot be part of another transformation. Supports both single-qubit and multi-qubit gates.
    
    Args:
        base_ops: List of operations as (gate, qubits, params) tuples
        noise: Dict mapping (gate1, gate2) → (new_gate1, new_gate2) or
                                          → ((new_gate1, new_params1), (new_gate2, new_params2))
            If None, uses default rules: HH→HX, XX→XZ, ZZ→ZH
            
            Simple form: ('h', 'h'): ('h', 'x')  # Keeps original params
            Extended form: ('h', 'h'): (('rx', [0.5]), ('rz', [0.3]))  # Custom params
    
    Returns:
        Modified list of operations with coherent gate errors applied.
        Returns original list if no transformations are applied.
        
    Note:
        For multi-qubit gates, qubit lists must match exactly in order.
        E.g., [0,1] will not match [1,0]. If order doesn't matter for your gates,
        normalize qubit lists before calling this function.
        
    Example:
        >>> ops = [('h', [0], []), ('h', [0], []), ('x', [1], [])]
        >>> noisy = apply_gate_sequence_noise(ops)
        >>> # Result: [('h', [0], []), ('x', [0], []), ('x', [1], [])]
    """
    if not base_ops:
        return []
    
    if noise is None:
        noise = {
            ('h', 'h'): ('h', 'x'),   # HH → HX
            ('x', 'x'): ('x', 'z'),   # XX → XZ
            ('z', 'z'): ('z', 'h'),   # ZZ → ZH
        }
    
    # Normalize rules: support both simple (gate names) and extended (gate, params) forms
    normalized_rules = {}
    for (g1, g2), replacement in noise.items():
        key = (g1.lower(), g2.lower())
        
        # Check if replacement is extended form with params
        if isinstance(replacement[0], tuple):
            # Extended form: ((gate1, params1), (gate2, params2))
            (r1, p1), (r2, p2) = replacement
            normalized_rules[key] = ((r1.lower(), p1), (r2.lower(), p2))
        else:
            # Simple form: (gate1, gate2) - inherit original params
            r1, r2 = replacement
            normalized_rules[key] = ((r1.lower(), None), (r2.lower(), None))
    
    # Pre-compute lowercase gate names to avoid repeated .lower() calls in loop
    gate_names_lower = [op[0].lower() for op in base_ops]
    
    # Track which gate indices have been transformed (global tracking, not per-qubit)
    # O(n) space instead of O(num_qubits × n)
    transformed = [False] * len(base_ops)
    
    # Lazy copy: only create noisy_ops when first transformation occurs
    noisy_ops = None
    
    # Scan for consecutive pairs using index-based loop (avoids O(n) slice copy)
    for i in range(len(base_ops) - 1):
        op1 = base_ops[i]
        op2 = base_ops[i + 1]
        
        gate1, qubits1, params1 = op1
        gate2, qubits2, params2 = op2
        
        # Check if gates operate on the same qubits (supports multi-qubit gates)
        # Note: requires exact qubit order match
        if tuple(qubits1) != tuple(qubits2):
            continue
        
        # Skip if either gate already part of a transformation
        if transformed[i] or transformed[i + 1]:
            continue
        
        # Check if this pair matches a transformation rule (using cached lowercase names)
        gate_pair = (gate_names_lower[i], gate_names_lower[i + 1])
        if gate_pair in normalized_rules:
            (new_gate1, new_params1), (new_gate2, new_params2) = normalized_rules[gate_pair]
            
            # Use custom params if provided, otherwise inherit original
            final_params1 = new_params1 if new_params1 is not None else params1
            final_params2 = new_params2 if new_params2 is not None else params2
            
            # Lazy copy: create noisy_ops on first transformation
            if noisy_ops is None:
                noisy_ops = list(base_ops)
            
            # Apply transformation
            noisy_ops[i] = (new_gate1, qubits1, final_params1)
            noisy_ops[i + 1] = (new_gate2, qubits2, final_params2)
            
            # Mark both gates as transformed
            transformed[i] = True
            transformed[i + 1] = True
    
    # Return original list if no transformations occurred (avoids unnecessary copy)
    return noisy_ops if noisy_ops is not None else base_ops


def apply_gate_sequence_noise_probabilistic(
    base_ops: List[Tuple],
    transformation_rules: Optional[Dict[Tuple[str, str], Tuple[str, str] | Tuple[Tuple[str, List], Tuple[str, List]]]] = None,
    error_probability: float = 1.0,
    seed: Optional[int] = None
) -> List[Tuple]:
    """
    Apply coherent noise probabilistically - only transform some matching pairs.
    
    Uses non-overlapping transformations: once a gate is part of a transformed pair,
    it cannot be part of another transformation. Supports both single-qubit and multi-qubit gates.
    
    Args:
        base_ops: List of operations as (gate, qubits, params) tuples
        transformation_rules: Dict mapping (gate1, gate2) → (new_gate1, new_gate2) or
                                                          → ((new_gate1, new_params1), (new_gate2, new_params2))
            If None, uses default rules: HH→HX, XX→XZ, ZZ→ZH
        error_probability: Probability [0, 1] that a matching pair will be transformed
        seed: Random seed for reproducibility
    
    Returns:
        Modified list of operations with probabilistic coherent errors.
        Returns original list if no transformations are applied.
        
    Note:
        For multi-qubit gates, qubit lists must match exactly in order.
        E.g., [0,1] will not match [1,0]. If order doesn't matter for your gates,
        normalize qubit lists before calling this function.
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
    
    # Normalize rules: support both simple (gate names) and extended (gate, params) forms
    normalized_rules = {}
    for (g1, g2), replacement in transformation_rules.items():
        key = (g1.lower(), g2.lower())
        
        # Check if replacement is extended form with params
        if isinstance(replacement[0], tuple):
            # Extended form: ((gate1, params1), (gate2, params2))
            (r1, p1), (r2, p2) = replacement
            normalized_rules[key] = ((r1.lower(), p1), (r2.lower(), p2))
        else:
            # Simple form: (gate1, gate2) - inherit original params
            r1, r2 = replacement
            normalized_rules[key] = ((r1.lower(), None), (r2.lower(), None))
    
    # Pre-compute lowercase gate names to avoid repeated .lower() calls in loop
    gate_names_lower = [op[0].lower() for op in base_ops]
    
    # Track which gate indices have been transformed (global tracking, not per-qubit)
    # O(n) space instead of O(num_qubits × n)
    transformed = [False] * len(base_ops)
    
    # Lazy copy: only create noisy_ops when first transformation occurs
    noisy_ops = None
    
    # Scan for consecutive pairs using index-based loop (avoids O(n) slice copy)
    for i in range(len(base_ops) - 1):
        op1 = base_ops[i]
        op2 = base_ops[i + 1]
        
        gate1, qubits1, params1 = op1
        gate2, qubits2, params2 = op2
        
        # Check if gates operate on the same qubits (supports multi-qubit gates)
        # Note: requires exact qubit order match
        if tuple(qubits1) != tuple(qubits2):
            continue
        
        # Skip if either gate already part of a transformation
        if transformed[i] or transformed[i + 1]:
            continue
        
        # Check if this pair matches a transformation rule (using cached lowercase names)
        gate_pair = (gate_names_lower[i], gate_names_lower[i + 1])
        if gate_pair in normalized_rules:
            # Probabilistically apply transformation
            if rng.random() < error_probability:
                (new_gate1, new_params1), (new_gate2, new_params2) = normalized_rules[gate_pair]
                
                # Use custom params if provided, otherwise inherit original
                final_params1 = new_params1 if new_params1 is not None else params1
                final_params2 = new_params2 if new_params2 is not None else params2
                
                # Lazy copy: create noisy_ops on first transformation
                if noisy_ops is None:
                    noisy_ops = list(base_ops)
                
                # Apply transformation
                noisy_ops[i] = (new_gate1, qubits1, final_params1)
                noisy_ops[i + 1] = (new_gate2, qubits2, final_params2)
                
                # Mark both gates as transformed
                transformed[i] = True
                transformed[i + 1] = True
                
                # Format qubit display (single or multi-qubit)
                qubit_str = f"qubit {qubits1[0]}" if len(qubits1) == 1 else f"qubits {qubits1}"
                print(f"  Coherent error: {gate1.upper()}{gate2.upper()} → "
                      f"{new_gate1.upper()}{new_gate2.upper()} on {qubit_str} "
                      f"(gates {i}→{i+1})")
    
    # Return original list if no transformations occurred (avoids unnecessary copy)
    return noisy_ops if noisy_ops is not None else base_ops


