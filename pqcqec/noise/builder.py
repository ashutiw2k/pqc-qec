import numpy as np
from ..simulate.statevector import (
    GATE_DICT, 
    GATE_H, GATE_X, GATE_Z, 
    GATE_RX, GATE_RY, GATE_RZ, 
    GATE_CX, GATE_CZ
)


def build_circuit(circuit_ops, dtype=np.float32):
    """
    Convert a high-level circuit description into parallel arrays for the executor.
    
    This function provides a user-friendly interface for constructing quantum
    circuits. Instead of manually building the parallel arrays required by the
    Numba-compiled functions, users can specify circuits using intuitive tuples.
    
    Parameters:
    -----------
    circuit_ops : list of tuples
        Circuit description as a list of gate operations in format (gate_name, [qubits], [params]):
        
        Single-qubit gates (no angle):
        - ('h', [q], [])      : Hadamard gate on qubit q
        - ('x', [q], [])      : Pauli-X gate on qubit q  
        - ('z', [q], [])      : Pauli-Z gate on qubit q
        
        Single-qubit rotation gates (with angle):
        - ('rx', [q], [θ])  : X-rotation by angle θ on qubit q
        - ('ry', [q], [θ])  : Y-rotation by angle θ on qubit q
        - ('rz', [q], [θ])  : Z-rotation by angle θ on qubit q
        
        Two-qubit gates:
        - ('cx', [c, t], [])  : CNOT with control c and target t
        - ('cz', [c, t], [])  : Controlled-Z with control c and target t
        
    dtype : numpy dtype, optional (default=np.float32)
        Data type for the theta array (angles)
        
    Returns:
    --------
    tuple of (gate_ids, wire1, wire2, theta)
        gate_ids : int32 array
            Gate type identifiers
        wire1 : int32 array  
            Primary qubit indices (target for 1q, control for 2q)
        wire2 : int32 array
            Secondary qubit indices (-1 for 1q, target for 2q)
        theta : float array
            Rotation angles (0.0 for non-rotation gates)
            
    Example:
    --------
    >>> ops = [
    ...     ('h', [0], []),           # Hadamard on qubit 0
    ...     ('cx', [0, 1], []),       # CNOT: control=0, target=1  
    ...     ('rz', [1], [0.5]),       # Z-rotation by 0.5 radians on qubit 1
    ... ]
    >>> gate_ids, w1, w2, theta = build_circuit(ops)
    
    Notes:
    ------
    - The function validates gate types and raises ValueError for unknown gates
    - All arrays are pre-allocated for optimal memory usage and speed
    - The wire2 array contains -1 for single-qubit gates (unused parameter)
    - The theta array contains 0.0 for non-parameterized gates
    """
    # Pre-allocate arrays for better memory efficiency and speed
    n = len(circuit_ops)
    gate_ids = np.empty(n, dtype=np.int32)
    w1 = np.empty(n, dtype=np.int32)
    w2 = np.empty(n, dtype=np.int32)
    th = np.empty(n, dtype=dtype)
    
    # Process each operation in the circuit
    for i, op in enumerate(circuit_ops):
        gate, qubits, param = op
        g = GATE_DICT[gate]  # Gate type identifier
        
        # Handle single-qubit gates without parameters
        if g in (GATE_X, GATE_Z, GATE_H):
            gate_ids[i] = g
            w1[i] = qubits[0]      # Target qubit
            w2[i] = -1             # No second qubit (unused)
            th[i] = 0.0            # No angle parameter
            
        # Handle parameterized single-qubit rotation gates  
        elif g in (GATE_RX, GATE_RY, GATE_RZ):
            gate_ids[i] = g
            w1[i] = qubits[0]      # Target qubit
            w2[i] = -1             # No second qubit (unused)
            th[i] = param[0]       # Rotation angle (no float() needed, numpy handles it)
            
        # Handle two-qubit controlled gates
        elif g in (GATE_CX, GATE_CZ):
            gate_ids[i] = g
            w1[i] = qubits[0]      # Control qubit
            w2[i] = qubits[1]      # Target qubit  
            th[i] = 0.0            # No angle parameter
            
        else:
            raise ValueError(f"Unknown gate code: {g}")
    
    return gate_ids, w1, w2, th


def build_regular_noisy_circuit(circuit_ops, x_noise:np.ndarray, z_noise:np.ndarray, return_tagged=False):
    """
    Build a noisy circuit by adding RX and RZ noise gates after each operation.
    
    For each gate in the circuit, noise is applied to all qubits involved in that gate.
    This models gate-level noise where both control and target qubits (for 2-qubit gates)
    experience coherent rotation errors.
    
    Parameters:
    -----------
    circuit_ops : list of tuples
        Original circuit operations in format (gate_name, [qubits], [params])
    x_noise : np.ndarray
        X-rotation noise values indexed by gate position (shape: [num_gates])
    z_noise : np.ndarray  
        Z-rotation noise values indexed by gate position (shape: [num_gates])
    return_tagged : bool, optional (default=False)
        If True, return tagged circuit operations with noise gates marked
        If False, return compiled Numba arrays (original behavior)
        
    Returns:
    --------
    If return_tagged=False (default):
        tuple of (gate_ids, wire1, wire2, theta)
            Compiled circuit arrays from build_circuit() with noise gates inserted
    If return_tagged=True:
        list of tuples
            Circuit operations with noise gates tagged as {'noise': True}
            Use with build_circuit_with_pqc(..., ignore_noise_gates=True)
    """
    # Pre-calculate total size: original gates + 2 noise gates per qubit per original gate
    total_qubits = sum(len(op[1]) for op in circuit_ops)
    noisy_circuit_ops = [None] * (len(circuit_ops) + 2 * total_qubits)
    
    idx = 0
    for i, op in enumerate(circuit_ops):
        noisy_circuit_ops[idx] = op
        idx += 1
        
        # Add noise to each qubit involved in this gate
        x_val = x_noise[i]
        z_val = z_noise[i]
        for q in op[1]:
            if return_tagged:
                noisy_circuit_ops[idx] = ('rx', [q], [x_val], {'noise': True})
                idx += 1
                noisy_circuit_ops[idx] = ('rz', [q], [z_val], {'noise': True})
                idx += 1
            else:
                noisy_circuit_ops[idx] = ('rx', [q], [x_val])
                idx += 1
                noisy_circuit_ops[idx] = ('rz', [q], [z_val])
                idx += 1

    if return_tagged:
        return noisy_circuit_ops
    return build_circuit(noisy_circuit_ops)
            
def build_idle_qubit_circuit(circuit_ops, num_qubits, idle_noise:np.ndarray, idle_threshold:int=1, return_tagged=False):
    """
    Build a noisy circuit by adding RX and RZ noise to qubits that have been idle for n gates.
    
    This models realistic idle qubit decoherence where qubits accumulate noise only after
    being idle for a threshold number of consecutive gates. Noise is applied when a qubit
    has been unused for 'idle_threshold' consecutive gates.
    
    Parameters:
    -----------
    circuit_ops : list of tuples
        Original circuit operations in format (gate_name, [qubits], [params])
    num_qubits : int
        Total number of qubits in the circuit
    idle_noise : np.ndarray
        Noise values indexed by gate position (shape: [num_gates])
        Same noise value used for both RX and RZ on each idle qubit
    idle_threshold : int, optional (default=1)
        Number of consecutive gates a qubit must be idle before noise is applied.
        - idle_threshold=1: Apply noise after every gate (original behavior)
        - idle_threshold=2: Apply noise only if qubit was idle for 2+ consecutive gates
        - idle_threshold=n: Apply noise only if qubit was idle for n+ consecutive gates
    return_tagged : bool, optional (default=False)
        If True, return tagged circuit operations with noise gates marked
        If False, return compiled Numba arrays (original behavior)
        
    Returns:
    --------
    If return_tagged=False (default):
        tuple of (gate_ids, wire1, wire2, theta)
            Compiled circuit arrays from build_circuit() with idle noise gates inserted
    If return_tagged=True:
        list of tuples
            Circuit operations with noise gates tagged as {'noise': True}
            Use with build_circuit_with_pqc(..., ignore_noise_gates=True)
    """
    # Track how many consecutive gates each qubit has been idle
    idle_counts = np.zeros(num_qubits, dtype=np.int32)
    
    # Build circuit in single pass - dynamically append as we go
    noisy_circuit_ops = []
    
    for i, op in enumerate(circuit_ops):
        active_qubits = set(op[1])
        
        # Determine which qubits get noise (idle >= threshold)
        # Create boolean mask for active qubits
        is_active = np.zeros(num_qubits, dtype=bool)
        is_active[list(active_qubits)] = True
        # print(is_active)
        
        # Update idle counts: increment for inactive qubits, reset for active ones
        idle_counts = np.where(is_active, 0, idle_counts + 1)
        # print(idle_counts)
        
        # Find qubits that need noise (idle >= threshold and not active)
        qubits_to_noise = np.where((idle_counts >= idle_threshold) & ~is_active)[0]
        # Insert the current gate
        noisy_circuit_ops.append(op)
        
        # print(qubits_to_noise)

        idle_counts[qubits_to_noise] = 0  # Reset idle counts for qubits that are getting noise

        # Add noise gates for idle qubits (sorted for deterministic order)
        if np.any(qubits_to_noise):
            noise_val = idle_noise[i]
            for q in sorted(qubits_to_noise):
                if return_tagged:
                    noisy_circuit_ops.append(('rx', [q], [noise_val], {'noise': True}))
                    noisy_circuit_ops.append(('rz', [q], [noise_val], {'noise': True}))
                else:
                    noisy_circuit_ops.append(('rx', [q], [noise_val]))
                    noisy_circuit_ops.append(('rz', [q], [noise_val]))


    # exit(0)
    if return_tagged:
        return noisy_circuit_ops
    return build_circuit(noisy_circuit_ops)


def build_circuit_with_pqc(circuit_ops, num_qubits, gate_blocks, pqc_gates, pqc_params, dtype=np.float32, return_numba=False, ignore_noise_gates=False, return_pqc_map=False):
    """
    Build a circuit with PQC (Parameterized Quantum Circuit) layers interleaved at specified intervals.
    
    This function takes a base circuit (potentially with noise) and interleaves PQC operations
    after every 'gate_blocks' gates. The PQC operations are applied to all qubits with 
    trainable parameters. Highly optimized with NumPy operations and minimal branching.
    
    Parameters:
    -----------
    circuit_ops : list of tuples
        Base circuit operations in format (gate_name, [qubits], [params])
        Can be output from noisy circuit builders
        
        Standard format: (gate_name, [qubits], [params])
        Tagged format: (gate_name, [qubits], [params], {'noise': True})
        
        When ignore_noise_gates=True, gates tagged with {'noise': True} are invisible
        to PQC block counting but remain in the final circuit.
        
    num_qubits : int
        Total number of qubits in the circuit
    gate_blocks : int
        Number of base circuit gates between PQC insertions
        PQC layers are inserted after every gate_blocks gates
        When ignore_noise_gates=True, only counts non-noise gates
    pqc_gates : list of str
        List of PQC gate names to apply (e.g., ['rx', 'ry', 'rz'])
        Applied in sequence to each qubit
    pqc_params : np.ndarray
        PQC parameters with shape: [num_blocks, num_qubits, num_pqc_gates]
        - num_blocks includes the final PQC block appended to the circuit
        - Must be a numpy array (for optimal performance)
    dtype : numpy dtype, optional (default=np.float32)
        Data type for the theta array
    return_numba : bool, optional (default=False)
        If True, return Numba-compatible arrays (gate_ids, w1, w2, theta)
        If False, return circuit operations list
    ignore_noise_gates : bool, optional (default=False)
        If True, gates tagged with {'noise': True} are not counted for PQC blocks
        but are preserved in the output circuit. This allows applying PQC to the
        logical circuit structure while keeping all noise gates.
        
    Returns:
    --------
    If return_numba=True:
        tuple of (gate_ids, wire1, wire2, theta)
            Compiled circuit arrays from build_circuit() with PQC layers inserted
    If return_numba=False:
        list of tuples
            Circuit operations with PQC gates interleaved
            
    Example:
    --------
    >>> # Regular usage
    >>> ops = [('h', [0], []), ('cx', [0,1], [])]
    >>> params = np.random.randn(2, 2, 3)
    >>> g, w1, w2, th = build_circuit_with_pqc(ops, 2, 1, ['rx','ry','rz'], params, return_numba=True)
    >>> 
    >>> # With noise gates (ignored for PQC counting)
    >>> noisy_ops = build_regular_noisy_circuit(ops, x_noise, z_noise)
    >>> # Convert to tagged format
    >>> tagged_ops = tag_noise_gates(ops, noisy_ops)
    >>> g, w1, w2, th = build_circuit_with_pqc(
    ...     tagged_ops, 2, 1, ['rx','ry','rz'], params, 
    ...     return_numba=True, ignore_noise_gates=True
    ... )
    
    Notes:
    ------
    - When ignore_noise_gates=True, PQC blocks are inserted based only on non-noise gates
    - Noise gates are preserved in their original positions
    - All gates (noise + logic + PQC) are compiled into the final circuit
    """
    # Pre-compute constants
    num_base_gates = len(circuit_ops)
    num_pqc_gates = len(pqc_gates)
    num_pqc_blocks = pqc_params.shape[0]
    pqc_ops_per_block = num_qubits * num_pqc_gates
    
    # If ignoring noise gates, filter to find logical gate positions
    if ignore_noise_gates:
        # Identify which gates are logical (non-noise) gates
        logical_mask = np.array([
            not (len(op) > 3 and isinstance(op[3], dict) and op[3].get('noise', False))
            for op in circuit_ops
        ], dtype=bool)
        
        logical_indices = np.where(logical_mask)[0]
        num_logical_gates = len(logical_indices)
        
        # Compute insertion points based on logical gates only
        # After every gate_blocks logical gates
        logical_insertion_points = np.arange(gate_blocks - 1, num_logical_gates, gate_blocks)
        
        # Map back to actual circuit indices
        # BUT: we need to insert AFTER the logical gate AND its associated noise gates
        # Find the last noise gate that follows each logical gate
        insertion_indices = []
        for logical_point in logical_insertion_points:
            logical_gate_idx = logical_indices[logical_point]
            
            # Scan forward to find where noise gates end (next logical gate or end of circuit)
            insert_after_idx = logical_gate_idx
            for j in range(logical_gate_idx + 1, num_base_gates):
                if logical_mask[j]:
                    # Hit next logical gate, insert before it
                    break
                # This is a noise gate, continue scanning
                insert_after_idx = j
            
            insertion_indices.append(insert_after_idx)
        
        insertion_indices = np.array(insertion_indices, dtype=np.int32)
        num_insertions = len(insertion_indices)
    else:
        # Original behavior: count all gates
        insertion_indices = np.arange(gate_blocks - 1, num_base_gates, gate_blocks)
        num_insertions = len(insertion_indices)
    
    # User has full control - num_pqc_blocks must match num_insertions exactly
    # No automatic final block added
    if num_pqc_blocks != num_insertions:
        raise ValueError(
            f"num_pqc_blocks mismatch! Got {num_pqc_blocks}, but there are {num_insertions} insertion points.\n"
            f"For {num_logical_gates if ignore_noise_gates else num_base_gates} gates with gate_blocks={gate_blocks}, \n"
            f"PQC will be inserted after gates at indices: {insertion_indices.tolist()}\n"
            f"Use: num_pqc_blocks = {num_insertions}"
        )
    
    # Total circuit size
    total_size = num_base_gates + num_pqc_blocks * pqc_ops_per_block
    
    # Pre-allocate the entire circuit list for optimal memory usage
    circuit_with_pqc = [None] * total_size
    
    # Build mapping arrays for vectorized copying
    # For each segment: [start_idx, end_idx, target_position]
    segments_start = np.concatenate(([0], insertion_indices + 1))
    segments_end = np.concatenate((insertion_indices + 1, [num_base_gates]))
    segments_len = segments_end - segments_start
    
    # Calculate cumulative target positions in output array
    # Each segment shifts by: num_prior_pqc_blocks * pqc_ops_per_block
    pqc_offsets = np.arange(num_insertions + 1) * pqc_ops_per_block
    target_starts = segments_start + pqc_offsets
    
    # Copy base circuit segments in batches
    write_idx = 0
    for seg_idx in range(len(segments_start)):
        start = segments_start[seg_idx]
        end = segments_end[seg_idx]
        seg_len = segments_len[seg_idx]
        
        # Batch copy operations (strip tags if present, only keep gate info)
        for i in range(start, end):
            op = circuit_ops[i]
            # Keep only the first 3 elements (gate, qubits, params) - strip metadata
            circuit_with_pqc[write_idx] = op[:3] if len(op) > 3 else op
            write_idx += 1
        
        # Insert PQC block after this segment (at insertion points only)
        if seg_idx < num_insertions:
            block_params = pqc_params[seg_idx]
            # Vectorized PQC generation using NumPy indexing
            q_indices = np.repeat(np.arange(num_qubits), num_pqc_gates)
            g_indices = np.tile(np.arange(num_pqc_gates), num_qubits)
            
            for i, (q, g) in enumerate(zip(q_indices, g_indices)):
                circuit_with_pqc[write_idx + i] = (pqc_gates[g], [q], [block_params[q, g]])
            
            write_idx += pqc_ops_per_block
    
    # Compile to Numba-compatible format
    if return_numba:
        result = build_circuit(circuit_with_pqc, dtype=dtype)
        if return_pqc_map:
            # Build PQC parameter map by tracking positions
            # We need to map from (block, qubit, gate) to theta_idx in compiled circuit
            pqc_map = []
            
            # Track through the compiled circuit structure
            # PQC gates are inserted at specific positions we can calculate
            compiled_idx = 0
            pqc_block = 0
            
            for seg_idx in range(len(segments_start)):
                # Skip base circuit segment
                seg_len = segments_end[seg_idx] - segments_start[seg_idx]
                compiled_idx += seg_len
                
                # PQC block follows at insertion points only
                if seg_idx < num_insertions:
                    for q in range(num_qubits):
                        for g in range(num_pqc_gates):
                            pqc_map.append((pqc_block, q, g, compiled_idx))
                            compiled_idx += 1
                    pqc_block += 1
            
            return result + (np.array(pqc_map, dtype=np.int32),)
        return result
    return circuit_with_pqc



def build_circuit_with_pqc_simplified(circuit_ops, num_qubits, gate_blocks, pqc_gates, pqc_params, 
                                      dtype=np.float32, return_numba=False, ignore_noise_gates=False, 
                                      return_pqc_map=False):
    """
    Simplified version: Build circuit with PQC layers interleaved at specified intervals.
    
    Key simplifications:
    1. Single-pass construction instead of pre-computing segment arrays
    2. Helper function extracts insertion point logic
    3. PQC map built during main loop instead of second pass
    4. Cleaner separation of concerns
    """
    from pqcqec.noise.builder import build_circuit
    
    # Helper: Determine insertion indices based on gate counting logic
    def get_insertion_indices():
        if not ignore_noise_gates:
            # Simple case: insert after every gate_blocks gates
            return np.arange(gate_blocks - 1, len(circuit_ops), gate_blocks)
        
        # Complex case: count only non-noise gates, insert after noise trailing each logical gate
        logical_indices = [i for i, op in enumerate(circuit_ops) 
                          if not (len(op) > 3 and isinstance(op[3], dict) and op[3].get('noise', False))]
        
        insertion_points = []
        for idx in range(gate_blocks - 1, len(logical_indices), gate_blocks):
            logical_gate_idx = logical_indices[idx]
            # Scan forward past trailing noise gates
            insert_after = logical_gate_idx
            for j in range(logical_gate_idx + 1, len(circuit_ops)):
                if j in logical_indices:
                    break  # Hit next logical gate
                insert_after = j
            insertion_points.append(insert_after)
        
        return np.array(insertion_points, dtype=np.int32)
    
    # Calculate where to insert PQC blocks
    insertion_indices = get_insertion_indices()
    num_insertions = len(insertion_indices)
    
    # Validate parameters
    if pqc_params.shape[0] != num_insertions:
        raise ValueError(
            f"num_pqc_blocks mismatch! Got {pqc_params.shape[0]}, need {num_insertions}.\n"
            f"PQC inserted after gate indices: {insertion_indices.tolist()}"
        )
    
    # Build circuit with PQC in single pass
    circuit_with_pqc = []
    pqc_map = [] if return_pqc_map else None
    insertion_set = set(insertion_indices)  # O(1) lookup
    pqc_block_idx = 0
    
    for i, op in enumerate(circuit_ops):
        # Add base gate (strip metadata tags)
        circuit_with_pqc.append(op[:3] if len(op) > 3 else op)
        
        # Insert PQC block after this gate if it's an insertion point
        if i in insertion_set:
            block_params = pqc_params[pqc_block_idx]
            
            # Add PQC gates for all qubits
            for q in range(num_qubits):
                for g_idx, gate_name in enumerate(pqc_gates):
                    circuit_with_pqc.append((gate_name, [q], [block_params[q, g_idx]]))
                    
                    # Track PQC parameter mapping if requested
                    if return_pqc_map:
                        compiled_idx = len(circuit_with_pqc) - 1
                        pqc_map.append((pqc_block_idx, q, g_idx, compiled_idx))
            
            pqc_block_idx += 1
    
    # Return in requested format
    if return_numba:
        result = build_circuit(circuit_with_pqc, dtype=dtype)
        if return_pqc_map:
            return result + (np.array(pqc_map, dtype=np.int32),)
        return result
    
    return circuit_with_pqc



def create_pqc_circuit_template(circuit_ops, num_qubits, gate_blocks, pqc_gates, num_pqc_blocks, dtype=np.float32, ignore_noise_gates=False):
    """
    Create a PQC circuit template with placeholder parameters.
    
    This function creates the circuit structure once, which can then be rapidly updated
    with new PQC parameters using update_pqc_circuit_template(). This is ideal for
    training loops where circuit structure is fixed but parameters change.
    
    Parameters:
    -----------
    circuit_ops : list of tuples
        Base circuit operations in format (gate_name, [qubits], [params])
        Can be tagged operations from noisy circuit builders
    num_qubits : int
        Total number of qubits in the circuit
    gate_blocks : int
        Number of base circuit gates between PQC insertions
        When ignore_noise_gates=True, only counts non-noise gates
    pqc_gates : list of str
        List of PQC gate names (e.g., ['rx', 'ry', 'rz'])
    num_pqc_blocks : int
        Number of PQC blocks (should be (num_logical_gates // gate_blocks) + 1)
    dtype : numpy dtype, optional (default=np.float32)
        Data type for arrays
    ignore_noise_gates : bool, optional (default=False)
        If True, gates tagged with {'noise': True} are not counted for PQC placement
        Use this with tagged circuits from noisy circuit builders
        
    Returns:
    --------
    dict with keys:
        'structure': Pre-compiled circuit structure information
        'pqc_indices': np.ndarray - indices where PQC parameters go in theta array
        'num_qubits': int - number of qubits
        'num_pqc_gates': int - number of PQC gates per qubit per block
        'gate_ids': np.ndarray - pre-allocated gate identifiers (fixed)
        'wire1': np.ndarray - pre-allocated wire1 array (fixed)
        'wire2': np.ndarray - pre-allocated wire2 array (fixed)
        'theta': np.ndarray - parameter array (to be updated)
        'dtype': numpy dtype
        
    Example:
    --------
    >>> # Create template once
    >>> template = create_pqc_circuit_template(
    ...     base_ops, num_qubits=2, gate_blocks=1, 
    ...     pqc_gates=['rx', 'ry', 'rz'], num_pqc_blocks=4
    ... )
    >>> 
    >>> # Update with new parameters many times (fast!)
    >>> for epoch in range(1000):
    ...     new_params = optimizer.get_params()  # Shape: [4, 2, 3]
    ...     gate_ids, w1, w2, theta = update_pqc_circuit_template(template, new_params)
    ...     # Run circuit...
    
    Notes:
    ------
    - Template creation is done ONCE before training
    - Updating template is 10-100x faster than rebuilding circuit
    - Only theta array is modified during updates
    - Perfect for gradient-based optimization loops
    """
    # Build initial circuit with dummy parameters
    dummy_params = np.zeros((num_pqc_blocks, num_qubits, len(pqc_gates)), dtype=dtype)
    
    # Build the full circuit to get structure AND PQC parameter map
    gate_ids_init, w1_init, w2_init, theta_init, pqc_param_map = build_circuit_with_pqc(
        circuit_ops, num_qubits, gate_blocks, pqc_gates, dummy_params, 
        dtype=dtype, return_numba=True, ignore_noise_gates=ignore_noise_gates,
        return_pqc_map=True  # Request PQC mapping
    )
    
    return {
        'gate_ids': gate_ids_init,
        'wire1': w1_init,
        'wire2': w2_init,
        'theta': theta_init,
        'pqc_param_map': pqc_param_map,  # Shape: [total_pqc_ops, 4] (block, qubit, gate, theta_idx)
        'num_qubits': num_qubits,
        'num_pqc_gates': len(pqc_gates),
        'num_pqc_blocks': num_pqc_blocks,
        'dtype': dtype
    }



def create_pqc_circuit_template_simplified(circuit_ops, num_qubits, gate_blocks, pqc_gates, num_pqc_blocks, 
                                           dtype=np.float32, ignore_noise_gates=False):
    """
    Simplified template creation - same as original but calls simplified builder.
    
    Creates a template dictionary that can be rapidly updated with new PQC parameters.
    This is the companion to build_circuit_with_pqc_simplified for training loops.
    """
    from pqcqec.noise.builder import build_circuit
    
    # Build initial circuit with dummy parameters
    dummy_params = np.zeros((num_pqc_blocks, num_qubits, len(pqc_gates)), dtype=dtype)
    
    # Build the full circuit to get structure AND PQC parameter map
    gate_ids_init, w1_init, w2_init, theta_init, pqc_param_map = build_circuit_with_pqc_simplified(
        circuit_ops, num_qubits, gate_blocks, pqc_gates, dummy_params, 
        dtype=dtype, return_numba=True, ignore_noise_gates=ignore_noise_gates,
        return_pqc_map=True
    )
    
    return {
        'gate_ids': gate_ids_init,
        'wire1': w1_init,
        'wire2': w2_init,
        'theta': theta_init,
        'pqc_param_map': pqc_param_map,
        'num_qubits': num_qubits,
        'num_pqc_gates': len(pqc_gates),
        'num_pqc_blocks': num_pqc_blocks,
        'dtype': dtype
    }



def update_pqc_circuit_template(template, pqc_params):
    """
    Update PQC circuit template with new parameters (ultra-fast).
    
    This function updates only the theta array in a pre-built circuit template.
    It's designed for maximum speed in training loops where the circuit structure
    is fixed but parameters change frequently.
    
    Parameters:
    -----------
    template : dict
        Template created by create_pqc_circuit_template()
    pqc_params : np.ndarray
        New PQC parameters with shape: [num_blocks, num_qubits, num_pqc_gates]
        
    Returns:
    --------
    tuple of (gate_ids, wire1, wire2, theta)
        Updated Numba-compatible circuit arrays
        Note: gate_ids, wire1, wire2 are the SAME arrays (not copied)
        Only theta is updated with new values
        
    Example:
    --------
    >>> # Create template once
    >>> template = create_pqc_circuit_template(base_ops, 2, 1, ['rx', 'ry', 'rz'], 4)
    >>> 
    >>> # Fast updates in training loop
    >>> for iteration in range(10000):
    ...     params = get_new_params()  # Shape: [4, 2, 3]
    ...     gate_ids, w1, w2, theta = update_pqc_circuit_template(template, params)
    ...     loss = run_and_compute_loss(gate_ids, w1, w2, theta)
    
    Notes:
    ------
    - 10-100x faster than rebuilding circuit from scratch
    - Only updates theta array (vectorized operation)
    - Returns references to existing arrays (no allocation)
    - Perfect for gradient descent / optimization loops
    - Thread-safe if each thread has its own template copy
    """
    # Extract template data
    gate_ids = template['gate_ids']
    wire1 = template['wire1']
    wire2 = template['wire2']
    theta = template['theta'].copy()  # Copy to avoid modifying template
    pqc_param_map = template['pqc_param_map']
    
    # Vectorized update: directly index into theta using the pre-computed map
    # pqc_param_map shape: [total_pqc_ops, 4] where columns are (block, qubit, gate, theta_idx)
    block_indices = pqc_param_map[:, 0]
    qubit_indices = pqc_param_map[:, 1]
    gate_indices = pqc_param_map[:, 2]
    theta_indices = pqc_param_map[:, 3]
    
    # Vectorized assignment: theta[theta_indices] = pqc_params[block_indices, qubit_indices, gate_indices]
    theta[theta_indices] = pqc_params[block_indices, qubit_indices, gate_indices]
    
    return gate_ids, wire1, wire2, theta

def decompile_circuit(gate_ids, wire1, wire2, theta):
    """
    Convert Numba-compatible circuit arrays back to high-level circuit operations.
    
    This function reverses the compilation process, converting the parallel arrays
    used by Numba-compiled simulators back into human-readable circuit operations.
    Useful for debugging, visualization, and analysis.
    
    Parameters:
    -----------
    gate_ids : np.ndarray (int32)
        Gate type identifiers
    wire1 : np.ndarray (int32)
        Primary qubit indices (target for 1q, control for 2q)
    wire2 : np.ndarray (int32)
        Secondary qubit indices (-1 for 1q, target for 2q)
    theta : np.ndarray (float)
        Rotation angles (0.0 for non-rotation gates)
        
    Returns:
    --------
    list of tuples
        Circuit operations in format (gate_name, [qubits], [params])
        Same format accepted by build_circuit()
        
    Example:
    --------
    >>> # Compile circuit
    >>> ops = [('h', [0], []), ('cx', [0, 1], []), ('rz', [1], [0.5])]
    >>> gate_ids, w1, w2, theta = build_circuit(ops)
    >>> 
    >>> # Decompile back
    >>> recovered_ops = decompile_circuit(gate_ids, w1, w2, theta)
    >>> assert ops == recovered_ops
    
    Notes:
    ------
    - Inverse operation of build_circuit()
    - Preserves gate order and parameters
    - Handles all gate types (single-qubit, rotation, two-qubit)
    - Returns standard format without metadata tags
    """
    # Create reverse lookup: gate_id -> gate_name
    GATE_ID_TO_NAME = {v: k for k, v in GATE_DICT.items()}
    
    circuit_ops = []
    n = len(gate_ids)
    
    for i in range(n):
        gid = gate_ids[i]
        gate_name = GATE_ID_TO_NAME[gid]
        
        # Single-qubit gates without parameters
        if gid in (GATE_X, GATE_Z, GATE_H):
            circuit_ops.append((gate_name, [int(wire1[i])], []))
            
        # Parameterized single-qubit rotation gates
        elif gid in (GATE_RX, GATE_RY, GATE_RZ):
            circuit_ops.append((gate_name, [int(wire1[i])], [float(theta[i])]))
            
        # Two-qubit controlled gates
        elif gid in (GATE_CX, GATE_CZ):
            circuit_ops.append((gate_name, [int(wire1[i]), int(wire2[i])], []))
            
        else:
            raise ValueError(f"Unknown gate ID: {gid}")
    
    return circuit_ops
