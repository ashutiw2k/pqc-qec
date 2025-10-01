import numpy as np
from ..simulate.statevector import GATE_DICT, GATE_H, GATE_X, GATE_Z, GATE_RX, GATE_RY, GATE_RZ, GATE_CX, GATE_CZ

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


def build_regularnoisy_circuit(circuit_ops, x_noise:np.ndarray, z_noise:np.ndarray):
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
        
    Returns:
    --------
    tuple of (gate_ids, wire1, wire2, theta)
        Compiled circuit arrays from build_circuit() with noise gates inserted
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
            noisy_circuit_ops[idx] = ('rx', [q], [x_val])
            idx += 1
            noisy_circuit_ops[idx] = ('rz', [q], [z_val])
            idx += 1

    return build_circuit(noisy_circuit_ops)
            
def build_idle_qubit_circuit(circuit_ops, num_qubits, idle_noise:np.ndarray, idle_threshold:int=1):
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
        
    Returns:
    --------
    tuple of (gate_ids, wire1, wire2, theta)
        Compiled circuit arrays from build_circuit() with idle noise gates inserted
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
                noisy_circuit_ops.append(('rx', [q], [noise_val]))
                noisy_circuit_ops.append(('rz', [q], [noise_val]))


    # exit(0)
    return build_circuit(noisy_circuit_ops)