import numpy as np
from numba import njit, prange
from numba.experimental import jitclass
from numba import types

from ..utils.constants import GateEnums, GATE_DICT

# Gate Enums as regualar ints

GATE_X  = GateEnums.GATE_X
GATE_Z  = GateEnums.GATE_Z
GATE_H  = GateEnums.GATE_H
GATE_RX = GateEnums.GATE_RX
GATE_RY = GateEnums.GATE_RY
GATE_RZ = GateEnums.GATE_RZ
GATE_CX = GateEnums.GATE_CX
GATE_CZ = GateEnums.GATE_CZ

# Move all functions outside the class and make them standalone njit functions
@njit
def _apply_1q_unitary(state, n_qubits, q, a, b, c, d):
    """Apply a general 1-qubit 2x2 unitary matrix [[a,b],[c,d]] to qubit q."""
    dim = state.shape[0]
    bit_pos = n_qubits - 1 - q
    mask = 1 << bit_pos
    
    indices_0 = np.arange(dim, dtype=np.int64)
    indices_0 = indices_0[indices_0 & mask == 0]
    indices_1 = indices_0 | mask
    
    u0 = state[indices_0]
    u1 = state[indices_1]
    
    state[indices_0] = a * u0 + b * u1
    state[indices_1] = c * u0 + d * u1

@njit
def _apply_x(state, n_qubits, q):
    """Apply Pauli-X gate."""
    _apply_1q_unitary(state, n_qubits, q,
                     0.0+0.0j, 1.0+0.0j,
                     1.0+0.0j, 0.0+0.0j)

@njit
def _apply_z(state, n_qubits, q):
    """Apply Pauli-Z gate."""
    dim = state.shape[0]
    bit_pos = n_qubits - 1 - q
    mask = 1 << bit_pos
    
    indices = np.arange(dim, dtype=np.int64)
    indices_1 = indices[indices & mask != 0]
    state[indices_1] *= -1.0

@njit
def _apply_h(state, n_qubits, q):
    """Apply Hadamard gate."""
    sqrt_half_real = 1.0 / np.sqrt(2.0)
    s = sqrt_half_real + 0.0j
    
    _apply_1q_unitary(state, n_qubits, q, s, s, s, -s)

@njit
def _apply_rx(state, n_qubits, q, theta):
    """Apply X-rotation gate."""
    half_theta = 0.5 * theta
    ct = np.cos(half_theta)
    st = np.sin(half_theta)
    
    a = ct + 0.0j
    b = 0.0 - 1j * st
    c = 0.0 - 1j * st
    d = ct + 0.0j
    
    _apply_1q_unitary(state, n_qubits, q, a, b, c, d)

@njit
def _apply_ry(state, n_qubits, q, theta):
    """Apply Y-rotation gate."""
    half_theta = 0.5 * theta
    ct = np.cos(half_theta)
    st = np.sin(half_theta)
    
    a = ct + 0.0j
    b = -st + 0.0j
    c = st + 0.0j
    d = ct + 0.0j
    
    _apply_1q_unitary(state, n_qubits, q, a, b, c, d)

@njit
def _apply_rz(state, n_qubits, q, theta):
    """Apply Z-rotation gate."""
    half_theta = 0.5 * theta
    
    cos_neg = np.cos(-half_theta)
    sin_neg = np.sin(-half_theta)
    cos_pos = np.cos(half_theta)
    sin_pos = np.sin(half_theta)
    
    e0 = cos_neg + 1j * sin_neg
    e1 = cos_pos + 1j * sin_pos
    
    dim = state.shape[0]
    bit_pos = n_qubits - 1 - q
    mask = 1 << bit_pos
    
    indices = np.arange(dim, dtype=np.int64)
    indices_0 = indices[indices & mask == 0]
    indices_1 = indices[indices & mask != 0]
    
    state[indices_0] *= e0
    state[indices_1] *= e1

@njit
def _apply_cx(state, n_qubits, control, target):
    """Apply CNOT gate."""
    dim = state.shape[0]
    control_bit_pos = n_qubits - 1 - control
    target_bit_pos = n_qubits - 1 - target
    
    mc = 1 << control_bit_pos
    mt = 1 << target_bit_pos
    
    indices = np.arange(dim, dtype=np.int64)
    control_1_target_0 = indices[(indices & mc != 0) & (indices & mt == 0)]
    control_1_target_1 = control_1_target_0 | mt
    
    temp = state[control_1_target_0].copy()
    state[control_1_target_0] = state[control_1_target_1]
    state[control_1_target_1] = temp

@njit
def _apply_cz(state, n_qubits, control, target):
    """Apply controlled-Z gate."""
    dim = state.shape[0]
    control_bit_pos = n_qubits - 1 - control
    target_bit_pos = n_qubits - 1 - target
    
    mc = 1 << control_bit_pos
    mt = 1 << target_bit_pos
    
    indices = np.arange(dim, dtype=np.int64)
    both_1_indices = indices[(indices & mc != 0) & (indices & mt != 0)]
    state[both_1_indices] *= -1.0

@njit
def run_circuit_with_state(state, n_qubits, gate_ids, wire1, wire2, theta):
    """Execute a quantum circuit in-place on an existing state vector."""
    L = gate_ids.shape[0]
    
    for k in range(L):
        g = gate_ids[k]
        a = wire1[k]
        b = wire2[k]
        t = theta[k]
        
        if g == GATE_X:
            _apply_x(state, n_qubits, a)
        elif g == GATE_Z:
            _apply_z(state, n_qubits, a)
        elif g == GATE_H:
            _apply_h(state, n_qubits, a)
        elif g == GATE_RX:
            _apply_rx(state, n_qubits, a, t)
        elif g == GATE_RY:
            _apply_ry(state, n_qubits, a, t)
        elif g == GATE_RZ:
            _apply_rz(state, n_qubits, a, t)
        elif g == GATE_CX:
            _apply_cx(state, n_qubits, a, b)
        elif g == GATE_CZ:
            _apply_cz(state, n_qubits, a, b)
    
    return state

@njit(parallel=True)
def run_many_states(n_qubits, gate_ids, wire1, wire2, theta, states_in, states_out):
    """Execute the same quantum circuit on a batch of input states in parallel."""
    B = states_in.shape[0]
    
    for b in prange(B):
        s = states_in[b].copy()
        run_circuit_with_state(s, n_qubits, gate_ids, wire1, wire2, theta)
        states_out[b] = s
    
    return states_out


def build_circuit(circuit_ops, dtype=np.float32):
    """
    Convert a high-level circuit description into parallel arrays for the executor.
    
    This function provides a user-friendly interface for constructing quantum
    circuits. Instead of manually building the parallel arrays required by the
    Numba-compiled functions, users can specify circuits using intuitive tuples.
    
    Parameters:
    -----------
    ops : list of tuples
        Circuit description as a list of gate operations:
        
        Single-qubit gates (no angle):
        - (GATE_H, q)      : Hadamard gate on qubit q
        - (GATE_X, q)      : Pauli-X gate on qubit q  
        - (GATE_Z, q)      : Pauli-Z gate on qubit q
        
        Single-qubit rotation gates (with angle):
        - (GATE_RX, q, θ)  : X-rotation by angle θ on qubit q
        - (GATE_RY, q, θ)  : Y-rotation by angle θ on qubit q
        - (GATE_RZ, q, θ)  : Z-rotation by angle θ on qubit q
        
        Two-qubit gates:
        - (GATE_CX, c, t)  : CNOT with control c and target t
        - (GATE_CZ, c, t)  : Controlled-Z with control c and target t
        
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
    ...     (GATE_H, 0),           # Hadamard on qubit 0
    ...     (GATE_CX, 0, 1),       # CNOT: control=0, target=1  
    ...     (GATE_RZ, 1, 0.5),     # Z-rotation by 0.5 radians on qubit 1
    ... ]
    >>> gate_ids, w1, w2, theta = build_circuit(ops)
    
    Notes:
    ------
    - The function validates gate types and raises ValueError for unknown gates
    - All arrays are converted to appropriate NumPy dtypes for Numba compatibility
    - The wire2 array contains -1 for single-qubit gates (unused parameter)
    - The theta array contains 0.0 for non-parameterized gates
    """
    # Initialize lists to collect circuit components
    gate_ids, w1, w2, th = [], [], [], []
    
    # Process each operation in the circuit
    for op in circuit_ops:
        gate, qubits, param = op
        g = GATE_DICT[gate]  # Gate type identifier
        
        
        # Handle single-qubit gates without parameters
        if g in (GATE_X, GATE_Z, GATE_H):
            gate_ids.append(g)
            w1.append(qubits[0])      # Target qubit
            w2.append(-1)         # No second qubit (unused)
            th.append(0.0)        # No angle parameter
            
        # Handle parameterized single-qubit rotation gates  
        elif g in (GATE_RX, GATE_RY, GATE_RZ):
            gate_ids.append(g)
            w1.append(qubits[0])      # Target qubit
            w2.append(-1)         # No second qubit (unused)
            th.append(float(param[0]))  # Rotation angle
            
        # Handle two-qubit controlled gates
        elif g in (GATE_CX, GATE_CZ):
            gate_ids.append(g)
            w1.append(qubits[0])      # Control qubit
            w2.append(qubits[1])      # Target qubit  
            th.append(0.0)        # No angle parameter
            
        else:
            raise ValueError(f"Unknown gate code: {g}")
    
    # Convert lists to NumPy arrays with appropriate dtypes
    return (
        np.asarray(gate_ids, dtype=np.int32),  # Gate identifiers
        np.asarray(w1, dtype=np.int32),        # Primary qubit indices
        np.asarray(w2, dtype=np.int32),        # Secondary qubit indices  
        np.asarray(th, dtype=dtype),           # Rotation angles
    )


# Numba JitClass version for zero-overhead OOP
spec = [
    ('n_qubits', types.int32),
    ('dim', types.int32),
]

@njit
def create_zero_state(n_qubits):
    """Create the |0...0⟩ computational basis state."""
    state = np.zeros((2**n_qubits,), dtype=np.complex64)
    state[0] = 1.0 + 0.0j
    return state

@jitclass(spec)
class FastStateVectorSimulator:
    """Fully compiled Numba jitclass quantum state vector simulator with zero Python overhead."""

    def __init__(self, num_qubits):
        self.n_qubits = num_qubits
        self.dim = 2 ** num_qubits
        
    def run_circuit_from_zero(self, gate_ids, wire1, wire2, theta):
        """Execute a quantum circuit starting from |0...0⟩ state."""
        input_state = create_zero_state(self.n_qubits)
        run_circuit_with_state(input_state, self.n_qubits, gate_ids, wire1, wire2, theta)
        return input_state
        
    def run_circuit(self, gate_ids, wire1, wire2, theta, input_state):
        """Execute a quantum circuit on a given input state."""
        run_circuit_with_state(input_state, self.n_qubits, gate_ids, wire1, wire2, theta)
        return input_state

    def run_circuit_with_state(self, state, gate_ids, wire1, wire2, theta):
        """Execute a quantum circuit in-place on an existing state vector."""
        return run_circuit_with_state(state, self.n_qubits, gate_ids, wire1, wire2, theta)

    def run_many_states(self, gate_ids, wire1, wire2, theta, states_in, states_out):
        """Execute the same quantum circuit on a batch of input states in parallel."""
        return run_many_states(self.n_qubits, gate_ids, wire1, wire2, theta, states_in, states_out)


class SimpleStateVectorSimulator:
    """Numba-optimized quantum state vector simulator."""

    def __init__(self, num_qubits):
        self.n_qubits = num_qubits
        self.dim = 2 ** num_qubits
        
    
    def run_circuit(self, gate_ids, wire1, wire2, theta, input_state=None):
        """Execute a quantum circuit."""
        if input_state is None:
            input_state = np.zeros((2**self.n_qubits,), dtype=np.complex64)
            input_state[0] = 1.0 + 0.0j

        self.run_circuit_with_state(input_state, gate_ids, wire1, wire2, theta)
        return input_state

    def run_circuit_with_state(self, state, gate_ids, wire1, wire2, theta):
        """Execute a quantum circuit in-place on an existing state vector."""
        return run_circuit_with_state(state, self.n_qubits, gate_ids, wire1, wire2, theta)

    def run_many_states(self, gate_ids, wire1, wire2, theta, states_in, states_out=None):
        """Execute the same quantum circuit on a batch of input states in parallel."""
        if states_out is None:
            states_out = np.empty_like(states_in)
        return run_many_states(self.n_qubits, gate_ids, wire1, wire2, theta, states_in, states_out)
