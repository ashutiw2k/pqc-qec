"""
Circuit template builder for efficient parametrized circuit generation.

This module provides a way to create circuit templates that can be efficiently
instantiated with different parameter values, avoiding repeated iteration over
the base circuit structure.
"""
import numpy as np
from typing import List, Tuple, Dict, Any


class CircuitTemplate:
    """
    A template for efficiently generating parametrized circuits.
    
    The template stores the circuit structure and parameter indices,
    allowing fast instantiation with different parameter values.
    """
    
    def __init__(self):
        self.gates = []  # List of (gate_name, qubits)
        self.param_indices = []  # List of parameter indices for each gate
        self.param_sources = []  # List of parameter source identifiers
        
    def add_gate(self, gate_name: str, qubits: List[int], 
                 param_source: str = None, param_idx: int = None):
        """
        Add a gate to the template.
        
        Args:
            gate_name: Name of the gate (e.g., 'rx', 'rz', 'cx')
            qubits: List of qubit indices the gate acts on
            param_source: Source of parameter (e.g., 'x_noise', 'pre_params', 'base')
            param_idx: Index into the parameter source array
        """
        self.gates.append((gate_name, qubits))
        self.param_sources.append(param_source)
        self.param_indices.append(param_idx)
        
    def instantiate(self, param_dict: Dict[str, np.ndarray]) -> List[Tuple]:
        """
        Instantiate the template with concrete parameter values.
        
        Args:
            param_dict: Dictionary mapping parameter source names to arrays
                       e.g., {'x_noise': array, 'z_noise': array, 'base': array}
        
        Returns:
            List of circuit operations in (gate, qubits, params) format
        """
        circuit_ops = []
        
        for i, (gate_name, qubits) in enumerate(self.gates):
            param_source = self.param_sources[i]
            param_idx = self.param_indices[i]
            
            if param_source is None or param_idx is None:
                # Gate with no parameters (e.g., CNOT)
                circuit_ops.append((gate_name, qubits, []))
            elif param_source == 'fixed_angle':
                # Gate with fixed angle parameter (param_idx is the angle value itself)
                circuit_ops.append((gate_name, qubits, [param_idx]))
            else:
                # Gate with parameters from param_dict
                if param_source not in param_dict:
                    raise ValueError(f"Parameter source '{param_source}' not found in param_dict")
                
                param_array = param_dict[param_source]
                
                # Validate array is not empty
                if hasattr(param_array, 'shape') and param_array.shape[0] == 0:
                    raise ValueError(
                        f"Empty parameter array for source '{param_source}' at gate {i}. "
                        f"Array shape: {param_array.shape}, param_idx: {param_idx}"
                    )
                
                # Handle different parameter indexing schemes
                if isinstance(param_idx, tuple):
                    # Multi-dimensional indexing (e.g., for pre_params[i][j])
                    param_value = param_array[param_idx]
                else:
                    # Single-dimensional indexing
                    param_value = param_array[param_idx]
                
                circuit_ops.append((gate_name, qubits, [param_value]))
        
        return circuit_ops
    
    def __len__(self):
        return len(self.gates)
    
    def __repr__(self):
        return f"CircuitTemplate(num_gates={len(self)}, param_sources={set(self.param_sources)})"


def add_rzrxrz_local_layer(template: CircuitTemplate, 
                           pqc_layer_idx: int, 
                           num_qubits: int,
                           param_source: str = 'pre_params'):
    """
    Add Rz-Rx-Rz local unitaries on all qubits.
    
    Args:
        template: CircuitTemplate to add gates to
        pqc_layer_idx: Which PQC layer index (for parameter indexing)
        num_qubits: Number of qubits in the circuit
        param_source: Parameter source name ('pre_params' or 'post_params')
    """
    for q in range(num_qubits):
        template.add_gate('rz', [q], param_source=param_source, param_idx=(pqc_layer_idx, q, 0))
        template.add_gate('rx', [q], param_source=param_source, param_idx=(pqc_layer_idx, q, 1))
        template.add_gate('rz', [q], param_source=param_source, param_idx=(pqc_layer_idx, q, 2))


def add_rxrzry_local_layer(template: CircuitTemplate, 
                           pqc_layer_idx: int, 
                           num_qubits: int,
                           param_source: str = 'pre_params'):
    """
    Add Rx-Rz-Ry local unitaries on all qubits.
    
    Args:
        template: CircuitTemplate to add gates to
        pqc_layer_idx: Which PQC layer index (for parameter indexing)
        num_qubits: Number of qubits in the circuit
        param_source: Parameter source name ('pre_params' or 'post_params')
    """
    for q in range(num_qubits):
        template.add_gate('rx', [q], param_source=param_source, param_idx=(pqc_layer_idx, q, 0))
        template.add_gate('rz', [q], param_source=param_source, param_idx=(pqc_layer_idx, q, 1))
        template.add_gate('ry', [q], param_source=param_source, param_idx=(pqc_layer_idx, q, 2))


def add_zz_ring_entangling_layer(template: CircuitTemplate, 
                                  pqc_layer_idx: int, 
                                  num_qubits: int):
    """
    Add ZZ entangling gates in ring topology (CNOT-Rz-CNOT).
    
    Args:
        template: CircuitTemplate to add gates to
        pqc_layer_idx: Which PQC layer index (for parameter indexing)
        num_qubits: Number of qubits in the circuit
    """
    for q in range(num_qubits):
        j = (q + 1) % num_qubits
        template.add_gate('cx', [q, j])
        template.add_gate('rz', [j], param_source='theta_zz', param_idx=(pqc_layer_idx, q, 0))
        template.add_gate('cx', [q, j])

def add_zxz_ring_entangling_layer(template: CircuitTemplate, 
                                  pqc_layer_idx: int, 
                                  num_qubits: int):
    """
    Add ZXZ entangling gates in ring topology (CNOT-Rz-CNOT-Rx-CNOT-Rz-CNOT).
    
    Args:
        template: CircuitTemplate to add gates to
        pqc_layer_idx: Which PQC layer index (for parameter indexing)
        num_qubits: Number of qubits in the circuit
    """
    for q in range(num_qubits):
        j = (q + 1) % num_qubits
        template.add_gate('cx', [q, j])
        template.add_gate('rz', [j], param_source='entangled_params', param_idx=(pqc_layer_idx, q, 0))
        template.add_gate('cx', [q, j])
        template.add_gate('rx', [j], param_source='entangled_params', param_idx=(pqc_layer_idx, q, 1))
        template.add_gate('cx', [q, j])
        template.add_gate('rz', [j], param_source='entangled_params', param_idx=(pqc_layer_idx, q, 2))
        template.add_gate('cx', [q, j])


def add_xx_ring_entangling_layer(template: CircuitTemplate, 
                                  pqc_layer_idx: int, 
                                  num_qubits: int):
    """
    Add XX entangling gates in ring topology.
    
    Implements exp(-i * theta * XX) using basis transformation:
    Ry(π/2) ⊗ Ry(π/2) → CNOT → Rz(θ) → CNOT → Ry(-π/2) ⊗ Ry(-π/2)
    
    Note: Uses theta_zz parameter source (same parameters as ZZ gates).
    Fixed angles (±π/2) are stored as special 'fixed_angle' parameter source.
    
    Args:
        template: CircuitTemplate to add gates to
        pqc_layer_idx: Which PQC layer index (for parameter indexing)
        num_qubits: Number of qubits in the circuit
    """
    import numpy as np
    for q in range(num_qubits):
        j = (q + 1) % num_qubits
        # Basis change to XX (fixed π/2 rotations)
        template.add_gate('ry', [q], param_source='fixed_angle', param_idx=np.pi/2)
        template.add_gate('ry', [j], param_source='fixed_angle', param_idx=np.pi/2)
        template.add_gate('cx', [q, j])
        template.add_gate('rz', [j], param_source='theta_zz', param_idx=(pqc_layer_idx, q, 0))
        template.add_gate('cx', [q, j])
        # Basis change back (fixed -π/2 rotations)
        template.add_gate('ry', [q], param_source='fixed_angle', param_idx=-np.pi/2)
        template.add_gate('ry', [j], param_source='fixed_angle', param_idx=-np.pi/2)


def add_yy_ring_entangling_layer(template: CircuitTemplate, 
                                  pqc_layer_idx: int, 
                                  num_qubits: int):
    """
    Add YY entangling gates in ring topology.
    
    Implements exp(-i * theta * YY) using basis transformation:
    Rx(π/2) ⊗ Rx(π/2) → CNOT → Rz(θ) → CNOT → Rx(-π/2) ⊗ Rx(-π/2)
    
    Note: Uses theta_zz parameter source (same parameters as ZZ gates).
    Fixed angles (±π/2) are stored as special 'fixed_angle' parameter source.
    
    Args:
        template: CircuitTemplate to add gates to
        pqc_layer_idx: Which PQC layer index (for parameter indexing)
        num_qubits: Number of qubits in the circuit
    """
    import numpy as np
    for q in range(num_qubits):
        j = (q + 1) % num_qubits
        # Basis change to YY (fixed π/2 rotations)
        template.add_gate('rx', [q], param_source='fixed_angle', param_idx=np.pi/2)
        template.add_gate('rx', [j], param_source='fixed_angle', param_idx=np.pi/2)
        template.add_gate('cx', [q, j])
        template.add_gate('rz', [j], param_source='theta_zz', param_idx=(pqc_layer_idx, q, 0))
        template.add_gate('cx', [q, j])
        # Basis change back (fixed -π/2 rotations)
        template.add_gate('rx', [q], param_source='fixed_angle', param_idx=-np.pi/2)
        template.add_gate('rx', [j], param_source='fixed_angle', param_idx=-np.pi/2)


def add_all_to_all_zz_entangling_layer(template: CircuitTemplate, 
                                        pqc_layer_idx: int, 
                                        num_qubits: int):
    """
    Add ZZ entangling gates in all-to-all topology (every pair of qubits).
    
    Args:
        template: CircuitTemplate to add gates to
        pqc_layer_idx: Which PQC layer index (for parameter indexing)
        num_qubits: Number of qubits in the circuit
    """
    pair_idx = 0
    for q in range(num_qubits):
        for j in range(q + 1, num_qubits):
            template.add_gate('cx', [q, j])
            template.add_gate('rz', [j], param_source='theta_zz', param_idx=(pqc_layer_idx, pair_idx, 0))
            template.add_gate('cx', [q, j])
            pair_idx += 1


def add_star_zz_entangling_layer(template: CircuitTemplate, 
                                  pqc_layer_idx: int, 
                                  num_qubits: int,
                                  center_qubit: int = 0):
    """
    Add ZZ entangling gates in star topology (center qubit connected to all others).
    
    Args:
        template: CircuitTemplate to add gates to
        pqc_layer_idx: Which PQC layer index (for parameter indexing)
        num_qubits: Number of qubits in the circuit
        center_qubit: Which qubit is at the center of the star (default: 0)
    """
    edge_idx = 0
    for q in range(num_qubits):
        if q != center_qubit:
            template.add_gate('cx', [center_qubit, q])
            template.add_gate('rz', [q], param_source='theta_zz', param_idx=(pqc_layer_idx, edge_idx, 0))
            template.add_gate('cx', [center_qubit, q])
            edge_idx += 1


def add_linear_zz_entangling_layer(template: CircuitTemplate, 
                                    pqc_layer_idx: int, 
                                    num_qubits: int):
    """
    Add ZZ entangling gates in linear topology (nearest-neighbor chain, no wraparound).
    
    Args:
        template: CircuitTemplate to add gates to
        pqc_layer_idx: Which PQC layer index (for parameter indexing)
        num_qubits: Number of qubits in the circuit
    """
    for q in range(num_qubits - 1):
        j = q + 1
        template.add_gate('cx', [q, j])
        template.add_gate('rz', [j], param_source='theta_zz', param_idx=(pqc_layer_idx, q, 0))
        template.add_gate('cx', [q, j])
        


def build_pqc_circuit_template(base_ops: List[Tuple], 
                               num_qubits: int,
                               num_gate_blocks: int,
                               add_noise: bool = True,
                               pqc_type: str = "rzrxrz_zz_ring") -> CircuitTemplate:
    """
    Build a circuit template from base operations with optional noise and PQC layers.
    
    Args:
        base_ops: List of base circuit operations
        num_qubits: Number of qubits in the circuit
        num_gate_blocks: Number of gates per block before adding PQC layer
        add_noise: Whether to add noise gates after each base gate
        pqc_type: Type of PQC architecture to use. Format: "{local}_{entangling}"
        
        Local unitary options:
            - "rzrxrz": Rz-Rx-Rz decomposition (default)
            - "rxrzry": Rx-Rz-Ry decomposition
            - "none": No local unitaries
        
        Entangling options:
            - "none": No entanglement (local only)
            - "zz_ring": ZZ entangling in ring topology (nearest-neighbor with wraparound)
            - "zz_linear": ZZ entangling in linear chain (nearest-neighbor, no wraparound)
            - "zz_all_to_all": ZZ entangling between all pairs of qubits
            - "zz_star": ZZ entangling in star topology (center qubit to all others)
            - "xx_ring": XX entangling in ring topology
            - "yy_ring": YY entangling in ring topology
            - "zxz_ring": ZXZ entangling in ring topology (3-param per pair)
        
        Examples:
            - "rzrxrz_zz_ring": Local RzRxRz + ZZ ring (default, LEL-ZZ)
            - "rxrzry_xx_ring": Local RxRzRy + XX ring
            - "rzrxrz": Just local RzRxRz (no entanglement)
            - "none_zz_all_to_all": Just ZZ all-to-all (no local pre/post)
            - "none": No PQC layers at all
    
    Returns:
        CircuitTemplate that can be instantiated with different parameters
    """
    template = CircuitTemplate()
    
    # Parse PQC type
    pqc_type_lower = pqc_type.lower()
    add_pqc = pqc_type_lower != "none"
    
    if not add_pqc:
        # No PQC layers at all
        local_gate_fn = None
        entangling_fn = None
        has_entanglement = False
    else:
        # Split into local and entangling parts
        parts = pqc_type_lower.split('_', 1)
        local_part = parts[0]
        entangling_part = parts[1] if len(parts) > 1 else "none"
        
        # Determine local gate type
        if local_part == "rzrxrz":
            local_gate_fn = add_rzrxrz_local_layer
        elif local_part == "rxrzry":
            local_gate_fn = add_rxrzry_local_layer
        elif local_part == "none":
            local_gate_fn = None
        else:
            raise ValueError(
                f"Unknown local unitary type: '{local_part}'. "
                f"Supported: 'rzrxrz', 'rxrzry', 'none'"
            )
        
        # Determine entangling layer type
        has_entanglement = entangling_part != "none"
        if entangling_part == "none":
            entangling_fn = None
        elif entangling_part == "zz_ring":
            entangling_fn = add_zz_ring_entangling_layer
        elif entangling_part == "zz_linear":
            entangling_fn = add_linear_zz_entangling_layer
        elif entangling_part == "zz_all_to_all":
            entangling_fn = add_all_to_all_zz_entangling_layer
        elif entangling_part == "zz_star":
            entangling_fn = add_star_zz_entangling_layer
        elif entangling_part == "xx_ring":
            entangling_fn = add_xx_ring_entangling_layer
        elif entangling_part == "yy_ring":
            entangling_fn = add_yy_ring_entangling_layer
        elif entangling_part == "zxz_ring":
            entangling_fn = add_zxz_ring_entangling_layer
        else:
            raise ValueError(
                f"Unknown entangling type: '{entangling_part}'. "
                f"Supported: 'none', 'zz_ring', 'zz_linear', 'zz_all_to_all', 'zz_star', "
                f"'xx_ring', 'yy_ring', 'zxz_ring'"
            )
    
    pqc_layer_idx = 0  # Track which PQC layer we're on
    for i, op in enumerate(base_ops):
        gate_name, qubits, base_params = op
        
        # Add the base gate
        if len(base_params) > 0:
            template.add_gate(gate_name, qubits, param_source='base', param_idx=i)
        else:
            template.add_gate(gate_name, qubits)
        
        # Add noise gates if requested
        if add_noise:
            for q in qubits:
                template.add_gate('rx', [q], param_source='x_noise', param_idx=i)
                template.add_gate('rz', [q], param_source='z_noise', param_idx=i)
        
        # Add PQC layer after each block
        if add_pqc and (i + 1) % num_gate_blocks == 0:
            # Pre-local unitaries (if specified)
            if local_gate_fn is not None:
                local_gate_fn(template, pqc_layer_idx, num_qubits, param_source='pre_params')
            
            # Entangling layer (only for multi-qubit circuits and if specified)
            if num_qubits > 1 and has_entanglement and entangling_fn is not None:
                entangling_fn(template, pqc_layer_idx, num_qubits)
                
                # Post-local unitaries (only added if we have both local and entanglement)
                if local_gate_fn is not None:
                    local_gate_fn(template, pqc_layer_idx, num_qubits, param_source='post_params')
            
            pqc_layer_idx += 1
    
    return template


def build_simple_noise_template(base_ops: List[Tuple]) -> CircuitTemplate:
    """
    Build a simpler template with just noise added to base operations.
    
    Args:
        base_ops: List of base circuit operations
    
    Returns:
        CircuitTemplate for noisy circuit
    """
    template = CircuitTemplate()
    
    for i, op in enumerate(base_ops):
        gate_name, qubits, base_params = op
        
        # Add the base gate
        if len(base_params) > 0:
            template.add_gate(gate_name, qubits, param_source='base', param_idx=i)
        else:
            template.add_gate(gate_name, qubits)
        
        # Add noise gates
        for q in qubits:
            template.add_gate('rx', [q], param_source='x_noise', param_idx=i)
            template.add_gate('rz', [q], param_source='z_noise', param_idx=i)
    
    return template
