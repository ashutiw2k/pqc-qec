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
            else:
                # Gate with parameters
                if param_source not in param_dict:
                    raise ValueError(f"Parameter source '{param_source}' not found in param_dict")
                
                param_array = param_dict[param_source]
                
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


def build_pqc_circuit_template(base_ops: List[Tuple], 
                               num_qubits: int,
                               num_gate_blocks: int,
                               add_noise: bool = True,
                               add_pqc_layers: bool = True) -> CircuitTemplate:
    """
    Build a circuit template from base operations with optional noise and PQC layers.
    
    Args:
        base_ops: List of base circuit operations
        num_qubits: Number of qubits in the circuit
        num_gate_blocks: Number of gates per block before adding PQC layer
        add_noise: Whether to add noise gates after each base gate
        add_pqc_layers: Whether to add PQC layers after each block
    
    Returns:
        CircuitTemplate that can be instantiated with different parameters
    """
    template = CircuitTemplate()
    
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
        if add_pqc_layers and (i + 1) % num_gate_blocks == 0:
            # Pre-local unitaries
            for q in range(num_qubits):
                template.add_gate('rz', [q], param_source='pre_params', param_idx=(pqc_layer_idx, q, 0))
                template.add_gate('rx', [q], param_source='pre_params', param_idx=(pqc_layer_idx, q, 1))
                template.add_gate('rz', [q], param_source='pre_params', param_idx=(pqc_layer_idx, q, 2))
            
            # ZZ entangling gates (ring topology)
            for q in range(num_qubits):
                j = (q + 1) % num_qubits
                template.add_gate('cx', [q, j])
                template.add_gate('rz', [j], param_source='theta_zz', param_idx=q)
                template.add_gate('cx', [q, j])
            
            # Post-local unitaries
            for q in range(num_qubits):
                template.add_gate('rz', [q], param_source='post_params', param_idx=(pqc_layer_idx, q, 0))
                template.add_gate('rx', [q], param_source='post_params', param_idx=(pqc_layer_idx, q, 1))
                template.add_gate('rz', [q], param_source='post_params', param_idx=(pqc_layer_idx, q, 2))
            
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
