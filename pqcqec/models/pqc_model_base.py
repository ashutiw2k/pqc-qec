"""
Unified base model class for PQC training.

This class handles all training-related logic (circuit execution, template management,
batched simulation) while delegating parameter structure to PQC architecture classes.
"""

import copy
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from ..circuits.templates import build_pqc_circuit_template
from ..simulate.jax_statevector import build_jax_circuit, jax_run_many_states
from ..utils.quaternions_utils import quaternion_to_zxz_angles, quaternion_to_xzy_angles
from ..noise.builder import apply_gate_sequence_noise, apply_gate_sequence_noise_probabilistic
from .pqc_architectures import PQCArchitectureBase


class PQCModelBase:
    """
    Base class for PQC models with unified training logic.
    
    This class handles:
    - Circuit template creation and caching
    - Batched statevector simulation (JAX)
    - Progressive/individual block training support
    - Parameter get/set with validation
    
    Architecture-specific parameter management is delegated to PQCArchitectureBase subclasses.
    """
    
    def __init__(self, 
                 base_circuit_ops: List,
                 num_qubits: int,
                 x_noise: np.ndarray,
                 z_noise: np.ndarray,
                 pqc_architecture: PQCArchitectureBase,
                 pqc_blocks: int = 1,
                 gate_blocks: int = 1,
                 pqc_type: str = 'zxz',
                 noise_type: str = 'rotation',
                 gate_sequence_noise_rules: Optional[Dict] = None,
                 gate_sequence_noise_prob: float = 1.0,
                 noise_seed: Optional[int] = None):
        """
        Initialize the PQC model.
        
        Args:
            base_circuit_ops: List of base circuit operations (without noise/PQC)
            num_qubits: Number of qubits
            x_noise: X-noise array for each gate (fixed during training)
            z_noise: Z-noise array for each gate (fixed during training)
            pqc_architecture: PQCArchitectureBase instance managing parameters
            pqc_blocks: Number of PQC blocks
            gate_blocks: Number of gates per block before adding PQC
            pqc_type: Type of PQC decomposition ('zxz' or 'xzy')
            noise_type: Type of noise model ('rotation', 'gate_sequence', or 'both')
                - 'rotation': Traditional RxRz noise (default)
                - 'gate_sequence': Coherent gate sequence transformations (HH→HX, etc.)
                - 'both': Apply both noise types
            gate_sequence_noise_rules: Custom transformation rules for gate sequence noise
                If None, uses defaults (HH→HX, XX→XZ, ZZ→ZH)
            gate_sequence_noise_prob: Probability of applying gate sequence transformations (0-1)
            noise_seed: Random seed for gate sequence noise (if probabilistic)
        """
        self.num_qubits = num_qubits
        self.base_circuit_ops = copy.deepcopy(base_circuit_ops)
        self.num_gates = len(self.base_circuit_ops)
        self.pqc_blocks = pqc_blocks
        self.gate_blocks = gate_blocks
        
        # Store noise configuration
        self.noise_type = noise_type.lower()
        self.gate_sequence_noise_rules = gate_sequence_noise_rules
        self.gate_sequence_noise_prob = gate_sequence_noise_prob
        self.noise_seed = noise_seed
        
        # Apply gate sequence noise if requested
        if self.noise_type in ['gate_sequence', 'both']:
            print(f"Applying gate sequence noise (type={self.noise_type}, prob={gate_sequence_noise_prob})")
            if gate_sequence_noise_prob < 1.0:
                # Probabilistic transformations
                self.base_circuit_ops = apply_gate_sequence_noise_probabilistic(
                    self.base_circuit_ops,
                    transformation_rules=gate_sequence_noise_rules,
                    error_probability=gate_sequence_noise_prob,
                    seed=noise_seed
                )
            else:
                # Deterministic transformations
                self.base_circuit_ops = apply_gate_sequence_noise(
                    self.base_circuit_ops,
                    noise=gate_sequence_noise_rules,
                    seed=noise_seed
                )
        
        # Store rotation noise arrays (fixed during training)
        # These are used only if noise_type is 'rotation' or 'both'
        self.x_noise = x_noise.astype(np.float32)
        self.z_noise = z_noise.astype(np.float32)
        
        # Store architecture handler
        self.pqc_arch = pqc_architecture
        
        # Calculate number of PQC layers
        self.num_pqc_layers = int(pqc_blocks * jnp.ceil(self.num_gates / gate_blocks))
        
        # Verify architecture has correct number of layers
        if self.pqc_arch.num_layers != self.num_pqc_layers:
            raise ValueError(f"Architecture has {self.pqc_arch.num_layers} layers, "
                           f"but model needs {self.num_pqc_layers} layers")
        
        # Set up quaternion conversion function (if needed)
        if pqc_type == 'zxz':
            self.quaternion_to_angles_fn = quaternion_to_zxz_angles
        elif pqc_type == 'xzy':
            self.quaternion_to_angles_fn = quaternion_to_xzy_angles
        else:
            raise ValueError(f"Unknown pqc_type: {pqc_type}")
        
        # Initialize parameters using architecture
        self.params = self.pqc_arch.initialize_params()
        
        # Store base circuit parameters
        self.base_params = np.array([
            op[2][0] if len(op[2]) > 0 else 0.0 
            for op in base_circuit_ops
        ], dtype=np.float32)
        
        # Build main circuit template
        template_type = self.pqc_arch.get_template_type()
        
        # Determine whether to add rotation noise to template
        # - If noise_type is 'rotation' or 'both': add RxRz noise gates
        # - If noise_type is 'gate_sequence': don't add rotation noise (already applied gate mods)
        add_rotation_noise = self.noise_type in ['rotation', 'both']
        
        self.template = build_pqc_circuit_template(
            base_ops=self.base_circuit_ops,  # Use potentially modified ops
            num_qubits=num_qubits,
            num_gate_blocks=gate_blocks,
            add_noise=add_rotation_noise,
            pqc_type=template_type
        )
        
        # Pre-build cached templates for progressive/individual training
        self.partial_templates = {}
        for idx in range(self.num_pqc_layers):
            self.partial_templates[idx] = self._build_partial_template(idx)
        
        self.individual_block_templates = {}
        for idx in range(self.num_pqc_layers):
            self.individual_block_templates[idx] = self._build_individual_block_template(idx)

    def get_pqc_params(self) -> jnp.ndarray:
        """Get all PQC parameters as a single array."""
        pqc_params = {}
        for key, value in self.params.items():
            if 'quaternion' in key:
                # Convert quaternions to angles
                angle_key = key.replace('quaternion', 'angle')
                pqc_params[angle_key] = self.convert_quaternions_to_angles(value).tolist()
            else:
                # theta_zz or other direct params
                pqc_params[key] = value.tolist()

        return pqc_params

    def get_model_params(self) -> Tuple:
        """Get all trainable model parameters as tuple in canonical order."""
        return self.pqc_arch.params_dict_to_tuple(self.params)
    
    def get_model_params_dict(self) -> Dict[str, jnp.ndarray]:
        """Get all trainable model parameters as dict."""
        return self.params.copy()
    
    def set_model_params(self, new_params: Union[Tuple, Dict[str, jnp.ndarray]]):
        """
        Set model parameters with NaN checking.
        
        Args:
            new_params: Either tuple in canonical order or dict with parameter names
        """
        # Convert to dict if tuple
        if isinstance(new_params, (tuple, list)):
            params_dict = self.pqc_arch.params_tuple_to_dict(new_params)
        else:
            params_dict = new_params
        
        # Validate and check for NaNs
        for name, param in params_dict.items():
            if jnp.isnan(param).any():
                raise ValueError(f"Parameter '{name}' contains NaNs")
            params_dict[name] = param.astype(jnp.float32)
        
        self.params = params_dict
    
    def convert_quaternions_to_angles(self, quaternions: jnp.ndarray) -> jnp.ndarray:
        """Convert batch of quaternions to PQC angles."""
        return jax.vmap(jax.vmap(self.quaternion_to_angles_fn))(quaternions)
    
    def _parse_params_flexible(self, args: tuple, kwargs: dict) -> Dict[str, jnp.ndarray]:
        """
        Parse parameters from flexible calling conventions.
        
        Supports:
        1. No args/kwargs: use self.params
        2. Single tuple arg: convert tuple to dict
        3. Single dict arg: use dict directly
        4. Multiple args: assume they're in canonical order (for LEL-ZZ: pre, theta, post)
        5. Kwargs: use as dict
        """
        if not args and not kwargs:
            # No params provided, use stored params
            return self.params
        
        if args:
            if len(args) == 1:
                # Single argument - could be tuple or dict
                param_arg = args[0]
                if isinstance(param_arg, dict):
                    return param_arg
                elif isinstance(param_arg, (tuple, list)):
                    return self.pqc_arch.params_tuple_to_dict(param_arg)
                else:
                    raise ValueError(f"Single parameter must be dict or tuple, got {type(param_arg)}")
            else:
                # Multiple arguments - assume canonical order
                return self.pqc_arch.params_tuple_to_dict(args)
        
        if kwargs:
            # Check if 'params' kwarg was provided
            if 'params' in kwargs:
                params = kwargs['params']
                if isinstance(params, dict):
                    return params
                elif isinstance(params, (tuple, list)):
                    return self.pqc_arch.params_tuple_to_dict(params)
            else:
                # Kwargs are the parameter dict directly
                return kwargs
        
        raise ValueError("Could not parse parameters from args/kwargs")
    
    def _prepare_param_dict_for_template(self, 
                                         params: Optional[Dict[str, jnp.ndarray]] = None,
                                         gate_slice: Optional[slice] = None) -> Dict:
        """
        Prepare parameter dictionary for template instantiation.
        
        This handles quaternion→angle conversion if needed and slices arrays appropriately.
        
        Args:
            params: Parameter dict (uses self.params if None)
            gate_slice: Slice object for base gates (None = all gates)
        
        Returns:
            Dict ready for template.instantiate()
        """
        if params is None:
            params = self.params
        
        # Start with base circuit params
        if gate_slice is not None:
            param_dict = {
                'base': self.base_params[gate_slice],
                'x_noise': self.x_noise[gate_slice],
                'z_noise': self.z_noise[gate_slice],
            }
        else:
            param_dict = {
                'base': self.base_params,
                'x_noise': self.x_noise,
                'z_noise': self.z_noise,
            }
        
        # Add PQC parameters, converting quaternions if present
        for key, value in params.items():
            if 'quaternion' in key:
                # Convert quaternions to angles
                angle_key = key.replace('quaternion', 'param')
                param_dict[angle_key] = self.convert_quaternions_to_angles(value)
            elif 'angle' in key:
                # Direct angle parametrization, just rename
                angle_key = key.replace('angle', 'param')
                param_dict[angle_key] = value
            else:
                # theta_zz or other direct params
                param_dict[key] = value
        
        return param_dict
    
    def run_model_batch(self, input_states: jnp.ndarray, 
                       params: Optional[Union[Tuple, Dict[str, jnp.ndarray]]] = None) -> jnp.ndarray:
        """
        Run the model on a batch of input states.
        
        Args:
            input_states: Batch of input quantum states (B, 2^n)
            params: Optional parameters as tuple or dict (uses self.params if None)
        
        Returns:
            output_states: Batch of output quantum states (B, 2^n)
        """
        # Convert tuple to dict if needed
        if isinstance(params, (tuple, list)):
            params = self.pqc_arch.params_tuple_to_dict(params)
        
        param_dict = self._prepare_param_dict_for_template(params)
        
        # Instantiate template with current parameters
        circuit_ops = self.template.instantiate(param_dict)
        
        # Convert to JAX format and run simulation
        gate_ids, wire1s, wire2s, thetas = build_jax_circuit(circuit_ops)
        output_states = jax_run_many_states(
            self.num_qubits, gate_ids, wire1s, wire2s, thetas, input_states
        )
        
        return output_states
    
    def run_single_block_batch(self, input_states: jnp.ndarray, block_idx: int,
                               *args, **kwargs) -> jnp.ndarray:
        """
        Run ONLY the specified block in isolation (not cascaded).
        
        This simulates: input → base_gates[block_idx] → PQC[block_idx] → output
        
        Args:
            input_states: Batch of input quantum states (B, 2^n)
            block_idx: Which block to simulate (0-indexed)
            
            Params can be passed in multiple ways (see run_model_batch_up_to_block docs)
        
        Returns:
            output_states: Batch of output quantum states (B, 2^n)
        """
        # Parse parameters from various calling conventions
        params = self._parse_params_flexible(args, kwargs)
        
        # Slice parameters for this block only
        block_params = {
            name: param[block_idx:block_idx+1] 
            for name, param in params.items()
        }
        
        # Calculate gate indices for this block
        gate_start = self.gate_blocks * block_idx
        gate_end = self.gate_blocks * (block_idx + 1)
        gate_slice = slice(gate_start, gate_end)
        
        # Prepare params for template
        param_dict = self._prepare_param_dict_for_template(block_params, gate_slice)
        
        # Get cached template and instantiate
        template = self.individual_block_templates[block_idx]
        circuit_ops = template.instantiate(param_dict)
        
        # Convert to JAX format and run simulation
        gate_ids, wire1s, wire2s, thetas = build_jax_circuit(circuit_ops)
        output_states = jax_run_many_states(
            self.num_qubits, gate_ids, wire1s, wire2s, thetas, input_states
        )
        
        return output_states
    
    def run_model_batch_up_to_block(self, input_states: jnp.ndarray, max_block_idx: int,
                                    *args, **kwargs) -> jnp.ndarray:
        """
        Run model but only simulate up to and including max_block_idx.
        
        Used for progressive training where we train blocks one at a time.
        
        Args:
            input_states: Batch of input quantum states (B, 2^n)
            max_block_idx: Last block to simulate (0-indexed)
            
            Params can be passed in multiple ways:
            1. As tuple: run_model_batch_up_to_block(states, idx, (pre, theta, post))
            2. As dict: run_model_batch_up_to_block(states, idx, params={'pre_quaternions': ...})
            3. As separate args: run_model_batch_up_to_block(states, idx, pre, theta, post)
            4. As kwargs: run_model_batch_up_to_block(states, idx, pre_quaternions=..., theta_zz=...)
            5. None (uses self.params)
        
        Returns:
            output_states: Batch of output quantum states (B, 2^n)
        """
        # Parse parameters from various calling conventions
        params = self._parse_params_flexible(args, kwargs)
        
        # Slice parameters to include blocks 0 through max_block_idx
        partial_params = {
            name: param[:max_block_idx + 1]
            for name, param in params.items()
        }
        
        # Calculate gate indices
        num_gates = self.gate_blocks * (max_block_idx + 1)
        gate_slice = slice(0, num_gates)
        
        # Prepare params for template
        param_dict = self._prepare_param_dict_for_template(partial_params, gate_slice)
        
        # Get cached template and instantiate
        template = self.partial_templates[max_block_idx]
        circuit_ops = template.instantiate(param_dict)
        
        # Convert to JAX format and run simulation
        gate_ids, wire1s, wire2s, thetas = build_jax_circuit(circuit_ops)
        output_states = jax_run_many_states(
            self.num_qubits, gate_ids, wire1s, wire2s, thetas, input_states
        )
        
        return output_states
    
    def get_circuit_tokens(self) -> List:
        """Get full circuit with current PQC parameters and Error Model as tokens."""
        param_dict = self._prepare_param_dict_for_template()
        pqc_ops = self.template.instantiate(param_dict)
        
        # Convert operations to serializable tokens
        tokens = []
        for gate, qubits, params in pqc_ops:
            # Convert JAX arrays to native Python floats for serialization
            if params and isinstance(params[0], jnp.ndarray):
                serialized_params = [np.float32(params[0].item())]
            else:
                serialized_params = params
            
            tokens.append((gate, qubits, serialized_params))
        
        return tokens
    
    def _build_individual_block_template(self, block_idx: int):
        """Build template for a single isolated block."""
        gate_start = self.gate_blocks * block_idx
        gate_end = self.gate_blocks * (block_idx + 1)
        block_base_ops = self.base_circuit_ops[gate_start:gate_end]
        
        template_type = self.pqc_arch.get_template_type()
        add_rotation_noise = self.noise_type in ['rotation', 'both']
        
        return build_pqc_circuit_template(
            base_ops=block_base_ops,
            num_qubits=self.num_qubits,
            num_gate_blocks=self.gate_blocks,
            add_noise=add_rotation_noise,
            pqc_type=template_type
        )
    
    def _build_partial_template(self, max_block_idx: int):
        """Build template up to and including max_block_idx."""
        num_gates_to_include = self.gate_blocks * (max_block_idx + 1)
        partial_base_ops = self.base_circuit_ops[:num_gates_to_include]
        
        template_type = self.pqc_arch.get_template_type()
        add_rotation_noise = self.noise_type in ['rotation', 'both']
        
        return build_pqc_circuit_template(
            base_ops=partial_base_ops,
            num_qubits=self.num_qubits,
            num_gate_blocks=self.gate_blocks,
            add_noise=add_rotation_noise,
            pqc_type=template_type
        )
