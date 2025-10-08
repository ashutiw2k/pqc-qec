"""PQC models using custom Numba statevector simulator with JAX autodiff.

This module provides drop-in replacements for Pennylane-based models that use
the high-performance Numba simulator while maintaining JAX compatibility for
automatic differentiation.

Key features:
- Same interface as Pennylane models (compatible with existing training code)
- Faster forward pass (Numba vs Pennylane overhead)
- JAX gradients via finite differences
- Supports quaternion parameterization and LEL-ZZ blocks
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from ..simulate.jax_statevector import statevec_simulate_jax
from ..noise.builder import (
    build_regular_noisy_circuit, 
    build_circuit,
    create_pqc_circuit_template_simplified,
    update_pqc_circuit_template,
    decompile_circuit
)
from ..utils.quaternions_utils import quaternion_to_zxz_angles, quaternion_to_xzy_angles


def _build_circuit_from_params_numpy(noisy_base_ops, pqc_gates, num_qubits, num_blocks, 
                                      gate_blocks, pre_angles, theta_zz, post_angles):
    """Build circuit from parameters (handles both NumPy and JAX arrays).
    
    This function builds the full circuit including PQC blocks with current parameters.
    It accepts either NumPy arrays or JAX tracers during gradient computation.
    
    Args:
        noisy_base_ops: List of base circuit operations (static)
        pqc_gates: List of PQC gate names (static)
        num_qubits, num_blocks, gate_blocks: Circuit structure (static)
        pre_angles: Pre-local Euler angles [num_blocks, num_qubits, 3] (NumPy/JAX)
        theta_zz: ZZ coupling angles [num_blocks, num_qubits] (NumPy/JAX)
        post_angles: Post-local Euler angles [num_blocks, num_qubits, 3] (NumPy/JAX)
    
    Returns:
        Tuple of (gate_ids, wire1, wire2, theta) as NumPy arrays
    """
    # NOTE: Don't convert to NumPy here - the arrays might be JAX tracers during gradient computation.
    # The float() conversions below will handle both NumPy and JAX arrays correctly.
    
    # Build circuit with current parameters
    full_circuit_ops = []
    logical_gate_count = 0
    block_idx = 0
    
    for op in noisy_base_ops:
        full_circuit_ops.append(op[:3])
        
        is_noise = (len(op) > 3 and isinstance(op[3], dict) and 
                   op[3].get('noise', False))
        
        if not is_noise:
            logical_gate_count += 1
            
            if (logical_gate_count % gate_blocks == 0 and block_idx < num_blocks):
                # Pre-local rotations
                for q in range(num_qubits):
                    for g_idx, gate_name in enumerate(pqc_gates):
                        full_circuit_ops.append(
                            (gate_name, [q], [float(pre_angles[block_idx, q, g_idx])])
                        )
                
                # ZZ entangling ring
                for q in range(num_qubits):
                    q_next = (q + 1) % num_qubits
                    full_circuit_ops.append(('cx', [q, q_next], []))
                    full_circuit_ops.append(('rz', [q_next], 
                                           [float(theta_zz[block_idx, q])]))
                    full_circuit_ops.append(('cx', [q, q_next], []))
                
                # Post-local rotations
                for q in range(num_qubits):
                    for g_idx, gate_name in enumerate(pqc_gates):
                        full_circuit_ops.append(
                            (gate_name, [q], [float(post_angles[block_idx, q, g_idx])])
                        )
                
                block_idx += 1
    
    return build_circuit(full_circuit_ops, dtype=np.float32)


class CustomStatevecComplexQuaternionModel:
    """LEL-ZZ PQC model using custom Numba simulator with JAX autodiff.
    
    This model implements the Local-Entangling Layer with ZZ gates (LEL-ZZ) architecture:
    - Pre-local rotations (quaternion → RZ-RX-RZ per qubit)
    - ZZ entangling ring (parameterized ZZ gates between adjacent qubits)
    - Post-local rotations (quaternion → RZ-RX-RZ per qubit)
    
    The model:
    - Uses Numba simulator for forward pass (fast)
    - Provides JAX gradients via finite differences
    - Maintains same interface as Pennylane version
    - Compatible with existing JAX training code (Optax optimizers, etc.)
    
    Args:
        circuit_ops: List of base circuit operations [(gate, qubits, params), ...]
        num_qubits: Number of qubits in the circuit
        noise_model: PennylaneNoisyGates instance (used only for sampling noise)
        pqc_blocks: Number of PQC insertions per gate_blocks base gates
        gate_blocks: Insert PQC after every gate_blocks base circuit gates
        seed: Random seed for parameter initialization
        pqc_type: 'zxz' or 'xzy' (Euler angle decomposition type)
    """
    
    def __init__(self, circuit_ops, num_qubits, noise_model, pqc_blocks, 
                 gate_blocks, seed, pqc_type='zxz'):
        self.num_qubits = num_qubits
        self.circuit_ops = circuit_ops
        self.pqc_blocks = pqc_blocks
        self.gate_blocks = gate_blocks
        self.num_gates = len(circuit_ops)
        
        # Sample noise ONCE (fixed for this circuit instance)
        # The noise is sampled from PennylaneNoisyGates but applied via Numba
        self.x_noise, self.z_noise = self._sample_noise_from_pennylane_model(
            noise_model, self.num_gates
        )
        
        # PQC decomposition type
        if pqc_type == 'zxz':
            self.pqc_gates = ['rz', 'rx', 'rz']
            self.quat_to_angles = quaternion_to_zxz_angles
        elif pqc_type == 'xzy':
            self.pqc_gates = ['rx', 'rz', 'ry']
            self.quat_to_angles = quaternion_to_xzy_angles
        else:
            raise ValueError(f"pqc_type must be 'zxz' or 'xzy', got {pqc_type}")
        
        # Initialize parameters (JAX arrays for autodiff)
        self.num_blocks = int(pqc_blocks * jnp.ceil(self.num_gates / gate_blocks))
        self.pre_quaternions = self._init_quaternions(seed + 1)
        self.post_quaternions = self._init_quaternions(seed + 2)
        self.theta_zz = self._init_angles(seed + 3)
        
        # Build noisy circuit structure (done once at initialization)
        self._build_noisy_base_circuit()
        
        # Pre-build circuit template for gradient-compatible execution
        self._build_circuit_template()
    
    def _sample_noise_from_pennylane_model(self, noise_model, num_gates):
        """Extract noise samples from PennylaneNoisyGates model.
        
        The noise model is only used to generate random noise values.
        The actual noise application happens via the Numba simulator.
        """
        x_noise = noise_model.rng.uniform(
            noise_model.x_noise_min, noise_model.x_noise_max, num_gates
        )
        z_noise = noise_model.rng.uniform(
            noise_model.z_noise_min, noise_model.z_noise_max, num_gates
        )
        return x_noise.astype(np.float32), z_noise.astype(np.float32)
    
    def _init_quaternions(self, seed):
        """Initialize unit quaternions as small rotations near identity.
        
        For uncomputation (U U†), PQC should start near identity and learn
        small corrections. Uses small random rotations (0.01-0.1 radians).
        
        Returns:
            JAX array of shape [num_blocks, num_qubits, 4] (w, x, y, z components)
        """
        key = jax.random.PRNGKey(seed)
        k_axis, k_ang = jax.random.split(key)
        
        shape = (self.num_blocks, self.num_qubits, 4)
        
        # Random unit axes
        axes = jax.random.normal(k_axis, shape[:-1] + (3,), dtype=jnp.float32)
        axes = axes / (jnp.linalg.norm(axes, axis=-1, keepdims=True) + 1e-12)
        
        # Small angles (0.01 to 0.1 radians)
        angles = jax.random.uniform(k_ang, shape[:-1] + (1,), 
                                    minval=0.01, maxval=0.1, dtype=jnp.float32)
        
        # Construct quaternions: q = [cos(θ/2), sin(θ/2) * axis]
        w = jnp.cos(0.5 * angles)
        v = axes * jnp.sin(0.5 * angles)
        q = jnp.concatenate([w, v], axis=-1)
        
        # Enforce w >= 0 to avoid q ~ -q ambiguity
        flip = jnp.where(q[..., 0:1] < 0, -1.0, 1.0)
        return (q * flip).astype(jnp.float32)
    
    def _init_angles(self, seed):
        """Initialize ZZ coupling angles (small values near zero)."""
        key = jax.random.PRNGKey(seed)
        return jax.random.normal(key, (self.num_blocks, self.num_qubits), 
                                dtype=jnp.float32) * 1e-3
    
    def _build_noisy_base_circuit(self):
        """Build the noisy base circuit (without PQC).
        
        This is done once at initialization. The PQC blocks are added
        dynamically during forward pass based on current parameters.
        """
        # Build noisy circuit with tagged noise gates
        self.noisy_base_ops = build_regular_noisy_circuit(
            self.circuit_ops, self.x_noise, self.z_noise, return_tagged=True
        )
        
        # Create circuit template info for analysis
        self._create_circuit_template()
    
    def _create_circuit_template(self):
        """Create PQC circuit template using the template builder.
        
        LEL-ZZ structure is flattened into a list of PQC gates:
        - Pre-local: RZ, RX, RZ (3 gates per qubit)
        - ZZ entangling: CX, RZ, CX for each pair (3 * num_qubits gates total)
        - Post-local: RZ, RX, RZ (3 gates per qubit)
        
        Total: 6*num_qubits + 3*num_qubits = 9*num_qubits gates per LEL-ZZ block
        """
        # Create flat list of PQC gates for LEL-ZZ structure
        # Pre-locals (per qubit): RZ, RX, RZ
        lel_zz_gates = []
        for q in range(self.num_qubits):
            lel_zz_gates.extend([('rz', [q]), ('rx', [q]), ('rz', [q])])
        
        # ZZ entangling ring (per adjacent pair): CX, RZ, CX  
        for q in range(self.num_qubits):
            q_next = (q + 1) % self.num_qubits
            lel_zz_gates.extend([
                ('cx', [q, q_next]),
                ('rz', [q_next]),
                ('cx', [q, q_next])
            ])
        
        # Post-locals (per qubit): RZ, RX, RZ
        for q in range(self.num_qubits):
            lel_zz_gates.extend([('rz', [q]), ('rx', [q]), ('rz', [q])])
        
        # Store gate structure for parameter mapping
        self.lel_zz_gates = lel_zz_gates
        self.num_lel_zz_params = len(lel_zz_gates)  # 9 * num_qubits
        
        # Extract gate names for template creation
        pqc_gate_names_for_template = [gate[0] for gate in lel_zz_gates]
        
        # Create template using the builder (done once)
        # Note: We pass dummy pqc_gates list for template, actual structure in lel_zz_gates
        # The template builder expects pqc_gates to be per-qubit, but LEL-ZZ is more complex
        # So we'll use build_circuit directly in forward pass with the template approach
        # Store for manual template updates
        self.circuit_template_gates = pqc_gate_names_for_template
    
    def _build_circuit_template(self):
        """Pre-build circuit structure with dummy parameters.
        
        This creates the FULL circuit structure (gate_ids, wire1, wire2) ONCE
        at initialization with placeholder theta values. During forward pass,
        we only update the theta array, which is JAX-compatible.
        
        Stores:
            template_gate_ids, template_wire1, template_wire2: Circuit structure (NumPy)
            param_indices: Dict mapping (block_idx, q, param_type) -> theta_index
        """
        # Use dummy angles to build the structure
        # pre/post_angles: [num_blocks, num_qubits, 3] for RZ-RX-RZ per qubit
        dummy_pre = np.zeros((self.num_blocks, self.num_qubits, 3), dtype=np.float32)
        dummy_zz = np.zeros((self.num_blocks, self.num_qubits), dtype=np.float32)
        dummy_post = np.zeros((self.num_blocks, self.num_qubits, 3), dtype=np.float32)
        
        # Build with dummy parameters to get structure
        gate_ids, wire1, wire2, theta = _build_circuit_from_params_numpy(
            self.noisy_base_ops, self.pqc_gates, self.num_qubits,
            self.num_blocks, self.gate_blocks, dummy_pre, dummy_zz, dummy_post
        )
        
        # Store template structure (these never change)
        self.template_gate_ids = gate_ids
        self.template_wire1 = wire1
        self.template_wire2 = wire2
        self.template_theta = theta  # Will be updated each forward pass
        
        # Build parameter index mapping for fast updates
        # We need to know: which theta[i] corresponds to which (block, qubit, gate)?
        self._build_param_index_mapping()
    
    def _build_param_index_mapping(self):
        """Build mapping from parameter indices to theta array indices.
        
        This allows us to quickly update theta values during forward pass
        without rebuilding the circuit structure.
        """
        # Trace through circuit building to identify parameter locations
        param_map = {}
        theta_idx = 0
        
        # Count logical gates in base circuit to find insertion points
        logical_count = 0
        block_idx = 0
        
        for op in self.noisy_base_ops:
            is_noise = (len(op) > 3 and isinstance(op[3], dict) and 
                       op[3].get('noise', False))
            
            # Base gate theta (skip if noise)
            if not is_noise:
                logical_count += 1
            theta_idx += 1
            
            # PQC insertion point?
            if not is_noise and (logical_count % self.gate_blocks == 0) and (block_idx < self.num_blocks):
                # Pre-local rotations (3 gates per qubit)
                for q in range(self.num_qubits):
                    for g_idx in range(3):  # RZ, RX, RZ
                        param_map[('pre', block_idx, q, g_idx)] = theta_idx
                        theta_idx += 1
                
                # ZZ gates (1 param per qubit, but 3 gates each: CX, RZ, CX)
                for q in range(self.num_qubits):
                    theta_idx += 1  # CX (no param)
                    param_map[('zz', block_idx, q)] = theta_idx
                    theta_idx += 1  # RZ with param
                    theta_idx += 1  # CX (no param)
                
                # Post-local rotations (3 gates per qubit)
                for q in range(self.num_qubits):
                    for g_idx in range(3):  # RZ, RX, RZ
                        param_map[('post', block_idx, q, g_idx)] = theta_idx
                        theta_idx += 1
                
                block_idx += 1
        
        self.param_index_map = param_map
    
    def _build_lel_zz_circuit_with_params(self, pre_angles, theta_zz, post_angles):
        """Build complete circuit with LEL-ZZ blocks using pre-built template.
        
        This is JAX-compatible! Instead of rebuilding the circuit structure,
        we use the pre-built template and only update theta values.
        This works during gradient computation because:
        - gate_ids, wire1, wire2 are static (pre-built NumPy arrays)
        - theta array is updated with JAX arrays (differentiable)
        
        Args:
            pre_angles: [num_blocks, num_qubits, 3] - Pre-local Euler angles (JAX array)
            theta_zz: [num_blocks, num_qubits] - ZZ coupling angles (JAX array)
            post_angles: [num_blocks, num_qubits, 3] - Post-local Euler angles (JAX array)
        
        Returns:
            Tuple of (gate_ids, wire1, wire2, theta) for Numba simulator
        """
        # Start with template theta (copy to avoid modifying template)
        theta = jnp.array(self.template_theta, dtype=jnp.float32)
        
        # Update parameter values in theta array using the index mapping
        updates = []
        for block_idx in range(self.num_blocks):
            # Pre-local rotations
            for q in range(self.num_qubits):
                for g_idx in range(3):
                    key = ('pre', block_idx, q, g_idx)
                    if key in self.param_index_map:
                        idx = self.param_index_map[key]
                        updates.append((idx, pre_angles[block_idx, q, g_idx]))
            
            # ZZ coupling angles
            for q in range(self.num_qubits):
                key = ('zz', block_idx, q)
                if key in self.param_index_map:
                    idx = self.param_index_map[key]
                    updates.append((idx, theta_zz[block_idx, q]))
            
            # Post-local rotations
            for q in range(self.num_qubits):
                for g_idx in range(3):
                    key = ('post', block_idx, q, g_idx)
                    if key in self.param_index_map:
                        idx = self.param_index_map[key]
                        updates.append((idx, post_angles[block_idx, q, g_idx]))
        
        # Apply all updates to theta array (JAX-compatible)
        indices = jnp.array([idx for idx, _ in updates], dtype=jnp.int32)
        values = jnp.array([val for _, val in updates], dtype=jnp.float32)
        theta = theta.at[indices].set(values)
        
        # Return template structure with updated theta
        return (self.template_gate_ids, self.template_wire1, 
                self.template_wire2, theta)
    
    def get_model_params(self):
        """Get model parameters as JAX dict (compatible with Optax optimizers).
        
        Returns:
            Dict with keys: 'pre_quaternions', 'theta_zz', 'post_quaternions'
        """
        return {
            'pre_quaternions': self.pre_quaternions,
            'theta_zz': self.theta_zz,
            'post_quaternions': self.post_quaternions
        }
    
    def set_model_params(self, params):
        """Set model parameters from JAX dict (called by training loop).
        
        Args:
            params: Dict with keys: 'pre_quaternions', 'theta_zz', 'post_quaternions'
        """
        self.pre_quaternions = params['pre_quaternions']
        self.theta_zz = params['theta_zz']
        self.post_quaternions = params['post_quaternions']
    
    def _quats_to_angles(self, quats):
        """Convert quaternions to Euler angles (vectorized).
        
        Uses JAX vmap to vectorize over blocks and qubits.
        
        Args:
            quats: [num_blocks, num_qubits, 4]
        
        Returns:
            angles: [num_blocks, num_qubits, 3]
        """
        return jax.vmap(jax.vmap(self.quat_to_angles))(quats)
    
    def run_model_batch(self, input_states, params=None):
        """Run model on batch of input states.
        
        This is the main forward pass that:
        1. Converts quaternions to Euler angles
        2. Builds circuit with current parameters
        3. Runs Numba simulator
        4. Returns output states
        
        The function is JAX-differentiable and can be used in loss functions.
        
        Args:
            input_states: [batch, 2^n] JAX array of input quantum states
            params: Optional dict of parameters (uses self.* if None)
        
        Returns:
            output_states: [batch, 2^n] JAX array of output quantum states
        """
        # Get parameters (use provided or internal)
        if params is None:
            pre_q = self.pre_quaternions
            theta_zz = self.theta_zz
            post_q = self.post_quaternions
        else:
            pre_q = params['pre_quaternions']
            theta_zz = params['theta_zz']
            post_q = params['post_quaternions']
        
        # Convert quaternions to Euler angles
        pre_angles = self._quats_to_angles(pre_q)   # [num_blocks, num_qubits, 3]
        post_angles = self._quats_to_angles(post_q)  # [num_blocks, num_qubits, 3]
        
        # Build circuit with current parameters using pre-built template
        # gate_ids, wire1, wire2 are static NumPy arrays (never change)
        # theta is a JAX array with updated parameter values (differentiable)
        gate_ids, wire1, wire2, theta = self._build_lel_zz_circuit_with_params(
            pre_angles, theta_zz, post_angles
        )
        
        # CRITICAL: gate_ids, wire1, wire2 MUST be NumPy arrays for nondiff_argnums
        # They come from self.template_* which are NumPy arrays
        # DO NOT convert them to JAX arrays - use them directly
        # Only theta needs to be a JAX array for gradients
        
        # Run simulation with custom statevector simulator
        # This calls the Numba backend with JAX gradient support
        return statevec_simulate_jax(input_states, gate_ids, wire1, 
                                      wire2, theta, self.num_qubits)
    
    def __call__(self, *args, **kwargs):
        """Allow model to be called like a function (compatible with training code)."""
        return self.run_model_batch(*args, **kwargs)
    
    def get_pqc_params(self):
        """Get PQC parameters in Euler angle form (for analysis/saving).
        
        Returns:
            Tuple of (pre_angles, theta_zz, post_angles)
        """
        pre_angles = self._quats_to_angles(self.pre_quaternions)
        post_angles = self._quats_to_angles(self.post_quaternions)
        return pre_angles, self.theta_zz, post_angles
    
    def get_circuit_tokens(self):
        """Get circuit tokens for visualization/saving.
        
        Builds the complete circuit with current parameters and decompiles it
        back to human-readable token format.
        
        Returns:
            List of circuit operations in format (gate_name, [qubits], [params])
        """
        # Convert quaternions to Euler angles
        pre_angles = self._quats_to_angles(self.pre_quaternions)
        post_angles = self._quats_to_angles(self.post_quaternions)
        
        # Build circuit with current parameters
        gate_ids, wire1, wire2, theta = self._build_lel_zz_circuit_with_params(
            pre_angles, self.theta_zz, post_angles
        )
        
        # Decompile back to token format using builder utility
        circuit_tokens = decompile_circuit(gate_ids, wire1, wire2, theta)
        
        return circuit_tokens
