"""
PQC Architecture parameter management classes.

These classes handle parameter initialization, naming, and structure for different
PQC architectures. They do NOT handle circuit execution or training logic.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional


class PQCArchitectureBase:
    """Base class for PQC architecture parameter management."""
    
    def __init__(self, num_layers: int, num_qubits: int, seed: int = 0):
        """
        Initialize base PQC architecture.
        
        Args:
            num_layers: Number of PQC layers
            num_qubits: Number of qubits
            seed: Random seed for initialization
        """
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.seed = seed
        
    def initialize_params(self) -> Dict[str, jnp.ndarray]:
        """Initialize and return all parameters as a dict."""
        raise NotImplementedError("Subclasses must implement initialize_params")
    
    def get_param_names(self) -> Tuple[str, ...]:
        """Return tuple of parameter names in order."""
        raise NotImplementedError("Subclasses must implement get_param_names")
    
    def get_template_type(self) -> str:
        """Return the template type string for circuit building."""
        raise NotImplementedError("Subclasses must implement get_template_type")
    
    def params_dict_to_tuple(self, params: Dict[str, jnp.ndarray]) -> Tuple:
        """Convert params dict to tuple in canonical order."""
        return tuple(params[name] for name in self.get_param_names())
    
    def params_tuple_to_dict(self, params: Tuple) -> Dict[str, jnp.ndarray]:
        """Convert params tuple to dict."""
        param_names = self.get_param_names()
        if len(params) != len(param_names):
            raise ValueError(f"Expected {len(param_names)} params, got {len(params)}")
        return {name: param for name, param in zip(param_names, params)}


class LELZZQuaternionArchitecture(PQCArchitectureBase):
    """
    LEL-ZZ architecture with quaternion parametrization.
    
    Structure: Pre-local (RzRxRz) → ZZ entangling ring → Post-local (RzRxRz)
    
    Parameters:
    - pre_quaternions: (num_layers, num_qubits, 4) - Pre-layer local unitaries
    - theta_zz: (num_layers, num_qubits, 1) - ZZ entangling angles
    - post_quaternions: (num_layers, num_qubits, 4) - Post-layer local unitaries
    """
    
    def __init__(self, num_layers: int, num_qubits: int, seed: int = 0, 
                 pqc_type: str = 'zxz'):
        """
        Initialize LEL-ZZ quaternion architecture.
        
        Args:
            num_layers: Number of PQC layers
            num_qubits: Number of qubits
            seed: Random seed for initialization
            pqc_type: Local unitary decomposition ('zxz' or 'xzy')
        """
        super().__init__(num_layers, num_qubits, seed)
        self.pqc_type = pqc_type
        
        # Determine template string based on pqc_type
        if pqc_type == 'zxz':
            self.local_type = 'rzrxrz'
        elif pqc_type == 'xzy':
            self.local_type = 'rxrzry'
        else:
            raise ValueError(f"Unknown pqc_type: {pqc_type}")
    
    def get_param_names(self) -> Tuple[str, ...]:
        """Return parameter names in canonical order."""
        return ('pre_quaternions', 'theta_zz', 'post_quaternions')
    
    def get_template_type(self) -> str:
        """Return template type string for circuit building."""
        return f"{self.local_type}_zz_ring"
    
    def initialize_params(self) -> Dict[str, jnp.ndarray]:
        """Initialize all parameters with moderate random rotations."""
        quaternions_shape = (self.num_layers, self.num_qubits, 4)
        zz_shape = (self.num_layers, self.num_qubits, 1)
        
        key = jax.random.PRNGKey(self.seed)
        key_pre_axis, key_pre_angle, key_post_axis, key_post_angle = jax.random.split(key, 4)
        
        # Initialize pre-layer quaternions
        axes_pre = jax.random.normal(key_pre_axis, quaternions_shape[:-1] + (3,), dtype=jnp.float32)
        axes_pre = axes_pre / (jnp.linalg.norm(axes_pre, axis=-1, keepdims=True) + 1e-12)
        angles_pre = jax.random.uniform(key_pre_angle, quaternions_shape[:-1] + (1,), 
                                       dtype=jnp.float32, minval=0.2, maxval=0.8)
        w_pre = jnp.cos(0.5 * angles_pre)
        v_pre = axes_pre * jnp.sin(0.5 * angles_pre)
        pre_quaternions = jnp.concatenate([w_pre, v_pre], axis=-1).astype(jnp.float32)
        
        # Initialize post-layer quaternions
        axes_post = jax.random.normal(key_post_axis, quaternions_shape[:-1] + (3,), dtype=jnp.float32)
        axes_post = axes_post / (jnp.linalg.norm(axes_post, axis=-1, keepdims=True) + 1e-12)
        angles_post = jax.random.uniform(key_post_angle, quaternions_shape[:-1] + (1,), 
                                        dtype=jnp.float32, minval=0.2, maxval=0.8)
        w_post = jnp.cos(0.5 * angles_post)
        v_post = axes_post * jnp.sin(0.5 * angles_post)
        post_quaternions = jnp.concatenate([w_post, v_post], axis=-1).astype(jnp.float32)
        
        # Initialize ZZ angles at zero
        theta_zz = jnp.zeros(zz_shape, dtype=jnp.float32)
        
        return {
            'pre_quaternions': pre_quaternions,
            'theta_zz': theta_zz,
            'post_quaternions': post_quaternions
        }


class EntangledZXZQuaternionArchitecture(PQCArchitectureBase):
    """
    LEL-ZXZ architecture with quaternion parametrization.

    Structure: Pre-local (RzRxRz) → ZXZ entangling ring → Post-local (RzRxRz)
    
    Parameters:
    - pre_quaternions: (num_layers, num_qubits, 4) - Pre-layer local unitaries
    - entangled_quaternions: (num_layers, num_qubits, 4) - ZXZ entangling angles
    - post_quaternions: (num_layers, num_qubits, 4) - Post-layer local unitaries
    """
    
    def __init__(self, num_layers: int, num_qubits: int, seed: int = 0, 
                 pqc_type: str = 'zxz'):
        """
        Initialize LEL-ZZ quaternion architecture.
        
        Args:
            num_layers: Number of PQC layers
            num_qubits: Number of qubits
            seed: Random seed for initialization
            pqc_type: Local unitary decomposition ('zxz' or 'xzy')
        """
        super().__init__(num_layers, num_qubits, seed)
        self.pqc_type = pqc_type
        
        # Determine template string based on pqc_type
        if pqc_type == 'zxz':
            self.local_type = 'rzrxrz'
        elif pqc_type == 'xzy':
            self.local_type = 'rxrzry'
        else:
            raise ValueError(f"Unknown pqc_type: {pqc_type}")
    
    def get_param_names(self) -> Tuple[str, ...]:
        """Return parameter names in canonical order."""
        return ('pre_quaternions', 'entangled_quaternions', 'post_quaternions')
    
    def get_template_type(self) -> str:
        """Return template type string for circuit building."""
        return f"{self.local_type}_zxz_ring"
    
    def initialize_params(self) -> Dict[str, jnp.ndarray]:
        """Initialize all parameters with moderate random rotations."""
        quaternions_shape = (self.num_layers, self.num_qubits, 4)
        zz_shape = (self.num_layers, self.num_qubits, 1)
        
        key = jax.random.PRNGKey(self.seed)
        key_pre_axis, key_pre_angle, key_post_axis, key_post_angle = jax.random.split(key, 4)
        
        # Initialize pre-layer quaternions
        axes_pre = jax.random.normal(key_pre_axis, quaternions_shape[:-1] + (3,), dtype=jnp.float32)
        axes_pre = axes_pre / (jnp.linalg.norm(axes_pre, axis=-1, keepdims=True) + 1e-12)
        angles_pre = jax.random.uniform(key_pre_angle, quaternions_shape[:-1] + (1,), 
                                       dtype=jnp.float32, minval=0.2, maxval=0.8)
        w_pre = jnp.cos(0.5 * angles_pre)
        v_pre = axes_pre * jnp.sin(0.5 * angles_pre)
        pre_quaternions = jnp.concatenate([w_pre, v_pre], axis=-1).astype(jnp.float32)
        
        # Initialize post-layer quaternions
        axes_post = jax.random.normal(key_post_axis, quaternions_shape[:-1] + (3,), dtype=jnp.float32)
        axes_post = axes_post / (jnp.linalg.norm(axes_post, axis=-1, keepdims=True) + 1e-12)
        angles_post = jax.random.uniform(key_post_angle, quaternions_shape[:-1] + (1,), 
                                        dtype=jnp.float32, minval=0.2, maxval=0.8)
        w_post = jnp.cos(0.5 * angles_post)
        v_post = axes_post * jnp.sin(0.5 * angles_post)
        post_quaternions = jnp.concatenate([w_post, v_post], axis=-1).astype(jnp.float32)
        
        # Initialize ZZ angles at zero
        entangled_quaternions = jnp.zeros(zz_shape, dtype=jnp.float32)
        
        return {
            'pre_quaternions': pre_quaternions,
            'entangled_quaternions': entangled_quaternions,
            'post_quaternions': post_quaternions
        }



class LocalOnlyQuaternionArchitecture(PQCArchitectureBase):
    """
    Local-only architecture with quaternion parametrization (no entanglement).
    
    Structure: Pre-local (RzRxRz) only
    
    Parameters:
    - pre_quaternions: (num_layers, num_qubits, 4) - Local unitaries
    """
    
    def __init__(self, num_layers: int, num_qubits: int, seed: int = 0,
                 pqc_type: str = 'zxz'):
        """
        Initialize local-only quaternion architecture.
        
        Args:
            num_layers: Number of PQC layers
            num_qubits: Number of qubits
            seed: Random seed for initialization
            pqc_type: Local unitary decomposition ('zxz' or 'xzy')
        """
        super().__init__(num_layers, num_qubits, seed)
        self.pqc_type = pqc_type
        
        # Determine template string based on pqc_type
        if pqc_type == 'zxz':
            self.local_type = 'rzrxrz'
        elif pqc_type == 'xzy':
            self.local_type = 'rxrzry'
        else:
            raise ValueError(f"Unknown pqc_type: {pqc_type}")
    
    def get_param_names(self) -> Tuple[str, ...]:
        """Return parameter names in canonical order."""
        return ('pre_quaternions',)
    
    def get_template_type(self) -> str:
        """Return template type string for circuit building."""
        return self.local_type
    
    def initialize_params(self) -> Dict[str, jnp.ndarray]:
        """Initialize parameters with moderate random rotations."""
        quaternions_shape = (self.num_layers, self.num_qubits, 4)
        
        key = jax.random.PRNGKey(self.seed)
        key_axis, key_angle = jax.random.split(key, 2)
        
        # Initialize quaternions
        axes = jax.random.normal(key_axis, quaternions_shape[:-1] + (3,), dtype=jnp.float32)
        axes = axes / (jnp.linalg.norm(axes, axis=-1, keepdims=True) + 1e-12)
        angles = jax.random.uniform(key_angle, quaternions_shape[:-1] + (1,), 
                                    dtype=jnp.float32, minval=0.2, maxval=0.8)
        w = jnp.cos(0.5 * angles)
        v = axes * jnp.sin(0.5 * angles)
        pre_quaternions = jnp.concatenate([w, v], axis=-1).astype(jnp.float32)
        
        return {'pre_quaternions': pre_quaternions}


class LocalOnlyAngleArchitecture(PQCArchitectureBase):
    """
    Local-only architecture with direct angle parametrization (no quaternions).
    
    Structure: Pre-local (RzRxRz) with direct angle parametrization
    
    Parameters:
    - pre_angles: (num_layers, num_qubits, 3) - Direct Euler angles (α, β, γ)
    """
    
    def __init__(self, num_layers: int, num_qubits: int, seed: int = 0,
                 pqc_type: str = 'zxz'):
        """
        Initialize local-only angle architecture.
        
        Args:
            num_layers: Number of PQC layers
            num_qubits: Number of qubits
            seed: Random seed for initialization
            pqc_type: Local unitary decomposition ('zxz' or 'xzy')
        """
        super().__init__(num_layers, num_qubits, seed)
        self.pqc_type = pqc_type
        
        # Determine template string based on pqc_type
        if pqc_type == 'zxz':
            self.local_type = 'rzrxrz'
        elif pqc_type == 'xzy':
            self.local_type = 'rxrzry'
        else:
            raise ValueError(f"Unknown pqc_type: {pqc_type}")
    
    def get_param_names(self) -> Tuple[str, ...]:
        """Return parameter names in canonical order."""
        return ('pre_angles',)
    
    def get_template_type(self) -> str:
        """Return template type string for circuit building."""
        return self.local_type
    
    def initialize_params(self) -> Dict[str, jnp.ndarray]:
        """Initialize angles uniformly in [-π, π]."""
        angles_shape = (self.num_layers, self.num_qubits, 3)
        
        key = jax.random.PRNGKey(self.seed)
        pre_angles = jax.random.uniform(key, angles_shape, dtype=jnp.float32, 
                                        minval=-jnp.pi, maxval=jnp.pi)
        
        return {'pre_angles': pre_angles}


# Factory function for easy architecture creation
def create_pqc_architecture(arch_type: str, num_qubits: int, num_gates: int, 
                            gate_blocks: int, pqc_blocks: int = 1,
                            seed: int = 0, pqc_type: str = 'zxz') -> PQCArchitectureBase:
    """
    Factory function to create PQC architecture instances.
    
    Args:
        arch_type: Architecture type ('lelzz_quat', 'local_quat', 'local_angle')
        num_layers: Number of PQC layers
        num_qubits: Number of qubits
        seed: Random seed for initialization
        pqc_type: Local unitary decomposition ('zxz' or 'xzy')
    
    Returns:
        PQCArchitectureBase instance
    
    Example:
        >>> arch = create_pqc_architecture('lelzz_quat', num_layers=5, num_qubits=3)
        >>> params = arch.initialize_params()
        >>> template_type = arch.get_template_type()  # 'rzrxrz_zz_ring'
    """
    arch_map = {
        'lelzz_quat': LELZZQuaternionArchitecture,
        'entangled_zxz': EntangledZXZQuaternionArchitecture,
        'local_quat': LocalOnlyQuaternionArchitecture,
        'local_angle': LocalOnlyAngleArchitecture,
    }
    
    if arch_type not in arch_map:
        raise ValueError(f"Unknown architecture type: {arch_type}. "
                        f"Available: {list(arch_map.keys())}")

    num_layers = int(pqc_blocks * jnp.ceil(num_gates / gate_blocks))

    return arch_map[arch_type](num_layers, num_qubits, seed, pqc_type)
