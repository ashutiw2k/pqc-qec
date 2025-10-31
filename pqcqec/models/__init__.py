"""
PQC Models module.

New architecture (recommended):
- Use pqc_architectures for parameter management
- Use PQCModelBase for training/execution

Backward compatibility:
- LELZZInterleavedQuaternionCustomStatevecModel
- ZXZInterleavedQuaternionCustomStatevecModel
"""

from .pqc_architectures import (
    PQCArchitectureBase,
    LELZZQuaternionArchitecture,
    LocalOnlyQuaternionArchitecture,
    LocalOnlyAngleArchitecture,
    create_pqc_architecture
)
from .pqc_model_base import PQCModelBase


# Backward compatibility wrappers
class LELZZInterleavedQuaternionCustomStatevecModel(PQCModelBase):
    """
    Backward compatibility wrapper for LEL-ZZ quaternion model.
    
    This creates a PQCModelBase with LELZZQuaternionArchitecture.
    """
    
    def __init__(self, base_circuit_ops, num_qubits, x_noise, z_noise,
                 pqc_blocks=1, gate_blocks=1, seed=0, pqc_type='zxz'):
        # Calculate number of layers
        num_gates = len(base_circuit_ops)
        num_pqc_layers = int(pqc_blocks * ((num_gates + gate_blocks - 1) // gate_blocks))
        
        # Create architecture
        arch = LELZZQuaternionArchitecture(num_pqc_layers, num_qubits, seed, pqc_type)
        
        # Initialize base model
        super().__init__(base_circuit_ops, num_qubits, x_noise, z_noise,
                        arch, pqc_blocks, gate_blocks, pqc_type)
        
        # For backward compatibility, expose parameters as attributes
        self._sync_params_to_attributes()
    
    def _sync_params_to_attributes(self):
        """Sync params dict to individual attributes for backward compatibility."""
        self.pre_quaternions = self.params['pre_quaternions']
        self.theta_zz = self.params['theta_zz']
        self.post_quaternions = self.params['post_quaternions']
    
    def set_model_params(self, new_params):
        """Override to maintain attribute sync."""
        super().set_model_params(new_params)
        self._sync_params_to_attributes()
    
    def get_model_params_to_store(self):
        """Backward compatibility for old method name."""
        return self.get_model_params_dict()
    
    def get_pqc_params(self):
        """Get PQC parameters as angles (for inspection/logging)."""
        pre_angles = self.convert_quaternions_to_angles(self.pre_quaternions)
        post_angles = self.convert_quaternions_to_angles(self.post_quaternions)
        return {
            'pre_angles': pre_angles,
            'theta_zz': self.theta_zz,
            'post_angles': post_angles
        }


class ZXZInterleavedQuaternionCustomStatevecModel(PQCModelBase):
    """
    Backward compatibility wrapper for local-only quaternion model.
    
    This creates a PQCModelBase with LocalOnlyQuaternionArchitecture.
    """
    
    def __init__(self, base_circuit_ops, num_qubits, x_noise, z_noise,
                 pqc_blocks=1, gate_blocks=1, seed=0, pqc_type='zxz'):
        # Calculate number of layers
        num_gates = len(base_circuit_ops)
        num_pqc_layers = int(pqc_blocks * ((num_gates + gate_blocks - 1) // gate_blocks))
        
        # Create architecture
        arch = LocalOnlyQuaternionArchitecture(num_pqc_layers, num_qubits, seed, pqc_type)
        
        # Initialize base model
        super().__init__(base_circuit_ops, num_qubits, x_noise, z_noise,
                        arch, pqc_blocks, gate_blocks, pqc_type)
        
        # For backward compatibility, expose parameters as attributes
        self._sync_params_to_attributes()
    
    def _sync_params_to_attributes(self):
        """Sync params dict to individual attributes for backward compatibility."""
        self.pre_quaternions = self.params['pre_quaternions']
    
    def set_model_params(self, new_params):
        """Override to maintain attribute sync."""
        super().set_model_params(new_params)
        self._sync_params_to_attributes()
    
    def get_model_params_to_store(self):
        """Backward compatibility for old method name."""
        return self.get_model_params_dict()
    
    def get_pqc_params(self):
        """Get PQC parameters as angles (for inspection/logging)."""
        pre_angles = self.convert_quaternions_to_angles(self.pre_quaternions)
        return {'pre_angles': pre_angles}


__all__ = [
    # New architecture
    'PQCArchitectureBase',
    'LELZZQuaternionArchitecture',
    'LocalOnlyQuaternionArchitecture',
    'LocalOnlyAngleArchitecture',
    'create_pqc_architecture',
    'PQCModelBase',
    # Backward compatibility
    'LELZZInterleavedQuaternionCustomStatevecModel',
    'ZXZInterleavedQuaternionCustomStatevecModel',
]
