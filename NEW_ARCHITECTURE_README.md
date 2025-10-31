# New PQC Model Architecture

## Overview

The PQC model code has been refactored into a cleaner, more maintainable structure:

1. **`pqc_architectures.py`**: Handles parameter structure, initialization, and naming
2. **`pqc_model_base.py`**: Handles training logic, circuit execution, and simulation
3. **`__init__.py`**: Provides backward-compatible wrappers for existing code

## New Usage (Recommended)

```python
from pqcqec.models import create_pqc_architecture, PQCModelBase

# 1. Create a PQC architecture
arch = create_pqc_architecture(
    arch_type='lelzz_quat',  # or 'local_quat', 'local_angle'
    num_layers=5,
    num_qubits=3,
    seed=42,
    pqc_type='zxz'  # or 'xzy'
)

# 2. Create the model
model = PQCModelBase(
    base_circuit_ops=circuit_ops,
    num_qubits=3,
    x_noise=x_noise,
    z_noise=z_noise,
    pqc_architecture=arch,
    pqc_blocks=1,
    gate_blocks=10,
    pqc_type='zxz'
)

# 3. Use the model
output_states = model.run_model_batch(input_states)

# 4. Access/modify parameters
params_dict = model.get_model_params_dict()
params_tuple = model.get_model_params()
model.set_model_params(new_params)
```

## Available Architectures

### 1. LEL-ZZ with Quaternions (`'lelzz_quat'`)
- **Structure**: Pre-local → ZZ entangling ring → Post-local
- **Parameters**: `pre_quaternions`, `theta_zz`, `post_quaternions`
- **Use case**: Most expressive, best for complex error correction

### 2. Local-Only with Quaternions (`'local_quat'`)
- **Structure**: Pre-local only (no entanglement)
- **Parameters**: `pre_quaternions`
- **Use case**: Faster training, less expressive

### 3. Local-Only with Direct Angles (`'local_angle'`)
- **Structure**: Pre-local with direct angle parametrization
- **Parameters**: `pre_angles`
- **Use case**: Fastest (no quaternion conversion), but may have gimbal lock issues

## Backward Compatibility

Existing code continues to work without changes:

```python
from pqcqec.models import LELZZInterleavedQuaternionCustomStatevecModel

# Old API still works
model = LELZZInterleavedQuaternionCustomStatevecModel(
    base_circuit_ops=circuit_ops,
    num_qubits=3,
    x_noise=x_noise,
    z_noise=z_noise,
    pqc_blocks=1,
    gate_blocks=10,
    seed=42
)
```

## Benefits of New Structure

1. **Separation of Concerns**: Parameter management is separate from training logic
2. **Easy to Add New Architectures**: Just create a new class in `pqc_architectures.py`
3. **No Code Duplication**: Shared training logic in `PQCModelBase`
4. **Type Safety**: Clear interfaces via base classes
5. **Testability**: Each component can be tested independently

## Adding a New Architecture

To add a new PQC architecture (e.g., "Hardware-Efficient" with specific connectivity):

1. Create a class in `pqc_architectures.py`:

```python
class HardwareEfficientArchitecture(PQCArchitectureBase):
    def get_param_names(self) -> Tuple[str, ...]:
        return ('rotation_angles', 'cx_angles')
    
    def get_template_type(self -> str:
        return 'hardware_efficient'
    
    def initialize_params(self) -> Dict[str, jnp.ndarray]:
        # Your initialization logic
        ...
```

2. Add to factory in `create_pqc_architecture()`:

```python
arch_map = {
    ...
    'hw_efficient': HardwareEfficientArchitecture,
}
```

3. Use it:

```python
arch = create_pqc_architecture('hw_efficient', num_layers=5, num_qubits=3)
model = PQCModelBase(..., pqc_architecture=arch, ...)
```

That's it! No need to duplicate any training logic.

## Migration Guide

If you have existing code using the old classes:

### Option 1: No Changes Needed
- The backward compatibility wrappers work identically
- Your existing code will continue to function

### Option 2: Migrate to New API
1. Replace class imports:
   ```python
   # Old
   from pqcqec.models.pqc_models import LELZZInterleavedQuaternionCustomStatevecModel
   
   # New
   from pqcqec.models import create_pqc_architecture, PQCModelBase
   ```

2. Update instantiation:
   ```python
   # Old
   model = LELZZInterleavedQuaternionCustomStatevecModel(...)
   
   # New
   arch = create_pqc_architecture('lelzz_quat', num_layers=..., ...)
   model = PQCModelBase(..., pqc_architecture=arch, ...)
   ```

3. Update parameter access:
   ```python
   # Old
   params = model.get_model_params()  # Returns tuple
   
   # New (both work)
   params_tuple = model.get_model_params()  # Returns tuple
   params_dict = model.get_model_params_dict()  # Returns dict (recommended)
   ```

## Examples

See `testnotebooks/` for usage examples with the new architecture.
