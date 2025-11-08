# PQCModelBase Integration with Gate Sequence Noise

## Summary

`PQCModelBase` now supports three noise types:

1. **`'rotation'`** (default) - Traditional RxRz noise gates
2. **`'gate_sequence'`** - Coherent gate sequence transformations (HH→HX, etc.)
3. **`'both'`** - Apply gate sequence transformations THEN add rotation noise

## New Parameters

```python
class PQCModelBase:
    def __init__(self,
                 ...,
                 noise_type: str = 'rotation',
                 gate_sequence_noise_rules: Optional[Dict] = None,
                 gate_sequence_noise_prob: float = 1.0,
                 noise_seed: Optional[int] = None)
```

### Parameters

- **`noise_type`**: `'rotation'`, `'gate_sequence'`, or `'both'`
- **`gate_sequence_noise_rules`**: Custom transformation dict `{(g1, g2): (g1, g2_new)}`
  - Default: `{('h','h'): ('h','x'), ('x','x'): ('x','z'), ('z','z'): ('z','h')}`
- **`gate_sequence_noise_prob`**: Probability in [0, 1] for applying transformations
  - `1.0` = deterministic (all matching pairs transformed)
  - `<1.0` = probabilistic (random subset transformed)
- **`noise_seed`**: Random seed for probabilistic noise

## Usage Examples

### Example 1: Traditional Rotation Noise (Default)

```python
model = PQCModelBase(
    base_circuit_ops=circuit_ops,
    num_qubits=3,
    x_noise=x_noise_array,
    z_noise=z_noise_array,
    pqc_architecture=pqc_arch,
    gate_blocks=5,
    noise_type='rotation'  # Adds RxRz gates
)
```

### Example 2: Pure Gate Sequence Noise

```python
model = PQCModelBase(
    base_circuit_ops=circuit_ops,
    num_qubits=3,
    x_noise=np.zeros(20),  # Not used
    z_noise=np.zeros(20),  # Not used
    pqc_architecture=pqc_arch,
    gate_blocks=10,
    noise_type='gate_sequence',  # HH→HX, XX→XZ, ZZ→ZH
    noise_seed=42
)
```

### Example 3: Both Noise Types

```python
model = PQCModelBase(
    base_circuit_ops=circuit_ops,
    num_qubits=3,
    x_noise=x_noise_array,
    z_noise=z_noise_array,
    pqc_architecture=pqc_arch,
    gate_blocks=5,
    noise_type='both',  # Gate sequence + rotation
    gate_sequence_noise_prob=1.0,
    noise_seed=123
)
```

### Example 4: Custom Transformation Rules

```python
custom_rules = {
    ('h', 'h'): ('h', 's'),  # HH → HS
    ('x', 'x'): ('x', 'y'),  # XX → XY
    ('h', 'x'): ('h', 'z'),  # HX → HZ
}

model = PQCModelBase(
    base_circuit_ops=circuit_ops,
    num_qubits=3,
    x_noise=np.zeros(15),
    z_noise=np.zeros(15),
    pqc_architecture=pqc_arch,
    gate_blocks=5,
    noise_type='gate_sequence',
    gate_sequence_noise_rules=custom_rules
)
```

### Example 5: Probabilistic Transformations

```python
model = PQCModelBase(
    base_circuit_ops=circuit_ops,
    num_qubits=3,
    x_noise=np.zeros(20),
    z_noise=np.zeros(20),
    pqc_architecture=pqc_arch,
    gate_blocks=10,
    noise_type='gate_sequence',
    gate_sequence_noise_prob=0.3,  # 30% of matching pairs
    noise_seed=999
)
```

## How It Works

### Initialization Flow

1. **Gate Sequence Noise Application** (if `noise_type` is `'gate_sequence'` or `'both'`)
   - Modifies `self.base_circuit_ops` in-place
   - Scans for consecutive gate pairs on same qubit
   - Applies transformations before template building

2. **Template Building**
   - Uses potentially modified `base_circuit_ops`
   - `add_noise` parameter determined by `noise_type`:
     - `'rotation'`: `add_noise=True` (adds RxRz)
     - `'gate_sequence'`: `add_noise=False` (no RxRz)
     - `'both'`: `add_noise=True` (adds RxRz to modified gates)

3. **Partial Template Caching**
   - Individual block templates use same `noise_type` logic
   - Progressive training templates respect noise configuration

## Integration with Experiment Runners

### Updating `pqc_experiment_custom_statevec_runner`

Add parameters to the function signature:

```python
def pqc_experiment_custom_statevec_runner(
    ...,
    noise_type: str = 'rotation',
    gate_sequence_noise_rules: Optional[Dict] = None,
    gate_sequence_noise_prob: float = 1.0,
    noise_seed: Optional[int] = None
):
    ...
    
    model = PQCModelBase(
        base_circuit_ops=uncomp_circuit_ops,
        num_qubits=num_qubits,
        x_noise=x_noise_arr,
        z_noise=z_noise_arr,
        pqc_architecture=pqc_arch,
        pqc_blocks=pqc_blocks,
        gate_blocks=gate_blocks,
        pqc_type='zxz',
        noise_type=noise_type,  # NEW
        gate_sequence_noise_rules=gate_sequence_noise_rules,  # NEW
        gate_sequence_noise_prob=gate_sequence_noise_prob,  # NEW
        noise_seed=noise_seed  # NEW
    )
```

### Command Line Usage

Add arguments to training scripts:

```python
parser.add_argument('--noise-type', type=str, default='rotation',
                    choices=['rotation', 'gate_sequence', 'both'],
                    help='Type of noise model')
parser.add_argument('--gate-noise-prob', type=float, default=1.0,
                    help='Probability for gate sequence noise')
```

## Benefits

### Gate Sequence Noise Advantages

1. **No circuit growth**: Circuit size unchanged (vs 3x with rotation noise)
2. **Coherent errors**: Models systematic calibration errors
3. **Learnable patterns**: PQC can potentially discover and compensate
4. **Physical realism**: Represents actual hardware miscalibration

### When to Use Each Type

- **Rotation noise**: Decoherence, random control errors
- **Gate sequence noise**: Calibration drift, systematic errors
- **Both**: Comprehensive error model (coherent + incoherent)

## Testing

Run the integration example:

```bash
.venv/bin/python example_gate_sequence_noise_integration.py
```

Expected output shows:
- ✅ Traditional rotation noise working
- ✅ Gate sequence transformations applied correctly
- ✅ Both noise types can be combined
- ✅ Custom rules work
- ✅ Probabilistic mode functional

## Files Modified

1. **`pqcqec/models/pqc_model_base.py`**
   - Added `noise_type`, `gate_sequence_noise_rules`, `gate_sequence_noise_prob`, `noise_seed` parameters
   - Apply gate sequence transformations in `__init__`
   - Conditionally add rotation noise based on `noise_type`
   - Updated partial template builders

2. **`pqcqec/noise/builder.py`**
   - Contains `apply_gate_sequence_noise()` and `apply_gate_sequence_noise_probabilistic()`
   - Called from `PQCModelBase.__init__()`

## Next Steps

To use in experiments:

1. Update experiment runner functions to accept new parameters
2. Add command-line arguments to training scripts
3. Run experiments comparing:
   - Rotation noise only
   - Gate sequence noise only
   - Both combined
4. Analyze which noise type PQC learns best
