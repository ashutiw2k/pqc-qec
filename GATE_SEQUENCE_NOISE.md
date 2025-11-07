# Gate Sequence Noise Model

## Overview

This implements a **coherent noise model** that modifies gate sequences instead of adding rotation errors (RxRz gates). This represents systematic calibration errors where consecutive gate operations on the same qubit are misimplemented.

## Key Differences from Traditional Noise

### Traditional Noise Model (RxRz)
- Adds small rotation errors after each gate
- Represents decoherence and control errors
- Circuit grows: `N gates → N + 2×N noise gates`
- Implemented in: `add_noise_to_base_ops()`

### Gate Sequence Noise Model (NEW)
- Modifies consecutive gate pairs on same qubit
- Represents coherent calibration errors
- Circuit size unchanged: `N gates → N gates`
- Implemented in: `apply_gate_sequence_noise()`

## Default Transformation Rules

When two consecutive gates of the same type appear on the same qubit:

| Input Pair | Output Pair | Interpretation |
|------------|-------------|----------------|
| `HH` | `HX` | Second Hadamard misimplemented as Pauli-X |
| `XX` | `XZ` | Second X-gate misimplemented as Z-gate |
| `ZZ` | `ZH` | Second Z-gate misimplemented as Hadamard |

## Implementation Details

### Core Function
```python
from pqcqec.noise.builder import apply_gate_sequence_noise

noisy_ops = apply_gate_sequence_noise(
    base_ops,                    # List of (gate, qubits, params) tuples
    transformation_rules=None,   # Optional custom rules
    seed=None                    # For future probabilistic extensions
)
```

### Algorithm
1. **Track last gate per qubit**: Maintain `{qubit_id: (gate_index, gate_name)}`
2. **Scan for matching pairs**: When gate `G2` appears on qubit `q`:
   - Check if previous gate on `q` was `G1`
   - Look up `(G1, G2)` in transformation rules
3. **Apply transformation**: If match found, replace `G2` with the transformed gate
4. **Update tracking**: Record current gate as "last gate" on that qubit

### Key Features
- ✅ **Qubit-independent**: Gates on different qubits don't interfere
- ✅ **Multi-qubit aware**: CNOT/CZ break the sequence chain on both qubits
- ✅ **Case-insensitive**: 'H', 'h', 'Hadamard' all match
- ✅ **Custom rules**: Pass your own `transformation_rules` dict
- ✅ **Deterministic**: Same circuit always produces same noisy circuit

## Advanced: Probabilistic Noise

For stochastic errors, use the probabilistic variant:

```python
from pqcqec.noise.builder import apply_gate_sequence_noise_probabilistic

noisy_ops = apply_gate_sequence_noise_probabilistic(
    base_ops,
    transformation_rules=None,
    error_probability=0.1,  # 10% chance to transform each matching pair
    seed=42                 # For reproducibility
)
```

## Usage Examples

### Example 1: Basic Usage
```python
circuit_ops = [
    ('h', [0], []),
    ('h', [0], []),  # This becomes 'x'
    ('x', [1], []),
    ('x', [1], []),  # This becomes 'z'
]

noisy_circuit = apply_gate_sequence_noise(circuit_ops)
# Result: [('h', [0], []), ('x', [0], []), ('x', [1], []), ('z', [1], [])]
```

### Example 2: Custom Rules
```python
custom_rules = {
    ('rx', 'rx'): ('rx', 'ry'),  # RxRx → RxRy
    ('s', 's'): ('s', 't'),      # SS → ST
}

noisy_circuit = apply_gate_sequence_noise(circuit_ops, transformation_rules=custom_rules)
```

### Example 3: Probabilistic Errors
```python
# Only 30% of matching pairs will be transformed
noisy_circuit = apply_gate_sequence_noise_probabilistic(
    circuit_ops,
    error_probability=0.3,
    seed=123
)
```

## Integration with Existing Pipeline

### Option 1: Replace Traditional Noise
In `pqcqec/circuits/templates.py`, instead of:
```python
if add_noise:
    template.add_gate('rx', [q], ['x_noise'])
    template.add_gate('rz', [q], ['z_noise'])
```

Use:
```python
# Apply sequence noise before building template
base_ops = apply_gate_sequence_noise(base_ops)
template = build_pqc_circuit_template(base_ops, add_noise=False)
```

### Option 2: Combine Both Models
Apply both coherent (sequence) and incoherent (rotation) noise:
```python
# First apply coherent errors (gate substitutions)
noisy_ops = apply_gate_sequence_noise(base_ops)

# Then apply incoherent errors (rotations)
template = build_pqc_circuit_template(noisy_ops, add_noise=True)
```

## Testing

Run the comprehensive test suite:
```bash
.venv/bin/python test_gate_sequence_noise.py
```

Tests cover:
- ✅ Basic transformations (HH→HX, XX→XZ, ZZ→ZH)
- ✅ Qubit independence (no cross-talk)
- ✅ Multi-qubit gate handling (chain breaking)
- ✅ Custom transformation rules
- ✅ Probabilistic transformations
- ✅ Complex realistic circuits

## Physical Interpretation

This noise model represents **coherent systematic errors** such as:

1. **Calibration drift**: Gate pulse slightly off, causing wrong unitary
2. **Crosstalk**: Adjacent operations interfere constructively
3. **Pulse leakage**: Intended gate partially implements different gate
4. **Control system bugs**: Software mapping gates incorrectly

Unlike decoherence (modeled by RxRz), these errors:
- Are **deterministic** (same every time)
- Can **accumulate coherently** (interfere constructively/destructively)
- May be **correctable** by PQC learning the systematic pattern
- Represent **addressable hardware issues** (can be fixed by recalibration)

## Future Extensions

Potential enhancements:
- [ ] 3-gate patterns: `HXH → HXZ`
- [ ] Qubit-specific rules: Different errors on different qubits
- [ ] Time-dependent: Rules that change during circuit execution
- [ ] Context-aware: Transform based on previous N gates, not just last one
- [ ] Parametric noise: Modify rotation angles instead of gate type

## References

- Traditional RxRz noise: `pqcqec/noise/simple_noise.py`
- Template system: `pqcqec/circuits/templates.py`
- Experiment runner: `pqcqec/experiment/pqc_experiment.py`
