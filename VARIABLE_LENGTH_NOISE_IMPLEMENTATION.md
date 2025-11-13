# Variable-Length Gate Sequence Noise Implementation

## Summary

Successfully implemented variable-length transformation support for `apply_gate_sequence_noise()` and `apply_gate_sequence_noise_probabilistic()` functions.

## What Changed

### New Capabilities

**Before:** Only 2→2 transformations
```python
('h', 'h'): ('h', 'x')  # HH → HX
```

**After:** Variable-length transformations (2→N)
```python
('h', 'h'): [('h', []), ('z', []), ('x', [])]  # HH → HZX (2→3)
('z', 'z'): [('i', [])]                        # ZZ → I (2→1)
```

### Three Rule Formats (All Supported)

1. **Simple 2-tuple** (backward compatible):
   ```python
   ('h', 'h'): ('h', 'x')  # Inherits params from originals
   ```

2. **Extended 2-tuple** (with custom params):
   ```python
   ('h', 'h'): (('rx', [0.5]), ('rz', [0.3]))
   ```

3. **List format** (variable length - NEW):
   ```python
   ('h', 'h'): [('h', []), ('z', []), ('x', [0.1])]
   ```

## Parameter Inheritance Rules

For list-based replacements with `None` parameters:

- **First element** (index 0): inherits params from gate1
- **Last element** (index -1): inherits params from gate2  
- **Middle elements**: use explicit params or `[]` if `None`

Example:
```python
base_ops = [('rz', [0], [0.5]), ('rz', [0], [0.3])]
rules = {('rz', 'rz'): [('rz', None), ('rx', [0.1]), ('rz', None)]}
# Result: [('rz', [0], [0.5]), ('rx', [0], [0.1]), ('rz', [0], [0.3])]
```

## Implementation Details

### Algorithm Change
- **Old:** In-place modification with index tracking
- **New:** Streaming builder that constructs output incrementally
- **Benefit:** Avoids index confusion when length changes

### Key Features Preserved
- ✅ Non-overlapping transformations (once transformed, gates can't be re-transformed)
- ✅ Lazy-copy optimization (returns original list if no transformations occur)
- ✅ Multi-qubit gate support
- ✅ Case-insensitive matching
- ✅ Backward compatibility with all existing code

### Performance
- O(n) time complexity (single pass)
- O(n) space for output
- Lazy-copy: O(1) when no transformations occur

## Testing

Added 7 new tests covering:
- ✅ Variable-length 2→3 transformations
- ✅ Variable-length 2→1 reductions
- ✅ Explicit parameters in sequences
- ✅ Parameter inheritance logic
- ✅ Non-overlapping with variable lengths
- ✅ Multiple consecutive transformations (HHHH → 2 triplets)
- ✅ Mixed rule types (old + new syntax)

All 22 tests pass (15 original + 7 new).

## Use Cases

### 1. Adding Small Rotation Errors
```python
# Simulate over-rotations after coherent errors
rules = {
    ('h', 'h'): [('h', []), ('x', []), ('rx', [0.05])]
}
```

### 2. Gate Fusion/Cancellation
```python
# Two Z gates cancel to identity
rules = {
    ('z', 'z'): [('i', [])]  # or remove entirely with []
}
```

### 3. Complex Error Channels
```python
# Model realistic error accumulation
rules = {
    ('h', 'h'): [('h', []), ('rz', [0.02]), ('x', []), ('rx', [0.01])]
}
```

### 4. Richer PQC Training Data
Since you insert PQCs only at the end, variable circuit length is not an issue. You can now train on more realistic noise models.

## Example Usage in Your Notebook

```python
# Your current usage (still works):
noisy_ops = apply_gate_sequence_noise(
    base_ops,
    noise={
        ('h', 'h'): ('h', 'x'),
        ('x', 'x'): ('x', 'z'),
        ('z', 'z'): ('z', 'h'),
    }
)

# New: Add small rotations
noisy_ops = apply_gate_sequence_noise(
    base_ops,
    noise={
        ('h', 'h'): [('h', []), ('x', []), ('rx', [0.01])],
        ('x', 'x'): [('x', []), ('z', []), ('rz', [0.01])],
        ('z', 'z'): [('z', []), ('h', [])],
    }
)

# Then add PQC at the end (as before)
noisy_ops.append(('rz', [0], [pqc_params[0]]))
noisy_ops.append(('rx', [0], [pqc_params[1]]))
noisy_ops.append(('rz', [0], [pqc_params[2]]))
```

## Files Modified

1. `pqcqec/noise/builder.py`:
   - `apply_gate_sequence_noise()` - supports list-based rules
   - `apply_gate_sequence_noise_probabilistic()` - same update

2. `tests/test_gate_sequence_noise_builder.py`:
   - Added 7 new tests for variable-length behavior

3. `examples/variable_length_noise_demo.py` (NEW):
   - Demonstration script showing 8 examples

## Next Steps (Optional)

1. **Experiment with noise strength**: Try different small rotation values (0.01, 0.05, 0.1) to see how PQC correction performs
2. **Compare training**: Train PQCs on simple HH→HX vs enriched HH→HZX+rotations noise models
3. **Data generation**: Update your circuit generation scripts to use richer noise models if desired
4. **Analysis**: Compare fidelity improvements between simple and complex noise models

## Compatibility Notes

- ✅ All existing code continues to work unchanged
- ✅ Default rules unchanged (HH→HX, XX→XZ, ZZ→ZH)
- ✅ All existing tests pass
- ✅ JSON data format compatible (just stores longer token sequences)
- ✅ Fidelity calculations work (state-based, length-independent)
