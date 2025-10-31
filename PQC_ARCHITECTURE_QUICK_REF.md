# Quick Reference: PQC Architecture Types

## Format
```
"{local}_{entangling}"
```

## Local Options
- `rzrxrz` - Rz-Rx-Rz (default)
- `rxrzry` - Rx-Rz-Ry
- `none` - No local gates

## Entangling Options

### ZZ-based (CNOT-Rz-CNOT)
- `zz_ring` - Ring topology ✓ Default
- `zz_linear` - Linear chain (no wraparound)
- `zz_all_to_all` - Full connectivity
- `zz_star` - Hub-and-spoke

### Pauli Variants
- `xx_ring` - XX entangling (with basis change)
- `yy_ring` - YY entangling (with basis change)

### Special
- `none` - No entanglement

## Common Combinations

```python
"rzrxrz_zz_ring"      # Default LEL-ZZ
"rzrxrz_zz_linear"    # LEL-ZZ linear
"rzrxrz_zz_all_to_all"  # LEL-ZZ full
"rzrxrz_xx_ring"      # LEL-XX
"rxrzry_zz_ring"      # XZY-ZZ
"rzrxrz"              # Local only
"none_zz_ring"        # Entangling only
"none"                # No PQC
```

## Usage
```python
from pqcqec.circuits.templates import build_pqc_circuit_template

template = build_pqc_circuit_template(
    base_ops=ops,
    num_qubits=4,
    num_gate_blocks=10,
    add_noise=True,
    pqc_type="rzrxrz_xx_ring"  # ← Choose architecture here
)
```
