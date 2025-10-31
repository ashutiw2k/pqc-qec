# PQC Entangling Architectures - Summary

## Overview

The `build_pqc_circuit_template` function now supports **multiple entangling architectures** beyond the original LEL-ZZ (Local-Entangle-Local with ZZ gates) design.

## Architecture Format

PQC architectures are specified using the format: `"{local}_{entangling}"`

### Local Unitary Options

- **`rzrxrz`**: Rz-Rx-Rz decomposition (ZXZ Euler angles) - Default
- **`rxrzry`**: Rx-Rz-Ry decomposition (XZY Euler angles)
- **`none`**: No local unitaries (entanglement only)

### Entangling Layer Options

#### ZZ-based Topologies
All use exp(-iθ·Z⊗Z) implemented as: CNOT → Rz(θ) → CNOT

- **`zz_ring`**: Ring topology (nearest-neighbor with wraparound)
  - Connects: 0↔1, 1↔2, ..., (n-1)↔0
  - Default architecture
  
- **`zz_linear`**: Linear chain (nearest-neighbor, no wraparound)
  - Connects: 0↔1, 1↔2, ..., (n-2)↔(n-1)
  - Fewer gates than ring
  
- **`zz_all_to_all`**: All-to-all connectivity
  - Connects every pair of qubits: 0↔1, 0↔2, ..., (n-2)↔(n-1)
  - Most expressive, most gates: n(n-1)/2 pairs
  
- **`zz_star`**: Star topology (center qubit to all others)
  - Center qubit 0 connects to: 1, 2, ..., n-1
  - Useful for hub-based architectures

#### Pauli Basis Variants

- **`xx_ring`**: XX-based entangling in ring topology
  - Implements exp(-iθ·X⊗X) via basis change: Ry(π/2) → ZZ → Ry(-π/2)
  - Uses same θ parameters as ZZ gates
  
- **`yy_ring`**: YY-based entangling in ring topology
  - Implements exp(-iθ·Y⊗Y) via basis change: Rx(π/2) → ZZ → Rx(-π/2)
  - Uses same θ parameters as ZZ gates

#### Multi-angle Entangling

- **`zxz_ring`**: Three-parameter entangling in ring topology
  - Full decomposition: CNOT → Rz → CNOT → Rx → CNOT → Rz → CNOT
  - More expressive than single-angle ZZ
  - **Note**: Currently not fully supported (needs dedicated parameter arrays)

#### No Entanglement

- **`none`**: No entangling layer (local unitaries only)

## Example Architectures

### 1. LEL-ZZ Ring (Default)
```python
pqc_type = "rzrxrz_zz_ring"
```
- Structure: Pre-Local (RzRxRz) → ZZ Ring → Post-Local (RzRxRz)
- Best for: General quantum error correction
- Gates per layer (4 qubits): 3×4 + 3×4 + 3×4 = 36 gates

### 2. LEL-ZZ Linear
```python
pqc_type = "rzrxrz_zz_linear"
```
- Structure: Pre-Local → ZZ Linear → Post-Local
- Best for: Linear qubit arrangements (no wraparound)
- Gates per layer (4 qubits): 3×4 + 3×3 + 3×4 = 33 gates

### 3. LEL-ZZ All-to-All
```python
pqc_type = "rzrxrz_zz_all_to_all"
```
- Structure: Pre-Local → ZZ All-to-All → Post-Local
- Best for: Maximum expressiveness
- Gates per layer (4 qubits): 3×4 + 3×6 + 3×4 = 42 gates

### 4. LEL-XX Ring
```python
pqc_type = "rzrxrz_xx_ring"
```
- Structure: Pre-Local → XX Ring → Post-Local
- Best for: Testing different Pauli bases
- Gates per layer (4 qubits): 3×4 + 8×4 + 3×4 = 56 gates (includes basis change)

### 5. Local Only
```python
pqc_type = "rzrxrz"
```
- Structure: Pre-Local only (no entanglement)
- Best for: Faster training, single-qubit error correction
- Gates per layer (4 qubits): 3×4 = 12 gates

### 6. Entangling Only
```python
pqc_type = "none_zz_ring"
```
- Structure: ZZ Ring only (no local pre/post)
- Best for: Pure entangling correction
- Gates per layer (4 qubits): 3×4 = 12 gates

## Usage

### With PQCModelBase

```python
from pqcqec.models import create_pqc_architecture, PQCModelBase

# Create architecture
arch = create_pqc_architecture(
    arch_type='lelzz_quat',
    num_qubits=4,
    num_gates=20,
    gate_blocks=10,
    pqc_blocks=1,
    seed=42
)

# Create model with specific entangling type
model = PQCModelBase(
    base_circuit_ops=circuit_ops,
    num_qubits=4,
    x_noise=x_noise,
    z_noise=z_noise,
    pqc_architecture=arch,
    pqc_blocks=1,
    gate_blocks=10,
    pqc_type='zxz'  # Local unitary decomposition type
)
```

### Building Templates Directly

```python
from pqcqec.circuits.templates import build_pqc_circuit_template

# Build template with specific architecture
template = build_pqc_circuit_template(
    base_ops=circuit_ops,
    num_qubits=4,
    num_gate_blocks=10,
    add_noise=True,
    pqc_type="rzrxrz_xx_ring"  # LEL-XX architecture
)

# Instantiate with parameters
circuit_ops = template.instantiate(param_dict)
```

## Gate Count Comparison

For 4 qubits, 20 base gates, 10 gates per block (2 PQC layers):

| Architecture | Gates/Layer | Total Gates | Description |
|-------------|-------------|-------------|-------------|
| `rzrxrz_zz_ring` | 36 | 132 | Default LEL-ZZ |
| `rzrxrz_zz_linear` | 33 | 126 | LEL-ZZ linear chain |
| `rzrxrz_zz_all_to_all` | 42 | 144 | LEL-ZZ full connectivity |
| `rzrxrz_zz_star` | 33 | 126 | LEL-ZZ star topology |
| `rzrxrz_xx_ring` | 56 | 164 | LEL-XX with basis change |
| `rzrxrz_yy_ring` | 56 | 164 | LEL-YY with basis change |
| `rzrxrz` (local only) | 12 | 84 | No entanglement |
| `none_zz_ring` | 12 | 84 | Entanglement only |

*Total includes: 20 base gates + 40 noise gates (2 per base) + PQC layers*

## Implementation Details

### Fixed Angle Gates
XX and YY entangling layers use fixed π/2 rotations for basis changes. These are handled via a special `'fixed_angle'` parameter source that stores the angle value directly in `param_idx`.

### Parameter Reuse
XX and YY rings reuse the `theta_zz` parameter source, meaning the same entangling angles are used but applied in different Pauli bases.

### Template Instantiation
The `CircuitTemplate.instantiate()` method now handles three parameter types:
1. Regular parameters: Indexed from param_dict arrays
2. Fixed angles: Stored directly in param_idx
3. No parameters: Gates like CNOT

## Testing

Run the test suite to verify all architectures:
```bash
PYTHONPATH="${PWD}:${PYTHONPATH}" .venv/bin/python tests/test_entangling_architectures.py
```

All 10 architectures should pass instantiation tests.

## Future Extensions

Potential additions:
- **Linear XX/YY topologies**: `xx_linear`, `yy_linear`
- **All-to-all XX/YY**: `xx_all_to_all`, `yy_all_to_all`
- **Dedicated ZXZ parameters**: Full 3-angle entangling support
- **Custom topologies**: User-defined connectivity graphs
- **Hardware-specific**: IBM/Google/IonQ native gate sets
