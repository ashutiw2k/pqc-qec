# Critical Bug Fix: Gate ID Mismatch in JAX Statevector Simulator

## 🐛 Bug Description

The JAX statevector simulator was applying **incorrect gates** due to an off-by-one error in gate ID indexing.

## Root Cause

The `GateEnums` class in `pqcqec/utils/constants.py` uses `enum.auto()` which starts counting from **1**:

```python
class GateEnums(enum.IntEnum):
    GATE_X  = enum.auto()  # = 1
    GATE_Z  = enum.auto()  # = 2
    GATE_H  = enum.auto()  # = 3
    GATE_RX = enum.auto()  # = 4
    GATE_RY = enum.auto()  # = 5
    GATE_RZ = enum.auto()  # = 6
    GATE_CX = enum.auto()  # = 7
    GATE_CZ = enum.auto()  # = 8
```

However, `jax.lax.switch` in `apply_gate()` expects **0-based indices**:

```python
return jax.lax.switch(
    gate_id,  # Was using 1-8, but switch expects 0-7!
    [
        lambda s: apply_x(...),   # Index 0
        lambda s: apply_z(...),   # Index 1
        lambda s: apply_h(...),   # Index 2
        ...
    ],
    state
)
```

## Impact

This caused every gate to be off by one position:
- **H gate** (ID=3) → Applied **RX** (index 3) ❌ instead of H (index 2) ✓
- **CNOT** (ID=7) → Applied **CZ** (index 7) ❌ instead of CNOT (index 6) ✓  
- **RZ** (ID=6) → Applied **CNOT** (index 6) ❌ instead of RZ (index 5) ✓

### Example Failure

Circuit: `H(0) → CNOT(0,1) → RZ(1, π/4) → CNOT(0,1)`

**Before Fix:**
- Applied: `RX(0,0) → CZ(0,1) → CNOT(1) → CZ(0,1)`
- Output: `[1+0j, 0+0j, 0+0j, 0+0j]` (incorrect - essentially no change!)

**After Fix:**
- Applied: `H(0) → CNOT(0,1) → RZ(1, π/4) → CNOT(0,1)`  
- Output: `[0.653-0.271j, 0, 0.653+0.271j, 0]` (correct!)

## The Fix

Modified `apply_gate()` in `pqcqec/simulate/jax_statevector.py`:

```python
# OLD (BROKEN):
return jax.lax.switch(
    gate_id,  # Wrong! 1-based enum
    [...],
    state
)

# NEW (FIXED):
return jax.lax.switch(
    gate_id - 1,  # Correct! Convert to 0-based index
    [...],
    state
)
```

## Why This Explains the PQC Training Failure

This bug directly explains why your PQC training wasn't working:

1. **ZZ entangling layer wasn't working** because:
   - ZZ = CNOT-RZ-CNOT sequence
   - But was actually applying CZ-CNOT-CZ sequence
   - This completely changed the circuit topology!

2. **Theta_zz gradients were zero** because:
   - The RZ gates (containing theta_zz) were being replaced by CNOTs
   - CNOTs have no parameters, so no gradients!

3. **Pre/post rotation gradients were tiny** because:
   - RZ gates (ID=6) were applying CNOT instead
   - RX gates (ID=4) were applying RY instead  
   - The entire circuit was scrambled

## Verification

Run the test notebook `test_jax_statevector_simulator.ipynb`:
- Test 8 should now **PASS** ✓
- Test 12 (PQC-like circuit) should show **non-zero theta_zz gradients** ✓
- All tests should pass ✓

## Next Steps

1. ✅ **Re-run Test 8** in the notebook to confirm the fix
2. ✅ **Re-run Test 12** to verify theta_zz gradients now flow
3. ✅ **Re-run your PQC training** - it should work correctly now!
4. Consider adding unit tests for gate ID mappings to prevent future regressions

## Files Changed

- `pqcqec/simulate/jax_statevector.py` - Fixed `apply_gate()` function (line ~205)

## Date

October 12, 2025
