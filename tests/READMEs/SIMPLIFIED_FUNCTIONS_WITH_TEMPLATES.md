# Understanding the Simplified Functions vs Template System

## The Confusion

You asked: **"So this simple function will not work with the pqc template file?"**

The answer is: **The simplified function WORKS, but you need to use it correctly!**

## Two Different Use Cases

### Use Case 1: **Direct Circuit Building** (One-Time Use)
```python
# Build circuit directly with specific PQC parameters
pqc_params = np.random.randn(2, 2, 3)  # [num_blocks, num_qubits, num_gates]

# Using simplified version
gate_ids, w1, w2, theta = build_circuit_with_pqc_simplified(
    circuit_ops, num_qubits=2, gate_blocks=1, 
    pqc_gates=['rx', 'ry', 'rz'], 
    pqc_params=pqc_params,  # ← Actual parameters
    return_numba=True
)
```

### Use Case 2: **Template System** (Training Loop Optimization)
```python
# Step 1: Create template ONCE (with num_pqc_blocks, not pqc_params)
template = create_pqc_circuit_template(
    circuit_ops, num_qubits=2, gate_blocks=1,
    pqc_gates=['rx', 'ry', 'rz'],
    num_pqc_blocks=2,  # ← Just the number
    ignore_noise_gates=True
)

# Step 2: Update template MANY TIMES (fast!)
for epoch in range(10000):
    pqc_params = optimizer.get_params()  # New parameters
    gate_ids, w1, w2, theta = update_pqc_circuit_template(template, pqc_params)
    # Run training...
```

---

## What Happened in Your Notebook

### ❌ What You Tried (Incorrect):
```python
pqc_noisy_circ_template = build_circuit_with_pqc_simplified(
    noisy_circ, num_qubits=NUM_QUBITS, gate_blocks=NUM_GATE_BLOCKS, 
    pqc_gates=['rz', 'rx', 'rz'], 
    num_pqc_blocks=PQC_BLOCKS,  # ← ERROR! This function doesn't take num_pqc_blocks
    dtype=np.float32, ignore_noise_gates=True
)
```

**Problem:** 
- `build_circuit_with_pqc_simplified()` takes `pqc_params` (array), not `num_pqc_blocks` (int)
- You wanted a **template dictionary** but this function returns compiled arrays or circuit ops

### ✅ What You Should Use (Correct):
```python
# For template system, use create_pqc_circuit_template()
pqc_noisy_circ_template = create_pqc_circuit_template(
    noisy_circ, num_qubits=NUM_QUBITS, gate_blocks=NUM_GATE_BLOCKS, 
    pqc_gates=['rz', 'rx', 'rz'], 
    num_pqc_blocks=PQC_BLOCKS,  # ← Correct! Template function takes num_pqc_blocks
    dtype=np.float32, ignore_noise_gates=True
)
```

---

## Function Comparison Table

| Function | Input Params | Returns | Use Case |
|----------|-------------|---------|----------|
| `build_circuit_with_pqc()` | `pqc_params` array | Circuit arrays or ops | One-time build |
| `build_circuit_with_pqc_simplified()` | `pqc_params` array | Circuit arrays or ops | One-time build (cleaner code) |
| `create_pqc_circuit_template()` | `num_pqc_blocks` int | Template dict | Create reusable template |
| `create_pqc_circuit_template_simplified()` | `num_pqc_blocks` int | Template dict | Create reusable template (cleaner code) |
| `update_pqc_circuit_template()` | Template dict + params | Circuit arrays | Fast parameter update |

---

## The Simplified Functions DO Work with Templates!

The simplified version works **exactly the same** as the original, just with cleaner code:

### Original Template Creation (Complex):
```python
def create_pqc_circuit_template(circuit_ops, num_qubits, gate_blocks, pqc_gates, num_pqc_blocks, ...):
    dummy_params = np.zeros((num_pqc_blocks, num_qubits, len(pqc_gates)), dtype=dtype)
    
    # Calls the COMPLEX 220-line function
    gate_ids_init, w1_init, w2_init, theta_init, pqc_param_map = build_circuit_with_pqc(
        circuit_ops, num_qubits, gate_blocks, pqc_gates, dummy_params, 
        dtype=dtype, return_numba=True, ignore_noise_gates=ignore_noise_gates,
        return_pqc_map=True
    )
    
    return {'gate_ids': gate_ids_init, 'wire1': w1_init, ...}
```

### Simplified Template Creation (Clean):
```python
def create_pqc_circuit_template_simplified(circuit_ops, num_qubits, gate_blocks, pqc_gates, num_pqc_blocks, ...):
    dummy_params = np.zeros((num_pqc_blocks, num_qubits, len(pqc_gates)), dtype=dtype)
    
    # Calls the SIMPLIFIED 90-line function
    gate_ids_init, w1_init, w2_init, theta_init, pqc_param_map = build_circuit_with_pqc_simplified(
        circuit_ops, num_qubits, gate_blocks, pqc_gates, dummy_params, 
        dtype=dtype, return_numba=True, ignore_noise_gates=ignore_noise_gates,
        return_pqc_map=True
    )
    
    return {'gate_ids': gate_ids_init, 'wire1': w1_init, ...}
```

**Both produce identical templates!** The only difference is the underlying builder is cleaner.

---

## Solution: Three Options

### Option 1: Keep Using Original Functions ✅
```python
# Your notebook now uses this (I fixed it)
template = create_pqc_circuit_template(...)  # Original
gate_ids, w1, w2, theta = update_pqc_circuit_template(template, params)
```

**Pros:** Already working, no changes needed  
**Cons:** Uses complex 220-line builder internally

### Option 2: Replace with Simplified Versions ✅✅ (Recommended)
```python
# 1. Add to builder.py:
from simplified_builder import (
    build_circuit_with_pqc_simplified,
    create_pqc_circuit_template_simplified
)

# 2. Use in notebook:
template = create_pqc_circuit_template_simplified(...)  # Simplified!
gate_ids, w1, w2, theta = update_pqc_circuit_template(template, params)
```

**Pros:** 60% less code, same performance, easier to maintain  
**Cons:** Requires adding simplified functions to codebase

### Option 3: Direct Build (No Template) ✅
```python
# For one-time use only
gate_ids, w1, w2, theta = build_circuit_with_pqc_simplified(
    circuit_ops, num_qubits=2, gate_blocks=1,
    pqc_gates=['rx', 'ry', 'rz'],
    pqc_params=actual_params_array,  # Not num_pqc_blocks!
    return_numba=True
)
```

**Pros:** Simplest for single builds  
**Cons:** Slow in training loops (rebuilds from scratch each time)

---

## Bottom Line

✅ **The simplified functions DO work with templates!**

The confusion was just about function names:
- `build_circuit_with_pqc_simplified()` → For direct building (takes `pqc_params` array)
- `create_pqc_circuit_template_simplified()` → For template creation (takes `num_pqc_blocks` int)

Both produce **identical results** to the originals, just with 60% less code!

Your notebook now uses the correct function (`create_pqc_circuit_template`) and should work perfectly. If you want to switch to the simplified version, you'd just need to add `create_pqc_circuit_template_simplified()` to your codebase.
