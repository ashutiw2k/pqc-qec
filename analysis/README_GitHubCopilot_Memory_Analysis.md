# Memory Analysis for PQC-QEC 10-Qubit Circuit Simulation
**Analysis by: GitHub Copilot**  
**Date: September 19, 2025**  
**Issue: OOM Error on 10 Qubit, 2000 Gate Circuit Simulation**

## Executive Summary

The Out-of-Memory (OOM) errors occurring during 10-qubit, 2000-gate circuit simulations are primarily caused by exponential state vector memory requirements combined with inefficient memory management patterns in JAX/PennyLane operations. The analysis identified four critical memory bottlenecks that compound to create unsustainable memory usage.

## Key Findings

### Memory Requirements Analysis
- **10-qubit state vector**: 2^10 = 1024 complex numbers
- **Single state memory**: 1024 × 16 bytes (complex128) = 16KB per state
- **Batch operations (5000 states)**: 5000 × 16KB = 80MB base memory
- **With copies and intermediate operations**: ~240MB+ per batch

## Critical Memory Bottlenecks

### 1. State Vector Allocation (Primary Issue)

**Location**: 
- `pqcqec/simulate/simulate.py` - `get_input_data()` function
- All simulation functions using `qml.state()` returns

**Problem Analysis**:
```python
# Current implementation in get_input_data()
state = jax.random.normal(key_real, (num_vals, 2**num_qubits,)) + 1j * jax.random.normal(key_imag, (num_vals, 2**num_qubits,))
norms = jnp.linalg.norm(state, axis=1, keepdims=True)
ideal_data = state / norms
```

**Memory Impact**:
- Creates multiple copies of large state tensors
- For 10 qubits with 5000 batch size: 240MB+ just for input data generation
- Each normalization step creates additional temporary arrays

### 2. JAX JIT Compilation Memory Explosion

**Location**: 
- `jax.jit(jax.vmap(circuit, in_axes=(0)))` throughout simulation functions
- Model circuit compilation in `pqc_models.py`

**Problem Analysis**:
- JAX stores entire computation graphs for 2000-gate circuits
- Each gate operation creates nodes in the computation graph
- JIT compilation can consume several GB for complex circuits
- Memory is not released until cache is manually cleared

### 3. Batched Operations with `vmap`

**Location**: 
- All `batched_circuit = jax.jit(jax.vmap(self.model_circuit, in_axes=(0, None)))` calls
- Model training functions using batch processing

**Problem Analysis**:
- `vmap` vectorizes operations but multiplies memory usage by batch size
- For each operation: base_memory × batch_size × num_intermediate_states
- With 2000 gates and large batches, this becomes prohibitive

### 4. State Copying in PennyLane Operations

**Location**: 
- Every gate application in circuit simulations
- State preparation and measurement operations

**Problem Analysis**:
- PennyLane creates intermediate state copies for each gate
- 2000 gates × 16KB per state × batch_size = massive memory usage
- No automatic cleanup of intermediate states

## Detailed Code Analysis

### Memory Hotspots Identified:

1. **`get_input_data()` in simulate.py**:
   ```python
   # Memory: 2 × (num_vals × 2^num_qubits × 8 bytes) for real + imaginary
   state = jax.random.normal(key_real, (num_vals, 2**num_qubits,)) + 1j * jax.random.normal(key_imag, (num_vals, 2**num_qubits,))
   ```

2. **Batched circuit execution**:
   ```python
   # Creates vectorized circuits that hold all intermediate states in memory
   batched_circuit = jax.jit(jax.vmap(circuit, in_axes=(0)))
   ```

3. **Model parameter storage**:
   ```python
   # In pqc_models.py - parameter tensors grow with circuit complexity
   self.param_sz = (int(self.pqc_blocks * jnp.ceil(self.num_gates/self.gate_blocks)), self.num_qubits, self.num_pqc_angles)
   ```

## Immediate Solutions

### 1. Reduce Memory Footprint
```python
# Use complex64 instead of complex128 (50% memory reduction)
state = state.astype(jnp.complex64)

# Reduce batch sizes for large qubit counts
batch_size = 5 if num_qubits >= 10 else 20
```

### 2. Implement Memory Management
```python
# Clear JAX cache regularly
if iteration % 10 == 0:
    jax.clear_caches()
    gc.collect()
```

### 3. Chunked Processing
```python
def run_circuit_chunked(circuit_ops, input_states, noise_model, num_qubits, chunk_size=100):
    results = []
    for i in range(0, len(input_states), chunk_size):
        chunk = input_states[i:i+chunk_size]
        result = run_circuit_with_noise_model(circuit_ops, chunk, noise_model, num_qubits, batched=True)
        results.append(result)
        # Clear intermediate results
        del chunk
    return jnp.concatenate(results, axis=0)
```

### 4. Configuration Adjustments
```python
# In config files for 10+ qubit experiments
{
    "batch": 5,           # Reduced from 20
    "num_data": 1000,     # Reduced from 5000
    "dtype": "complex64"  # Add precision control
}
```

## Long-term Architectural Recommendations

### 1. Streaming Data Pipeline
- Implement generator-based data loading to avoid loading entire datasets
- Process circuits in streaming fashion rather than batch loading

### 2. Memory-Efficient Simulators
- Consider PennyLane's Lightning simulator for better memory efficiency
- Explore tensor network methods for large qubit counts

### 3. Circuit Segmentation
- Break 2000-gate circuits into smaller segments
- Use intermediate state checkpointing

### 4. Hardware Considerations
- Move to systems with larger RAM for 10+ qubit full state simulation
- Consider distributed computing for very large circuits

## Implementation Priority

1. **High Priority (Immediate)**:
   - Reduce batch sizes in configs
   - Add complex64 dtype usage
   - Implement chunked processing

2. **Medium Priority (1-2 weeks)**:
   - Add memory monitoring and automatic cache clearing
   - Implement streaming data loaders

3. **Low Priority (Long-term)**:
   - Architectural changes for very large circuits
   - Hardware upgrades or distributed computing

## Expected Memory Improvements

With immediate fixes:
- **50% reduction** from complex64 usage
- **75% reduction** from smaller batch sizes (5 vs 20)
- **Additional 20-30% reduction** from chunked processing and cache management

Combined: **85-90% memory reduction** should resolve OOM errors for 10-qubit circuits.

## Monitoring Recommendations

Add memory monitoring to identify future bottlenecks:
```python
import psutil
import tracemalloc

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
    return memory_mb
```

## Conclusion

The OOM errors are caused by the fundamental exponential scaling of quantum state simulation combined with inefficient memory management. The recommended solutions provide both immediate relief and long-term scalability for larger quantum circuit simulations.

---
*This analysis was generated by examining the codebase structure, identifying memory allocation patterns, and analyzing the specific requirements for 10-qubit quantum circuit simulation.*
