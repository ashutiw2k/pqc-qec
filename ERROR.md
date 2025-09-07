# Case Study: Debugging a JAX/PennyLane Memory Leak (FINAL FIX NOT IMPLEMENTED)

This document provides a comprehensive summary of the process used to diagnose and fix a critical memory leak in a parallelized Python script for quantum circuit simulation.

## 1. The Initial Problem: A Growing Memory Footprint

The script was designed to run thousands of unique quantum circuit experiments in parallel. While functionally correct, it exhibited a severe memory leak, with RAM usage growing steadily over time, threatening to crash the system.

**Symptom:** The resident memory (`RES`) of each worker process, as seen in `htop`, would grow continuously without ever stabilizing.



### Original Problematic Code

The leak originated from a design pattern where complex, compilable objects were created inside a loop that ran thousands of times. The core issue was in the `pqc_experiment_runner` and the `StateInputModelInterleavedQuaternionModel` class it called.

```python
# Original Leaky Model Structure
class StateInputModelInterleavedQuaternionModel:
    def __init__(self, circuit_ops:List, num_qubits:int, noise_model:PennylaneNoisyGates, ...):
        # ... other initializations ...
        self.circuit_ops = circuit_ops
        self.noise_model = noise_model

        # PROBLEM: A new device and QNode are created for every single experiment.
        self.qdev_cpu = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(self.qdev_cpu, interface='jax', diff_method="backprop")
        def model_circuit(state, pqc_params):
            # The circuit structure is defined here using self.circuit_ops
            for i, op in enumerate(self.circuit_ops):
                gate, qubit, param = op
                self.noise_model.apply_gate(gate, qubit, angle=param)
                # ... PQC logic ...
            return qml.state()
        
        # JAX compiles and caches a new function on every instantiation.
        self.batched_model_circuit = jax.jit(jax.vmap(self.model_circuit, in_axes=(0, None)))
    
    def __call__(self, in_state, params=None):
        # ... logic to get pqc_angles ...
        return self.batched_model_circuit(in_state, pqc_angles)

# Original Experiment Runner (Simplified)
def pqc_experiment_runner(seed, ...):
    # This function is called in a loop, once for each of the 10,000 seeds.
    
    # 1. A unique circuit is generated for each seed.
    circuit_ops = generate_random_circuit(seed=seed, ...) 
    
    # 2. A new, heavy model object is created, triggering recompilation.
    model = StateInputModelInterleavedQuaternionModel(circuit_ops=circuit_ops, ...)
    
    # 3. The model is trained.
    train_pqc_model_with_uncomp(model, ...)
```

## 2. The Debugging Journey: A Series of Errors

The fix involved creating a single, static JIT-compiled function. This process revealed several deep integration challenges between JAX and PennyLane.

### Error 1: `TypeError: len() of unsized object`

After the first refactor to a static QNode, this error appeared.

**Traceback:**
```
TypeError: len() of unsized object
  File ".../pqcqec/noise/simple_noise.py", line 76, in apply_noise
    x_noise = self.rng.uniform(size=(len(wires),), ...)
```
**Diagnosis:** The static circuit was passing qubit wires to the `noise_model` as single integers (e.g., `0`) instead of lists (e.g., `[0]`). The `apply_noise` function then tried to call `len()` on the integer, which is invalid.

**Fix:** All single-qubit `wires` arguments were wrapped in a list within the static circuit's `lambda` functions.
```python
# Incorrect code in the JIT-compiled function's branches
lambda: noise_model.apply_gate('h', qubit_indices[i, 0])

# Corrected code
lambda: noise_model.apply_gate('h', [qubit_indices[i, 0]])
```

### Error 2: `pennylane.wires.WireError: abstract wires are present`

After fixing the `TypeError`, a more complex error surfaced.

**Traceback:**
```
pennylane.wires.WireError: Cannot run circuit(s) on default.qubit as abstract wires are present in the tape: Wires([..., Traced<ShapedArray(int32[])>...])
```
**Diagnosis:** This error revealed a fundamental conflict. We were using JAX's `jax.lax.switch` for conditional logic. JAX's compiler was abstracting the qubit numbers (wires) into placeholders ("tracers"), but PennyLane's circuit builder required concrete integer values for the `wires` argument *during* the compilation trace.

**Attempted Fix:** The code was modified to use PennyLane's native conditional, `qml.cond`, instead of `jax.lax.switch`, hoping it would handle the wire tracing correctly.

### Error 3: `jax.errors.TracerBoolConversionError`

The switch to `qml.cond` resolved the `WireError` but immediately caused a new, opposite error.

**Traceback:**
```
jax.errors.TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
The error occurred while tracing the function static_circuit_executor ... because it depends on the value of the argument gate_types.
```
**Diagnosis:** This error showed that `qml.cond` was trying to use a dynamic JAX value (the condition, e.g., `gate_id == 0`) in a context that required a static Python `True` or `False`. This is not possible during JIT compilation. This confirmed that creating a single "universal" executor for structurally different circuits was not a viable approach.

## 3. The Final, Robust Solution: Grouping by Structure

The iterative fixes revealed that the core strategy needed to change. The final solution was to solve the memory leak at an architectural level.

**The Strategy:** Instead of compiling one function for all circuits, we **group seeds that share the same circuit structure** and compile a specialized function for each unique group.

### Final Code Architecture

**1. Grouping Logic (in `main()`):** Before starting the workers, the main process pre-generates all circuit structures and groups the seeds.

```python
import collections

# ...
print("--- Pre-generating and grouping circuit structures ---")
grouped_tasks = collections.defaultdict(list)

for seed in range(num_circs):
    circuit_ops = generate_random_circuit(seed=seed, ...)
    # Add uncomputation if needed
    if add_uncomputation:
        circuit_ops = circuit_ops + circuit_ops[::-1]

    # Create a hashable key representing the structure (gate names and wires)
    structure_key = tuple((op[0], tuple(op[1])) for op in circuit_ops)
    grouped_tasks[structure_key].append(seed)

print(f"Reduced {num_circs} tasks to {len(grouped_tasks)} unique structures.")

pool_tasks = []
for structure_key, seeds in grouped_tasks.items():
    circuit_ops = [(gate, list(wires), []) for gate, wires in structure_key]
    pool_tasks.append((circuit_ops, seeds))

# ... start multiprocessing pool with pool_tasks ...
```

**2. New Worker Function:** The worker function now receives a shared circuit structure and a list of seeds. It can now use the **original, simple `pqc_experiment_runner` code**, as it's only called once per unique structure, containing the leak.

```python
def process_circuit_group(args):
    shared_circuit_ops, seeds_in_group = args
    
    print(f"Worker compiling new structure for {len(seeds_in_group)} seeds...")
    
    # Loop efficiently through all seeds in the group
    for seed in seeds_in_group:
        print(f"  Processing seed {seed} with pre-compiled circuit.")
        
        # Call your ORIGINAL pqc_experiment_runner, adapting it to accept
        # the pre-computed circuit operations.
        pqc_experiment_runner(
            seed=seed,
            precomputed_circuit_ops=shared_circuit_ops,
            # ... other parameters ...
        )

    return {"status": "group_complete", "num_seeds": len(seeds_in_group)}
```

This final architecture resolves the memory leak by reducing the number of JIT compilations from 10,000 to a much smaller number, while also simplifying the core scientific code.