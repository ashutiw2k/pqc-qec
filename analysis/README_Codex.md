# Memory Analysis (Codex)

## High-Impact Hotspots
- `pqcqec/models/pqc_models.py:203` vectorizes the QNode with `jax.vmap`, so every gate materializes an intermediate `(batch, 2**num_qubits)` state in `complex128`; with 10 qubits, 2000 gates, and batch 32 you hit ~0.5 MB per op → >3 GB retained during forward/grad.
- `pqcqec/simulate/simulate.py:22` repeats the same `jax.vmap` pattern for noisy circuit evaluation, so the “10-qubit, 2000-gate” run stores the full batched state after each gate.
- `pqcqec/noise/simple_noise.py:75` injects two extra rotations per wire for every logical gate, multiplying the tape length (and intermediate states) several-fold, pushing allocations into multi-GB territory.
- `pqcqec/training/jax_train_functions.py:14` wraps the batched model call in `jax.value_and_grad`, so JAX keeps the entire forward trace (all those batched states) for backprop; the backward sweep doubles peak memory.

## Why It OOMs
- PennyLane’s `default.qubit` defaults to `complex128`, doubling bytes per amplitude relative to `complex64`; a 10-qubit batched state is ~0.5 MB, and thousands of gates exhaust RAM when every intermediate is retained.
- The experiment driver (`pqcqec/experiment/pqc_experiment.py:96`) runs `run_circuit_with_noise_model(..., batched=True)` with large test batches, so even evaluation inherits the same batched-tape footprint.

## Mitigations
1. Run simulations batch-by-batch (drop `vmap`, loop in Python or use `lax.scan`) or reduce `batch_size` drastically when circuits exceed a few hundred gates.
2. Instantiate the device with `dtype=jnp.complex64` (or use `default.qubit.jax`/`lightning.qubit` supporting `complex64`) to halve per-amplitude memory.
3. Consider gradient checkpointing (`jax.remat`) or a parameter-shift/adjoint diff method to avoid caching the full forward pass.

## Next Steps
- Re-run the 10-qubit/2000-gate case with `batch_size=1` or `complex64` to validate the peak allocation drop before layering additional changes.
