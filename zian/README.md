## `main_training_loop` (Multi-Initial-State Focus Only)

This section explains what each step in `main_training_loop` does, focusing solely on the multi initial state (Monte Carlo) training path. Single-initial-state caching logic has been intentionally removed from the explanation because we only deploy the multi-state variant.

### 0. Key Inputs
* `tokenized_data_x`, `extracted_sched`: Encoded base gate sequences plus PQC schedule (qubit indices, gate types, insertion positions `after_idx`).
* `n_init_states` (>1 in our setting): Number of random initial states S per data sample.
* `fixed_multi_init`, `fixed_multi_base_ideal`: (Precomputed) tensors holding the S raw random initial states and their base-only ideal evolutions.

### 1. Data Split (setup:data_splits)
Create train / validation datasets and DataLoaders. Pure infrastructure for batching.

### 2. Model Construction (setup:model)
Instantiate the Transformer encoder (`EncoderWithQuerySlotsPacked`). It consumes padded base gate tokens and outputs sine/cosine pairs for up to `K_MAX` parameterized PQC gates. Optional `torch.compile` may be applied for performance.

### 3. Optimizer & (Optional) Scheduler (setup:optim_sched)
Create `AdamW` and (if requested) attach a cosine schedule with warmup. Not logically coupled to quantum simulation—just optimization plumbing.

### 4. Noise Models (setup:noise_models)
Construct two noise model objects:
* `noisy_model`: Injects stochastic Rx/Rz perturbations immediately after each base gate.
* `ideal_model`: Zero-noise reference used only for producing the base-only ideal states (already baked into the precomputation phase below).

### 5. Multi Initial State Precomputation (core)
Call `precompute_fixed_multi_initial_states` (done once before training epochs):
1. For each dataset batch generate S random complex normalized states per sample.
2. Evolve each through the base circuit under `ideal_model` (no noise) using `simulate_base_only_multi`.
3. Store:
   * `fixed_multi_init[idx]`  → shape `[S, 2^n]`: raw random initial states.
   * `fixed_multi_base_ideal[idx]` → shape `[S, 2^n]`: base-only ideal outputs.
These remain fixed across epochs for stable targets (unless you choose to resample in a future enhancement).

### 6. Epoch Loop
`for ep in range(1, epochs+1)` — each epoch = training phase + validation phase (+ scheduler step + best checkpoint logic).

#### 6.1 Training Phase (`train_one_epoch`)
For each batch:
1. Move batch tensors to device; collect sample indices `idx`.
2. Forward pass: model outputs `pred_sincos` (shape `[B, K_MAX, 2]`).
3. Fidelity loss (multi-state path only):
   * Slice `init_states = fixed_multi_init[idx]` and `base_ideal = fixed_multi_base_ideal[idx]` giving `[B, S, 2^n]`.
   * Convert predicted sin/cos → angles.
   * Run `simulate_interleaved_with_params_multi` using predicted angles, inserting parameter gates according to `pqc_after_idx` between base gates and injecting noise only after base gates.
   * Compute fidelity per (sample, initial_state): `F[b,s] = |⟨ base_ideal[b,s] | noisy_pred[b,s] ⟩|^2`.
   * Average over S, then over batch: `loss = mean_b(1 - mean_s F[b,s])`.
4. Backpropagation: automatic mixed precision (autocast + GradScaler) → gradient step (`optimizer.step()`). Optional gradient clipping may be applied.
5. Accumulate running training loss; optionally record timing stats for profiling.

#### 6.2 Validation Phase (`evaluate`)
Same data flow as training phase but with `torch.no_grad()`: no gradient computation, just forward simulation + fidelity loss evaluation to obtain average validation loss.

#### 6.3 Scheduler Step (optional)
If a learning rate scheduler was configured, call `scheduler.step()` once per epoch.

#### 6.4 Best Checkpoint
If current validation loss improves the best seen, persist `model.state_dict()` to `save_path`.

### 7. Final Benchmark (optional)
`evaluate_fidelity_benchmark` re-runs fidelity evaluation (train + val sets) for reporting consistency and final metrics snapshot.

### 8. Timing Report
If profiling enabled, print aggregated per-section wall times (I/O, tokenization, base simulation, interleaved simulation, loss, backward, etc.).

### 9. Return
Return the trained model (weights correspond to last epoch; best checkpoint is stored on disk when improvement occurred).

---

### Core Pipeline Recap (Multi-State Only)
1. Random multi initial states (S per sample) + deterministic base-only evolution cached.
2. Forward: Transformer predicts parameter gate angles (sin/cos → angle).
3. Simulation: Interleave predicted parameter gates with base gates; inject noise after base gates only.
4. Fidelity Objective: Maximize overlap with cached noise-free base-only evolved states → implemented as minimizing `1 - mean_s(|⟨ψ_base|ψ_noisy_pred⟩|^2)`.

This constitutes a label-free self-supervised objective: the model learns angles that best counteract noise so the final noisy circuit reproduces the ideal base-only outputs for a diverse distribution of random input states.

Future improvements (not implemented yet): epoch-level resampling of initial states, full vectorization of parameter gate insertion across batch & S, gate fusion, low-discrepancy (Sobol) initial state sampling, and custom GPU kernels for batched gate application.

