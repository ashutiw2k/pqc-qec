# pqc-qec

Learning Parameterized Quantum Circuits for Quantum Error Correction.

## Setting Up the environment

Linux (Ubuntu):

```bash
virtualenv .venv
source .venv/bin/activate
```

Ensure your virtual environment is activated (you may have to install `visrtualenv` using `sudo update && sudo upgrade && apt install virtualenv`)

MacOS:

```bash
python -m virtualenv .venv
source .venv/bin/activate
```

Now, install dependencies

```bash
pip install --upgrade pip
pip install --upgrade pennylane pennylane-catalyst qiskit[visualization] jax optax torch jupyter
```

## Repository Structure

- **checkpoints**: Placeholder for saved model or run artifacts (currently empty).
- **data**: JSON datasets used for experiments.
  - `data/json_data/3q_10g_5blk_data`: `config.json`, `0.json`, `1.json`, `2.json`, `3.json`, `4.json`, `5.json`, `poor_fid_params.json`.
  - `data/json_data/10q_50g_10blk_data`: `config.json`, `0.json`, `4.json`, `7.json`, `8.json`, `9.json`, `poor_fid_params.json`.
- **plots**: Generated figures and results.
  - `Fidelity_3q_10g_seed5.png`
- **pqcqec**: Core Python package for circuits, training, simulation, experiments, utils, and noise (source files only; caches omitted).
  - `pqcqec/__init__.py`
  - `pqcqec/circuits`: `__init__.py`, `generate.py`, `modify.py`, `pqc_circuits.py`
  - `pqcqec/models`: `__init__.py`, `pqc_models.py`
  - `pqcqec/training`: `__init__.py`, `jax_loss_functions.py`, `jax_train_functions.py`
  - `pqcqec/simulate`: `__init__.py`, `simulate.py`
  - `pqcqec/experiment`: `__init__.py`, `pqc_experiment.py`
  - `pqcqec/utils`: `__init__.py`, `args.py`, `constants.py`, `jax_utils.py`
  - `pqcqec/noise`: `__init__.py`, `simple_noise.py`
- **scripts**: Utility scripts for training and plotting.
  - `plot_fidelity_experiement.py`
  - `train_tokenize_circuits_mp.py`
- **testnotebooks**: Example and exploratory Jupyter notebooks.
  - `jax/JAX_CPU_NewNoiseModel.ipynb`
  - `tokenizer/NewInputStateGenerator.ipynb`
  - `tokenizer/RunTokenizedCircuits.ipynb`
  - `tokenizer/TokenizeQuantumCircuit.ipynb`
- **nogit**: Placeholder for ignored local artifacts (currently empty).
- **Root files**: `README.md`, `.gitignore`.

## pqcqec Package Details

This section explains what each module in the `pqcqec` package does so new users can quickly navigate and extend the code.

### Top-level
- `pqcqec/__init__.py`: Marks the directory as a Python package.

### circuits
- `pqcqec/circuits/generate.py`: Utilities to synthesize random circuits.
  - `generate_random_circuit_qiskit(num_qubits, num_gates, gate_dist)`: Returns a Qiskit `QuantumCircuit` by sampling gates according to `gate_dist`.
  - `generate_random_circuit_pennylane(...)`: Returns a list of PennyLane operations constructed from sampled gates.
  - `generate_random_circuit_list(...)`: Returns a token list of `(gate_name, [wires])` without framework objects.
  - `generate_random_circuit(..., backend)`: Convenience wrapper to select backend: `'qiskit' | 'pennylane' | 'list'` with optional `seed` and default uniform gate distribution.
- `pqcqec/circuits/modify.py`: Circuit post-processing helpers.
  - `tokenize_qiskit_circuit(circuit)`: Converts a Qiskit circuit to tokens `(gate_name, [wires], [params])`.
  - `pennylane_state_embedding(input_state, num_qubits)`: Embeds an arbitrary complex state via `qml.StatePrep` across all wires.
- `pqcqec/circuits/pqc_circuits.py`: Parametrized PQC building blocks.
  - `pennylane_PQC_RZRXRZ_unique(num_qubits, params)`: Applies one layer per qubit of RZ → RX → RZ using a flat parameter vector of length `3 * num_qubits`.

### models
- `pqcqec/models/pqc_models.py`: Trainable PQC models that interleave learned gates with a target circuit under noise.
  - `StateInputModelInterleavedPQCModel`:
    - Initializes with a tokenized uncompensated circuit `circuit_ops`, `num_qubits`, a `noise_model` (see `noise/simple_noise.py`), and layout controls `pqc_blocks`, `gate_blocks`.
    - Builds a PennyLane `QNode` that: (1) embeds an input state, (2) iterates the circuit tokens applying noisy gates, interleaving learned PQC gates (RZ/RX/RZ) after every `gate_blocks` operations, and (3) returns the output state.
    - Parameters shape: `(ceil(num_gates/gate_blocks) * pqc_blocks, num_qubits, 3)`; trained via JAX/Optax.
    - Methods: `run_model_batch` (batched forward via `vmap`), `draw_mpl` (matplotlib circuit drawing), `get_circuit_tokens` (returns original tokens interleaved with PQC parameter tokens).

### training
- `pqcqec/training/jax_train_functions.py`: Training loop utilities using JAX + Optax.
  - `train_pqc_model(model, dataloader, optimizer, schedule, main_loss_fn, epochs)`: Per-epoch loop with a JIT’d `update_step` computing loss, gradients, parameter updates, and per-batch fidelity. Displays running metrics via `tqdm` and prints epoch summaries.
- `pqcqec/training/jax_loss_functions.py`: Differentiable JAX loss and fidelity functions.
  - `jax_pure_state_fidelity(psi, phi)`: |⟨ψ|φ⟩|² for statevectors (normalized internally).
  - `jax_mixed_state_fidelity(rho, sigma)`: Uhlmann fidelity for density matrices.
  - Phase-invariant and complex MSE losses: `jax_mse_complex_loss`, `jax_mse_complex_loss_aligned`, `jax_l2_loss_ignore_global_phase`.
  - Alternatives: `jax_fidelity_loss`, `jax_mixed_fidelity_loss`, `jax_density_trace_loss`, `jax_hilbert_schmidt_density_loss`.

### simulate
- `pqcqec/simulate/simulate.py`: Data generation and noisy circuit execution.
  - `get_input_data(num_qubits, num_vals, seed)`: Samples random complex vectors and normalizes to valid quantum states.
  - `run_circuit_with_noise_model(circuit_ops, input_state, noise_model, num_qubits, device, batched)`: Builds a PennyLane circuit that embeds `input_state`, applies tokenized gates through the provided `noise_model`, and returns the final state (supports batched vmap execution).

### experiment
- `pqcqec/experiment/pqc_experiment.py`: End-to-end experiment orchestration.
  - `pqc_experiment_runner(...)`:
    - Seeds PRNGs; generates train/test data via `simulate.get_input_data`.
    - Instantiates `PennylaneNoisyGates` from `noise.simple_noise`.
    - Synthesizes a random Qiskit circuit, forms its inverse, composes an uncompensated circuit, and tokenizes it.
    - Builds `StateInputModelInterleavedPQCModel` with interleaved learned gates.
    - Defines a warmup + cosine-restart learning-rate schedule and an Optax optimizer chain.
    - Trains the model; evaluates batched fidelities on test data vs. noisy and PQC outputs; returns metrics and final parameters.

### utils
- `pqcqec/utils/constants.py`: Common mappings/constants.
  - Gate maps for Qiskit/PennyLane (`QISKIT_GATES`, `PENNYLANE_GATES`), arity (`QUBITS_FOR_GATES`), and placeholders for PQC/model registries.
- `pqcqec/utils/args.py`: Flexible CLI/config handling for experiments.
  - Central `ARG_DEFINITIONS` registry; parser creation; config loading; argument precedence (CLI > config > defaults); normalization helpers for ranges; output directory setup and device selection; returns a validated config dict.
- `pqcqec/utils/jax_utils.py`: Lightweight dataset/dataloader for JAX arrays.
  - `JAXStateDataset` (indexable array wrapper) and `JAXDataLoader` (permutes indices, batches, stacks tensors, supports `drop_last`).

### noise
- `pqcqec/noise/simple_noise.py`: Noisy gate model for PennyLane simulations.
  - `PennylaneNoisyGates`: Applies ideal gates then injects random RX/RZ over-rotations per wire; supports `X/Z/CX/CZ/H` plus parameterized `RX/RZ` for PQC layers. Noise magnitudes are configurable; exposes `apply_gate(gate_name, wires, angle=None)` to route tokens to the correct noisy/parametrized implementation.

## scripts Folder Details

Utilities to run experiments, serialize tokenized circuits/parameters, and visualize fidelity.

- `scripts/train_tokenize_circuits.py`: Single‑process training + tokenization over seeds.
  - Purpose: Iterates seeds serially for each `(qubits, gates)` configuration, runs `pqc_experiment_runner`, and writes per‑seed outputs when PQC fidelity exceeds 0.95.
  - Flow: Creates `figure_output/{qubits}q_{gates}g_{gate_blocks}blk_data`, writes `config.json`, loads/updates `poor_fid_params.json`, loops seeds with redo/force/skip logic, saves `{seed}.json` containing tokens, PQC params, and QASM for base/PQC circuits; accumulates poor‑fidelity tuples.
  - Helpers: `create_circuit_from_ops` (tokens → Qiskit circuit), `deep_tuple` (normalize nested lists to tuples for comparison/storage).
  - CLI (via `utils/args.py`): `--qubit_range`, `--gate_range`, `--gate_blocks`, `--pqc_blocks`, `--epochs`, `--num_data`, `--num_test`, `--batch`, `--gpu`, `--gate_dist`, `--figure_output`, `--seed`, `--force`, `--redo`.

- `scripts/train_tokenize_circuits_mp.py`: Multiprocessing variant across seeds.
  - Purpose: Same outputs as the serial script, but dispatches per‑seed runs across a pool (half of CPU cores) using `process_seed` workers.
  - Key pieces: `process_seed(args)` returns status tuples for each seed (`success/poor_fidelity/skipped/error`) to the main process, which writes `{seed}.json`, updates `poor_fid_params.json`, and collects run stats.
  - Helpers/CLI: Shares the same helpers and argument set as the serial script; additionally writes `mp_stats.json` when `--force` is used.

- `scripts/plot_fidelity_experiement.py`: Single‑run training and fidelity plot.
  - Purpose: Runs one configuration using the first entries of `--qubit_range` and `--gate_range`, retrieves `(fidelity_noisy, fidelity_pqc)` from `pqc_experiment_runner(return_fidelity=True)`, and saves a comparison plot.
  - Output: `figure_output/Fidelity_{qubits}q_{gates}g_seed{seed}.png` with both series.
