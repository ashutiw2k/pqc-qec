# PQC-QEC Project Instructions for AI Agents

## Project Overview
**Learning Parametrized Quantum Circuits (PQC) for Quantum Error Correction**

This project trains parametrized quantum circuits to compensate for coherent noise in quantum circuits. The core idea: insert trainable PQC layers between blocks of noisy gates, optimizing them to restore fidelity.

### Key Architecture Concepts

**Three Simulator Backends:**
1. **PennyLane** (`pqcqec/models/pqc_models.py::StateInputModelInterleaved*`) - Original JAX+PennyLane implementation, slower but easier to debug
2. **Custom Numba** (`pqcqec/simulate/statevector.py`) - Fast C-compiled statevector simulator for forward passes
3. **Custom JAX** (`pqcqec/simulate/jax_statevector.py`) - Pure JAX implementation, fully differentiable, **PREFERRED for new models**

**Always use the JAX statevector backend** (`LELZZInterleavedQuaternionCustomStatevecModel`) for new work - it's 10-50x faster than PennyLane while maintaining full differentiability.

**Circuit Template System** (`pqcqec/circuits/templates.py`):
- Templates separate circuit *structure* from *parameters*
- Build template once with `build_pqc_circuit_template()`, instantiate many times with different parameters
- Critical for performance: avoid rebuilding circuit structure in training loops
- When adding new PQC architectures, create separate functions like `add_lel_zz_pqc_layer()` rather than complex conditionals

**PQC Layer Types:**
- **LEL-ZZ**: Pre-local unitaries (RzRxRz) + ZZ entangling ring + Post-local unitaries. Default and most effective.
- **Simple Local**: Just RzRxRz per qubit, no entanglement. Faster, less expressive.
- Quaternion parametrization for local unitaries (avoids gimbal lock), direct angles for ZZ gates

## Critical Patterns & Conventions

### Model Parameter Management
```python
# Models expose THREE parameter groups (NOT one flat array!)
params = model.get_model_params()  # Returns dict with 'pre_quaternions', 'theta_zz', 'post_quaternions'
model.set_model_params(pre_quats, theta_zz, post_quats)  # Takes 3 separate arrays

# In training loops, extract all three:
pre_q, theta, post_q = params['pre_quaternions'], params['theta_zz'], params['post_quaternions']
```

### Training Strategies (See `pqcqec/training/jax_train_functions.py`)
1. **With Uncomputation** (`train_*_with_uncomp`): Circuit is U·U†, target is input state (identity)
2. **Without Uncomputation** (`train_*_no_uncomp`): Circuit is just U, target is ideal noiseless output
3. **Progressive** (`train_*_progressive`): Train blocks 0→N sequentially, freeze previous blocks with `jax.lax.stop_gradient()`
4. **Individual** (`train_*_individual`): Train each block in isolation on its own gates (non-cascading)

**Always** use progressive or individual training for circuits >50 gates - simultaneous training causes gradient vanishing.

### Data Flow Pattern
```
Input States (get_input_data) 
  → Noisy Circuit (base gates + noise) 
  → PQC Model (learnable correction gates) 
  → Compare to Target (input for uncomp, ideal_noiseless for no-uncomp)
  → Optimize PQC parameters
```

### Noise Model
- `PennylaneNoisyGates` applies over-rotations: ideal_gate → RX(noise) → RZ(noise)
- Noise arrays (`x_noise`, `z_noise`) are **fixed per circuit** (not trainable), shape `(num_gates,)`
- For uncomputation: concatenate noise arrays for U·U† circuit

## Development Workflows

### Running Experiments
```bash
# Quick test (5 qubits, 20 gates, 10 per block)
python scripts/train_tokenize_circuits_mp.py -q 5 -g 20 -k 10 -n 5000 -e 5 -a 20 -t 20 --seed 25 -o nogit/json_data/

# Full experiment sweep (multiprocessing over seeds)
python scripts/train_tokenize_circuits_mp.py -q 3 5 10 -g 10 20 50 -k 10 -n 10000 -e 10 --seed 0-50
```
**Output**: JSON files in `nogit/json_data/{qubits}q_{gates}g_{blocks}blk_data/` with circuit tokens, trained PQC params, and fidelity metrics.

### Testing
```bash
# Run all tests with verbose output
PYTHONPATH="${PYTHONPATH}:${PWD}" pytest -vv

# Test specific module
pytest tests/test_custom_statevec_runner.py -v

# Test progressive training
pytest tests/test_progressive_training.py -v
```
**Critical**: Always run tests after modifying `templates.py`, loss functions, or training loops.

### Debugging Fidelity Issues
1. Check `poor_fid_params.json` for circuits that didn't save (fidelity < 0.95 threshold)
2. Use `CheckFidOfFineTunedCircuit.ipynb` to inspect specific seed failures
3. Common causes: NaN gradients (quaternion singularities), noise too high, insufficient epochs

## File Organization

**Data Folders:**
- `data/json_data/` - Committed datasets (small, for testing)
- `nogit/json_data/` - Generated data (gitignored, for experiments)
- `plots/` - Fidelity visualizations

**Key Modules:**
- `pqcqec/circuits/` - Circuit generation, templates, PQC layers
- `pqcqec/models/` - Trainable PQC models (use `LELZZInterleavedQuaternionCustomStatevecModel`)
- `pqcqec/training/` - Loss functions (JAX), training loops
- `pqcqec/experiment/` - End-to-end experiment orchestration
- `pqcqec/simulate/` - Statevector backends (prefer `jax_statevector.py`)

**Documentation:**
- `PROGRESSIVE_TRAINING_IMPLEMENTATION.md` - Block-by-block training details
- `QUATERNION_ANALYSIS.md` - Why quaternions avoid gimbal lock
- `nogit_README/` - Various implementation notes (check for context)

## Common Pitfalls

1. **Don't rebuild templates in training loops** - Build once, instantiate many times
2. **Don't forget to JIT compile** - Wrap update steps in `@jax.jit` for 10x speedup
3. **Sanitize gradients** - Use `jnp.nan_to_num()` after grad computation (quaternions can produce NaNs)
4. **Learning rate matters** - Use warmup + cosine decay with restarts (see experiment runners for schedule)
5. **Batch size vs GPU memory** - Start with 32, reduce if OOM
6. **Progressive training can't use uncomputation yet** - Raises `NotImplementedError`, use `add_uncomputation=False`

## Environment Setup
```bash
python -m virtualenv .venv
source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```
**Dependencies**: JAX, Optax, PennyLane, Qiskit, PyTorch (for models, not actual training), Numba

## When Modifying Templates

If adding new PQC architecture types (e.g., "yxy" gates, star topology):
1. Create helper function like `add_yxy_pqc_layer(template, layer_idx, num_qubits, param_sources)`
2. Modify `build_pqc_circuit_template()` to accept string `pqc_type` parameter
3. Route to appropriate helper based on `pqc_type`
4. Update models to pass `pqc_type` through to template builder
5. Add corresponding quaternion conversion if needed (see `quaternions_utils.py`)
6. **Write tests** - See `tests/test_circuits.py` for examples

## Performance Expectations
- 3q/10g circuit: ~5s per epoch (1000 samples, batch=32)
- 5q/50g circuit: ~30s per epoch
- 10q/100g circuit: ~2min per epoch
- Progressive training: multiply by number of blocks
- Use `train_tokenize_circuits_mp.py` for parallel seed processing (uses half available cores)
