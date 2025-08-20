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
