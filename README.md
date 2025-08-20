# pqc-qec

Learning Parameterized Quantum Circuits for Quantum Error Correction.

## Setting Up the environment

Linux (Ubuntu):
```
virtualenv .venv
source .venv/bin/activate
```
Ensure your virtual environment is activated (you may have to install `virtualenv` using `sudo update && sudo upgrade && apt install virtualenv`)

MacOS:
```
python -m virtualenv .venv
source .venv/bin/activate
```

Now, install dependencies 
```
pip install --upgrade pip
pip install --upgrade pennylane pennylane-catalyst qiskit[visualization] jax optax torch jupyter
```