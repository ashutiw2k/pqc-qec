# ==============================================================================
# PROFILING AND UTILITIES
# ==============================================================================
import cProfile
import pstats
import io
import time
from contextlib import contextmanager
from typing import Generator, Optional

# Set up line_profiler
# To get line-by-line statistics, you must add the @profile decorator
# to the function(s) you want to inspect. Then, run the script from
# your terminal using: kernprof -l -v your_script_name.py
try:
    # This will succeed if running with kernprof
    from line_profiler import profile
except ImportError:
    # If not, create a dummy decorator that does nothing
    def profile(func):
        return func

@contextmanager
def performance_profile(
    description: str = "Execution Profile",
    sort_by: str = 'cumulative',
    print_stats: int = 25,
    dump_file: Optional[str] = None
) -> Generator[cProfile.Profile, None, None]:
    """A flexible context manager for profiling Python code execution with cProfile."""
    prof = cProfile.Profile()
    start_time = time.time()
    try:
        prof.enable()
        yield prof
    finally:
        prof.disable()
        end_time = time.time()

        print(f"\n--- {description} ---")
        print(f"Total Wall Time: {end_time - start_time:.4f} seconds")

        if print_stats > 0:
            s = io.StringIO()
            stats = pstats.Stats(prof, stream=s).sort_stats(sort_by)
            print(f"Top {print_stats} functions sorted by '{sort_by}':")
            stats.print_stats(print_stats)
            print(s.getvalue())

        if dump_file:
            prof.dump_stats(dump_file)
            print(f"💾 Full profiling stats saved to '{dump_file}'")
        
        print("-" * (len(description) + 6))

# ==============================================================================
# ORIGINAL SCRIPT LOGIC (with profiling decorators added)
# ==============================================================================
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
import numpy as np
import json
import pennylane as qml
import os
import sys
from pathlib import Path

# Add project root to Python path
# Note: This might need adjustment depending on your project structure.
# Using an absolute path or setting PYTHONPATH is often more robust.
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from pqcqec.utils.constants import QUBITS_FOR_GATES, QISKIT_GATES, GATE_IS_DIRECTIONAL, PENNYLANE_GATES
    from pqcqec.noise.simple_noise import PennylaneNoisyGates
    from pqcqec.circuits.modify import pennylane_state_embedding
except (ImportError, NameError):
    print("Warning: Could not import project-specific modules. Using placeholder PENNYLANE_GATES.")
    # Define a placeholder if the imports fail, allowing the script to be runnable standalone.
    PENNYLANE_GATES = {
        'x': qml.PauliX, 'y': qml.PauliY, 'z': qml.PauliZ,
        'h': qml.Hadamard, 's': qml.S, 't': qml.T,
        'rx': qml.RX, 'ry': qml.RY, 'rz': qml.RZ,
        'cx': qml.CNOT, 'cz': qml.CZ
    }


# --- Configuration ---
PQC_GATES = ['rz', 'rx', 'rz']
DATA_PATH = 'nogit/circuit_tokens/no_uncomp/5q_500g_circuit_data/'
GOOD_DATA_PATH = DATA_PATH + 'per_seed_data/'
BAD_DATA_PATH = DATA_PATH + 'poor_fidelity/'
CONFIG_PATH = DATA_PATH + 'config.json'

# Load config with error handling
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print(f"Warning: Config file not found at {CONFIG_PATH}. Using default values.")
    CONFIG = {}

NUM_QUBITS = CONFIG.get("qubits", [5])[0]
NUM_GATES = 500
GATE_BLOCKS = CONFIG.get("gate_blocks", 4)
try:
    VALID_GATES = list(CONFIG.get("gate_dist", {}).keys())
except Exception as e:
    print(f"Error loading valid gates: {e}")
    VALID_GATES = ['x', 'z', 'h', 'cx', 'cz']

print(f"Using {NUM_QUBITS} qubits, {NUM_GATES} gates, {GATE_BLOCKS} gate blocks, {VALID_GATES} valid gates.")

NOISE_DIST = {"x_rad": 0.01, "z_rad": 0.01, "delta_x": 0, "delta_z": 0}
MAX_CIRCUITS = 1000

# Add the @profile decorator for line-by-line analysis with kernprof
@profile
def simple_circuit_generator(circuit_ops, num_qubits, x_noise, z_noise):
    """
    Generates a PennyLane QNode for a given quantum circuit.
    This function is decorated for line-profiling.
    """
    qdevice = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(qdevice)
    def circuit(state):
        qml.StatePrep(state, wires=range(num_qubits), normalize=True)
        for i, op in enumerate(circuit_ops):
            gate, wires, param = op
            PENNYLANE_GATES[gate](wires=wires) # Assuming param is not used if None
            for wire in wires:
                qml.RX(x_noise[i], wires=[wire])
                qml.RZ(z_noise[i], wires=[wire])
        return qml.state()

    return circuit

# Add the @profile decorator to the main execution logic as well
@profile
def main():
    """Main execution script with integrated profiling."""
    
    os.makedirs("profs", exist_ok=True)

    # --- Data Loading ---
    good_data = []
    if os.path.exists(GOOD_DATA_PATH):
        for i, filename in enumerate(os.listdir(GOOD_DATA_PATH)):
            if i >= 10000: # Limit data loading for speed
                break
            try:
                with open(os.path.join(GOOD_DATA_PATH, filename), 'r') as f:
                    token_dict = json.load(f)
                    good_data.append(token_dict['base_circuit_tokens'])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping file {filename} due to error: {e}")
    else:
        print(f"Warning: Data path not found: {GOOD_DATA_PATH}. Proceeding with no data.")
        # Create dummy data if path doesn't exist, to make script runnable
        good_data = [[['h', [0], None], ['cx', [0, 1], None]] for _ in range(MAX_CIRCUITS)]


    print(f"Number of good data samples: {len(good_data)}")

    # --- Initial State and Noise Setup ---
    input_states = np.zeros((100, 2**NUM_QUBITS), dtype=np.complex128)
    input_states[:, 0] = 1.0
    x_noise = np.ones(NUM_GATES) * 0.01
    z_noise = np.ones(NUM_GATES) * 0.01

    # --- Profiling Circuit Creation ---
    circuit_nodes = []
    with performance_profile("Phase 1: Circuit Creation", dump_file="profs/exec_profile_creation.prof"):
        # Limit the loop to MAX_CIRCUITS to match execution phase
        for data in tqdm(good_data[:MAX_CIRCUITS], desc="Creating circuits"):
            exec_node = simple_circuit_generator(data, NUM_QUBITS, x_noise, z_noise)
            circuit_nodes.append(exec_node)

    if not circuit_nodes:
        print("No circuits were created. Exiting.")
        return

    # --- Profiling Circuit Compilation/Warm-up ---
    print('Running circuit exec nodes with 1 state to set up compiled nodes...')
    with performance_profile("Phase 2: JIT Compilation / Warm-up", dump_file="profs/exec_profile_warmup.prof"):
        for exec_node in tqdm(circuit_nodes, desc="Warming up"):
            _ = exec_node(input_states[0])

    # --- Profiling Main Circuit Execution ---
    print('Running circuit exec nodes with all 100 states...')
    with performance_profile("Phase 3: Main Execution (100 States)", dump_file="profs/exec_profile_main_exec.prof"):
        for exec_node in tqdm(circuit_nodes, desc="Executing all states"):
            _ = exec_node(input_states)

    # --- Specific Timing Checks (as in original script) ---
    print("\n--- Specific Single-Node Timings ---")
    start_time = time.time()
    circuit_nodes[0](input_states)
    end_time = time.time()
    print(f"Execution time for one node (100 states): {end_time - start_time:.4f} seconds")

    start_time = time.time()
    circuit_nodes[0](input_states[:10])
    end_time = time.time()
    print(f"Execution time for one node (10 states): {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()