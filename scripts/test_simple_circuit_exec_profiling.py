# ==============================================================================
# PROFILING AND UTILITIES
# ==============================================================================
import cProfile
import pstats
import io
import time
from contextlib import contextmanager
from typing import Generator, Optional

# --- PROFILER TOGGLE ---
# Set this to True when you want to use line_profiler with `kernprof`.
# Set it to False to use the cProfile context manager.
USE_LINE_PROFILER = True # <-- CHANGE THIS FLAG TO SWITCH PROFILERS

# Set up line_profiler
try:
    from line_profiler import profile
except ImportError:
    # If not using kernprof, create a dummy decorator
    def profile(func):
        return func

@contextmanager
def performance_profile(
    description: str = "Execution Profile",
    sort_by: str = 'cumulative',
    print_stats: int = 25,
    dump_file: Optional[str] = None
) -> Generator[Optional[cProfile.Profile], None, None]:
    """
    A flexible context manager for profiling with cProfile.
    This profiler is AUTOMATICALLY DISABLED if USE_LINE_PROFILER is True.
    """
    # If the line_profiler is active, this context manager does nothing.
    if USE_LINE_PROFILER:
        yield None
        return

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
# ORIGINAL SCRIPT LOGIC (Unchanged from previous version)
# ==============================================================================
import numpy as np
import json
import pennylane as qml
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from pqcqec.utils.constants import PENNYLANE_GATES
except (ImportError, NameError):
    print("Warning: Could not import project-specific modules. Using placeholder PENNYLANE_GATES.")
    PENNYLANE_GATES = {
        'x': qml.PauliX, 'y': qml.PauliY, 'z': qml.PauliZ,
        'h': qml.Hadamard, 's': qml.S, 't': qml.T,
        'rx': qml.RX, 'ry': qml.RY, 'rz': qml.RZ,
        'cx': qml.CNOT, 'cz': qml.CZ
    }

# --- Configuration ---
DATA_PATH = 'nogit/circuit_tokens/no_uncomp/5q_500g_circuit_data/'
GOOD_DATA_PATH = DATA_PATH + 'per_seed_data/'
CONFIG_PATH = DATA_PATH + 'config.json'

try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print(f"Warning: Config file not found at {CONFIG_PATH}. Using default values.")
    CONFIG = {}

NUM_QUBITS = CONFIG.get("qubits", [5])[0]
NUM_GATES = 500
MAX_CIRCUITS = 100


@profile
def simple_circuit_generator(circuit_ops, num_qubits, x_noise, z_noise):
    """Generates a PennyLane QNode for a given quantum circuit."""
    qdevice = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(qdevice)
    def circuit(state):
        @profile
        def prepare_state():
            qml.StatePrep(state, wires=range(num_qubits), normalize=True)

            for i, op in enumerate(circuit_ops):
                gate, wires, param = op
                # PENNYLANE_GATES[gate](wires=wires)

                if gate == 'cx':
                    qml.CNOT(wires=wires)
                elif gate == 'cz':
                    qml.CZ(wires=wires)
                elif gate == 'x':
                    qml.PauliX(wires=wires)
                elif gate == 'z':
                    qml.PauliZ(wires=wires)
                elif gate == 'h':
                    qml.Hadamard(wires=wires)

                for wire in wires:
                    qml.RX(x_noise[i], wires=[wire])
                    qml.RZ(z_noise[i], wires=[wire])
            return qml.state()
        return prepare_state()
    return circuit

@profile
def main():
    """Main execution script."""
    good_data = []
    if os.path.exists(GOOD_DATA_PATH):
        for i, filename in enumerate(os.listdir(GOOD_DATA_PATH)):
            if i >= 10000: break
            with open(os.path.join(GOOD_DATA_PATH, filename), 'r') as f:
                good_data.append(json.load(f)['base_circuit_tokens'])
    else:
        print(f"Warning: Data path not found: {GOOD_DATA_PATH}.")
        good_data = [[['h', [0], None], ['cx', [0, 1], None]] for _ in range(MAX_CIRCUITS)]

    print(f"Number of good data samples: {len(good_data)}")

    input_states = np.zeros((100, 2**NUM_QUBITS), dtype=np.complex128)
    input_states[:, 0] = 1.0
    x_noise = np.ones(NUM_GATES) * 0.01
    z_noise = np.ones(NUM_GATES) * 0.01

    circuit_nodes = []
    with performance_profile("Phase 1: Circuit Creation"):
        for data in tqdm(good_data[:MAX_CIRCUITS], desc="Creating circuits"):
            circuit_nodes.append(simple_circuit_generator(data, NUM_QUBITS, x_noise, z_noise))
    
    if not circuit_nodes:
        print("No circuits created. Exiting.")
        return
        
    # with performance_profile("Phase 2: JIT Compilation / Warm-up"):
    #     for exec_node in tqdm(circuit_nodes, desc="Warming up"):
    #         _ = exec_node(input_states[0])

    with performance_profile("Phase 3: Main Execution (100 States)"):
        for exec_node in tqdm(circuit_nodes, desc="Executing all states"):
            _ = exec_node(input_states)
            
    # print("\n--- Specific Single-Node Timings ---")
    # start_time = time.time()
    # circuit_nodes[0](input_states)
    # print(f"Execution time for one node (100 states): {time.time() - start_time:.4f} seconds")

    # start_time = time.time()
    # circuit_nodes[0](input_states[:10])
    # print(f"Execution time for one node (10 states): {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    main()