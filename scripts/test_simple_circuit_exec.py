import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
import numpy as np
import json
import time

import pennylane as qml

import os
import sys
from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))  # Makes other scripts and functions discoverable

from pqcqec.utils.constants import QUBITS_FOR_GATES, QISKIT_GATES, GATE_IS_DIRECTIONAL, PENNYLANE_GATES
from pqcqec.noise.simple_noise import PennylaneNoisyGates
# from pqcqec.simulate.simulate import run_circuit_with_noise_model
from pqcqec.circuits.modify import pennylane_state_embedding


PQC_GATES = ['rz', 'rx', 'rz']
DATA_PATH = 'nogit/circuit_tokens/no_uncomp/5q_500g_circuit_data/'
GOOD_DATA_PATH = DATA_PATH + 'per_seed_data/'
BAD_DATA_PATH = DATA_PATH + 'poor_fidelity/'
CONFIG_PATH = DATA_PATH + 'config.json'

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)


NUM_QUBITS = CONFIG.get("qubits", 3)[0]
# NUM_GATES = CONFIG.get("gates", 4)[0] # Multiply by 2 for uncomp gates. 
NUM_GATES = 500
GATE_BLOCKS = CONFIG.get("gate_blocks", 4)
VALID_GATES = CONFIG.get("gate_dist", QISKIT_GATES)
if VALID_GATES:
    VALID_GATES = list(VALID_GATES.keys())
else:
    VALID_GATES = ['x', 'z', 'h', 'cx', 'cz']

print(f"Using {NUM_QUBITS} qubits, {NUM_GATES} gates, {GATE_BLOCKS} gate blocks, {VALID_GATES} valid gates.")

NOISE_DIST = {"x_rad": 0.01, "z_rad": 0.01, "delta_x": 0, "delta_z": 0}

PAD_ID = 0
UNDIRECTED_GATES = [gate for gate in VALID_GATES if not GATE_IS_DIRECTIONAL.get(gate, False)]
print(UNDIRECTED_GATES)

TRAIN_SZ = 0.8
VAL_SZ = 0.1
TEST_SZ = 0.1
BATCH_SIZE = 64

LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-3

MAX_CIRCUITS = 1000




# def simple_circuit_simulator(circuit_ops, input_state, num_qubits, x_noise, z_noise):
#     """Runs a quantum circuit with a noise model using PennyLane and PyTorch."""
#     qdevice = qml.device("default.qubit", wires=num_qubits)
#     input_state_time_arr = []
#     circuit_loop_time_arr = []
#     output_state_time_arr = []


#     @qml.qnode(qdevice)
#     def circuit(state):
#         # pennylane_state_embedding(state, num_qubits)
#         input_state_time = time.time()
#         qml.StatePrep(state, wires=range(num_qubits), normalize=True, id='arbitrary_state_prep')
#         input_state_time = time.time() - input_state_time
#         input_state_time_arr.append(input_state_time)


#         circuit_loop_time = time.time()
#         for i, op in enumerate(circuit_ops):
#             gate, wires, param = op
#             # noise_model.apply_gate(gate, wires, angle=param)
#             PENNYLANE_GATES[gate](wires=wires)
#             for wire in wires:
#                 qml.RX(x_noise[i], wires=[wire])  # Example noise application
#                 qml.RZ(z_noise[i], wires=[wire])  # Example noise application

#         circuit_loop_time = time.time() - circuit_loop_time
#         circuit_loop_time_arr.append(circuit_loop_time)

#         output_state_time = time.time()
#         output = qml.state()
#         output_state_time = time.time() - output_state_time
#         output_state_time_arr.append(output_state_time)

#         return output

#     # The function now directly returns the output of the torch interface qnode
#     return circuit(input_state), input_state_time_arr, circuit_loop_time_arr, output_state_time_arr


def simple_circuit_generator(circuit_ops, num_qubits, x_noise, z_noise):
    """Runs a quantum circuit with a noise model using PennyLane and PyTorch."""
    qdevice = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(qdevice)
    def circuit(state):
        # pennylane_state_embedding(state, num_qubits)
        qml.StatePrep(state, wires=range(num_qubits), normalize=True, id='arbitrary_state_prep')
        for i, op in enumerate(circuit_ops):
            gate, wires, param = op
            # noise_model.apply_gate(gate, wires, angle=param)
            PENNYLANE_GATES[gate](wires=wires)
            for wire in wires:
                qml.RX(x_noise[i], wires=[wire])  # Example noise application
                qml.RZ(z_noise[i], wires=[wire])  # Example noise application

        return qml.state()

    # The function now directly returns the interface qnode
    return circuit


def main():

    good_data = []

    for i, filename in enumerate(os.listdir(GOOD_DATA_PATH)):
        if i > 10000:
            break
        with open(GOOD_DATA_PATH + filename, 'r') as f:
            token_dict = json.load(f)
            good_data.append(token_dict['base_circuit_tokens'])
            f.close()

    print(f"Number of good data samples: {len(good_data)}")

    input_states = np.zeros((100, 2**NUM_QUBITS))
    input_states[:,0] = 1.0
    x_noise = np.ones(NUM_GATES) * 0.01
    z_noise = np.ones(NUM_GATES) * 0.01


    circuit_nodes = []
    start_time = time.time()
    for data in tqdm(good_data):
        exec_node = simple_circuit_generator(data, NUM_QUBITS, x_noise, z_noise)
        circuit_nodes.append(exec_node)
    end_time = time.time()
    print(f"Execution time - circuit creation: {end_time - start_time} seconds")


    print(f'Running circuit exec nodes with 1 state to set up compiled nodes...')
    for exec_node in tqdm(circuit_nodes[:MAX_CIRCUITS]):
        output_state = exec_node(input_states[0])

    print(f'Running circuit exec nodes with all 100 states...')

    for exec_node in tqdm(circuit_nodes[:MAX_CIRCUITS]):
        output_state = exec_node(input_states)


    start_time = time.time()
    circuit_nodes[0](input_states)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time - 100 states: {execution_time} seconds")


    start_time = time.time()
    circuit_nodes[0](input_states[:10])
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time - 10 states: {execution_time} seconds")

if __name__ == "__main__":
    main()