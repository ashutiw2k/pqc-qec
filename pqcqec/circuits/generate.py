from qiskit import QuantumCircuit
import random

from ..utils.constants import PENNYLANE_GATES, QISKIT_GATES, QUBITS_FOR_GATES


def generate_random_circuit_qiskit(num_qubits: int, num_gates: int, gate_dist:dict):
    circuit = QuantumCircuit(num_qubits)

    gates = random.choices(
        population=list(gate_dist.keys()), 
        weights=list(gate_dist.values()), 
        k=num_gates)
    
    for gate in gates:
        gate_q = QUBITS_FOR_GATES.get(gate)
        gate_f = QISKIT_GATES.get(gate)
        q = random.sample(population=range(num_qubits), k=gate_q)

        circuit.append(gate_f(), q)

    return circuit


def generate_random_circuit_pennylane(num_qubits: int, num_gates: int, gate_dist:dict):
    circuit_ops = []

    gates = random.choices(
        population=list(gate_dist.keys()), 
        weights=list(gate_dist.values()), 
        k=num_gates)
    
    for gate in gates:
        gate_q = QUBITS_FOR_GATES.get(gate)
        gate_f = PENNYLANE_GATES.get(gate)
        q = random.sample(population=range(num_qubits), k=gate_q)
        circuit_ops.append(gate_f(q))

    return circuit_ops

def generate_random_circuit_list(num_qubits: int, num_gates: int, gate_dist:dict=None):
    circuit_ops = []

    if gate_dist is None:
        gate_dist = {gate:1/len(PENNYLANE_GATES) for gate in PENNYLANE_GATES} # Uniform distribution of gates

    gates = random.choices(
        population=list(gate_dist.keys()), 
        weights=list(gate_dist.values()), 
        k=num_gates)
    
    for gate in gates:
        gate_q = QUBITS_FOR_GATES.get(gate)
        # gate_f = PENNYLANE_GATES.get(gate)
        q = random.sample(population=range(num_qubits), k=gate_q)
        circuit_ops.append((gate, q))

    return circuit_ops


def generate_random_circuit(num_qubits: int, num_gates: int, gate_dist:dict=None, seed=None, backend='qiskit'):

    if seed is not None:
        random.seed(seed)

    if gate_dist is None:
        gate_dist = {gate:1/len(QISKIT_GATES) for gate in QISKIT_GATES}

    if backend == 'qiskit':
        return generate_random_circuit_qiskit(num_qubits, num_gates, gate_dist)
    elif backend == 'pennylane':
        return generate_random_circuit_pennylane(num_qubits, num_gates, gate_dist)
    elif backend == 'list':
        return generate_random_circuit_list(num_qubits, num_gates, gate_dist)

    raise Exception(f"Backend {backend} not supported")


def create_qiskit_circuit_from_ops(ops_list, num_qubits):
    """
    Creates a Qiskit QuantumCircuit from a list of operations.

    Args:
        ops_list (list): List of operations in the format (op, params, qubits).
        num_qubits (int): Number of qubits in the circuit.

    Returns:
        QuantumCircuit: The constructed quantum circuit.
    """

    circuit = QuantumCircuit(num_qubits)
    for op, qubits, params in ops_list:
        if hasattr(circuit, op):
            getattr(circuit, op)(*params, *qubits)
        else:
            raise ValueError(f"Unsupported operation: {op}")
    return circuit


