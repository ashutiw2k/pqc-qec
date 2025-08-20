from qiskit import QuantumCircuit
import pennylane as qml

def tokenize_qiskit_circuit(circuit:QuantumCircuit) -> list:
    """
    Tokenizes a Qiskit circuit into a list of gate names and qubit indices.
    Returns a list of tokens, where each token is a tuple (gate_name, qubits, params).
    """
    tokens = []
    for instruction in circuit.data:
        gate_name = instruction.operation.name
        params = instruction.operation.params
        qubits = [qubit._index for qubit in instruction.qubits]
        tokens.append((gate_name, qubits, params))
    return tokens

def pennylane_state_embedding(input_state, num_qubits):
    """Prepares an arbitrary state as input to the circuit."""
    qml.StatePrep(input_state, wires=range(num_qubits), normalize=True, id='arbitrary_state_prep')
