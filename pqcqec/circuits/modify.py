from qiskit import QuantumCircuit
import pennylane as qml
import torch

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


def interleave_tensor_pqc_in_circuit_torch(base_ops: list, qubits: int, blocks: int,
                                           pqc_gates: list, params: torch.Tensor):
    """Interleaves PQC operations into a base circuit using PyTorch tensors."""
    interleaved_circuit = []
    param_idx = 0
    for i, op in enumerate(base_ops):
        interleaved_circuit.append(op)
        if (i + 1) % blocks == 0:
            k = (i + 1) // blocks - 1
            for q in range(qubits):
                for j, g in enumerate(pqc_gates):
                    # Indexing the PyTorch tensor directly
                    interleaved_circuit.append((g, [q], params[k, q, j]))
    return interleaved_circuit
