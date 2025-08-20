import jax.numpy as jnp
import pennylane as qml
from qiskit import QuantumCircuit

from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit, pennylane_state_embedding
from pqcqec.circuits.pqc_circuits import pennylane_PQC_RZRXRZ_unique


def test_generate_random_circuit_list_backend_deterministic_and_shape():
    num_qubits, num_gates = 3, 5
    gate_dist = {"x": 1.0}
    ops1 = generate_random_circuit(num_qubits, num_gates, gate_dist=gate_dist, seed=42, backend="list")
    ops2 = generate_random_circuit(num_qubits, num_gates, gate_dist=gate_dist, seed=42, backend="list")
    assert ops1 == ops2
    assert len(ops1) == num_gates
    for gate, wires in ops1:
        assert gate == "x"
        assert all(0 <= w < num_qubits for w in wires)


def test_generate_random_circuit_qiskit_backend_gatecount():
    num_qubits, num_gates = 2, 4
    gate_dist = {"x": 1.0}
    qc = generate_random_circuit(num_qubits, num_gates, gate_dist=gate_dist, seed=0, backend="qiskit")
    assert isinstance(qc, QuantumCircuit)
    assert len(qc.data) == num_gates
    assert all(instr.operation.name == "x" for instr in qc.data)


def test_generate_random_circuit_pennylane_backend_types():
    num_qubits, num_gates = 2, 3
    gate_dist = {"x": 1.0}
    ops = generate_random_circuit(num_qubits, num_gates, gate_dist=gate_dist, seed=0, backend="pennylane")
    assert len(ops) == num_gates
    # Generated items should be Operation instances with 1 wire
    for op in ops:
        assert isinstance(op, qml.operation.Operation)
        assert len(op.wires) == 1


def test_tokenize_qiskit_circuit_roundtrip_simple():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.cx(0, 1)
    tokens = tokenize_qiskit_circuit(qc)
    assert tokens[0][0] == "x" and tokens[0][1] == [0]
    assert tokens[1][0] in ("cx", "cnot")
    # 2-qubit op wires should include both qubits
    assert set(tokens[1][1]) == {0, 1}


def test_pennylane_state_embedding_and_pqc_layer():
    dev = qml.device("default.qubit", wires=1)
    params = jnp.array([0.1, 0.2, 0.3])

    @qml.qnode(dev, interface="jax")
    def circuit(state):
        pennylane_state_embedding(state, num_qubits=1)
        pennylane_PQC_RZRXRZ_unique(1, params)
        return qml.state()

    # Start in |1> and ensure state is normalized after applying PQC
    input_state = jnp.array([0.0 + 0.0j, 1.0 + 0.0j])
    out = circuit(input_state)
    norm = jnp.linalg.norm(out)
    assert jnp.isclose(norm, 1.0, atol=1e-6)

