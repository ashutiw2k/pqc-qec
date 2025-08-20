# from circuits.pqc_circuits import *
# from models.pqc_models import *

import pennylane as qml
from qiskit.circuit.library import XGate, HGate, ZGate, CXGate, CZGate

PQC_MAPPINGS = {
    # 'unique_rzrxrz' : {'qiskit': qiskit_PQC_RZRXRZ_unique, 'pennylane':pennylane_PQC_RZRXRZ_unique, 'mult':3},
    # 'unique_rzrx' : {'qiskit': qiskit_PQC_RZRX_unique, 'pennylane':pennylane_PQC_RZRX_unique, 'mult':2},
    # 'unique_u3' : {'qiskit': qiskit_PQC_U3_unique, 'pennylane':pennylane_PQC_U3_unique, 'mult':3},
    # 'unique_u3u3' : {'qiskit': qiskit_PQC_U3U3_unique, 'pennylane':pennylane_PQC_U3U3_unique, 'mult':6},
    # 'unique_rzrxrz_cz' : {'qiskit': qiskit_PQC_RZRXRZ_CZ_unique, 'pennylane':pennylane_PQC_RZRXRZ_CZ_unique, 'mult':3},
}


PENNYLANE_MODELS = {
    # 'simple_quantum_probs' : {'model': SimplePennylaneQuantumProbsModel, 'dataset':FidelityQASMDataset},
    # 'simple_unitary_state' : {'model': SimplePennylaneUnitaryStateModel, 'dataset':FidelityQASMDataset},
    # 'simple_circuit_op'    : {'model': SimplePennylaneCircuitOpsStateModel, 'dataset':FidelityCircuitOpsDataset},
    # 'interleave_circuit_op' : {'model': InterleavePennylaneCircuitOpsStateModel, 'dataset':FidelityCircuitOpsDataset},
}



QISKIT_GATES = {'x':XGate, 'h':HGate, 'z':ZGate, 'cx': CXGate, 'cz': CZGate}
PENNYLANE_GATES = {'x':qml.PauliX, 'h':qml.Hadamard, 'z':qml.PauliZ, 'cx': qml.CNOT, 'cz': qml.CZ}

QUBITS_FOR_GATES = {'x':1, 'h':1, 'z':1, 'cx': 2, 'cz':2}
