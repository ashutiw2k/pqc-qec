# from circuits.pqc_circuits import *
# from models.pqc_models import *

import pennylane as qml
from qiskit.circuit.library import XGate, HGate, ZGate, YGate, RXGate, RYGate, RZGate, CXGate, CZGate, CCXGate, SwapGate
import enum

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



QISKIT_GATES = {'x':XGate, 'h':HGate, 'z':ZGate, 'y':YGate,
                'rx': RXGate, 'ry': RYGate, 'rz': RZGate,
                'cx': CXGate, 'cz': CZGate, 'swap': SwapGate, 'ccx': CCXGate}

PENNYLANE_GATES = {'x':qml.PauliX, 'h':qml.Hadamard, 'z':qml.PauliZ, 'cx': qml.CNOT, 'cz': qml.CZ}

QUBITS_FOR_GATES = {'x':1, 'h':1, 'z':1, 'y':1,
                    'rx': 1, 'ry': 1, 'rz': 1,
                    'cx': 2, 'cy': 2, 'cz':2,
                    'ccx': 3}

GATE_IS_DIRECTIONAL = {'cx': True, 'cz': False, 'ccx': True, 'swap': False}  # True means control->target matters, False means it doesn't


# Gate ENUMS
class GateEnums(enum.IntEnum):
    GATE_X  = enum.auto()
    GATE_Z  = enum.auto()
    GATE_H  = enum.auto()
    GATE_RX = enum.auto()
    GATE_RY = enum.auto()
    GATE_RZ = enum.auto()
    GATE_CX = enum.auto()
    GATE_CZ = enum.auto()


GATE_DICT = {
    'x': GateEnums.GATE_X,
    'z': GateEnums.GATE_Z,
    'h': GateEnums.GATE_H,
    'rx': GateEnums.GATE_RX,
    'ry': GateEnums.GATE_RY,
    'rz': GateEnums.GATE_RZ,
    'cx': GateEnums.GATE_CX,
    'cnot': GateEnums.GATE_CX,  # alias for cx
    'cz': GateEnums.GATE_CZ
}