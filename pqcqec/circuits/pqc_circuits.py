import jax.numpy as jnp
import pennylane as qml
import numpy as np

def pennylane_PQC_RZRXRZ_unique(num_qubits:int, params:jnp.ndarray):

    for i in range(num_qubits):
        qml.RZ(params[3*i], i, id='PQC')
        qml.RX(params[3*i + 1], i, id='PQC')
        qml.RZ(params[3*i + 2], i, id='PQC')


def list_LEL_ZZ(num_qubits:int, pre_params:np.ndarray, theta_zz: np.ndarray, post_params:np.ndarray):

    pqc_circuit_ops = []

    # Pre-local unitaries
    for i in range(num_qubits):
        pqc_circuit_ops.append(('rz', [i], [pre_params[i][0]]))
        pqc_circuit_ops.append(('rx', [i], [pre_params[i][1]]))
        pqc_circuit_ops.append(('rz', [i], [pre_params[i][2]]))

    # ZZ entangling gates
    for i in range(num_qubits):
        j = (i + 1) % num_qubits  # Assuming a ring topology
        # RZZ(θ) = CNOT-RZ(θ)-CNOT
        pqc_circuit_ops.append(('cnot', [i, j], []))
        pqc_circuit_ops.append(('rz', [j], [theta_zz[i]] ))
        pqc_circuit_ops.append(('cnot', [i, j], []))

    # Post-local unitaries
    for i in range(num_qubits):
        pqc_circuit_ops.append(('rz', [i], [post_params[i][0]]))
        pqc_circuit_ops.append(('rx', [i], [post_params[i][1]]))
        pqc_circuit_ops.append(('rz', [i], [post_params[i][2]]))

    return pqc_circuit_ops