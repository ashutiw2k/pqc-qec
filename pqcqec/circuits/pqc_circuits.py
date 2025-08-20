import jax.numpy as jnp
import pennylane as qml

def pennylane_PQC_RZRXRZ_unique(num_qubits:int, params:jnp.ndarray):

    for i in range(num_qubits):
        qml.RZ(params[3*i], i, id='PQC')
        qml.RX(params[3*i + 1], i, id='PQC')
        qml.RZ(params[3*i + 2], i, id='PQC')
