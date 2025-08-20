import jax
import jax.numpy as jnp
import pennylane as qml
from typing import List

from ..circuits.modify import pennylane_state_embedding
from ..noise.simple_noise import PennylaneNoisyGates

def get_input_data(num_qubits, num_vals, seed=0):
    """Generate ideal data for a Pennylane circuit with angle embedding."""

    key_real, key_imag = jax.random.split(jax.random.PRNGKey(seed))

    state =  jax.random.normal(key_real, (num_vals, 2**num_qubits,)) + 1j * jax.random.normal(key_imag, (num_vals, 2**num_qubits,))
    norms = jnp.linalg.norm(state, axis=1, keepdims=True)
    ideal_data = state / norms

    return ideal_data


def run_circuit_with_noise_model(circuit_ops:List, input_state:jnp.ndarray, 
                           noise_model:PennylaneNoisyGates, num_qubits:int, 
                           device='default.qubit', batched=False):

    @qml.qnode(qml.device(device), interface='jax')
    def circuit(input_state):
        pennylane_state_embedding(input_state, num_qubits)
        for op in circuit_ops:
            gate, wires, param = op
            # Apply the noisy gate:
            
            noise_model.apply_gate(gate, wires, angle=param)
        return qml.state()

    batched_circuit = jax.jit(jax.vmap(circuit, in_axes=(0)))

    if batched:
        return batched_circuit(input_state)
    else:
        return circuit(input_state)