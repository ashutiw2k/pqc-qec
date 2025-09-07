import jax
import jax.numpy as jnp
import pennylane as qml
from typing import List, Tuple, Optional, Dict

from ..circuits.modify import pennylane_state_embedding
from ..noise.simple_noise import PennylaneNoisyGates
from ..utils.constants import PENNYLANE_GATES

def get_input_data(num_qubits: int, num_vals: int, seed: int = 0) -> jnp.ndarray:
    """
    Generate a batch of random, normalized complex state vectors.
    """
    key = jax.random.PRNGKey(seed)
    shape = (num_vals, 2**num_qubits)

    # IMPROVEMENT: Generate complex numbers directly for cleaner, more efficient code.
    state = jax.random.normal(key, shape, dtype=jnp.complex64)
    
    # Normalize the state vectors
    norms = jnp.linalg.norm(state, axis=1, keepdims=True)
    return state / norms


def create_optimized_circuit_executor(
    circuit_ops: List[Tuple],
    num_qubits: int,
    noise_model: Optional[PennylaneNoisyGates] = None,
    device: str = 'default.qubit'
) -> Dict:
    """
    A "factory" that defines, compiles, and returns circuit execution functions.
    This function should be called ONLY ONCE per unique circuit structure.

    Args:
        circuit_ops: The list of operations defining the circuit.
        num_qubits: The number of qubits.
        noise_model: An optional noise model. If None, an ideal circuit is created.
        device: The PennyLane device to use.

    Returns:
        A dictionary containing 'single' and 'batched' JIT-compiled execution functions.
    """
    dev = qml.device(device, wires=num_qubits)

    @qml.qnode(dev, interface='jax', diff_method=None)
    def circuit(input_state: jnp.ndarray) -> jnp.ndarray:
        """The static quantum circuit definition."""
        pennylane_state_embedding(input_state, num_qubits)
        
        for op in circuit_ops:
            gate_name, wires, params = op
            
            # This single block handles both noisy and ideal (noiseless) cases.
            if noise_model:
                # Use the noise model to apply the gate
                noise_model.apply_gate(gate_name, wires, angle=params)
            else:
                # Apply the ideal gate
                gate_fn = PENNYLANE_GATES[gate_name]
                # Correctly handle parameterized ideal gates.
                if params:
                    gate_fn(params[0], wires=wires)
                else:
                    gate_fn(wires=wires)

        return qml.state()

    # Compile both single and batched versions efficiently and return them.
    return {
        'single': jax.jit(circuit),
        'batched': jax.jit(jax.vmap(circuit, in_axes=(0,)))
    }


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
    

def run_ideal_circuit(circuit_ops:List, input_state:jnp.ndarray, 
                           num_qubits:int, 
                           device='default.qubit', batched=False):

    @qml.qnode(qml.device(device), interface='jax')
    def circuit(input_state):
        pennylane_state_embedding(input_state, num_qubits)
        for op in circuit_ops:
            gate, wires, param = op
            # Apply the ideal gate:
            PENNYLANE_GATES[gate](wires)
            
        return qml.state()

    batched_circuit = jax.jit(jax.vmap(circuit, in_axes=(0)))

    if batched:
        return batched_circuit(input_state)
    else:
        return circuit(input_state)
    

