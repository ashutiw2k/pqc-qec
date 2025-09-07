import jax
from typing import List, Tuple, Dict, Optional
import jax.numpy as jnp
import pennylane as qml

from ..noise.simple_noise import PennylaneNoisyGates

def pennylane_PQC_RZRXRZ_unique(num_qubits:int, params:jnp.ndarray):

    for i in range(num_qubits):
        qml.RZ(params[3*i], i, id='PQC')
        qml.RX(params[3*i + 1], i, id='PQC')
        qml.RZ(params[3*i + 2], i, id='PQC')


# ==============================================================================
# STEP 1: Implement robust, non-placeholder helper functions
# ==============================================================================

# This dictionary maps gate strings to unique integers.
GATE_MAP: Dict[str, int] = {
    'rx': 0, 'ry': 1, 'rz': 2, 'h': 3, 'x': 4,
    'z': 5, 'cx': 6, 'cz': 7
}

def numerically_encode_circuit(
    tokenized_circuit: List[Tuple], num_gates: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Converts a tokenized circuit into three numerical JAX arrays."""
    gate_types, qubit_indices, gate_params = [], [], []
    for gate_name, qubits, params in tokenized_circuit:
        gate_types.append(GATE_MAP[gate_name])
        q_indices = list(qubits) + [-1] * (2 - len(qubits))
        qubit_indices.append(q_indices[:2])
        gate_params.append(params[0] if params else 0.0)

    if len(gate_types) != num_gates:
        raise ValueError(f"Circuit length mismatch: expected {num_gates}, got {len(gate_types)}")

    return (
        jnp.array(gate_types, dtype=jnp.int32),
        jnp.array(qubit_indices, dtype=jnp.int32),
        jnp.array(gate_params, dtype=jnp.float32)
    )


# ==============================================================================
# STEP 2: The static circuit factory (now with noise)
# ==============================================================================
def create_static_pqc_circuit(
    num_qubits: int, num_gates: int, gate_blocks: int,
    pqc_gate_names: List[str], noise_model: 'PennylaneNoisyGates'
):
    """
    Creates and JIT-compiles a static QNode that applies the provided noise model.
    This version uses jax.lax.switch and correct wire slicing to be fully JIT-compatible.
    """
    device = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(device, interface='jax', diff_method="backprop")
    def static_circuit_executor(
        state: jnp.ndarray, pqc_params: jnp.ndarray, gate_types: jnp.ndarray,
        qubit_indices: jnp.ndarray, gate_params: jnp.ndarray
    ):
        qml.StatePrep(state, wires=range(num_qubits))

        for i in range(num_gates):
            # This helper function now uses jax.lax.switch and passes wires as JAX array slices.
            def apply_noisy_gate(gate_id):
                # --- THIS IS THE FIX ---
                # The `wires` argument is now a JAX array slice (e.g., qubit_indices[i, 0:1]),
                # not a Python list, which makes it fully compatible with JAX tracing.
                branches = [
                    lambda: noise_model.apply_gate('rx', qubit_indices[i, 0:1], angle=gate_params[i]),
                    lambda: noise_model.apply_gate('ry', qubit_indices[i, 0:1], angle=gate_params[i]),
                    lambda: noise_model.apply_gate('rz', qubit_indices[i, 0:1], angle=gate_params[i]),
                    lambda: noise_model.apply_gate('h',  qubit_indices[i, 0:1]),
                    lambda: noise_model.apply_gate('x',  qubit_indices[i, 0:1]),
                    lambda: noise_model.apply_gate('z',  qubit_indices[i, 0:1]),
                    lambda: noise_model.apply_gate('cx', qubit_indices[i, 0:2]), # Slice to get both qubits
                    lambda: noise_model.apply_gate('cz', qubit_indices[i, 0:2]), # Slice to get both qubits
                ]
                jax.lax.switch(gate_id, branches)
            
            apply_noisy_gate(gate_types[i])

            # Apply interleaved PQC blocks
            if (i + 1) % gate_blocks == 0:
                pqc_params_block = pqc_params[i // gate_blocks]
                for qubit in range(num_qubits):
                    for j, pqc_gate_name in enumerate(pqc_gate_names):
                        angle = pqc_params_block[qubit, j]
                        # This call must also pass wires as a list or array
                        noise_model.apply_gate(pqc_gate_name, [qubit], angle=angle)

        return qml.state()

    batched_fn = jax.vmap(static_circuit_executor, in_axes=(0, None, None, None, None))
    return jax.jit(batched_fn, donate_argnums=(0,))
