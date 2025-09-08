import copy
import jax
import jax.numpy as jnp
import pennylane as qml

from typing import List

from ..circuits.modify import pennylane_state_embedding
from ..noise.simple_noise import PennylaneNoisyGates
from ..utils.quaternions_utils import quaternion_to_zxz_angles, quaternion_to_xzy_angles

class StateInputModelInterleavedPQCModel:
    """A class to define the PQC model."""
    
    def __init__(self, circuit_ops:List, num_qubits:int, noise_model:PennylaneNoisyGates,
                 pqc_blocks=1, gate_blocks=1, seed=0):
        """
        Initialize the PQC model with the given parameters.
        Args:
            circuit_ops (List): List of circuit operations to be applied (circuit and its inverse).
            num_qubits (int): Number of qubits in the circuit.
            noise_model (PennylaneNoisyGates): Noise model to be applied.
            pqc_blocks (int): Number of PQC blocks.
            gate_blocks (int): Number of gates per block.
            seed (int): Random seed for parameter initialization.
        """

        self.num_qubits = num_qubits
        # self.pqc_arch = pennylane_PQC_RZRXRZ_unique
        self.circuit_ops = copy.deepcopy(circuit_ops)   
        self.pqc_blocks = pqc_blocks
        self.gate_blocks = gate_blocks
        self.seed = seed
        self.noise_model = noise_model
        # self.uncomp_circuit = circuit_ops + circuit_ops[::-1]        # self.uncomp_circuit.extend([qml.adjoint(op) for op in self.circuit_ops[::-1]])
        self.num_gates = len(self.circuit_ops)

        self.qdev_cpu = qml.device("default.qubit", wires=self.num_qubits)
        self.diff_method = "backprop"  # Use backpropagation for differentiation

        self.pqc_gates = ['rz', 'rx', 'rz']
        self.num_pqc_angles = len(self.pqc_gates)

        self.param_sz = (int(self.pqc_blocks * jnp.ceil(self.num_gates/self.gate_blocks)), self.num_qubits, self.num_pqc_angles)

        # self.pqc_params = jax.random.uniform(jax.random.PRNGKey(self.seed), self.param_sz, jnp.float32, -jnp.pi, jnp.pi)
        self.pqc_params = jnp.zeros(self.param_sz, dtype=jnp.float32)
        # self.pqc_params = pnp.array(init_params, requires_grad=True, dtype=jnp.float32)


        
        @qml.qnode(self.qdev_cpu, interface='jax', diff_method=self.diff_method)
        def model_circuit(state, pqc_params):
            """Define the PQC model circuit."""
            # 1) Apply state embedding:
            pennylane_state_embedding(state, self.num_qubits)

            # @qml.for_loop(0, self.num_gates)
            for i, op in enumerate(self.circuit_ops):
            # def loop_body(i):
                gate, qubit, param = op
                # Apply the noisy gate:
                # if not param:
                self.noise_model.apply_gate(gate, qubit, angle=param)

                # Apply PQC to the qubit:
                if (i+1) % self.gate_blocks == 0:
                    # 2) Apply the PQC gates:
                    # print(f"Applying PQC block {i // self.gate_blocks + 1} with params: {pqc_params[i // self.gate_blocks]}")
                    pqc_params_block = pqc_params[i // self.gate_blocks]
                    # self.pqc_arch(self.num_qubits, pqc_params_block)
                    for qubit in range(self.num_qubits):
                        for j, pqc in enumerate(self.pqc_gates):
                            self.noise_model.apply_gate(pqc, qubit, angle=pqc_params_block[qubit, j])

            # 3) Return the output state:
            return qml.state()
        

        self.model_circuit = model_circuit
        self.batched_model_circuit = jax.jit(jax.vmap(self.model_circuit, in_axes=(0, None)))

    def get_model_params(self):
        """Get the model parameters."""
        return self.pqc_params
    
    def set_model_params(self, new_params):
        """Set the model parameters."""
        self.pqc_params = new_params

    def get_pqc_params(self):
        return self.pqc_params

    def run_model_batch(self, in_state, params=None):
        """Run the model circuit on the BATCHED parameters and return the output state."""
        if params is None:
            params = self.pqc_params
        return self.batched_model_circuit(in_state, params)

    def __call__(self, *args, **kwds):
        return self.run_model_batch(*args, **kwds)
    
    def __str__(self):
        return str(self.circuit_ops)

    def draw_mpl(self, in_state, params=None):
        """Draw the model circuit using matplotlib."""
 
        if params is None:
            params = self.pqc_params

        print(f"Drawing circuit with params: {params}")
        print(f"Input state: {in_state}")
        print(f'Model: {self}')

        return qml.draw_mpl(self.model_circuit, decimals=4)(in_state, params)

    def get_circuit_tokens(self):
        """Get the circuit tokens."""
        tokens = []
        for i, op in enumerate(self.circuit_ops):
        # def loop_body(i):
            tokens.append(op)
            
            if (i+1) % self.gate_blocks == 0:
                # 2) Apply the PQC gates:
                # print(f"Applying PQC block {i // self.gate_blocks + 1} with params: {pqc_params[i // self.gate_blocks]}")
                pqc_params_block = self.pqc_params[i // self.gate_blocks]
                # Add PQC parameters to the tokens:
                for qubit in range(self.num_qubits):
                    for j, pqc in enumerate(self.pqc_gates):
                        tokens.append((pqc, [qubit], [pqc_params_block[qubit, j].item()]))

        # 3) Return the circuit tokens with PQC params:
        return tokens
    



class StateInputModelInterleavedQuaternionModel:
    """A class to define the Quaternion PQC model."""
    
    def __init__(self, circuit_ops:List, num_qubits:int, noise_model:PennylaneNoisyGates,
                 pqc_blocks=1, gate_blocks=1, seed=0, pqc_type='zxz'):
        """
        Initialize the PQC model with the given parameters.
        Args:
            circuit_ops (List): List of circuit operations to be applied (circuit and its inverse).
            num_qubits (int): Number of qubits in the circuit.
            noise_model (PennylaneNoisyGates): Noise model to be applied.
            pqc_blocks (int): Number of PQC blocks.
            gate_blocks (int): Number of gates per block.
            seed (int): Random seed for parameter initialization.
        """

        self.num_qubits = num_qubits
        # self.pqc_arch = pennylane_PQC_RZRXRZ_unique
        self.circuit_ops = copy.deepcopy(circuit_ops)   
        self.pqc_blocks = pqc_blocks
        self.gate_blocks = gate_blocks
        self.seed = seed
        self.noise_model = noise_model
        # self.uncomp_circuit = circuit_ops + circuit_ops[::-1]        # self.uncomp_circuit.extend([qml.adjoint(op) for op in self.circuit_ops[::-1]])
        self.num_gates = len(self.circuit_ops)

        self.qdev_cpu = qml.device("default.qubit", wires=self.num_qubits)
        self.diff_method = "backprop"  # Use backpropagation for differentiation

        # self.pqc_gates = ['rz', 'rx', 'rz']
        # self.pqc_gates = ['rx', 'rz', 'ry']

        if pqc_type == 'zxz':
            self.pqc_gates = ['rz', 'rx', 'rz']
            self.quaternion_to_pqc_angles_fn = quaternion_to_zxz_angles
        elif pqc_type == 'xzy':
            self.pqc_gates = ['rx', 'rz', 'ry']
            self.quaternion_to_pqc_angles_fn = quaternion_to_xzy_angles

        self.num_quaternion_values = 4

        self.param_sz = (int(self.pqc_blocks * jnp.ceil(self.num_gates/self.gate_blocks)), self.num_qubits, self.num_quaternion_values)

        # Initialize unit quaternions with a moderate random rotation to avoid
        # ZXZ gimbal-lock singularities (β ≈ 0 or π) that can yield NaN gradients.
        # Sample a random axis u and an angle a ∈ [a_min, a_max], then form
        # q = [cos(a/2), u*sin(a/2)].
        key = jax.random.PRNGKey(self.seed)
        key_axis, key_angle = jax.random.split(key)
        axes = jax.random.normal(key_axis, self.param_sz[:-1] + (3,), dtype=jnp.float32)
        axes = axes / (jnp.linalg.norm(axes, axis=-1, keepdims=True) + 1e-12)
        a_min, a_max = 0.2, 0.8  # radians
        angles = jax.random.uniform(key_angle, self.param_sz[:-1] + (1,), dtype=jnp.float32,
                                    minval=a_min, maxval=a_max)
        w = jnp.cos(0.5 * angles)
        v = axes * jnp.sin(0.5 * angles)
        self.quaternions = jnp.concatenate([w, v], axis=-1).astype(jnp.float32)
        # print(self.quaternions)
        # self.pqc_params = pnp.array(init_params, requires_grad=True, dtype=jnp.float32)
        


        
        @qml.qnode(self.qdev_cpu, interface='jax', diff_method=self.diff_method)
        def model_circuit(state, pqc_params):
            """Define the PQC model circuit."""
            # 1) Apply state embedding:
            pennylane_state_embedding(state, self.num_qubits)

            # @qml.for_loop(0, self.num_gates)
            for i, op in enumerate(self.circuit_ops):
            # def loop_body(i):
                gate, qubit, param = op
                # Apply the noisy gate:
                # if not param:
                self.noise_model.apply_gate(gate, qubit, angle=param)

                # Apply PQC to the qubit:
                if (i+1) % self.gate_blocks == 0:
                    # 2) Apply the PQC gates:
                    # print(f"Applying PQC block {i // self.gate_blocks + 1} with params: {pqc_params[i // self.gate_blocks]}")
                    # print(f'PQC Params Block Shape for {i+1} : {pqc_params.shape}')

                    pqc_params_block = pqc_params[i // self.gate_blocks]
                    # self.pqc_arch(self.num_qubits, pqc_params_block)
                    for qubit in range(self.num_qubits):
                        for j, pqc in enumerate(self.pqc_gates):
                            self.noise_model.apply_gate(pqc, qubit, angle=pqc_params_block[qubit, j])

            # 3) Return the output state:
            return qml.state()
        

        self.model_circuit = model_circuit
        self.batched_model_circuit = jax.jit(jax.vmap(self.model_circuit, in_axes=(0, None)))

    def get_model_params(self):
        """Get the model parameters."""
        return self.quaternions

    def set_model_params(self, new_params: jnp.ndarray):
        """Set the model parameters."""
        if jnp.isnan(new_params).any():
            nan_indices = jnp.argwhere(jnp.isnan(new_params))
            raise ValueError(
                f"New parameters contain NaNs. Shape: {new_params.shape}. "
                f"NaN indices: {nan_indices.tolist()}"
            )
        self.quaternions = new_params.astype(jnp.float32)

    def get_pqc_params(self):
        return self.get_pqc_params_from_all_quaternions()

    def get_pqc_params_from_block_quaternions(self, quaternions):
        """Convert quaternions to PQC parameters."""
        angles = jax.vmap(self.quaternion_to_pqc_angles_fn)(quaternions)
        return angles

    def get_pqc_params_from_all_quaternions(self):
        """Convert all quaternions to PQC parameters."""
        block_angles = jax.vmap(self.get_pqc_params_from_block_quaternions)(self.quaternions)
        return block_angles
    
    def run_model_batch(self, in_state, params=None):
        """Run the model circuit on the BATCHED parameters and return the output state.

        Accepts either quaternion parameters of shape (blocks, qubits, 4) or none. 
        """
        if params is None:
            quats = self.quaternions
        else:
            quats = params

        pqc_angles = jax.vmap(self.get_pqc_params_from_block_quaternions)(quats)
        return self.batched_model_circuit(in_state, pqc_angles)

    def __call__(self, *args, **kwds):
        return self.run_model_batch(*args, **kwds)
    
    def __str__(self):
        return str(self.circuit_ops)

    def draw_mpl(self, in_state, params=None):
        """Draw the model circuit using matplotlib."""
 
        if params is None:
            params = self.get_pqc_params_from_all_quaternions()

        print(f"Drawing circuit with params: {params}")
        print(f"Input state: {in_state}")
        print(f'Model: {self}')

        return qml.draw_mpl(self.model_circuit, decimals=4)(in_state, params)

    def get_circuit_tokens(self):
        """Get the circuit tokens."""
        tokens = []
        for i, op in enumerate(self.circuit_ops):
        # def loop_body(i):
            tokens.append(op)
            
            if (i+1) % self.gate_blocks == 0:
                # 2) Apply the PQC gates:
                # print(f"Applying PQC block {i // self.gate_blocks + 1} with params: {pqc_params[i // self.gate_blocks]}")
                pqc_params_block = self.get_pqc_params_from_block_quaternions(self.quaternions[i // self.gate_blocks])
                # Add PQC parameters to the tokens:
                for qubit in range(self.num_qubits):
                    for j, pqc in enumerate(self.pqc_gates):
                        tokens.append((pqc, [qubit], [pqc_params_block[qubit, j].item()]))

        # 3) Return the circuit tokens with PQC params:
        return tokens
