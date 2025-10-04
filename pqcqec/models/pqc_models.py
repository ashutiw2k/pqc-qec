import copy
import jax
import jax.numpy as jnp
import pennylane as qml

from typing import List, Dict, Tuple

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

        # Initialize unit quaternions with SMALL random rotations near identity
        # to avoid adding noise at initialization. For uncomputation (U U†), PQC
        # should start near identity and learn small corrections.
        # Sample a random axis u and a SMALL angle a, then form q = [cos(a/2), u*sin(a/2)].
        key = jax.random.PRNGKey(self.seed)
        key_axis, key_angle = jax.random.split(key)
        axes = jax.random.normal(key_axis, self.param_sz[:-1] + (3,), dtype=jnp.float32)
        axes = axes / (jnp.linalg.norm(axes, axis=-1, keepdims=True) + 1e-12)
        # CRITICAL FIX: Use MUCH smaller angles (0.01-0.1 rad instead of 0.2-0.8)
        # This keeps initial PQC near identity to avoid adding noise
        a_min, a_max = 0.01, 0.1  # radians (was 0.2-0.8)
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
                        
                    for qubit in range(self.num_qubits):
                        qml.CNOT([qubit, ((qubit+1) % self.num_qubits)])
                    

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


class StateInputModelInterleavedComplexQuaternionModel:
    """Quaternion-parameterized PQC with LEL–ZZ blocks:
       [locals(q_pre)] → [ZZ ring(θ)] → [locals(q_post)]
    """

    def __init__(
        self,
        circuit_ops: List,
        num_qubits: int,
        noise_model: PennylaneNoisyGates,
        pqc_blocks: int = 1,
        gate_blocks: int = 1,
        seed: int = 0,
        pqc_type: str = 'zxz'
    ):
        self.num_qubits = num_qubits
        self.circuit_ops = copy.deepcopy(circuit_ops)
        self.pqc_blocks = pqc_blocks
        self.gate_blocks = gate_blocks
        self.seed = seed
        self.noise_model = noise_model
        self.num_gates = len(self.circuit_ops)

        # Device / autodiff
        self.qdev_cpu = qml.device("default.qubit", wires=self.num_qubits)
        self.diff_method = "backprop"

        # Local decomposition choice for quaternion → Euler angles
        if pqc_type == 'zxz':
            self.pqc_gates = ['rz', 'rx', 'rz']
            self.quaternion_to_pqc_angles_fn = quaternion_to_zxz_angles
        elif pqc_type == 'xzy':
            self.pqc_gates = ['rx', 'rz', 'ry']
            self.quaternion_to_pqc_angles_fn = quaternion_to_xzy_angles
        else:
            raise ValueError("pqc_type must be 'zxz' or 'xzy'")

        # ---- Parameter shapes ----
        # number of PQC insertions = pqc_blocks * ceil(num_gates / gate_blocks)
        self.num_blocks = int(self.pqc_blocks * jnp.ceil(self.num_gates / self.gate_blocks))
        Q = self.num_qubits

        # (A) Quaternion locals: pre and post, each (blocks, qubits, 4)
        self.pre_quaternions  = self._init_small_quaternions((self.num_blocks, Q, 4), seed=self.seed + 1)
        self.post_quaternions = self._init_small_quaternions((self.num_blocks, Q, 4), seed=self.seed + 2)

        # (B) ZZ-ring angles: one per edge in ring, shape (blocks, qubits)
        self.theta_zz = self._init_small_angles((self.num_blocks, Q), seed=self.seed + 3)

        # ---------------- QNode ----------------
        @qml.qnode(self.qdev_cpu, interface='jax', diff_method=self.diff_method)
        def model_circuit(state, pqc_params):
            """
            pqc_params = (pre_angles, theta_zz, post_angles)
              pre_angles:  (blocks, Q, 3)
              theta_zz:    (blocks, Q)
              post_angles: (blocks, Q, 3)
            """
            pre_angles, theta_zz, post_angles = pqc_params
            Q = self.num_qubits

            # 1) Embed input state
            pennylane_state_embedding(state, self.num_qubits)

            # 2) Walk through circuit and interleave LEL–ZZ blocks
            for i, op in enumerate(self.circuit_ops):
                gate, qubit, param = op
                # Apply noisy original gate
                self.noise_model.apply_gate(gate, qubit, angle=param)

                # Insert PQC block after every gate_blocks gates
                if (i + 1) % self.gate_blocks == 0:
                    b = i // self.gate_blocks  # block index

                    # --- Pre locals: apply Rz–Rx–Rz from pre_angles[b, q, :]
                    for q in range(Q):
                        a, x, g = pre_angles[b, q, 0], pre_angles[b, q, 1], pre_angles[b, q, 2]
                        # Route through your noise_model for consistency with previous design
                        self.noise_model.apply_gate(self.pqc_gates[0], q, angle=a)
                        self.noise_model.apply_gate(self.pqc_gates[1], q, angle=x)
                        self.noise_model.apply_gate(self.pqc_gates[2], q, angle=g)

                    # --- ZZ ring (parameterized non-local core)
                    for q in range(Q):
                        i0, i1 = q, (q + 1) % Q
                        # Use native ZZ if available; otherwise synthesize with CNOT–RZ–CNOT
                        # Here we use qml.IsingZZ directly (as in previous examples).
                        qml.IsingZZ(theta_zz[b, q], wires=[i0, i1])

                    # --- Post locals
                    for q in range(Q):
                        a, x, g = post_angles[b, q, 0], post_angles[b, q, 1], post_angles[b, q, 2]
                        self.noise_model.apply_gate(self.pqc_gates[0], q, angle=a)
                        self.noise_model.apply_gate(self.pqc_gates[1], q, angle=x)
                        self.noise_model.apply_gate(self.pqc_gates[2], q, angle=g)

            return qml.state()

        self.model_circuit = model_circuit
        # batch over input states; params shared across batch
        self.batched_model_circuit = jax.jit(jax.vmap(self.model_circuit, in_axes=(0, None)))

    # ---------- Initialization helpers ----------
    def _init_small_quaternions(self, shape, seed=0, angle_min=0.01, angle_max=0.10):
        """
        Sample unit quaternions close to identity: small rotation about random axis.
        shape: (blocks, qubits, 4)
        """
        key = jax.random.PRNGKey(seed)
        k_axis, k_ang = jax.random.split(key)
        axes = jax.random.normal(k_axis, shape[:-1] + (3,), dtype=jnp.float32)
        axes = axes / (jnp.linalg.norm(axes, axis=-1, keepdims=True) + 1e-12)
        angles = jax.random.uniform(k_ang, shape[:-1] + (1,), minval=angle_min, maxval=angle_max, dtype=jnp.float32)
        w = jnp.cos(0.5 * angles)
        v = axes * jnp.sin(0.5 * angles)
        q = jnp.concatenate([w, v], axis=-1)
        # enforce w ≥ 0 to avoid q ~ -q ambiguity drifting
        flip = jnp.where(q[..., 0:1] < 0, -1.0, 1.0)
        return (q * flip).astype(jnp.float32)

    def _init_small_angles(self, shape, seed=0, sigma=1e-3):
        key = jax.random.PRNGKey(seed)
        return jax.random.normal(key, shape, dtype=jnp.float32) * sigma

    # ---------- Parameter getters/setters ----------
    def get_model_params(self) -> Dict[str, jnp.ndarray]:
        """Return a dict of raw model params (quaternions + ZZ)."""
        return {
            "pre_quaternions": self.pre_quaternions,
            "theta_zz": self.theta_zz,
            "post_quaternions": self.post_quaternions,
        }

    def set_model_params(self, new_params: Dict[str, jnp.ndarray]):
        """Set model params (expects dict keys: pre_quaternions, theta_zz, post_quaternions)."""
        required = {"pre_quaternions", "theta_zz", "post_quaternions"}
        if not required.issubset(set(new_params.keys())):
            raise ValueError(f"set_model_params expects keys {required}")
        for k, arr in new_params.items():
            if jnp.isnan(arr).any():
                nan_idx = jnp.argwhere(jnp.isnan(arr))
                raise ValueError(f"{k} contains NaNs. Shape: {arr.shape}, NaN indices: {nan_idx.tolist()}")
        self.pre_quaternions = new_params["pre_quaternions"].astype(jnp.float32)
        self.theta_zz        = new_params["theta_zz"].astype(jnp.float32)
        self.post_quaternions= new_params["post_quaternions"].astype(jnp.float32)

    # ---------- Quaternion -> Euler conversion ----------
    def _quats_to_angles_block(self, quats_block: jnp.ndarray) -> jnp.ndarray:
        """quats_block: (Q,4) -> angles: (Q,3) for chosen pqc_type"""
        return jax.vmap(self.quaternion_to_pqc_angles_fn)(quats_block)

    def _all_quats_to_angles(self, quats_all: jnp.ndarray) -> jnp.ndarray:
        """quats_all: (B,Q,4) -> angles: (B,Q,3)"""
        return jax.vmap(self._quats_to_angles_block)(quats_all)

    # ---------- Public API ----------
    def get_pqc_params(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns (pre_angles, theta_zz, post_angles)
          pre_angles:  (blocks, Q, 3)
          theta_zz:    (blocks, Q)
          post_angles: (blocks, Q, 3)
        """
        pre_angles  = self._all_quats_to_angles(self.pre_quaternions)
        post_angles = self._all_quats_to_angles(self.post_quaternions)
        return pre_angles, self.theta_zz, post_angles

    def run_model_batch(self, in_state, params: Dict[str, jnp.ndarray] = None):
        """
        Run batched inputs with shared params.
        - in_state: (B, 2**n)
        - params: optional dict with raw params; if None, use internal state.
        """
        if params is None:
            pre_q, th_zz, post_q = self.pre_quaternions, self.theta_zz, self.post_quaternions
        else:
            pre_q, th_zz, post_q = params["pre_quaternions"], params["theta_zz"], params["post_quaternions"]

        pre_angles  = self._all_quats_to_angles(pre_q)    # (Bks,Q,3)
        post_angles = self._all_quats_to_angles(post_q)   # (Bks,Q,3)
        pqc_params  = (pre_angles, th_zz, post_angles)

        return self.batched_model_circuit(in_state, pqc_params)

    def __call__(self, *args, **kwargs):
        return self.run_model_batch(*args, **kwargs)

    def __str__(self):
        return str(self.circuit_ops)

    def draw_mpl(self, in_state, params: Dict[str, jnp.ndarray] = None):
        """
        Draw with current params (converting quaternions to angles).
        """
        if params is None:
            pre_q, th_zz, post_q = self.pre_quaternions, self.theta_zz, self.post_quaternions
        else:
            pre_q, th_zz, post_q = params["pre_quaternions"], params["theta_zz"], params["post_quaternions"]

        pre_a  = self._all_quats_to_angles(pre_q)
        post_a = self._all_quats_to_angles(post_q)
        pqc_params = (pre_a, th_zz, post_a)

        return qml.draw_mpl(self.model_circuit, decimals=4)(in_state, pqc_params)

    def get_circuit_tokens(self):
        """
        Return circuit tokens including PQC blocks as:
        - ('rz'/'rx'/'rz', [q], [angle])
        - ('zz', [q, (q+1)%Q], [theta])
        """
        tokens = []
        Q = self.num_qubits
        pre_a, th_zz, post_a = self.get_pqc_params()

        for i, op in enumerate(self.circuit_ops):
            tokens.append(op)
            if (i + 1) % self.gate_blocks == 0:
                b = i // self.gate_blocks
                # Pre locals
                for q in range(Q):
                    a, x, g = pre_a[b, q]
                    tokens.append((self.pqc_gates[0], [q], [float(a)]))
                    tokens.append((self.pqc_gates[1], [q], [float(x)]))
                    tokens.append((self.pqc_gates[2], [q], [float(g)]))
                # ZZ ring
                for q in range(Q):
                    j = (q + 1) % Q
                    tokens.append(('zz', [q, j], [float(th_zz[b, q])]))
                # Post locals
                for q in range(Q):
                    a, x, g = post_a[b, q]
                    tokens.append((self.pqc_gates[0], [q], [float(a)]))
                    tokens.append((self.pqc_gates[1], [q], [float(x)]))
                    tokens.append((self.pqc_gates[2], [q], [float(g)]))
        return tokens
    

