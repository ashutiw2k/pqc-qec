import jax
import jax.numpy as jnp
import pennylane as qml
import numpy as np

class PennylaneNoisyGates:
    def __init__(
        self,
        x_rad: float = jnp.pi / 30,
        z_rad: float = jnp.pi / 30,
        delta_x: float = 5,
        delta_z: float = 5,
        seed: int = 0
    ):
        # Store the *nominal* over-rotation angles
        self.x_noise = x_rad
        self.z_noise = z_rad

        # Convert the percentage deltas into absolute radian offsets
        self.delta_x = delta_x * self.x_noise / 100.0
        self.delta_z = delta_z * self.z_noise / 100.0

        self.x_noise_max = x_rad + self.delta_x
        self.x_noise_min = x_rad - self.delta_x
        self.z_noise_max = z_rad + self.delta_z
        self.z_noise_min = z_rad - self.delta_z

        self.seed = seed
        self.x_key, self.z_key = jax.random.split(jax.random.PRNGKey(seed))

        self.noisy_gates = {
            "X": self.noisyX, "PauliX": self.noisyX, "x": self.noisyX,
            "Z": self.noisyZ, "PauliZ": self.noisyZ, "z": self.noisyZ,
            "CX": self.noisyCX, "CNOT": self.noisyCX, "cx": self.noisyCX, "cnot": self.noisyCX,
            "CZ": self.noisyCZ, "cz": self.noisyCZ,
            "H": self.noisyH, "Hadamard": self.noisyH, "h": self.noisyH
        }

        self.pqc_gates = {
            "RX": self.pqcRX, "rx": self.pqcRX,
            "RZ": self.pqcRZ, "rz": self.pqcRZ,
        }

        self.X = self.noisyX
        self.PauliX = self.noisyX
        self.Z = self.noisyZ
        self.PauliZ = self.noisyZ
        self.CX = self.noisyCX
        self.CNOT = self.noisyCX
        self.CZ = self.noisyCZ
        self.H = self.noisyH
        self.Hadamard = self.noisyH

        self.RX = self.pqcRX
        self.RZ = self.pqcRZ

    def apply_gate(self, gate_name, wires, angle=None):
        """Apply the noisy gate based on its name."""
        if gate_name in self.noisy_gates:
            self.noisy_gates[gate_name](wires)
        elif gate_name in self.pqc_gates:
            assert angle is not None, f"Angle must be provided for parameterized {gate_name} gates."
            self.pqc_gates[gate_name](wires, angle)
        else:
            raise ValueError(f"Gate {gate_name} is not supported by the noisy gates model.")    

    def apply_noise(self, wires):
        x_noise = np.random.uniform(size=(len(wires),), low=self.x_noise_min, high=self.x_noise_max)
        z_noise = np.random.uniform(size=(len(wires),), low=self.z_noise_min, high=self.z_noise_max)

        for i, wire in enumerate(wires):
            # Apply the noisy X gate with random over-rotation
            qml.RX(x_noise[i], wires=wire, id="x_noise")
            # Apply the noisy Z gate with random over-rotation
            qml.RZ(z_noise[i], wires=wire, id="z_noise")

    def pqcRX(self, wires, angle):
        """Parameterized noisy RX gate with random over-rotation."""
        qml.RX(angle, wires=wires)
        
    def pqcRZ(self, wires, angle):
        """Parameterized noisy RZ gate with random over-rotation."""
        qml.RZ(angle, wires=wires)
        
    def noisyX(self, wires):
        """Noisy X gate with random over-rotation."""
        qml.PauliX(wires=wires)
        self.apply_noise(wires)

    def noisyZ(self, wires):
        """Noisy Z gate with random over-rotation."""
        qml.PauliZ(wires=wires)
        self.apply_noise(wires)

    def noisyCX(self, wires):
        """Noisy CNOT gate with random over-rotation."""
        qml.CNOT(wires=wires)
        self.apply_noise(wires)

    def noisyCZ(self, wires):
        """Noisy CZ gate with random over-rotation."""
        qml.CZ(wires=wires)
        self.apply_noise(wires)

    def noisyH(self, wires):
        """Noisy Hadamard gate with random over-rotation."""
        qml.Hadamard(wires=wires)
        self.apply_noise(wires)