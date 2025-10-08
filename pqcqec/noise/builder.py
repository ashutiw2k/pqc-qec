import numpy as np

def add_noise_to_base_ops(base_ops, x_noise:np.ndarray, z_noise:np.ndarray):
    """Add noise operations to a list of base operations according to a noise model."""
    noisy_ops = []
    for i, op in enumerate(base_ops):
        noisy_ops.append(op)
        gate, qubits, params = op
        for q in qubits:
            if x_noise[q] > 0:
                noisy_ops.append(('rx', [q], [x_noise[i]]))  # Add X error
            if z_noise[q] > 0:
                noisy_ops.append(('rz', [q], [z_noise[i]]))  # Add Z error

    return noisy_ops


