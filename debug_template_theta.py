"""Debug script to understand the theta mismatch"""
import numpy as np
import sys
sys.path.insert(0, '/Users/ashutoshtiwari/Desktop/Research-Code/pqc-qec')

from pqcqec.noise.builder import (
    build_regular_noisy_circuit,
    create_pqc_circuit_template,
    update_pqc_circuit_template,
    build_circuit_with_pqc,
)

# Simple test
logical = [('h', [0], []), ('h', [1], [])]
x_noise = np.array([0.01, 0.02], dtype=np.float32)
z_noise = np.array([0.03, 0.04], dtype=np.float32)

# Tagged noisy
tagged_noisy = build_regular_noisy_circuit(logical, x_noise, z_noise, return_tagged=True)

print("Tagged noisy circuit:")
for i, op in enumerate(tagged_noisy):
    print(f"  {i}: {op}")

# Build template
num_pqc_blocks = 3  # (2 // 1) + 1
template = create_pqc_circuit_template(
    tagged_noisy, 2, 1, ['rx'], num_pqc_blocks, ignore_noise_gates=True
)

print(f"\nTemplate gate_ids: {template['gate_ids']}")
print(f"Template theta: {template['theta']}")
print(f"PQC param map shape: {template['pqc_param_map'].shape}")
print(f"PQC param map:\n{template['pqc_param_map']}")

# Update
params = np.random.randn(num_pqc_blocks, 2, 1).astype(np.float32)  # [blocks, qubits, gates]
print(f"\nParams shape: {params.shape}")
g1, w1, w2, th1 = update_pqc_circuit_template(template, params)

# Full rebuild
g2, w1_2, w2_2, th2 = build_circuit_with_pqc(
    tagged_noisy, 2, 1, ['rx'], params,
    return_numba=True, ignore_noise_gates=True
)

print(f"\nTemplate theta: {th1}")
print(f"Rebuild theta:  {th2}")
print(f"Match: {np.allclose(th1, th2)}")

# Find differences
if not np.allclose(th1, th2):
    diff = np.abs(th1 - th2)
    print(f"\nDifferences at indices: {np.where(diff > 1e-6)[0]}")
    for idx in np.where(diff > 1e-6)[0]:
        print(f"  {idx}: th1={th1[idx]:.6f}, th2={th2[idx]:.6f}, diff={diff[idx]:.6f}")
