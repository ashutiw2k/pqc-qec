"""
Check why PQC has such low initial fidelity
"""
import jax
import jax.numpy as jnp

from pqcqec.models.pqc_models import StateInputModelInterleavedQuaternionModel
from pqcqec.noise.simple_noise import PennylaneNoisyGates
from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit
from pqcqec.training.jax_loss_functions import jax_pure_state_fidelity
from pqcqec.simulate.simulate import get_input_data

print("="*80)
print("CHECKING INITIAL PQC BEHAVIOR")
print("="*80)

# Small test case
num_qubits = 3
num_gates = 20
gate_blocks = 5
pqc_blocks = 1

# Create model
qc = generate_random_circuit(num_qubits, num_gates, seed=42)
qc_uncomp = qc.compose(qc.inverse())
circuit_ops = tokenize_qiskit_circuit(qc_uncomp)
# Use WEAKER noise (π/100 instead of π/30 default)
noise_model = PennylaneNoisyGates(x_rad=jnp.pi/100, z_rad=jnp.pi/100, seed=42)

model = StateInputModelInterleavedQuaternionModel(
    circuit_ops=circuit_ops,
    num_qubits=num_qubits,
    noise_model=noise_model,
    pqc_blocks=pqc_blocks,
    gate_blocks=gate_blocks,
    pqc_type='xzy',  # Use XZY instead of ZXZ
    seed=42
)

print(f"\nCircuit has {len(circuit_ops)} operations (including inverse)")
print(f"PQC inserted every {gate_blocks} gates")
print(f"Total PQC blocks: {len(circuit_ops) // gate_blocks}")

# Get test data
test_states = get_input_data(num_qubits, 10, seed=123)
print(f"\nTest states shape: {test_states.shape}")

# Test 1: Identity - input should equal output
print("\n[TEST 1] Testing uncomputation property...")
print("Expected: Output ≈ Input (fidelity ≈ 1.0)")

# Run without PQC (just the noisy circuit)
from pqcqec.simulate.simulate import run_circuit_with_noise_model
noisy_output = run_circuit_with_noise_model(
    circuit_ops,
    test_states,
    noise_model,
    num_qubits,
    batched=True
)

print(f"\nNoisy circuit (U U†) fidelity with input:")
batched_fidelity = jax.vmap(jax_pure_state_fidelity, in_axes=(0, 0))
fid_noisy = batched_fidelity(test_states, noisy_output)
print(f"  Mean: {jnp.mean(fid_noisy):.6f}")
print(f"  Min:  {jnp.min(fid_noisy):.6f}")
print(f"  Max:  {jnp.max(fid_noisy):.6f}")

if jnp.mean(fid_noisy) < 0.8:
    print("  ⚠️  WARNING: Noise is TOO STRONG! Noisy circuit loses too much fidelity.")
    print("  ⚠️  PQC needs to correct massive errors - may be impossible!")

# Run with PQC
print(f"\nPQC model fidelity with input:")
pqc_output = model.run_model_batch(test_states, params=model.get_model_params())
fid_pqc = batched_fidelity(test_states, pqc_output)
print(f"  Mean: {jnp.mean(fid_pqc):.6f}")
print(f"  Min:  {jnp.min(fid_pqc):.6f}")
print(f"  Max:  {jnp.max(fid_pqc):.6f}")

if jnp.mean(fid_pqc) < 0.3:
    print("  🔴 CRITICAL: PQC is WORSE than noisy circuit!")
    print("  🔴 This suggests PQC initialization is adding MORE noise!")

# Test 2: Check PQC angles
print("\n[TEST 2] Checking PQC parameter initialization...")
quaternions = model.get_model_params()
print(f"Quaternion shape: {quaternions.shape}")
print(f"Quaternion stats:")
print(f"  Mean: {jnp.mean(quaternions):.6f}")
print(f"  Std:  {jnp.std(quaternions):.6f}")
print(f"  Norms: {jnp.mean(jnp.linalg.norm(quaternions, axis=-1)):.6f}")

# Check w component (should be close to 1 for small rotations)
w_values = quaternions[..., 0]
print(f"Quaternion w component (cos(θ/2)):")
print(f"  Mean: {jnp.mean(w_values):.6f}")
print(f"  Std:  {jnp.std(w_values):.6f}")
print(f"  Min:  {jnp.min(w_values):.6f}")
print(f"  Max:  {jnp.max(w_values):.6f}")

if jnp.mean(w_values) < 0.95:
    print("  ⚠️  w values are too far from 1 - rotations are too large!")

pqc_angles = model.get_pqc_params()
print(f"\nPQC angles shape: {pqc_angles.shape}")
print(f"PQC angles stats:")
print(f"  Mean: {jnp.mean(pqc_angles):.6f}")
print(f"  Std:  {jnp.std(pqc_angles):.6f}")
print(f"  Min:  {jnp.min(pqc_angles):.6f}")
print(f"  Max:  {jnp.max(pqc_angles):.6f}")

# Check if angles are too random
if jnp.std(pqc_angles) > 0.5:
    print("  ⚠️  PQC angles have high variance - causing large rotations!")
    print("  ⚠️  Initial angles should be SMALL (near identity) for uncomputation!")

# Test 3: What happens with zero PQC angles (identity)?
print("\n[TEST 3] Testing with identity PQC (all angles = 0)...")
zero_quats = jnp.array([[[1.0, 0.0, 0.0, 0.0] for _ in range(num_qubits)] 
                         for _ in range(pqc_angles.shape[0])], dtype=jnp.float32)
pqc_output_zero = model.run_model_batch(test_states, params=zero_quats)
fid_pqc_zero = batched_fidelity(test_states, pqc_output_zero)
print(f"Fidelity with identity PQC:")
print(f"  Mean: {jnp.mean(fid_pqc_zero):.6f}")

if jnp.mean(fid_pqc_zero) > jnp.mean(fid_pqc) + 0.2:
    print("  🔴 CRITICAL: Identity PQC is MUCH BETTER than random initialization!")
    print("  🔴 Solution: Initialize quaternions closer to identity!")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
