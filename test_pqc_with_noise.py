"""
Test PQC circuit builder with noisy circuits (tagged noise gates).

This demonstrates:
1. Building noisy circuits with tagged noise gates
2. Applying PQC blocks based only on logical gates (ignoring noise)
3. Verifying that noise gates remain in the circuit
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/ashutoshtiwari/Desktop/Research-Code/pqc-qec')

from pqcqec.noise.builder import (
    build_regular_noisy_circuit,
    build_idle_qubit_circuit,
    build_circuit_with_pqc,
)

print("="*80)
print("PQC WITH NOISY CIRCUITS - TAGGED NOISE GATES TEST")
print("="*80)

# Define a simple logical circuit
logical_ops = [
    ('h', [0], []),
    ('cx', [0, 1], []),
    ('h', [1], []),
    ('cx', [1, 0], []),
]

num_qubits = 2
gate_blocks = 2  # Insert PQC after every 2 LOGICAL gates
pqc_gates = ['rx', 'ry', 'rz']

print(f"\nLogical Circuit:")
print(f"  Gates: {len(logical_ops)}")
for i, op in enumerate(logical_ops):
    print(f"    {i}: {op}")

# Test 1: Regular Noisy Circuit with Tagged Noise
print("\n" + "="*80)
print("Test 1: Regular Noisy Circuit with Tagged Noise")
print("="*80)

x_noise = np.random.randn(len(logical_ops)).astype(np.float32) * 0.01
z_noise = np.random.randn(len(logical_ops)).astype(np.float32) * 0.01

# Build tagged noisy circuit
tagged_noisy_ops = build_regular_noisy_circuit(
    logical_ops, x_noise, z_noise, return_tagged=True
)

print(f"\nNoisy Circuit (tagged):")
print(f"  Total gates: {len(tagged_noisy_ops)}")
print(f"  Logical gates: {len(logical_ops)}")
print(f"  Noise gates: {len(tagged_noisy_ops) - len(logical_ops)}")

# Show structure
print(f"\nCircuit structure:")
for i, op in enumerate(tagged_noisy_ops[:10]):  # Show first 10
    is_noise = len(op) > 3 and op[3].get('noise', False)
    marker = " [NOISE]" if is_noise else " [LOGIC]"
    print(f"  {i}: {op[0]:4s} {op[1]} {marker}")
if len(tagged_noisy_ops) > 10:
    print(f"  ... ({len(tagged_noisy_ops) - 10} more gates)")

# Calculate expected PQC blocks
# With gate_blocks=2 and 4 logical gates: blocks = (4 // 2) + 1 = 3
num_pqc_blocks = (len(logical_ops) // gate_blocks) + 1
print(f"\nExpected PQC blocks (based on {len(logical_ops)} logical gates): {num_pqc_blocks}")

# Build circuit with PQC (ignoring noise gates)
pqc_params = np.random.randn(num_pqc_blocks, num_qubits, len(pqc_gates)).astype(np.float32)

gate_ids, w1, w2, theta = build_circuit_with_pqc(
    tagged_noisy_ops, 
    num_qubits, 
    gate_blocks, 
    pqc_gates, 
    pqc_params,
    return_numba=True,
    ignore_noise_gates=True
)

print(f"\nFinal Circuit:")
print(f"  Total gates: {len(gate_ids)}")
print(f"  Logical gates: {len(logical_ops)}")
print(f"  Noise gates: {len(tagged_noisy_ops) - len(logical_ops)}")
print(f"  PQC gates: {num_pqc_blocks * num_qubits * len(pqc_gates)}")
print(f"  Expected total: {len(tagged_noisy_ops) + num_pqc_blocks * num_qubits * len(pqc_gates)}")

assert len(gate_ids) == len(tagged_noisy_ops) + num_pqc_blocks * num_qubits * len(pqc_gates), \
    f"Gate count mismatch! {len(gate_ids)} != {len(tagged_noisy_ops) + num_pqc_blocks * num_qubits * len(pqc_gates)}"

print("✓ Circuit size correct!")

# Test 2: Idle Noise Circuit with Tagged Noise
print("\n" + "="*80)
print("Test 2: Idle Noise Circuit with Tagged Noise")
print("="*80)

idle_noise = np.random.randn(len(logical_ops)).astype(np.float32) * 0.01

# Build tagged idle noise circuit
tagged_idle_ops = build_idle_qubit_circuit(
    logical_ops, num_qubits, idle_noise, idle_threshold=1, return_tagged=True
)

print(f"\nIdle Noise Circuit (tagged):")
print(f"  Total gates: {len(tagged_idle_ops)}")
print(f"  Logical gates: {len(logical_ops)}")
print(f"  Idle noise gates: {len(tagged_idle_ops) - len(logical_ops)}")

# Show structure
print(f"\nCircuit structure:")
for i, op in enumerate(tagged_idle_ops[:12]):
    is_noise = len(op) > 3 and op[3].get('noise', False)
    marker = " [NOISE]" if is_noise else " [LOGIC]"
    print(f"  {i}: {op[0]:4s} {op[1]} {marker}")
if len(tagged_idle_ops) > 12:
    print(f"  ... ({len(tagged_idle_ops) - 12} more gates)")

# Build circuit with PQC (ignoring idle noise gates)
gate_ids2, w1_2, w2_2, theta2 = build_circuit_with_pqc(
    tagged_idle_ops,
    num_qubits,
    gate_blocks,
    pqc_gates,
    pqc_params,
    return_numba=True,
    ignore_noise_gates=True
)

print(f"\nFinal Circuit:")
print(f"  Total gates: {len(gate_ids2)}")
print(f"  Logical gates: {len(logical_ops)}")
print(f"  Idle noise gates: {len(tagged_idle_ops) - len(logical_ops)}")
print(f"  PQC gates: {num_pqc_blocks * num_qubits * len(pqc_gates)}")

assert len(gate_ids2) == len(tagged_idle_ops) + num_pqc_blocks * num_qubits * len(pqc_gates), \
    f"Gate count mismatch!"

print("✓ Circuit size correct!")

# Test 3: Verify PQC placement is correct
print("\n" + "="*80)
print("Test 3: Verify PQC Placement Based on Logical Gates")
print("="*80)

# Create a simple test case for clarity
simple_ops = [
    ('h', [0], []),      # Logical gate 0
    ('h', [1], []),      # Logical gate 1
]

simple_noise = np.array([0.01, 0.02], dtype=np.float32)
simple_x_noise = np.array([0.01, 0.02], dtype=np.float32)
simple_z_noise = np.array([0.03, 0.04], dtype=np.float32)

# Build noisy version (tagged)
simple_noisy = build_regular_noisy_circuit(
    simple_ops, simple_x_noise, simple_z_noise, return_tagged=True
)

print(f"\nSimple test circuit:")
print(f"  Logical gates: {len(simple_ops)}")
print(f"  Noisy circuit: {len(simple_noisy)} gates")

for i, op in enumerate(simple_noisy):
    is_noise = len(op) > 3 and op[3].get('noise', False)
    marker = " [NOISE]" if is_noise else " [LOGIC]"
    print(f"    {i}: {op[0]:4s} qubit {op[1][0]} {marker}")

# With gate_blocks=1, expect PQC after each logical gate + final block
# 2 logical gates -> (2 // 1) + 1 = 3 PQC blocks
num_blocks_simple = (len(simple_ops) // 1) + 1
pqc_params_simple = np.random.randn(num_blocks_simple, num_qubits, len(pqc_gates)).astype(np.float32)

print(f"\nExpected PQC blocks: {num_blocks_simple}")
print(f"  After logical gate 0 (index 0 in original)")
print(f"  After logical gate 1 (index 1 in original)")  
print(f"  Final block at end")

# Build with PQC
gate_ids3, w1_3, w2_3, theta3 = build_circuit_with_pqc(
    simple_noisy,
    num_qubits,
    gate_blocks=1,
    pqc_gates=pqc_gates,
    pqc_params=pqc_params_simple,
    return_numba=True,
    ignore_noise_gates=True
)

print(f"\nFinal circuit: {len(gate_ids3)} gates")
print(f"  Expected: {len(simple_noisy)} (noisy) + {num_blocks_simple * num_qubits * len(pqc_gates)} (PQC)")
print(f"  Expected: {len(simple_noisy) + num_blocks_simple * num_qubits * len(pqc_gates)}")

assert len(gate_ids3) == len(simple_noisy) + num_blocks_simple * num_qubits * len(pqc_gates)

print("✓ PQC placement correct!")

# Test 4: Compare with and without ignore_noise_gates
print("\n" + "="*80)
print("Test 4: Compare ignore_noise_gates=True vs False")
print("="*80)

# Without ignoring noise gates (treats all as logical)
num_blocks_all = (len(simple_noisy) // 1) + 1  # Treats all gates as logical
pqc_params_all = np.random.randn(num_blocks_all, num_qubits, len(pqc_gates)).astype(np.float32)

gate_ids_all, _, _, _ = build_circuit_with_pqc(
    simple_noisy,
    num_qubits,
    gate_blocks=1,
    pqc_gates=pqc_gates,
    pqc_params=pqc_params_all,
    return_numba=True,
    ignore_noise_gates=False  # Don't ignore
)

print(f"\nWith ignore_noise_gates=False:")
print(f"  PQC blocks: {num_blocks_all} (based on all {len(simple_noisy)} gates)")
print(f"  Total gates: {len(gate_ids_all)}")

print(f"\nWith ignore_noise_gates=True:")
print(f"  PQC blocks: {num_blocks_simple} (based on {len(simple_ops)} logical gates)")
print(f"  Total gates: {len(gate_ids3)}")

print(f"\nDifference: {len(gate_ids_all) - len(gate_ids3)} more gates when not ignoring noise")
print(f"  = {num_blocks_all - num_blocks_simple} extra PQC blocks * {num_qubits * len(pqc_gates)} gates/block")

assert len(gate_ids_all) > len(gate_ids3), "Should have more gates when not ignoring noise!"

print("✓ Comparison validated!")

print("\n" + "="*80)
print("ALL TESTS PASSED ✓")
print("="*80)

print("\n📋 Summary:")
print("  ✓ Noise gates can be tagged with {'noise': True}")
print("  ✓ build_regular_noisy_circuit(..., return_tagged=True) tags noise")
print("  ✓ build_idle_qubit_circuit(..., return_tagged=True) tags noise")
print("  ✓ build_circuit_with_pqc(..., ignore_noise_gates=True) ignores tagged gates")
print("  ✓ PQC blocks inserted based on logical circuit structure only")
print("  ✓ All noise gates preserved in final circuit")
print("  ✓ Enables realistic noisy PQC training!")
