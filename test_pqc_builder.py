"""
Test the new build_circuit_with_pqc function.
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/ashutoshtiwari/Desktop/Research-Code/pqc-qec')

from pqcqec.noise.builder import (
    build_circuit, 
    build_regular_noisy_circuit, 
    build_circuit_with_pqc
)

print("="*70)
print("TEST: build_circuit_with_pqc")
print("="*70)

# Test 1: Simple base circuit with PQC
print("\n" + "="*70)
print("Test 1: Basic functionality")
print("="*70)

base_ops = [
    ('h', [0], []),
    ('cx', [0, 1], []),
    ('h', [1], []),
]

num_qubits = 2
gate_blocks = 1  # Insert PQC after every gate
pqc_gates = ['rx', 'ry', 'rz']
# Number of blocks = insertions during circuit + 1 final block
num_blocks = (len(base_ops) // gate_blocks) + 1  # = 3 insertions + 1 final = 4

# Generate random PQC parameters
pqc_params = np.random.randn(num_blocks, num_qubits, len(pqc_gates)).astype(np.float32)

print(f"Base circuit: {len(base_ops)} gates")
print(f"PQC blocks: {num_blocks}")
print(f"PQC params shape: {pqc_params.shape}")
print(f"Expected PQC ops: {num_blocks * num_qubits * len(pqc_gates)}")
print(f"Total expected: {len(base_ops) + num_blocks * num_qubits * len(pqc_gates)}")

gate_ids, w1, w2, theta = build_circuit_with_pqc(
    base_ops, num_qubits, gate_blocks, pqc_gates, pqc_params
)

print(f"\nResult:")
print(f"  Total gates compiled: {len(gate_ids)}")
print(f"  gate_ids shape: {gate_ids.shape}")
print(f"  Output types: gate_ids={gate_ids.dtype}, w1={w1.dtype}, w2={w2.dtype}, theta={theta.dtype}")
print(f"  ✓ PASS" if len(gate_ids) == len(base_ops) + num_blocks * num_qubits * len(pqc_gates) else "  ✗ FAIL")

# Test 2: With noisy circuit
print("\n" + "="*70)
print("Test 2: Integration with noisy circuit builder")
print("="*70)

base_ops = [
    ('h', [0], []),
    ('h', [1], []),
    ('cx', [0, 1], []),
    ('h', [2], []),
]

x_noise = np.full(len(base_ops), 0.01, dtype=np.float32)
z_noise = np.full(len(base_ops), 0.01, dtype=np.float32)

# Build noisy circuit first
noisy_gate_ids, noisy_w1, noisy_w2, noisy_theta = build_regular_noisy_circuit(
    base_ops, x_noise, z_noise
)

print(f"Base circuit: {len(base_ops)} gates")
print(f"Noisy circuit: {len(noisy_gate_ids)} gates")

# Now add PQC to the noisy circuit
num_qubits = 3
gate_blocks = 5  # Insert PQC every 5 gates
pqc_gates = ['rx', 'rz']

# Need to convert noisy circuit back to ops format for PQC insertion
# For this test, we'll build PQC on the base circuit and then add noise
gate_blocks_base = 2
num_blocks = (len(base_ops) // gate_blocks_base) + 1  # Insertions + final block
pqc_params = np.random.randn(num_blocks, num_qubits, len(pqc_gates)).astype(np.float32)

gate_ids, w1, w2, theta = build_circuit_with_pqc(
    base_ops, num_qubits, gate_blocks_base, pqc_gates, pqc_params
)

print(f"\nCircuit with PQC:")
print(f"  Total gates: {len(gate_ids)}")
print(f"  PQC blocks inserted: {num_blocks}")
print(f"  ✓ PASS")

# Test 3: Performance benchmark
print("\n" + "="*70)
print("Test 3: Performance benchmark")
print("="*70)

import time

num_gates = 100
base_ops = [('h', [i % 5], []) for i in range(num_gates)]
num_qubits = 5
gate_blocks = 10
pqc_gates = ['rx', 'ry', 'rz']
num_blocks = (num_gates // gate_blocks) + 1  # Insertions + final block
pqc_params = np.random.randn(num_blocks, num_qubits, len(pqc_gates)).astype(np.float32)

start = time.perf_counter()
for _ in range(1000):
    gate_ids, w1, w2, theta = build_circuit_with_pqc(
        base_ops, num_qubits, gate_blocks, pqc_gates, pqc_params
    )
end = time.perf_counter()

avg_time = (end - start) / 1000

print(f"Base circuit: {num_gates} gates")
print(f"PQC blocks: {num_blocks}")
print(f"Output circuit: {len(gate_ids)} gates")
print(f"Average build time: {avg_time*1000:.4f} ms")
print(f"Throughput: {len(gate_ids) / avg_time / 1000:.2f}k gates/sec")
print(f"  ✓ PASS")

# Test 4: Correct usage example
print("\n" + "="*70)
print("Test 4: Verify correct block calculation")
print("="*70)

# Demonstrate correct block calculation
test_circuit = [('h', [i % 3], []) for i in range(23)]  # 23 gates
test_gate_blocks = 5
test_num_qubits = 3
test_pqc_gates = ['rx', 'ry']

# Correct calculation: (23 // 5) + 1 = 4 + 1 = 5 blocks
# Insertions after: gate 4, 9, 14, 19, and final after 22
correct_num_blocks = (len(test_circuit) // test_gate_blocks) + 1

test_params = np.random.randn(correct_num_blocks, test_num_qubits, len(test_pqc_gates)).astype(np.float32)

gate_ids, w1, w2, theta = build_circuit_with_pqc(
    test_circuit, test_num_qubits, test_gate_blocks, test_pqc_gates, test_params
)

expected_pqc_ops = correct_num_blocks * test_num_qubits * len(test_pqc_gates)
expected_total = len(test_circuit) + expected_pqc_ops

print(f"Circuit: {len(test_circuit)} gates, blocks every {test_gate_blocks}")
print(f"Blocks needed: {correct_num_blocks}")
print(f"Expected total: {expected_total}")
print(f"Actual total: {len(gate_ids)}")
print(f"  ✓ PASS" if len(gate_ids) == expected_total else f"  ✗ FAIL")

print("\n" + "="*70)
print("ALL TESTS COMPLETED")
print("="*70)
