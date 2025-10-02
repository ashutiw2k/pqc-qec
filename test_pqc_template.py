"""
Test PQC circuit template functionality for ultra-fast parameter updates.
"""
import numpy as np
import time
import sys
sys.path.insert(0, '/Users/ashutoshtiwari/Desktop/Research-Code/pqc-qec')

from pqcqec.noise.builder import (
    build_circuit_with_pqc,
    create_pqc_circuit_template,
    update_pqc_circuit_template
)

print("="*70)
print("PQC CIRCUIT TEMPLATE - PERFORMANCE TEST")
print("="*70)

# Setup test circuit
base_ops = [
    ('h', [0], []),
    ('cx', [0, 1], []),
    ('h', [1], []),
    ('cx', [1, 0], []),
    ('h', [0], []),
]

num_qubits = 2
gate_blocks = 2
pqc_gates = ['rx', 'ry', 'rz']
num_blocks = (len(base_ops) // gate_blocks) + 1  # = 3 blocks

print(f"\nCircuit Configuration:")
print(f"  Base gates: {len(base_ops)}")
print(f"  Qubits: {num_qubits}")
print(f"  Gate blocks: {gate_blocks}")
print(f"  PQC gates: {pqc_gates}")
print(f"  PQC blocks: {num_blocks}")

# Test 1: Verify correctness
print("\n" + "="*70)
print("Test 1: Correctness Verification")
print("="*70)

# Create template
template = create_pqc_circuit_template(
    base_ops, num_qubits, gate_blocks, pqc_gates, num_blocks
)

print(f"Template created:")
print(f"  Total gates: {len(template['gate_ids'])}")
print(f"  PQC param map shape: {template['pqc_param_map'].shape}")

# Generate test parameters
test_params_1 = np.random.randn(num_blocks, num_qubits, len(pqc_gates)).astype(np.float32)
test_params_2 = np.random.randn(num_blocks, num_qubits, len(pqc_gates)).astype(np.float32)

# Method 1: Template update
g1_t, w1_t, w2_t, theta1_t = update_pqc_circuit_template(template, test_params_1)
g2_t, w1_t, w2_t, theta2_t = update_pqc_circuit_template(template, test_params_2)

# Method 2: Full rebuild
g1_f, w1_f, w2_f, theta1_f = build_circuit_with_pqc(
    base_ops, num_qubits, gate_blocks, pqc_gates, test_params_1, return_numba=True
)
g2_f, w1_f, w2_f, theta2_f = build_circuit_with_pqc(
    base_ops, num_qubits, gate_blocks, pqc_gates, test_params_2, return_numba=True
)

# Verify gate structure is identical
assert np.array_equal(g1_t, g1_f), "Gate IDs don't match!"
assert np.array_equal(w1_t, w1_f), "Wire1 doesn't match!"
assert np.array_equal(w2_t, w2_f), "Wire2 doesn't match!"
assert np.allclose(theta1_t, theta1_f, rtol=1e-6), "Theta 1 doesn't match!"
assert np.allclose(theta2_t, theta2_f, rtol=1e-6), "Theta 2 doesn't match!"

print(f"✓ Template update produces IDENTICAL results to full rebuild")

# Test 2: Performance benchmark
print("\n" + "="*70)
print("Test 2: Performance Comparison")
print("="*70)

num_iterations = 10000

# Benchmark template update
params_list = [np.random.randn(num_blocks, num_qubits, len(pqc_gates)).astype(np.float32) 
               for _ in range(num_iterations)]

start = time.perf_counter()
for params in params_list:
    g, w1, w2, theta = update_pqc_circuit_template(template, params)
end = time.perf_counter()
template_time = (end - start) / num_iterations

print(f"Template Update Method:")
print(f"  Average time: {template_time*1000:.6f} ms")
print(f"  Throughput: {1/template_time:.0f} updates/sec")

# Benchmark full rebuild
start = time.perf_counter()
for params in params_list:
    g, w1, w2, theta = build_circuit_with_pqc(
        base_ops, num_qubits, gate_blocks, pqc_gates, params, return_numba=True
    )
end = time.perf_counter()
rebuild_time = (end - start) / num_iterations

print(f"\nFull Rebuild Method:")
print(f"  Average time: {rebuild_time*1000:.6f} ms")
print(f"  Throughput: {1/rebuild_time:.0f} updates/sec")

speedup = rebuild_time / template_time
print(f"\n{'='*70}")
print(f"SPEEDUP: {speedup:.1f}x faster with template!")
print(f"{'='*70}")

# Test 3: Memory efficiency
print("\n" + "="*70)
print("Test 3: Memory Efficiency")
print("="*70)

import sys

# Calculate template memory footprint
template_arrays_size = (
    template['gate_ids'].nbytes +
    template['wire1'].nbytes +
    template['wire2'].nbytes +
    template['theta'].nbytes +
    template['pqc_param_map'].nbytes
)

print(f"Template size: {template_arrays_size / 1024:.2f} KB")
print(f"  - Stores structure once, reuse forever")
print(f"  - Only updates theta array ({template['theta'].nbytes} bytes)")

single_circuit_size = (g1_f.nbytes + w1_f.nbytes + w2_f.nbytes + theta1_f.nbytes)
print(f"\nSingle circuit arrays: {single_circuit_size / 1024:.2f} KB")
print(f"  - Must rebuild for each parameter update")

# Test 4: Realistic training scenario
print("\n" + "="*70)
print("Test 4: Realistic Training Scenario (1000 epochs)")
print("="*70)

num_epochs = 1000
batch_size = 32

print(f"Configuration:")
print(f"  Epochs: {num_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Total updates: {num_epochs * batch_size}")

# Simulate training with template
start = time.perf_counter()
for epoch in range(num_epochs):
    for batch in range(batch_size):
        params = np.random.randn(num_blocks, num_qubits, len(pqc_gates)).astype(np.float32)
        g, w1, w2, theta = update_pqc_circuit_template(template, params)
        # Simulate circuit execution (just a small operation)
        _ = theta.sum()
end = time.perf_counter()
template_training_time = end - start

print(f"\nTemplate-based training:")
print(f"  Total time: {template_training_time:.3f} seconds")
print(f"  Time per epoch: {template_training_time/num_epochs*1000:.2f} ms")
print(f"  Time per update: {template_training_time/(num_epochs*batch_size)*1000:.4f} ms")

# Simulate training without template
start = time.perf_counter()
for epoch in range(num_epochs):
    for batch in range(batch_size):
        params = np.random.randn(num_blocks, num_qubits, len(pqc_gates)).astype(np.float32)
        g, w1, w2, theta = build_circuit_with_pqc(
            base_ops, num_qubits, gate_blocks, pqc_gates, params, return_numba=True
        )
        _ = theta.sum()
end = time.perf_counter()
rebuild_training_time = end - start

print(f"\nFull rebuild training:")
print(f"  Total time: {rebuild_training_time:.3f} seconds")
print(f"  Time per epoch: {rebuild_training_time/num_epochs*1000:.2f} ms")
print(f"  Time per update: {rebuild_training_time/(num_epochs*batch_size)*1000:.4f} ms")

time_saved = rebuild_training_time - template_training_time
print(f"\n{'='*70}")
print(f"TIME SAVED: {time_saved:.2f} seconds ({time_saved/60:.1f} minutes)")
print(f"Speedup: {rebuild_training_time/template_training_time:.1f}x")
print(f"{'='*70}")

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nKey Takeaways:")
print(f"  • Template creation: One-time cost")
print(f"  • Template updates: {speedup:.1f}x faster than rebuilding")
print(f"  • Perfect for training loops with fixed circuit structure")
print(f"  • Saves {time_saved/60:.1f} minutes per 1000 training epochs!")
