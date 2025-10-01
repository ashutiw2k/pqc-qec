"""
Performance test to demonstrate the improvements in build_circuit functions.
"""
import numpy as np
import time
from pqcqec.noise.builder import build_circuit, build_regularnoisy_circuit, build_idle_qubit_circuit

# Generate a large test circuit
np.random.seed(42)
num_gates = 10000

# Create test circuit with mixed gates
circuit_ops = []
for i in range(num_gates):
    gate_type = np.random.choice(['h', 'x', 'z', 'rx', 'ry', 'rz', 'cx', 'cz'], p=[0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1])
    
    if gate_type in ['h', 'x', 'z']:
        circuit_ops.append((gate_type, [np.random.randint(0, 10)], []))
    elif gate_type in ['rx', 'ry', 'rz']:
        circuit_ops.append((gate_type, [np.random.randint(0, 10)], [np.random.rand()]))
    else:  # cx, cz
        q1, q2 = np.random.choice(10, 2, replace=False)
        circuit_ops.append((gate_type, [q1, q2], []))

print(f"Testing with {num_gates} gates...")
print("=" * 60)

# Test build_circuit performance
start = time.perf_counter()
for _ in range(100):
    gate_ids, w1, w2, theta = build_circuit(circuit_ops)
end = time.perf_counter()
avg_time_build = (end - start) / 100

print(f"build_circuit:")
print(f"  Average time: {avg_time_build*1000:.4f} ms")
print(f"  Output shapes: gate_ids={gate_ids.shape}, w1={w1.shape}, w2={w2.shape}, theta={theta.shape}")
print(f"  Memory (approx): {(gate_ids.nbytes + w1.nbytes + w2.nbytes + theta.nbytes) / 1024:.2f} KB")

# Test build_regularnoisy_circuit performance
x_noise = np.random.rand(num_gates).astype(np.float32)
z_noise = np.random.rand(num_gates).astype(np.float32)

start = time.perf_counter()
for _ in range(100):
    gate_ids, w1, w2, theta = build_regularnoisy_circuit(circuit_ops, x_noise, z_noise)
end = time.perf_counter()
avg_time_noisy = (end - start) / 100

print(f"\nbuild_regularnoisy_circuit:")
print(f"  Average time: {avg_time_noisy*1000:.4f} ms")
print(f"  Output shapes: gate_ids={gate_ids.shape}, w1={w1.shape}, w2={w2.shape}, theta={theta.shape}")
print(f"  Memory (approx): {(gate_ids.nbytes + w1.nbytes + w2.nbytes + theta.nbytes) / 1024:.2f} KB")

# Test build_idle_qubit_circuit performance with different thresholds
num_qubits = 10
idle_noise = np.random.rand(num_gates).astype(np.float32)

print(f"\nbuild_idle_qubit_circuit (with different idle_threshold values):")
print("-" * 60)

for threshold in [1, 2, 5, 10]:
    start = time.perf_counter()
    for _ in range(100):
        gate_ids, w1, w2, theta = build_idle_qubit_circuit(circuit_ops, num_qubits, idle_noise, idle_threshold=threshold)
    end = time.perf_counter()
    avg_time_idle = (end - start) / 100
    
    print(f"  idle_threshold={threshold}:")
    print(f"    Average time: {avg_time_idle*1000:.4f} ms")
    print(f"    Output size: {gate_ids.shape[0]:,} gates")
    print(f"    Memory: {(gate_ids.nbytes + w1.nbytes + w2.nbytes + theta.nbytes) / 1024:.2f} KB")
    print(f"    Expansion: {gate_ids.shape[0] / num_gates:.2f}x")

print("\n" + "=" * 60)
print("Optimizations applied to all functions:")
print("  ✓ Pre-allocated arrays/lists (no dynamic growth)")
print("  ✓ Direct indexing (faster than append)")
print("  ✓ Cached values (reduced array accesses)")
print("  ✓ Pre-calculated sizes (single allocation)")
print("  ✓ Better memory locality and cache efficiency")
print("  ✓ Comprehensive documentation added")
