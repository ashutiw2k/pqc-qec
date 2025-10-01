"""
Visual demonstration of how idle_threshold affects noise application.
"""
import numpy as np
from pqcqec.noise.builder import build_idle_qubit_circuit

# Simple example circuit
circuit_ops = [
    ('h', [0], []),      # Gate 0: H on qubit 0
    ('h', [1], []),      # Gate 1: H on qubit 1
    ('h', [2], []),      # Gate 2: H on qubit 2
    ('cx', [0, 1], []),  # Gate 3: CNOT on qubits 0,1
    ('h', [0], []),      # Gate 4: H on qubit 0
]

num_qubits = 4
num_gates = len(circuit_ops)
noise = np.full(num_gates, 0.01, dtype=np.float32)

print("="*70)
print("IDLE QUBIT NOISE THRESHOLD DEMONSTRATION")
print("="*70)
print(f"\nCircuit: {num_gates} gates on {num_qubits} qubits")
print("\nGate sequence:")
for i, op in enumerate(circuit_ops):
    gate_name, qubits, _ = op
    print(f"  Gate {i}: {gate_name.upper():4s} on qubit(s) {qubits}")

print("\n" + "="*70)
print("IDLE QUBIT TRACKING")
print("="*70)

# Manually track idle counts to show what happens
for threshold in [1, 2, 3]:
    print(f"\n{'='*70}")
    print(f"THRESHOLD = {threshold} (noise applied after {threshold}+ consecutive idle gates)")
    print(f"{'='*70}\n")
    
    idle_counts = np.zeros(num_qubits, dtype=int)
    total_noise_ops = 0
    
    for i, op in enumerate(circuit_ops):
        gate_name, active_qubits, _ = op
        
        print(f"Gate {i}: {gate_name.upper():4s} on {active_qubits}")
        
        # Check idle status before gate
        noise_applied_to = []
        for q in range(num_qubits):
            if q not in active_qubits:
                idle_counts[q] += 1
                status = f"idle({idle_counts[q]})"
                if idle_counts[q] >= threshold:
                    noise_applied_to.append(q)
                    status += " → NOISE APPLIED ✓"
            else:
                status = "ACTIVE (reset counter)"
                idle_counts[q] = 0
            
            print(f"  Qubit {q}: {status}")
        
        if noise_applied_to:
            total_noise_ops += len(noise_applied_to) * 2  # RX + RZ per qubit
            print(f"  → Added {len(noise_applied_to)*2} noise gates (RX+RZ on qubits {noise_applied_to})")
        print()
    
    # Build actual circuit and verify
    gate_ids, w1, w2, theta = build_idle_qubit_circuit(
        circuit_ops, num_qubits, noise, idle_threshold=threshold
    )
    
    print(f"Summary for threshold={threshold}:")
    print(f"  Original gates: {num_gates}")
    print(f"  Noise gates added: {total_noise_ops}")
    print(f"  Total circuit size: {len(gate_ids)}")
    print(f"  Expansion factor: {len(gate_ids)/num_gates:.2f}x")
    print(f"  Verified: {'✓ PASS' if len(gate_ids) == num_gates + total_noise_ops else '✗ FAIL'}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. Higher thresholds → Fewer noise gates → Smaller, faster circuits
2. Threshold controls how "patient" we are before applying noise
3. Active qubits reset their idle counter (no accumulated noise)
4. Different thresholds model different decoherence timescales
5. Threshold=1 is most conservative (noise at every opportunity)
""")
