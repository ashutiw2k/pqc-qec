"""
Example: Using variable-length noise in PQC-QEC training workflow.

This demonstrates how to integrate the new variable-length noise transformations
into your existing training pipeline without breaking anything.
"""

import sys
sys.path.insert(0, '../')

from pqcqec.noise.builder import apply_gate_sequence_noise

# Your typical workflow: base circuit → apply noise → add PQC correction → evaluate

print("="*80)
print("INTEGRATING VARIABLE-LENGTH NOISE IN YOUR WORKFLOW")
print("="*80)

# Scenario 1: Your current approach (still works exactly the same)
print("\n1. CURRENT APPROACH (unchanged)")
print("-" * 60)
base_ops = [
    ('h', [0], []),
    ('h', [0], []),
    ('x', [0], []),
    ('x', [0], []),
    ('z', [0], []),
]

# Apply coherent gate errors
noisy_ops = apply_gate_sequence_noise(
    base_ops,
    noise={
        ('h', 'h'): ('h', 'x'),
        ('x', 'x'): ('x', 'z'),
        ('z', 'z'): ('z', 'h'),
    }
)

# Add PQC correction at the end (your current pattern)
pqc_params = [0.123, 0.456, 0.789]  # From training
noisy_ops.append(('rz', [0], [pqc_params[0]]))
noisy_ops.append(('rx', [0], [pqc_params[1]]))
noisy_ops.append(('rz', [0], [pqc_params[2]]))

print(f"Base circuit: {[op[0].upper() for op in base_ops]}")
print(f"After noise:  {[op[0].upper() for op in noisy_ops[:-3]]}")
print(f"With PQC:     {[op[0].upper() for op in noisy_ops[-3:]]}")
print(f"Total length: {len(noisy_ops)}")

# Scenario 2: Enhanced noise model (more realistic)
print("\n2. ENHANCED NOISE MODEL (with small rotations)")
print("-" * 60)
base_ops = [
    ('h', [0], []),
    ('h', [0], []),
    ('x', [0], []),
    ('x', [0], []),
]

# Apply coherent errors + small over-rotations
noisy_ops = apply_gate_sequence_noise(
    base_ops,
    noise={
        ('h', 'h'): [('h', []), ('x', []), ('rx', [0.02])],      # HH → HX + small rotation
        ('x', 'x'): [('x', []), ('z', []), ('rz', [0.015])],     # XX → XZ + small rotation
        ('z', 'z'): [('z', []), ('h', [])],                      # ZZ → ZH (no extra rotation)
    }
)

# Add PQC correction at the end (same as before)
noisy_ops.append(('rz', [0], [pqc_params[0]]))
noisy_ops.append(('rx', [0], [pqc_params[1]]))
noisy_ops.append(('rz', [0], [pqc_params[2]]))

print(f"Base circuit: {[op[0].upper() for op in base_ops]}")
print(f"After noise:  {[f'{op[0].upper()}({op[2][0]:.3f})' if op[2] else op[0].upper() for op in noisy_ops[:-3]]}")
print(f"With PQC:     {[op[0].upper() for op in noisy_ops[-3:]]}")
print(f"Total length: {len(noisy_ops)} (note: longer due to extra rotation gates)")

# Scenario 3: Experiment suggestion - compare simple vs rich noise
print("\n3. EXPERIMENTAL COMPARISON SETUP")
print("-" * 60)
print("""
To test if richer noise models improve PQC generalization:

1. Train PQCs on SIMPLE noise (HH→HX):
   - Use current default rules
   - Train for N epochs
   - Save PQC parameters

2. Train PQCs on RICH noise (HH→HX+rotations):
   - Use enhanced rules with small rotations
   - Train for N epochs
   - Save PQC parameters

3. Test both PQC models on:
   - Simple noise circuits
   - Rich noise circuits
   - Real hardware noise (if available)

4. Compare fidelities:
   - Does simple-trained PQC work on rich noise? (generalization)
   - Does rich-trained PQC work on simple noise? (overfitting?)
   - Which achieves better fidelity on realistic noise?

This helps determine if training on more complex noise models
improves real-world performance.
""")

# Scenario 4: Noise strength sweep
print("\n4. NOISE STRENGTH SWEEP (for sensitivity analysis)")
print("-" * 60)
base_ops = [('h', [0], []), ('h', [0], [])]

for noise_strength in [0.0, 0.01, 0.05, 0.1]:
    noisy_ops = apply_gate_sequence_noise(
        base_ops,
        noise={
            ('h', 'h'): [('h', []), ('x', []), ('rx', [noise_strength])]
        }
    )
    print(f"Noise strength {noise_strength:.2f}: {[f'{op[0].upper()}({op[2][0]:.2f})' if op[2] else op[0].upper() for op in noisy_ops]}")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. Your existing code works without any changes
2. You can gradually add richer noise models
3. PQC insertion at the end is unaffected by variable circuit length
4. Fidelity calculation is state-based (circuit length doesn't matter)
5. You can experiment with noise complexity without breaking anything

Recommendation:
- Start with current simple noise model
- Once baseline is established, try adding small rotations
- Compare fidelity improvements empirically
- Choose model complexity based on results
""")
