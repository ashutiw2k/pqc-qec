#!/usr/bin/env python3
"""
Demonstration of variable-length gate sequence noise transformations.

Shows how to use the new list-based replacement format to create
flexible noise models that can increase or decrease circuit length.
"""

import sys
sys.path.insert(0, '../')

from pqcqec.noise.builder import apply_gate_sequence_noise

print("="*80)
print("VARIABLE-LENGTH GATE SEQUENCE NOISE DEMONSTRATIONS")
print("="*80)

# Example 1: Traditional 2→2 transformation (backward compatible)
print("\n1. Traditional 2→2 transformation (HH → HX)")
print("-" * 60)
base_ops = [('h', [0], []), ('h', [0], []), ('x', [1], [])]
rules = {('h', 'h'): ('h', 'x')}
noisy = apply_gate_sequence_noise(base_ops, noise=rules)
print(f"Input:  {[op[0].upper() for op in base_ops]}")
print(f"Output: {[op[0].upper() for op in noisy]}")
print(f"Length: {len(base_ops)} → {len(noisy)}")

# Example 2: Expansion 2→3 (add error gate)
print("\n2. Expansion 2→3 (HH → HZX)")
print("-" * 60)
base_ops = [('h', [0], []), ('h', [0], []), ('x', [1], [])]
rules = {('h', 'h'): [('h', []), ('z', []), ('x', [])]}
noisy = apply_gate_sequence_noise(base_ops, noise=rules)
print(f"Input:  {[op[0].upper() for op in base_ops]}")
print(f"Output: {[op[0].upper() for op in noisy]}")
print(f"Length: {len(base_ops)} → {len(noisy)}")

# Example 3: Reduction 2→1 (gate fusion/cancellation)
print("\n3. Reduction 2→1 (ZZ → I, where I is represented as empty)")
print("-" * 60)
base_ops = [('h', [0], []), ('z', [0], []), ('z', [0], []), ('x', [0], [])]
rules = {('z', 'z'): [('i', [])]}  # Identity gate (or could be empty list)
noisy = apply_gate_sequence_noise(base_ops, noise=rules)
print(f"Input:  {[op[0].upper() for op in base_ops]}")
print(f"Output: {[op[0].upper() for op in noisy]}")
print(f"Length: {len(base_ops)} → {len(noisy)}")

# Example 4: With explicit parameters (adding small rotations)
print("\n4. Adding small rotation errors (HH → H + Rz(0.05) + X)")
print("-" * 60)
base_ops = [('h', [0], []), ('h', [0], [])]
rules = {('h', 'h'): [('h', []), ('rz', [0.05]), ('x', [])]}
noisy = apply_gate_sequence_noise(base_ops, noise=rules)
print(f"Input:  {[f'{op[0].upper()}' for op in base_ops]}")
print(f"Output: {[f'{op[0].upper()}({op[2][0]:.2f})' if op[2] else op[0].upper() for op in noisy]}")
print(f"Length: {len(base_ops)} → {len(noisy)}")

# Example 5: Parameter inheritance
print("\n5. Parameter inheritance (first and last inherit from originals)")
print("-" * 60)
base_ops = [('rz', [0], [0.5]), ('rz', [0], [0.3])]
rules = {('rz', 'rz'): [('rz', None), ('rx', [0.1]), ('rz', None)]}
noisy = apply_gate_sequence_noise(base_ops, noise=rules)
print(f"Input:  {[f'{op[0].upper()}({op[2][0]:.1f})' for op in base_ops]}")
print(f"Output: {[f'{op[0].upper()}({op[2][0]:.1f})' for op in noisy]}")
print("Note: First and last inherit params, middle has explicit 0.1")

# Example 6: Non-overlapping with variable length
print("\n6. Non-overlapping behavior (HHH → HZX + H)")
print("-" * 60)
base_ops = [('h', [0], [])] * 3
rules = {('h', 'h'): [('h', []), ('z', []), ('x', [])]}
noisy = apply_gate_sequence_noise(base_ops, noise=rules)
print(f"Input:  {[op[0].upper() for op in base_ops]}")
print(f"Output: {[op[0].upper() for op in noisy]}")
print("First pair (H,H) → (H,Z,X), third H unchanged")

# Example 7: Multiple transformations (HHHH → HZX + HZX)
print("\n7. Multiple transformations (HHHH → HZX + HZX)")
print("-" * 60)
base_ops = [('h', [0], [])] * 4
rules = {('h', 'h'): [('h', []), ('z', []), ('x', [])]}
noisy = apply_gate_sequence_noise(base_ops, noise=rules)
print(f"Input:  {[op[0].upper() for op in base_ops]}")
print(f"Output: {[op[0].upper() for op in noisy]}")
print(f"Length: {len(base_ops)} → {len(noisy)}")

# Example 8: Mixed rule types (backward compatible)
print("\n8. Mixed rule types (old 2-tuple + new list syntax)")
print("-" * 60)
base_ops = [('h', [0], []), ('h', [0], []), ('x', [0], []), ('x', [0], [])]
rules = {
    ('h', 'h'): ('h', 'x'),  # Old style
    ('x', 'x'): [('x', []), ('z', []), ('rz', [0.01])]  # New style
}
noisy = apply_gate_sequence_noise(base_ops, noise=rules)
print(f"Input:  {[op[0].upper() for op in base_ops]}")
print(f"Output: {[op[0].upper() for op in noisy]}")
print(f"Length: {len(base_ops)} → {len(noisy)}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
The new list-based format allows:
  • Variable-length transformations (2→1, 2→2, 2→3, 2→N)
  • Parameter inheritance (first from gate1, last from gate2)
  • Explicit parameters for intermediate gates
  • Full backward compatibility with existing 2-tuple syntax
  • Non-overlapping behavior preserved
  
Use cases:
  • Adding small rotation errors after gate pairs
  • Modeling gate fusion or cancellation
  • Simulating complex error channels
  • Creating richer noise models for PQC training
""")
