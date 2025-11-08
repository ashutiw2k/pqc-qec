#!/usr/bin/env python3
"""
Test script for gate sequence noise model.

This demonstrates the new coherent noise method that modifies gate sequences
instead of adding rotation errors.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from pqcqec.noise.builder import apply_gate_sequence_noise, apply_gate_sequence_noise_probabilistic


def test_basic_transformations():
    """Test the basic HH→HX, XX→XZ, ZZ→ZH transformations."""
    print("=" * 60)
    print("Test 1: Basic Transformations")
    print("=" * 60)
    
    # Create a circuit with matching pairs
    circuit_ops = [
        ('h', [0], []),   # First H on qubit 0
        ('h', [0], []),   # Second H on qubit 0 → should become X
        ('x', [1], []),   # First X on qubit 1
        ('x', [1], []),   # Second X on qubit 1 → should become Z
        ('z', [2], []),   # First Z on qubit 2
        ('z', [2], []),   # Second Z on qubit 2 → should become H
    ]
    
    print("\nOriginal circuit:")
    for i, (gate, qubits, params) in enumerate(circuit_ops):
        print(f"  {i}: {gate.upper()} on qubit {qubits[0]}")
    
    print("\nApplying gate sequence noise...")
    noisy_circuit = apply_gate_sequence_noise(circuit_ops)
    
    print("\nNoisy circuit:")
    for i, (gate, qubits, params) in enumerate(noisy_circuit):
        print(f"  {i}: {gate.upper()} on qubit {qubits[0]}")
    
    # Verify transformations
    assert noisy_circuit[0] == ('h', [0], []), "First H should be unchanged"
    assert noisy_circuit[1] == ('x', [0], []), "Second H should become X"
    assert noisy_circuit[2] == ('x', [1], []), "First X should be unchanged"
    assert noisy_circuit[3] == ('z', [1], []), "Second X should become Z"
    assert noisy_circuit[4] == ('z', [2], []), "First Z should be unchanged"
    assert noisy_circuit[5] == ('h', [2], []), "Second Z should become H"
    
    print("\n✓ All transformations correct!")


def test_no_interference_between_qubits():
    """Test that gate pairs on different qubits don't interfere."""
    print("\n" + "=" * 60)
    print("Test 2: Qubit Independence")
    print("=" * 60)
    
    circuit_ops = [
        ('h', [0], []),   # H on qubit 0
        ('h', [1], []),   # H on qubit 1 (different qubit, no transformation)
        ('h', [0], []),   # Second H on qubit 0 → should transform
        ('x', [2], []),   # X on qubit 2
        ('h', [1], []),   # Second H on qubit 1 → should transform
    ]
    
    print("\nOriginal circuit:")
    for i, (gate, qubits, params) in enumerate(circuit_ops):
        print(f"  {i}: {gate.upper()} on qubit {qubits[0]}")
    
    print("\nApplying gate sequence noise...")
    noisy_circuit = apply_gate_sequence_noise(circuit_ops)
    
    print("\nNoisy circuit:")
    for i, (gate, qubits, params) in enumerate(noisy_circuit):
        print(f"  {i}: {gate.upper()} on qubit {qubits[0]}")
    
    # Verify independent transformations
    assert noisy_circuit[2] == ('x', [0], []), "HH pair on qubit 0 should transform"
    assert noisy_circuit[4] == ('x', [1], []), "HH pair on qubit 1 should transform"
    assert noisy_circuit[3] == ('x', [2], []), "Single X on qubit 2 unchanged"
    
    print("\n✓ Qubits transform independently!")


def test_multi_qubit_gates():
    """Test that multi-qubit gates are handled correctly."""
    print("\n" + "=" * 60)
    print("Test 3: Multi-Qubit Gates")
    print("=" * 60)
    
    circuit_ops = [
        ('h', [0], []),
        ('cx', [0, 1], []),  # CNOT - should break the chain
        ('h', [0], []),      # This H doesn't pair with first H (CNOT in between)
        ('x', [1], []),
        ('x', [1], []),      # This should pair with previous X → XZ
    ]
    
    print("\nOriginal circuit:")
    for i, (gate, qubits, params) in enumerate(circuit_ops):
        qubit_str = f"{qubits}" if len(qubits) > 1 else f"{qubits[0]}"
        print(f"  {i}: {gate.upper()} on qubit(s) {qubit_str}")
    
    print("\nApplying gate sequence noise...")
    noisy_circuit = apply_gate_sequence_noise(circuit_ops)
    
    print("\nNoisy circuit:")
    for i, (gate, qubits, params) in enumerate(noisy_circuit):
        qubit_str = f"{qubits}" if len(qubits) > 1 else f"{qubits[0]}"
        print(f"  {i}: {gate.upper()} on qubit(s) {qubit_str}")
    
    # HH pair is broken by CNOT
    assert noisy_circuit[2] == ('h', [0], []), "Second H shouldn't transform (CNOT breaks chain)"
    # XX pair should still work on qubit 1
    assert noisy_circuit[4] == ('z', [1], []), "XX pair on qubit 1 should transform to XZ"
    
    print("\n✓ Multi-qubit gates handled correctly!")


def test_custom_rules():
    """Test custom transformation rules."""
    print("\n" + "=" * 60)
    print("Test 4: Custom Transformation Rules")
    print("=" * 60)
    
    # Define custom rules: XX→XI, ZZ→ZY, HH→HS
    custom_rules = {
        ('x', 'x'): ('x', 'i'),   # XX → XI (X followed by identity)
        ('z', 'z'): ('z', 'y'),   # ZZ → ZY
        ('h', 'h'): ('h', 's'),   # HH → HS (H followed by S gate)
    }
    
    circuit_ops = [
        ('x', [0], []),
        ('x', [0], []),
        ('z', [1], []),
        ('z', [1], []),
        ('h', [2], []),
        ('h', [2], []),
    ]
    
    print("\nOriginal circuit:")
    for i, (gate, qubits, params) in enumerate(circuit_ops):
        print(f"  {i}: {gate.upper()} on qubit {qubits[0]}")
    
    print("\nApplying custom transformation rules...")
    print("  Rules: XX→XI, ZZ→ZY, HH→HS")
    noisy_circuit = apply_gate_sequence_noise(circuit_ops, noise=custom_rules)
    
    print("\nNoisy circuit:")
    for i, (gate, qubits, params) in enumerate(noisy_circuit):
        print(f"  {i}: {gate.upper()} on qubit {qubits[0]}")
    
    assert noisy_circuit[1] == ('i', [0], []), "XX should become XI"
    assert noisy_circuit[3] == ('y', [1], []), "ZZ should become ZY"
    assert noisy_circuit[5] == ('s', [2], []), "HH should become HS"
    
    print("\n✓ Custom rules work correctly!")


def test_probabilistic_noise():
    """Test probabilistic application of transformations."""
    print("\n" + "=" * 60)
    print("Test 5: Probabilistic Transformations")
    print("=" * 60)
    
    circuit_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('h', [1], []),
        ('h', [1], []),
        ('h', [2], []),
        ('h', [2], []),
    ]
    
    print("\nOriginal circuit (3 HH pairs):")
    for i, (gate, qubits, params) in enumerate(circuit_ops):
        print(f"  {i}: {gate.upper()} on qubit {qubits[0]}")
    
    # Test with 50% probability
    print("\nApplying probabilistic noise (p=0.5, seed=42)...")
    noisy_circuit = apply_gate_sequence_noise_probabilistic(
        circuit_ops, 
        error_probability=0.5,
        seed=42
    )
    
    print("\nNoisy circuit:")
    for i, (gate, qubits, params) in enumerate(noisy_circuit):
        print(f"  {i}: {gate.upper()} on qubit {qubits[0]}")
    
    # Count how many were transformed
    transforms = sum(1 for i, op in enumerate(noisy_circuit) 
                    if i % 2 == 1 and op[0] == 'x')
    print(f"\n{transforms}/3 pairs were transformed")
    
    # Test with 100% probability (should be deterministic)
    print("\nApplying probabilistic noise (p=1.0)...")
    noisy_circuit_full = apply_gate_sequence_noise_probabilistic(
        circuit_ops,
        error_probability=1.0,
        seed=42
    )
    
    assert noisy_circuit_full[1] == ('x', [0], [])
    assert noisy_circuit_full[3] == ('x', [1], [])
    assert noisy_circuit_full[5] == ('x', [2], [])
    
    print("\n✓ Probabilistic noise works correctly!")


def test_complex_circuit():
    """Test a more realistic circuit with mixed gates."""
    print("\n" + "=" * 60)
    print("Test 6: Complex Circuit")
    print("=" * 60)
    
    circuit_ops = [
        ('h', [0], []),
        ('cx', [0, 1], []),
        ('h', [0], []),    # H on qubit 0 (after CNOT, breaks HH chain)
        ('x', [1], []),
        ('h', [1], []),
        ('x', [2], []),
        ('x', [2], []),    # XX pair on qubit 2
        ('z', [0], []),
        ('z', [0], []),    # ZZ pair on qubit 0
        ('h', [1], []),    # HH pair on qubit 1
    ]
    
    print("\nOriginal circuit:")
    for i, (gate, qubits, params) in enumerate(circuit_ops):
        qubit_str = f"{qubits}" if len(qubits) > 1 else f"{qubits[0]}"
        print(f"  {i}: {gate.upper()} on qubit(s) {qubit_str}")
    
    print("\nApplying gate sequence noise...")
    noisy_circuit = apply_gate_sequence_noise(circuit_ops)
    
    print("\nNoisy circuit:")
    for i, (gate, qubits, params) in enumerate(noisy_circuit):
        qubit_str = f"{qubits}" if len(qubits) > 1 else f"{qubits[0]}"
        print(f"  {i}: {gate.upper()} on qubit(s) {qubit_str}")
    
    print("\n✓ Complex circuit processed successfully!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GATE SEQUENCE NOISE MODEL TESTS")
    print("=" * 60)
    
    test_basic_transformations()
    test_no_interference_between_qubits()
    test_multi_qubit_gates()
    test_custom_rules()
    test_probabilistic_noise()
    test_complex_circuit()
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe gate sequence noise model is working correctly.")
    print("\nUsage:")
    print("  from pqcqec.noise.builder import apply_gate_sequence_noise")
    print("  noisy_ops = apply_gate_sequence_noise(base_ops)")
    print("\nThis replaces the traditional RxRz noise with coherent gate errors.")
    print("=" * 60 + "\n")
