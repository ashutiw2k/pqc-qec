#!/usr/bin/env python3
"""
Example: Using Gate Sequence Noise with PQCModelBase

This demonstrates how to use the new coherent gate sequence noise
instead of (or in addition to) traditional rotation noise.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np

from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit
from pqcqec.models.pqc_model_base import PQCModelBase
from pqcqec.models.pqc_architectures import create_pqc_architecture


def example_rotation_noise():
    """Example 1: Traditional rotation noise (default behavior)"""
    print("=" * 70)
    print("Example 1: Traditional Rotation Noise (RxRz gates)")
    print("=" * 70)
    
    # Generate a simple circuit
    circuit = generate_random_circuit(
        num_qubits=3,
        num_gates=10,
        gate_dist={'h': 0.5, 'x': 0.5},
        seed=42
    )
    base_ops = tokenize_qiskit_circuit(circuit)
    
    # Create noise arrays
    x_noise = np.random.uniform(0.01, 0.05, size=10).astype(np.float32)
    z_noise = np.random.uniform(0.01, 0.05, size=10).astype(np.float32)
    
    # Create architecture
    pqc_arch = create_pqc_architecture(
        arch_type='lelzz_quat',
        num_qubits=3,
        num_gates=10,
        gate_blocks=5,
        seed=42
    )
    
    # Create model with rotation noise (default)
    model = PQCModelBase(
        base_circuit_ops=base_ops,
        num_qubits=3,
        x_noise=x_noise,
        z_noise=z_noise,
        pqc_architecture=pqc_arch,
        gate_blocks=5,
        noise_type='rotation'  # Default: adds RxRz gates
    )
    
    print(f"Circuit has {len(model.base_circuit_ops)} base gates")
    print("Template will add RxRz noise gates after each base gate")
    print(f"Total operations in noisy circuit: ~{len(model.base_circuit_ops) * 3}")
    print()


def example_gate_sequence_noise():
    """Example 2: Pure gate sequence noise (no rotation noise)"""
    print("=" * 70)
    print("Example 2: Gate Sequence Noise (HH→HX, XX→XZ, ZZ→ZH)")
    print("=" * 70)
    
    # Generate circuit with repeated gates
    circuit = generate_random_circuit(
        num_qubits=3,
        num_gates=20,
        gate_dist={'h': 0.4, 'x': 0.3, 'z': 0.3},
        seed=123
    )
    base_ops = tokenize_qiskit_circuit(circuit)
    
    print(f"\nOriginal circuit ({len(base_ops)} gates):")
    for i, (gate, qubits, params) in enumerate(base_ops[:10]):
        qubit_str = qubits[0] if len(qubits) == 1 else qubits
        print(f"  {i}: {gate.upper()} on qubit {qubit_str}")
    if len(base_ops) > 10:
        print(f"  ... ({len(base_ops) - 10} more gates)")
    
    # Create minimal noise arrays (won't be used for gate sequence noise)
    x_noise = np.zeros(20, dtype=np.float32)
    z_noise = np.zeros(20, dtype=np.float32)
    
    # Create architecture
    pqc_arch = create_pqc_architecture(
        arch_type='lelzz_quat',
        num_qubits=3,
        num_gates=20,
        gate_blocks=10,
        seed=123
    )
    
    # Create model with gate sequence noise
    model = PQCModelBase(
        base_circuit_ops=base_ops,
        num_qubits=3,
        x_noise=x_noise,
        z_noise=z_noise,
        pqc_architecture=pqc_arch,
        gate_blocks=10,
        noise_type='gate_sequence',  # NEW: coherent gate errors
        gate_sequence_noise_prob=1.0,  # Apply to all matching pairs
        noise_seed=123
    )
    
    print(f"\nAfter gate sequence noise ({len(model.base_circuit_ops)} gates - same count!):")
    for i, (gate, qubits, params) in enumerate(model.base_circuit_ops[:10]):
        qubit_str = qubits[0] if len(qubits) == 1 else qubits
        print(f"  {i}: {gate.upper()} on qubit {qubit_str}")
    if len(model.base_circuit_ops) > 10:
        print(f"  ... ({len(model.base_circuit_ops) - 10} more gates)")
    
    print("\nNote: Circuit size unchanged (no RxRz gates added)")
    print()


def example_both_noise_types():
    """Example 3: Combine both noise types"""
    print("=" * 70)
    print("Example 3: Both Noise Types (Gate Sequence + Rotation)")
    print("=" * 70)
    
    circuit = generate_random_circuit(
        num_qubits=3,
        num_gates=15,
        gate_dist={'h': 0.4, 'x': 0.3, 'z': 0.3},
        seed=456
    )
    base_ops = tokenize_qiskit_circuit(circuit)
    
    # Meaningful noise arrays
    x_noise = np.random.uniform(0.02, 0.06, size=15).astype(np.float32)
    z_noise = np.random.uniform(0.02, 0.06, size=15).astype(np.float32)
    
    pqc_arch = create_pqc_architecture(
        arch_type='lelzz_quat',
        num_qubits=3,
        num_gates=15,
        gate_blocks=5,
        seed=456
    )
    
    # Apply both noise types
    model = PQCModelBase(
        base_circuit_ops=base_ops,
        num_qubits=3,
        x_noise=x_noise,
        z_noise=z_noise,
        pqc_architecture=pqc_arch,
        gate_blocks=5,
        noise_type='both',  # Apply gate sequence transformations THEN add RxRz
        gate_sequence_noise_prob=1.0,
        noise_seed=456
    )
    
    print("Base gates: ", len(base_ops))
    print("After gate sequence transformations: ", len(model.base_circuit_ops))
    print("Template will also add RxRz noise gates")
    print("Result: Coherent errors (gate mods) + incoherent errors (rotations)")
    print()


def example_custom_rules():
    """Example 4: Custom transformation rules"""
    print("=" * 70)
    print("Example 4: Custom Gate Sequence Rules")
    print("=" * 70)
    
    circuit = generate_random_circuit(
        num_qubits=3,
        num_gates=12,
        gate_dist={'h': 0.5, 'x': 0.5},
        seed=789
    )
    base_ops = tokenize_qiskit_circuit(circuit)
    
    x_noise = np.zeros(12, dtype=np.float32)
    z_noise = np.zeros(12, dtype=np.float32)
    
    pqc_arch = create_pqc_architecture(
        arch_type='lelzz_quat',
        num_qubits=3,
        num_gates=12,
        gate_blocks=6,
        seed=789
    )
    
    # Define custom transformation rules
    custom_rules = {
        ('h', 'h'): ('h', 's'),  # HH → HS (instead of HX)
        ('x', 'x'): ('x', 'y'),  # XX → XY (instead of XZ)
        ('h', 'x'): ('h', 'y'),  # HX → HY (new rule!)
    }
    
    print("Using custom rules:")
    print("  HH → HS")
    print("  XX → XY")
    print("  HX → HY")
    print()
    
    model = PQCModelBase(
        base_circuit_ops=base_ops,
        num_qubits=3,
        x_noise=x_noise,
        z_noise=z_noise,
        pqc_architecture=pqc_arch,
        gate_blocks=6,
        noise_type='gate_sequence',
        gate_sequence_noise_rules=custom_rules,  # Custom rules!
        gate_sequence_noise_prob=1.0,
        noise_seed=789
    )

    print(f"\nAfter custom gate sequence noise ({len(model.base_circuit_ops)} gates):")
    for i, (gate, qubits, params) in enumerate(model.base_circuit_ops[:10]):
        qubit_str = qubits[0] if len(qubits) == 1 else qubits
        print(f"  {i}: {gate.upper()} on qubit {qubit_str}")

    
    print("Custom transformations applied successfully!")
    print()


def example_probabilistic_noise():
    """Example 5: Probabilistic gate sequence noise"""
    print("=" * 70)
    print("Example 5: Probabilistic Gate Sequence Noise")
    print("=" * 70)
    
    circuit = generate_random_circuit(
        num_qubits=3,
        num_gates=20,
        gate_dist={'h': 0.5, 'x': 0.5},
        seed=999
    )
    base_ops = tokenize_qiskit_circuit(circuit)
    
    x_noise = np.zeros(20, dtype=np.float32)
    z_noise = np.zeros(20, dtype=np.float32)
    
    pqc_arch = create_pqc_architecture(
        arch_type='lelzz_quat',
        num_qubits=3,
        num_gates=20,
        gate_blocks=10,
        seed=999
    )
    
    # Only 30% of matching pairs will be transformed
    model = PQCModelBase(
        base_circuit_ops=base_ops,
        num_qubits=3,
        x_noise=x_noise,
        z_noise=z_noise,
        pqc_architecture=pqc_arch,
        gate_blocks=10,
        noise_type='gate_sequence',
        gate_sequence_noise_prob=0.3,  # 30% probability
        noise_seed=999
    )

    print(f"\nAfter probabilistic gate sequence noise ({len(model.base_circuit_ops)} gates):")
    for i, (gate, qubits, params) in enumerate(model.base_circuit_ops[:10]):
        qubit_str = qubits[0] if len(qubits) == 1 else qubits
        print(f"  {i}: {gate.upper()} on qubit {qubit_str}")
    
    print("Only some matching gate pairs were transformed (p=0.3)")
    print("This models intermittent calibration errors")
    print()


def main():
    print("\n" + "=" * 70)
    print("GATE SEQUENCE NOISE INTEGRATION EXAMPLES")
    print("=" * 70)
    print()
    
    example_rotation_noise()
    example_gate_sequence_noise()
    example_both_noise_types()
    example_custom_rules()
    example_probabilistic_noise()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
To use the new noise type in your experiments:

1. Rotation Noise (default):
   model = PQCModelBase(..., noise_type='rotation')
   
2. Gate Sequence Noise only:
   model = PQCModelBase(..., noise_type='gate_sequence')
   
3. Both noise types:
   model = PQCModelBase(..., noise_type='both')
   
4. Custom rules:
   model = PQCModelBase(..., noise_type='gate_sequence',
                        gate_sequence_noise_rules={('h', 'h'): ('h', 's')})
   
5. Probabilistic:
   model = PQCModelBase(..., noise_type='gate_sequence',
                        gate_sequence_noise_prob=0.3)

The gate sequence noise is applied to base_circuit_ops BEFORE building
the template, so the PQC will learn to compensate for the modified gates!
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
