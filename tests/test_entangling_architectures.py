"""
Test script to verify new entangling PQC architectures work correctly.
"""

import jax.numpy as jnp
from qiskit import QuantumCircuit
from pqcqec.models import create_pqc_architecture, PQCModelBase
from pqcqec.circuits.modify import tokenize_qiskit_circuit

# Setup test case
num_qubits = 4
num_gates = 20
gate_blocks = 10

# Create simple base circuit using Qiskit
qc = QuantumCircuit(num_qubits)
for i in range(num_gates):
    qc.rz(0.1, i % num_qubits)

base_circuit_ops = tokenize_qiskit_circuit(qc)

# Create noise arrays
x_noise = jnp.zeros(num_gates, dtype=jnp.float32)
z_noise = jnp.zeros(num_gates, dtype=jnp.float32)

# Create test input (|0000> state, batch of 4)
input_states = jnp.zeros((4, 2**num_qubits), dtype=jnp.complex64)
input_states = input_states.at[:, 0].set(1.0 + 0.0j)

print("Testing different entangling PQC architectures...")
print("=" * 70)

# List of architectures to test
architectures = [
    ("rzrxrz_zz_ring", "LEL-ZZ Ring (default)"),
    ("rzrxrz_zz_linear", "LEL-ZZ Linear"),
    ("rzrxrz_zz_all_to_all", "LEL-ZZ All-to-All"),
    ("rzrxrz_zz_star", "LEL-ZZ Star"),
    ("rzrxrz_xx_ring", "LEL-XX Ring"),
    ("rzrxrz_yy_ring", "LEL-YY Ring"),
    ("rxrzry_zz_ring", "XZY-ZZ Ring"),
    ("rxrzry_xx_ring", "XZY-XX Ring"),
    ("rzrxrz", "Local only (no entanglement)"),
    ("none_zz_ring", "Entangling only (no local pre/post)"),
]

for pqc_type, description in architectures:
    print(f"\n{description} ({pqc_type}):")
    try:
        # Create architecture
        arch = create_pqc_architecture(
            arch_type='lelzz_quat',
            num_qubits=num_qubits,
            num_gates=num_gates,
            gate_blocks=gate_blocks,
            pqc_blocks=1,
            seed=42
        )
        
        # Create model with specific PQC type
        model = PQCModelBase(
            base_circuit_ops=base_circuit_ops,
            num_qubits=num_qubits,
            x_noise=x_noise,
            z_noise=z_noise,
            pqc_architecture=arch,
            pqc_blocks=1,
            gate_blocks=gate_blocks,
            pqc_type='zxz'
        )
        
        # Build template with this PQC type
        from pqcqec.circuits.templates import build_pqc_circuit_template
        template = build_pqc_circuit_template(
            base_ops=base_circuit_ops,
            num_qubits=num_qubits,
            num_gate_blocks=gate_blocks,
            add_noise=True,
            pqc_type=pqc_type
        )
        
        # Get number of gates in template
        num_template_gates = len(template)
        
        print(f"   ✓ Template built successfully")
        print(f"     Total gates in circuit: {num_template_gates}")
        
        # Try to instantiate the template
        param_dict = model._prepare_param_dict_for_template()
        circuit_ops = template.instantiate(param_dict)
        
        print(f"   ✓ Template instantiated successfully")
        print(f"     Circuit operations: {len(circuit_ops)}")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("All architecture tests completed!")
