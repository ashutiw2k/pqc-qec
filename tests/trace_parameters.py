"""
Trace through the exact parameter flow from theta_zz to circuit execution.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit
from pqcqec.circuits.templates import build_pqc_circuit_template
from pqcqec.simulate.jax_statevector import build_jax_circuit


def trace_parameter_flow():
    """Trace parameters from theta_zz through the entire pipeline."""
    
    print("="*60)
    print("Tracing Parameter Flow")
    print("="*60)
    
    # Setup
    num_qubits = 3
    num_gates = 5
    gate_blocks = 5
    
    # Generate circuit
    qiskit_circuit = generate_random_circuit(
        num_qubits=num_qubits,
        num_gates=num_gates,
        seed=42
    )
    
    qiskit_uncomp = qiskit_circuit.compose(qiskit_circuit.inverse())
    circuit_ops = tokenize_qiskit_circuit(qiskit_uncomp)
    
    print(f"\nBase circuit: {len(circuit_ops)} operations")
    
    # Build template
    template = build_pqc_circuit_template(
        base_ops=circuit_ops,
        num_qubits=num_qubits,
        num_gate_blocks=gate_blocks,
        add_noise=True,
        add_pqc_layers=True
    )
    
    print(f"Template: {len(template)} gates")
    
    # Create test parameters with DISTINCT theta_zz values
    num_layers = int(np.ceil(len(circuit_ops) / gate_blocks))
    
    theta_zz_test = jnp.array([0.5, 0.7, 0.9], dtype=jnp.float32)
    
    param_dict = {
        'base': np.zeros(len(circuit_ops), dtype=np.float32),
        'x_noise': np.zeros(len(circuit_ops), dtype=np.float32),
        'z_noise': np.zeros(len(circuit_ops), dtype=np.float32),
        'pre_params': jnp.zeros((num_layers, num_qubits, 3), dtype=jnp.float32),
        'theta_zz': theta_zz_test,  # ← DISTINCT VALUES
        'post_params': jnp.zeros((num_layers, num_qubits, 3), dtype=jnp.float32),
    }
    
    print(f"\nTest theta_zz: {theta_zz_test}")
    
    # Step 1: Instantiate template
    print(f"\n" + "="*60)
    print("Step 1: Template Instantiation")
    print("="*60)
    
    circuit_with_params = template.instantiate(param_dict)
    
    print(f"Instantiated circuit: {len(circuit_with_params)} operations")
    
    # Find all RZ gates and their parameters
    rz_gates = []
    for i, op in enumerate(circuit_with_params):
        gate, qubits, params = op
        if gate == 'rz' and len(params) > 0:
            param_val = params[0]
            is_jax = isinstance(param_val, (jnp.ndarray, jax.Array))
            rz_gates.append((i, qubits, param_val, is_jax))
    
    print(f"\nFound {len(rz_gates)} RZ gates with parameters:")
    
    # Check if any match our theta_zz values
    theta_matches = []
    for i, qubits, param_val, is_jax in rz_gates:
        for j, theta_val in enumerate(theta_zz_test):
            # Check for match
            if is_jax:
                param_float = float(param_val)
            else:
                param_float = float(param_val) if not isinstance(param_val, (list, tuple)) else float(param_val[0])
            
            theta_float = float(theta_val)
            
            if abs(param_float - theta_float) < 1e-6:
                theta_matches.append((i, qubits, j, param_float, is_jax))
                print(f"  Gate {i} on {qubits}: {param_float:.4f} matches theta_zz[{j}] (JAX: {is_jax})")
    
    print(f"\nFound {len(theta_matches)} RZ gates using theta_zz values")
    
    if len(theta_matches) == 0:
        print("⚠️  WARNING: No RZ gates are using theta_zz values!")
        print("This means the ZZ layer is not being created properly.")
        
        # Show first few RZ gates
        print("\nFirst 10 RZ gates:")
        for idx, (i, qubits, param_val, is_jax) in enumerate(rz_gates[:10]):
            if is_jax:
                print(f"  Gate {i} on {qubits}: {float(param_val):.6f} (JAX array)")
            else:
                print(f"  Gate {i} on {qubits}: {param_val} (type: {type(param_val)})")
    
    # Step 2: Build JAX circuit
    print(f"\n" + "="*60)
    print("Step 2: Build JAX Circuit")
    print("="*60)
    
    gate_ids, wire1s, wire2s, thetas = build_jax_circuit(circuit_with_params)
    
    print(f"Built JAX circuit:")
    print(f"  {len(gate_ids)} gates")
    print(f"  Thetas shape: {thetas.shape}")
    print(f"  Thetas dtype: {thetas.dtype}")
    
    # Check if theta_zz values appear in thetas array
    theta_in_circuit = []
    for i, theta_val in enumerate(thetas):
        for j, target_val in enumerate(theta_zz_test):
            if abs(float(theta_val) - float(target_val)) < 1e-6:
                theta_in_circuit.append((i, j, float(theta_val)))
    
    print(f"\nFound {len(theta_in_circuit)} gates in JAX circuit using theta_zz:")
    for gate_idx, theta_idx, val in theta_in_circuit[:10]:
        print(f"  Gate {gate_idx}: theta_zz[{theta_idx}] = {val:.4f}")
    
    if len(theta_in_circuit) == 0:
        print("⚠️  WARNING: theta_zz values did NOT make it into the JAX circuit!")
        print("This confirms the parameter flow is broken.")
        
        # Show non-zero thetas
        nonzero_thetas = [(i, float(t)) for i, t in enumerate(thetas) if abs(float(t)) > 1e-6]
        print(f"\nNon-zero theta values ({len(nonzero_thetas)}):")
        for i, val in nonzero_thetas[:10]:
            print(f"  Gate {i}: {val:.6f}")
    
    # Step 3: Test gradient flow
    print(f"\n" + "="*60)
    print("Step 3: Test Gradient Flow")
    print("="*60)
    
    def circuit_output_norm(theta_zz):
        """Compute circuit output as function of theta_zz."""
        param_dict_grad = {
            'base': np.zeros(len(circuit_ops), dtype=np.float32),
            'x_noise': np.zeros(len(circuit_ops), dtype=np.float32),
            'z_noise': np.zeros(len(circuit_ops), dtype=np.float32),
            'pre_params': jnp.zeros((num_layers, num_qubits, 3), dtype=jnp.float32),
            'theta_zz': theta_zz,
            'post_params': jnp.zeros((num_layers, num_qubits, 3), dtype=jnp.float32),
        }
        
        circuit_ops_inst = template.instantiate(param_dict_grad)
        gate_ids, wire1s, wire2s, thetas = build_jax_circuit(circuit_ops_inst)
        
        # Just return sum of thetas as a simple test
        return jnp.sum(jnp.abs(thetas))
    
    try:
        output_val = circuit_output_norm(theta_zz_test)
        grad_val = jax.grad(circuit_output_norm)(theta_zz_test)
        
        print(f"Output: {output_val:.6f}")
        print(f"Gradient: {grad_val}")
        print(f"Gradient is non-zero: {not jnp.allclose(grad_val, 0.0)}")
        
        if jnp.allclose(grad_val, 0.0):
            print("\n✗ Gradients are zero - parameter flow is broken!")
        else:
            print("\n✓ Gradients are flowing!")
    except Exception as e:
        print(f"\n✗ Gradient computation failed: {e}")
    
    print("\n" + "="*60)
    print("Diagnosis")
    print("="*60)
    
    if len(theta_matches) == 0:
        print("\n⚠️  ROOT CAUSE: theta_zz parameters are NOT being inserted into the circuit!")
        print("\nPossible reasons:")
        print("1. Template build_pqc_circuit_template() is not adding ZZ layer correctly")
        print("2. Template parameter indexing is wrong")
        print("3. Circuit has no PQC layers (gate_blocks setting)")
        
        # Check if PQC layers should be added
        num_expected_pqc_layers = int(np.ceil(len(circuit_ops) / gate_blocks))
        print(f"\nExpected PQC layers: {num_expected_pqc_layers}")
        print(f"Gate blocks: {gate_blocks}")
        print(f"Circuit ops: {len(circuit_ops)}")
        
        if len(circuit_ops) < gate_blocks:
            print("\n✗ BUG FOUND: Circuit has fewer operations than gate_blocks!")
            print(f"   Circuit ops: {len(circuit_ops)} < gate_blocks: {gate_blocks}")
            print("   This means NO PQC layers are being added!")
    elif len(theta_in_circuit) == 0:
        print("\n⚠️  Parameters are in instantiated circuit but lost in build_jax_circuit!")
        print("This is a build_jax_circuit issue.")
    else:
        print("\n✓ Parameters are flowing correctly through the pipeline")
    
    print("="*60)


if __name__ == "__main__":
    trace_parameter_flow()
