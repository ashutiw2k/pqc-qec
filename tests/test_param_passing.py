"""
Test script to verify flexible parameter passing to PQCModelBase methods.
"""

import jax.numpy as jnp
from qiskit import QuantumCircuit
from pqcqec.models import create_pqc_architecture, PQCModelBase
from pqcqec.circuits.modify import tokenize_qiskit_circuit

# Setup simple test case
num_qubits = 3
num_gates = 10
gate_blocks = 10

# Create simple base circuit using Qiskit
qc = QuantumCircuit(num_qubits)
for i in range(num_gates):
    qc.rz(0.1, i % num_qubits)

base_circuit_ops = tokenize_qiskit_circuit(qc)

# Create noise arrays
x_noise = jnp.zeros(num_gates, dtype=jnp.float32)
z_noise = jnp.zeros(num_gates, dtype=jnp.float32)

# Create architecture and model
arch = create_pqc_architecture(
    arch_type='lelzz_quat',
    num_qubits=num_qubits,
    num_gates=num_gates,
    gate_blocks=gate_blocks,
    pqc_blocks=1,
    seed=42
)

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

# Create test input (|000> state, batch of 4)
input_states = jnp.zeros((4, 2**num_qubits), dtype=jnp.complex64)
input_states = input_states.at[:, 0].set(1.0 + 0.0j)

print("Testing parameter passing conventions...")
print("=" * 60)

# Test 1: No params (uses stored params)
print("\n1. No params (uses model.params)")
try:
    output1 = model.run_model_batch(input_states)
    print(f"   ✓ Success! Output shape: {output1.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Tuple params
print("\n2. Tuple params")
try:
    params_tuple = model.get_model_params()
    output2 = model.run_model_batch(input_states, params_tuple)
    print(f"   ✓ Success! Output shape: {output2.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Dict params
print("\n3. Dict params")
try:
    params_dict = model.get_model_params_dict()
    output3 = model.run_model_batch(input_states, params_dict)
    print(f"   ✓ Success! Output shape: {output3.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: run_model_batch_up_to_block with separate args (like progressive training)
print("\n4. run_model_batch_up_to_block with separate args")
try:
    params_dict = model.get_model_params_dict()
    pre_q = params_dict['pre_quaternions'][:1]
    theta = params_dict['theta_zz'][:1]
    post_q = params_dict['post_quaternions'][:1]
    
    output4 = model.run_model_batch_up_to_block(input_states, 0, pre_q, theta, post_q)
    print(f"   ✓ Success! Output shape: {output4.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 5: run_single_block_batch with separate args (like individual training)
print("\n5. run_single_block_batch with separate args")
try:
    params_dict = model.get_model_params_dict()
    pre_q = params_dict['pre_quaternions'][:1]
    theta = params_dict['theta_zz'][:1]
    post_q = params_dict['post_quaternions'][:1]
    
    output5 = model.run_single_block_batch(input_states, 0, pre_q, theta, post_q)
    print(f"   ✓ Success! Output shape: {output5.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 6: run_model_batch_up_to_block with tuple
print("\n6. run_model_batch_up_to_block with tuple")
try:
    partial_tuple = (pre_q, theta, post_q)
    output6 = model.run_model_batch_up_to_block(input_states, 0, partial_tuple)
    print(f"   ✓ Success! Output shape: {output6.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 7: run_model_batch_up_to_block with no params
print("\n7. run_model_batch_up_to_block with no params (uses stored)")
try:
    output7 = model.run_model_batch_up_to_block(input_states, 0)
    print(f"   ✓ Success! Output shape: {output7.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
