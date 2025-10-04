"""
Check if gradients are flowing properly through the PQC model
"""
import jax
import jax.numpy as jnp

from pqcqec.models.pqc_models import StateInputModelInterleavedQuaternionModel
from pqcqec.noise.simple_noise import PennylaneNoisyGates
from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit
from pqcqec.training.jax_loss_functions import jax_fidelity_loss
from pqcqec.simulate.simulate import get_input_data

print("="*80)
print("GRADIENT FLOW CHECK")
print("="*80)

# Small test case
num_qubits = 3
num_gates = 20
gate_blocks = 5
pqc_blocks = 1

# Create model
qc = generate_random_circuit(num_qubits, num_gates, seed=42)
qc_uncomp = qc.compose(qc.inverse())
circuit_ops = tokenize_qiskit_circuit(qc_uncomp)
noise_model = PennylaneNoisyGates(seed=42)

model = StateInputModelInterleavedQuaternionModel(
    circuit_ops=circuit_ops,
    num_qubits=num_qubits,
    noise_model=noise_model,
    pqc_blocks=pqc_blocks,
    gate_blocks=gate_blocks,
    seed=42
)

print(f"Model has {model.get_model_params().size} parameters")
print(f"Parameter shape: {model.get_model_params().shape}")

# Get test data
test_states = get_input_data(num_qubits, 5, seed=123)

print(f"\nTest data shape: {test_states.shape}")

# Check forward pass
print("\n[1] Forward pass check...")
try:
    # Use the actual test_states which are properly formatted
    print(f"  Test states shape: {test_states.shape}")
    single_state = test_states[0:1]  # Keep batch dimension
    print(f"  Single state shape: {single_state.shape}")
    output = model(single_state, params=model.get_model_params())
    print(f"  ✅ Forward pass successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm: {jnp.linalg.norm(output):.6f}")
except Exception as e:
    print(f"  ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Check gradient computation
print("\n[2] Gradient computation check...")
def loss_fn(params):
    output = model(single_state, params=params)
    return jax_fidelity_loss(single_state[0], output[0])

try:
    loss_value, grads = jax.value_and_grad(loss_fn)(model.get_model_params())
    print(f"  ✅ Gradient computation successful")
    print(f"  Loss value: {loss_value:.6f}")
    print(f"  Gradient shape: {grads.shape}")
    print(f"  Gradient stats:")
    print(f"    - Mean: {jnp.mean(grads):.6e}")
    print(f"    - Std:  {jnp.std(grads):.6e}")
    print(f"    - Min:  {jnp.min(grads):.6e}")
    print(f"    - Max:  {jnp.max(grads):.6e}")
    print(f"    - Norm: {jnp.linalg.norm(grads):.6e}")
    
    # Check for issues
    nan_count = jnp.sum(jnp.isnan(grads))
    inf_count = jnp.sum(jnp.isinf(grads))
    zero_count = jnp.sum(jnp.abs(grads) < 1e-10)
    
    print(f"\n  Gradient health check:")
    print(f"    - NaN values: {nan_count} / {grads.size}")
    print(f"    - Inf values: {inf_count} / {grads.size}")
    print(f"    - Near-zero values (<1e-10): {zero_count} / {grads.size} ({100*zero_count/grads.size:.1f}%)")
    
    if nan_count > 0:
        print(f"  ⚠️  WARNING: Gradients contain NaN values!")
    if inf_count > 0:
        print(f"  ⚠️  WARNING: Gradients contain Inf values!")
    if zero_count > grads.size * 0.9:
        print(f"  ⚠️  WARNING: >90% of gradients are near zero - vanishing gradient problem!")
    if jnp.std(grads) < 1e-8:
        print(f"  ⚠️  WARNING: Gradient std is very small - may have learning issues!")
    
except Exception as e:
    print(f"  ❌ Gradient computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Check gradient flow through batched operation
print("\n[3] Batched gradient check...")
# Use test_states for batched testing
batch_states = test_states

def batched_loss_fn(params):
    outputs = model.run_model_batch(batch_states, params=params)
    batched_fidelity = jax.vmap(jax_fidelity_loss, in_axes=(0, 0))
    losses = batched_fidelity(batch_states, outputs)
    return jnp.mean(losses)

try:
    batch_loss, batch_grads = jax.value_and_grad(batched_loss_fn)(model.get_model_params())
    print(f"  ✅ Batched gradient computation successful")
    print(f"  Batch loss: {batch_loss:.6f}")
    print(f"  Batch gradient norm: {jnp.linalg.norm(batch_grads):.6e}")
except Exception as e:
    print(f"  ❌ Batched gradient computation failed: {e}")
    import traceback
    traceback.print_exc()

# Test parameter update
print("\n[4] Parameter update test...")
try:
    # Simple gradient descent step
    lr = 0.01
    new_params = model.get_model_params() - lr * grads
    
    # Check if parameters are still valid quaternions
    new_norms = jnp.linalg.norm(new_params, axis=-1)
    print(f"  Initial param norm: {jnp.mean(jnp.linalg.norm(model.get_model_params(), axis=-1)):.6f}")
    print(f"  Updated param norm: {jnp.mean(new_norms):.6f}")
    print(f"  Norm deviation from 1.0: {jnp.mean(jnp.abs(new_norms - 1.0)):.6e}")
    
    if jnp.mean(jnp.abs(new_norms - 1.0)) > 0.1:
        print(f"  ⚠️  WARNING: Quaternions are drifting from unit norm!")
        print(f"  ⚠️  This will cause representation issues - need normalization!")
    
    # Test forward pass with updated params
    new_output = model(single_state, params=new_params)
    new_loss = loss_fn(new_params)
    
    print(f"  Old loss: {loss_value:.6f}")
    print(f"  New loss: {new_loss:.6f}")
    print(f"  Change:   {new_loss - loss_value:.6e} {'✅ (improved)' if new_loss < loss_value else '❌ (worse)'}")
    
except Exception as e:
    print(f"  ❌ Parameter update test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("GRADIENT CHECK COMPLETE")
print("="*80)
