"""
Debug script to diagnose why PQC fidelity is stuck at 0.6-0.7
"""
import jax
import jax.numpy as jnp
import numpy as np

from pqcqec.experiment.pqc_experiment import pqc_experiment_runner

# Test configuration
config = {
    'num_qubits': 5,
    'num_gates': 500,
    'gate_blocks': 10,  # PQC every 10 gates
    'pqc_blocks': 1,
    'epochs': 2,  # Short test
    'num_data': 500,  # Small dataset for quick test
    'num_test': 20,
    'batch_size': 50,
    'seed': 42,
}

print("="*80)
print("DIAGNOSTIC TESTS FOR PQC TRAINING")
print("="*80)

# Test 1: Check gradient flow
print("\n[TEST 1] Checking gradient magnitudes and NaN issues...")
print(f"Configuration: {config['num_qubits']}q, {config['num_gates']}g, PQC every {config['gate_blocks']} gates")
print(f"Parameters: ~{config['pqc_blocks'] * (config['num_gates'] // config['gate_blocks']) * config['num_qubits'] * 4} quaternion params")

# Test 2: Check noise magnitude
print("\n[TEST 2] Testing noise model strength...")
noise_tests = [
    {'name': 'Default (π/30)', 'noise': None},
    {'name': 'Low noise (π/100)', 'noise': {'x_rad': jnp.pi/100, 'z_rad': jnp.pi/100}},
    {'name': 'High noise (π/10)', 'noise': {'x_rad': jnp.pi/10, 'z_rad': jnp.pi/10}},
]

for test in noise_tests[:1]:  # Run only default for now
    print(f"\n  Testing {test['name']}...")
    try:
        fid_noisy, fid_pqc = pqc_experiment_runner(
            **config,
            noise_dist=test['noise'],
            return_fidelity=True
        )
        print(f"    Baseline (noisy circuit): {jnp.mean(fid_noisy):.4f}")
        print(f"    PQC improvement: {jnp.mean(fid_pqc):.4f}")
        print(f"    Gap to close: {1 - jnp.mean(fid_noisy):.4f}")
        
        if jnp.mean(fid_noisy) > 0.95:
            print("    ⚠️  WARNING: Noise is too weak! PQC has little to learn.")
        elif jnp.mean(fid_noisy) < 0.3:
            print("    ⚠️  WARNING: Noise is too strong! Problem may be too hard.")
            
    except Exception as e:
        print(f"    ❌ Error: {e}")

# Test 3: Check if PQC placement matters
print("\n[TEST 3] Testing PQC block frequency...")
freq_tests = [
    {'gates_per_block': 5, 'desc': 'Very frequent (every 5 gates)'},
    {'gates_per_block': 20, 'desc': 'Less frequent (every 20 gates)'},
]

for test in freq_tests[:1]:  # Run only first for speed
    print(f"\n  Testing {test['desc']}...")
    config_copy = config.copy()
    config_copy['gate_blocks'] = test['gates_per_block']
    config_copy['epochs'] = 1
    config_copy['num_data'] = 200
    
    try:
        fid_noisy, fid_pqc = pqc_experiment_runner(
            **config_copy,
            return_fidelity=True
        )
        print(f"    PQC fidelity: {jnp.mean(fid_pqc):.4f}")
        print(f"    Num PQC blocks: {config_copy['num_gates'] // test['gates_per_block']}")
    except Exception as e:
        print(f"    ❌ Error: {e}")

# Test 4: Check parameter initialization
print("\n[TEST 4] Checking quaternion parameter statistics...")
from pqcqec.models.pqc_models import StateInputModelInterleavedQuaternionModel
from pqcqec.noise.simple_noise import PennylaneNoisyGates
from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit

qc = generate_random_circuit(config['num_qubits'], config['num_gates'], seed=config['seed'])
qc_uncomp = qc.compose(qc.inverse())
circuit_ops = tokenize_qiskit_circuit(qc_uncomp)
noise_model = PennylaneNoisyGates(seed=42)

model = StateInputModelInterleavedQuaternionModel(
    circuit_ops=circuit_ops,
    num_qubits=config['num_qubits'],
    noise_model=noise_model,
    pqc_blocks=config['pqc_blocks'],
    gate_blocks=config['gate_blocks'],
    seed=42
)

initial_params = model.get_model_params()
pqc_angles = model.get_pqc_params()

print(f"  Quaternion shape: {initial_params.shape}")
print(f"  Quaternion stats: mean={jnp.mean(initial_params):.4f}, std={jnp.std(initial_params):.4f}")
print(f"  Quaternion norms: mean={jnp.mean(jnp.linalg.norm(initial_params, axis=-1)):.4f}")
print(f"  PQC angles shape: {pqc_angles.shape}")
print(f"  PQC angles stats: mean={jnp.mean(pqc_angles):.4f}, std={jnp.std(pqc_angles):.4f}")
print(f"  PQC angles range: [{jnp.min(pqc_angles):.4f}, {jnp.max(pqc_angles):.4f}]")

# Check if angles are too small (underflow) or too uniform
if jnp.std(pqc_angles) < 0.1:
    print("  ⚠️  WARNING: PQC angles have very low variance - may not be expressive enough!")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
