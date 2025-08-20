import jax
import jax.numpy as jnp
import optax

from ..circuits.generate import generate_random_circuit
from ..circuits.modify import tokenize_qiskit_circuit

from ..models.pqc_models import StateInputModelInterleavedPQCModel
from ..noise.simple_noise import PennylaneNoisyGates
from ..simulate.simulate import get_input_data, run_circuit_with_noise_model

from ..training.jax_loss_functions import jax_pure_state_fidelity

from ..training.jax_train_functions import train_pqc_model
from ..utils.jax_utils import JAXStateDataset, JAXDataLoader

def pqc_experiment_runner(
    num_qubits, num_gates, gate_blocks, pqc_blocks, 
    epochs, num_data, num_test, noise_dist=None,
    gate_dist=None, gpu=False, seed=0, batch_size=32,
    return_fidelity=False
):
    """Run the full experiment with the given parameters."""
    
    # Set random seed for reproducibility
    jax_prng_keys = jax.random.split(jax.random.PRNGKey(seed), 3).flatten() # Split gives us (3,2) shape, flatten to (6,) 
    print(f"Using Seed and JAX PRNG Keys: {seed, jax_prng_keys}")

    # Generate ideal data
    ideal_train_data = get_input_data(num_qubits, num_data, seed=jax_prng_keys[0])
    
    # Generate noise
    # train_noise = JAXNoise(x_rad=jnp.pi/100, z_rad=jnp.pi/100, shape=(num_data, num_gates * 2), seed=jax_prng_keys[1])
    if noise_dist:
        noise_model = PennylaneNoisyGates(*noise_dist, seed=jax_prng_keys[1])
    else:
        noise_model = PennylaneNoisyGates(seed=jax_prng_keys[1])

    # Create dataset and dataloader
    train_dataset = JAXStateDataset(ideal_train_data)
    train_dataloader = JAXDataLoader(train_dataset, batch_size=batch_size, shuffle=True, seed=jax_prng_keys[2])

    # Generate random circuit list
    qiskit_random_circuit = generate_random_circuit(
        num_qubits=num_qubits,
        num_gates=num_gates,
        gate_dist=gate_dist,
        seed=seed
    )

    qiskit_adjoint_circuit = qiskit_random_circuit.inverse()
    qiskit_uncomp_circuit = qiskit_random_circuit.compose(qiskit_adjoint_circuit)

    uncomp_circuit_ops = tokenize_qiskit_circuit(qiskit_uncomp_circuit)

    # Initialize model
    model = StateInputModelInterleavedPQCModel(circuit_ops=uncomp_circuit_ops,
                                            num_qubits=num_qubits,
                                            noise_model=noise_model,
                                            pqc_blocks=pqc_blocks,
                                            gate_blocks=gate_blocks,
                                            seed=jax_prng_keys[4])
    
    print(f"Model Parameters Shape: {model.pqc_params.shape}")
    print(f"Model Parameter Count: {model.pqc_params.size}")

    # Define optimizer
    TOTAL_STEPS = int(num_data / batch_size)
    WARMUP_STEPS = int(0.1 * TOTAL_STEPS)
    RESTART_PERIOD = int(0.25 * TOTAL_STEPS)

    INIT_LR = 1e-5
    PEAK_LR = 1e-2
    MIN_LR = 5e-5

    # 1. Warmup schedule
    warmup = optax.linear_schedule(
        init_value=INIT_LR,
        end_value=PEAK_LR,
        transition_steps=WARMUP_STEPS
    )

    # 2. Cosine decay with restarts
    def cosine_with_restart_schedule(step):
        step_in_period = step % RESTART_PERIOD
        cosine = 0.5 * (1 + jnp.cos(jnp.pi * step_in_period / RESTART_PERIOD))
        return MIN_LR + (PEAK_LR - MIN_LR) * cosine

    # 3. Stitch warmup + cosine
    schedule = optax.join_schedules(
        schedules=[warmup, cosine_with_restart_schedule],
        boundaries=[WARMUP_STEPS]
    )

    # 4. Optimizer chain
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-8),
        optax.add_decayed_weights(weight_decay=1e-4),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)
    )
    
    # Train the model
    train_pqc_model(model, train_dataloader, optimizer, schedule, epochs=epochs)

    # Test the model

    # Generate test data
    ideal_test_data = get_input_data(num_qubits, num_test, seed=jax_prng_keys[5])

    print(f'Ideal Test Data Shape: {ideal_test_data.shape}')
    print(f'Running circuit with noise model on test data...')
    noisy_state = run_circuit_with_noise_model(
        uncomp_circuit_ops,
        ideal_test_data,
        noise_model,
        num_qubits,
        batched=True,
    )

    print(f'Running PQC model on test data...')
    pqc_state = model.run_model_batch(ideal_test_data)
    batched_fidelity = jax.vmap(jax_pure_state_fidelity, in_axes=(0, 0))    

    fidelity_ideal_noisy = batched_fidelity(ideal_test_data, noisy_state)
    fidelity_ideal_pqc = batched_fidelity(ideal_test_data, pqc_state)

    print(f"Fidelity (Ideal, Noisy): {jnp.mean(fidelity_ideal_noisy):.4e}")
    print(f"Fidelity (Ideal, PQC): {jnp.mean(fidelity_ideal_pqc):.4e}")
    # print(f"Test MSE Loss (Noisy): {mse_complex_loss(ideal_test_data, noisy_state):.4e}")
    # print(f'Model Parameters: {model.pqc_params}')
    if return_fidelity:
        return fidelity_ideal_noisy, fidelity_ideal_pqc

    return uncomp_circuit_ops, model.get_circuit_tokens(), jnp.mean(fidelity_ideal_pqc).item(), model.pqc_params
