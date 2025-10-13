import jax
import jax.numpy as jnp
import optax
import numpy as np

from ..circuits.generate import generate_random_circuit
from ..circuits.modify import tokenize_qiskit_circuit

from ..models.pqc_models import StateInputModelInterleavedPQCModel, StateInputModelInterleavedQuaternionModel
from ..noise.simple_noise import PennylaneNoisyGates
from ..simulate.simulate import get_input_data, run_circuit_with_noise_model

from ..training.jax_loss_functions import jax_pure_state_fidelity, jax_mse_complex_loss

from ..training.jax_train_functions import (
    train_pqc_model_with_uncomp, train_pqc_model_no_uncomp, 
    train_lel_zz_custom_statevec_with_uncomp, train_lel_zz_custom_statevec_no_uncomp
)
from ..utils.jax_utils import JAXStateDataset, JAXDataLoader, JAXStateMeasuredDataset

from ..simulate.jax_statevector import build_jax_circuit, run_many_states

def pqc_experiment_runner(
    num_qubits, num_gates, gate_blocks, pqc_blocks, 
    epochs, num_data, num_test, noise_dist=None,
    gate_dist=None, gpu=False, seed=0, batch_size=32,
    return_fidelity=False, add_uncomputation=True
):
    """Run the full experiment with the given parameters."""
    
    # Set random seed for reproducibility
    jax_prng_keys = jax.random.split(jax.random.PRNGKey(seed), 3).flatten() # Split gives us (3,2) shape, flatten to (6,) 
    print(f"Using Seed and JAX PRNG Keys: {seed, jax_prng_keys}")
    

    # Generate ideal data
    ideal_train_data = get_input_data(num_qubits, num_data, seed=jax_prng_keys[0])
    
    # Generate noise
    # train_noise = JAXNoise(x_rad=jnp.pi/100, z_rad=jnp.pi/100, shape=(num_data, num_gates * 2), seed=jax_prng_keys[1])
    # print(noise_dist)
    if noise_dist:
        noise_model = PennylaneNoisyGates(**noise_dist, seed=jax_prng_keys[1])
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

    if add_uncomputation:
        print("Using Uncomputation (U U†)")
        qiskit_adjoint_circuit = qiskit_random_circuit.inverse()
        qiskit_uncomp_circuit = qiskit_random_circuit.compose(qiskit_adjoint_circuit)
    else:
        print("Not using Uncomputation")
        qiskit_uncomp_circuit = qiskit_random_circuit

    uncomp_circuit_ops = tokenize_qiskit_circuit(qiskit_uncomp_circuit)

    # Initialize model
    # model = StateInputModelInterleavedPQCModel(circuit_ops=uncomp_circuit_ops,
    #                                         num_qubits=num_qubits,
    #                                         noise_model=noise_model,
    #                                         pqc_blocks=pqc_blocks,
    #                                         gate_blocks=gate_blocks,
    #                                         seed=jax_prng_keys[4])

    model = StateInputModelInterleavedQuaternionModel(circuit_ops=uncomp_circuit_ops,
                                            num_qubits=num_qubits,
                                            noise_model=noise_model,
                                            pqc_blocks=pqc_blocks,
                                            gate_blocks=gate_blocks,
                                            seed=jax_prng_keys[4])

    model_params = model.get_model_params()
    print(f"Model Parameters Shape: {model_params.shape}")
    print(f"Model Parameter Count: {model_params.size}")

    # Define optimizer
    TOTAL_STEPS = int(num_data / batch_size)
    WARMUP_STEPS = int(0.1 * TOTAL_STEPS)
    RESTART_PERIOD = int(0.25 * TOTAL_STEPS)

    INIT_LR = 1e-4
    PEAK_LR = 1e-2
    MIN_LR = 5e-4

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
        optax.add_decayed_weights(weight_decay=1e-5),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)
    )
    
    # Train the model
    if add_uncomputation:
        train_pqc_model_with_uncomp(model, train_dataloader, optimizer, schedule, epochs=epochs)
    else:
        train_pqc_model_no_uncomp(model, train_dataloader, optimizer, schedule, epochs=epochs)

    # Test the model

    # Generate test data
    ideal_test_input_data = get_input_data(num_qubits, num_test, seed=jax_prng_keys[5])

    print(f'Ideal Test Data Shape: {ideal_test_input_data.shape}')
    print(f'Running circuit with noise model on test data...')
    noisy_state = run_circuit_with_noise_model(
        uncomp_circuit_ops,
        ideal_test_input_data,
        noise_model,
        num_qubits,
        batched=True,
    )

    if not add_uncomputation:
        no_noise_model = PennylaneNoisyGates(x_rad=0, z_rad=0, delta_x=0, delta_z=0, seed=0)

        ideal_out_state = run_circuit_with_noise_model(
            uncomp_circuit_ops,
            ideal_test_input_data,
            no_noise_model,
            num_qubits,
            batched=True,
        )
    else:
        ideal_out_state = ideal_test_input_data

    print(f'Running PQC model on test data...')
    pqc_state = model.run_model_batch(ideal_test_input_data)
    batched_fidelity = jax.vmap(jax_pure_state_fidelity, in_axes=(0, 0))    

    fidelity_ideal_noisy = batched_fidelity(ideal_out_state, noisy_state)
    fidelity_ideal_pqc = batched_fidelity(ideal_out_state, pqc_state)

    print(f"Fidelity (Ideal, Noisy): {jnp.mean(fidelity_ideal_noisy):.4e}")
    print(f"Fidelity (Ideal, PQC): {jnp.mean(fidelity_ideal_pqc):.4e}")
    # print(f"Test MSE Loss (Noisy): {jax_mse_complex_loss(ideal_out_state, noisy_state):.4e}")
    # print(f'Model Parameters: \n{model.get_model_params()}')
    if return_fidelity:
        return fidelity_ideal_noisy, fidelity_ideal_pqc

    return uncomp_circuit_ops, model.get_circuit_tokens(), jnp.mean(fidelity_ideal_pqc).item(), model.get_pqc_params()


def pqc_experiment_custom_statevec_runner(
    num_qubits, num_gates, gate_blocks, pqc_blocks, 
    epochs, num_data, num_test, noise_dist=None,
    gate_dist=None, gpu=False, seed=0, batch_size=32,
    return_fidelity=False, add_uncomputation=True
):
    """
    Run the full experiment with custom Numba statevector backend.
    
    This uses LELZZInterleavedQuaternionCustomStatevecModel for fast simulation
    with the custom Numba backend while maintaining JAX/Optax training.
    """
    from ..models.pqc_models import LELZZInterleavedQuaternionCustomStatevecModel
    
    # Set random seed for reproducibility
    jax_prng_keys = jax.random.split(jax.random.PRNGKey(seed), 3).flatten() # Split gives us (3,2) shape, flatten to (6,) 
    print(f"Using Seed and JAX PRNG Keys: {seed, jax_prng_keys}")
    
    # Generate ideal training data (input states)
    ideal_train_data = get_input_data(num_qubits, num_data, seed=jax_prng_keys[0])
    
    # Generate noise parameters
    if noise_dist:
        noise_model = PennylaneNoisyGates(**noise_dist, seed=jax_prng_keys[1])
    else:
        noise_model = PennylaneNoisyGates(seed=jax_prng_keys[1])
    
    # Sample noise arrays (fixed during training)
    np.random.seed(seed)
    x_noise_arr = np.random.uniform(noise_model.x_noise_min, noise_model.x_noise_max, 
                                    (num_gates,)).astype(np.float32)
    z_noise_arr = np.random.uniform(noise_model.z_noise_min, noise_model.z_noise_max, 
                                    (num_gates,)).astype(np.float32)
    
    print(f"X-noise range: [{x_noise_arr.min():.4f}, {x_noise_arr.max():.4f}]")
    print(f"Z-noise range: [{z_noise_arr.min():.4f}, {z_noise_arr.max():.4f}]")

    # Generate random circuit
    qiskit_random_circuit = generate_random_circuit(
        num_qubits=num_qubits,
        num_gates=num_gates,
        gate_dist=gate_dist,
        seed=seed
    )

    if add_uncomputation:
        print("Using Uncomputation (U U†)")
        qiskit_adjoint_circuit = qiskit_random_circuit.inverse()
        qiskit_uncomp_circuit = qiskit_random_circuit.compose(qiskit_adjoint_circuit)
        # Double the noise arrays for the adjoint circuit
        x_noise_arr = np.concatenate([x_noise_arr, x_noise_arr])
        z_noise_arr = np.concatenate([z_noise_arr, z_noise_arr])
    else:
        print("Not using Uncomputation")
        qiskit_uncomp_circuit = qiskit_random_circuit
        

    uncomp_circuit_ops = tokenize_qiskit_circuit(qiskit_uncomp_circuit)
    print(f"Circuit has {len(uncomp_circuit_ops)} operations")

    # Initialize model with custom statevector backend
    model = LELZZInterleavedQuaternionCustomStatevecModel(
        base_circuit_ops=uncomp_circuit_ops,
        num_qubits=num_qubits,
        x_noise=x_noise_arr,
        z_noise=z_noise_arr,
        pqc_blocks=pqc_blocks,
        gate_blocks=gate_blocks,
        seed=jax_prng_keys[4]
    )
    
    params = model.get_model_params()
    total_params = (params['pre_quaternions'].size + params['theta_zz'].size + 
                   params['post_quaternions'].size)
    print(f"Model initialized with {total_params} trainable parameters")
    print(f"  Pre-quaternions: {params['pre_quaternions'].shape}")
    print(f"  Theta_zz: {params['theta_zz'].shape}")
    print(f"  Post-quaternions: {params['post_quaternions'].shape}")

    # Create dataset and dataloader

    if add_uncomputation:
        # For uncomputation, measured states are the same as input states
        ideal_train_outputs = ideal_train_data
    else:
        # For no uncomputation, we need ideal noiseless states
        # Generate them using the base circuit without noise
        print("Generating ideal target states for training...")
        base_jax_ops = build_jax_circuit(uncomp_circuit_ops)
        ideal_train_outputs = run_many_states(num_qubits, *base_jax_ops, ideal_train_data)  


    train_dataset = JAXStateMeasuredDataset(ideal_train_data, ideal_train_outputs)
    train_dataloader = JAXDataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                    seed=jax_prng_keys[2])

    # Define optimizer with learning rate schedule
    TOTAL_STEPS = int(num_data / batch_size)
    WARMUP_STEPS = int(0.1 * TOTAL_STEPS)
    RESTART_PERIOD = int(0.25 * TOTAL_STEPS)

    INIT_LR = 1e-4
    PEAK_LR = 1e-2
    MIN_LR = 5e-4

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
        optax.add_decayed_weights(weight_decay=1e-5),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)
    )
    
    print(f"\nStarting training with {epochs} epochs...")
    
    # Train the model
    if add_uncomputation:
        train_lel_zz_custom_statevec_with_uncomp(model, train_dataloader, optimizer, 
                                                 schedule, epochs=epochs)
    else:
        # For no uncomputation, we need target states (ideal noiseless outputs)
        # Generate them using the base circuit without noise
        # print("Generating ideal target states for training...")
        
        train_lel_zz_custom_statevec_no_uncomp(model, train_dataloader, 
                                               optimizer, schedule, epochs=epochs)

    # Test the model
    print(f"\nGenerating test data...")
    ideal_test_input_data = get_input_data(num_qubits, num_test, seed=jax_prng_keys[5])

    print(f'Ideal Test Data Shape: {ideal_test_input_data.shape}')
    
    # Generate noisy outputs using custom backend for comparison
    print(f'Running circuit with noise using custom backend on test data...')
    test_circuit_with_noise_ops = uncomp_circuit_ops.copy()
    # Add noise gates to circuit
    noisy_test_ops = []
    for i, op in enumerate(test_circuit_with_noise_ops):
        noisy_test_ops.append(op)
        gate, qubits, params = op
        for q in qubits:
            noisy_test_ops.append(('rx', [q], [float(x_noise_arr[min(i, len(x_noise_arr)-1)])]))
            noisy_test_ops.append(('rz', [q], [float(z_noise_arr[min(i, len(z_noise_arr)-1)])]))
    
    noisy_test_jax_ops = build_jax_circuit(noisy_test_ops)
    noisy_state = run_many_states(num_qubits, *noisy_test_jax_ops, ideal_test_input_data)

    # Determine ideal output state
    if not add_uncomputation:
        print(f'Generating ideal (noiseless) output states...')
        base_test_jax_ops = build_jax_circuit(uncomp_circuit_ops)
        ideal_out_state = run_many_states(num_qubits, *base_test_jax_ops, ideal_test_input_data)
    else:
        ideal_out_state = ideal_test_input_data

    print(f'Running PQC model on test data...')
    pqc_state = model.run_model_batch(ideal_test_input_data)
    
    # Compute fidelities
    batched_fidelity = jax.vmap(jax_pure_state_fidelity, in_axes=(0, 0))    
    fidelity_ideal_noisy = batched_fidelity(ideal_out_state, noisy_state)
    fidelity_ideal_pqc = batched_fidelity(ideal_out_state, pqc_state)

    print(f"\n=== Test Results ===")
    print(f"Fidelity (Ideal, Noisy): {jnp.mean(fidelity_ideal_noisy):.4e} ± {jnp.std(fidelity_ideal_noisy):.4e}")
    print(f"Fidelity (Ideal, PQC): {jnp.mean(fidelity_ideal_pqc):.4e} ± {jnp.std(fidelity_ideal_pqc):.4e}")
    
    if return_fidelity:
        return fidelity_ideal_noisy, fidelity_ideal_pqc

    return uncomp_circuit_ops, model.get_circuit_tokens(), jnp.mean(fidelity_ideal_pqc).item(), model.get_pqc_params()

