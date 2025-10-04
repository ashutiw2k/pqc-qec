import jax
import jax.numpy as jnp
import optax

from ..circuits.generate import generate_random_circuit
from ..circuits.modify import tokenize_qiskit_circuit

from ..models.pqc_models import StateInputModelInterleavedPQCModel, StateInputModelInterleavedQuaternionModel, StateInputModelInterleavedComplexQuaternionModel
from ..noise.simple_noise import PennylaneNoisyGates
from ..simulate.simulate import get_input_data, run_circuit_with_noise_model

from ..training.jax_loss_functions import jax_pure_state_fidelity, jax_mse_complex_loss, jax_fidelity_loss, jax_hilbert_schmidt_density_loss

from ..training.jax_train_functions import train_pqc_model_with_uncomp, train_pqc_model_no_uncomp, train_complex_pqc_model_no_uncomp, train_complex_pqc_model_with_uncomp
from ..utils.jax_utils import JAXStateDataset, JAXDataLoader

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
    # For 500 gates, default π/30 noise is TOO STRONG (compounds to ~70% fidelity loss)
    # Use much weaker noise for learnable error correction
    if noise_dist:
        noise_model = PennylaneNoisyGates(**noise_dist, seed=jax_prng_keys[1])
    else:
        # Default: π/100 instead of π/30 for 500-gate circuits
        noise_model = PennylaneNoisyGates(x_rad=jnp.pi/100, z_rad=jnp.pi/100, seed=jax_prng_keys[1])

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

    # Use XZY decomposition instead of ZXZ to avoid gimbal lock issues
    # # XZY is more numerically stable for small rotations near identity
    # model = StateInputModelInterleavedQuaternionModel(circuit_ops=uncomp_circuit_ops,
    #                                         num_qubits=num_qubits,
    #                                         noise_model=noise_model,
    #                                         pqc_blocks=pqc_blocks,
    #                                         gate_blocks=gate_blocks,
    #                                         pqc_type='zxz',  # Use ZXZ
    #                                         seed=jax_prng_keys[4])

    
    # print(f"Model Parameters Shape: {model_params.shape}")
    # print(f"Model Parameter Count: {model_params.size}")

    
    model = StateInputModelInterleavedComplexQuaternionModel(circuit_ops=uncomp_circuit_ops,
                                            num_qubits=num_qubits,
                                            noise_model=noise_model,
                                            pqc_blocks=pqc_blocks,
                                            gate_blocks=gate_blocks,
                                            pqc_type='zxz',  # Use ZXZ
                                            seed=jax_prng_keys[4])

    model_params = model.get_model_params()
    # Complex model returns dict, not array
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(model_params))
    print(f"Model Parameters: {list(model_params.keys())}")
    print(f"  - pre_quaternions: {model_params['pre_quaternions'].shape}")
    print(f"  - theta_zz: {model_params['theta_zz'].shape}")
    print(f"  - post_quaternions: {model_params['post_quaternions'].shape}")
    print(f"Total Parameter Count: {total_params}")

    # Define optimizer
    TOTAL_STEPS = int(num_data / batch_size) * epochs  # Total steps across all epochs
    WARMUP_STEPS = int(0.1 * TOTAL_STEPS)
    DECAY_STEPS = TOTAL_STEPS - WARMUP_STEPS

    # Scale learning rate with batch size (linear scaling rule)
    # For batch_size=10: use lower LR, for batch_size=64+: can use higher LR
    INIT_LR = 1e-7
    PEAK_LR = 1e-3 if batch_size >= 32 else 5e-4  # Higher LR for larger batches
    MIN_LR = 1e-7   # Lower floor for fine-tuning

    # 1. Warmup schedule
    warmup = optax.linear_schedule(
        init_value=INIT_LR,
        end_value=PEAK_LR,
        transition_steps=WARMUP_STEPS
    )

    # 2. Cosine decay schedule
    cosine_decay = optax.cosine_decay_schedule(
        init_value=PEAK_LR,
        decay_steps=DECAY_STEPS,
        alpha=MIN_LR / PEAK_LR
    )

    # 3. Stitch warmup + cosine decay
    schedule = optax.join_schedules(
        schedules=[warmup, cosine_decay],
        boundaries=[WARMUP_STEPS]
    )

    # 4. Optimizer chain
    # For small batch sizes, use gradient accumulation or increase batch norm momentum
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-8, b1=0.9, b2=0.999),  # Standard Adam hyperparams
        optax.add_decayed_weights(weight_decay=1e-5),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)
    )
    
    # Train the model
    # For better convergence, consider using jax_hilbert_schmidt_density_loss
    # which is more sensitive to small deviations than fidelity loss
    if add_uncomputation:
        # train_pqc_model_with_uncomp(model, train_dataloader, optimizer, schedule, 
        #                             main_loss_fn=jax_fidelity_loss, epochs=epochs)
        train_complex_pqc_model_with_uncomp(model, train_dataloader, optimizer, schedule, 
                                    main_loss_fn=jax_fidelity_loss, epochs=epochs)
    else:
        # train_pqc_model_no_uncomp(model, train_dataloader, optimizer, schedule, 
        #                           main_loss_fn=jax_fidelity_loss, epochs=epochs)
        train_complex_pqc_model_no_uncomp(model, train_dataloader, optimizer, schedule, 
                                  main_loss_fn=jax_fidelity_loss, epochs=epochs)

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
    
    # Print model parameter statistics (for complex quaternion model)
    print("\nFinal Model Parameters Summary:")
    final_params = model.get_model_params()
    for key, val in final_params.items():
        print(f"  {key}: shape={val.shape}, mean={jnp.mean(jnp.abs(val)):.6f}, std={jnp.std(val):.6f}")
    
    if return_fidelity:
        return fidelity_ideal_noisy, fidelity_ideal_pqc

    return uncomp_circuit_ops, model.get_circuit_tokens(), jnp.mean(fidelity_ideal_pqc).item(), model.get_pqc_params()
