import jax
import jax.numpy as jnp
import optax

from ..circuits.generate import generate_random_circuit
from ..circuits.modify import tokenize_qiskit_circuit
from ..circuits.pqc_circuits import create_static_pqc_circuit, numerically_encode_circuit

from ..models.pqc_models import StateInputLightweightInterleavedQuaternionModel, StateInputModelInterleavedPQCModel, StateInputModelInterleavedQuaternionModel
from ..noise.simple_noise import PennylaneNoisyGates
from ..simulate.simulate import get_input_data, run_circuit_with_noise_model

from ..training.jax_loss_functions import jax_pure_state_fidelity, jax_mse_complex_loss

from ..training.jax_train_functions import train_pqc_model_no_uncomp_optimized, train_pqc_model_with_uncomp, train_pqc_model_no_uncomp, train_pqc_model_with_uncomp_optimized
from ..utils.jax_utils import JAXStateDataset, JAXDataLoader

def pqc_experiment_runner(
    num_qubits, num_gates, gate_blocks, pqc_blocks, 
    epochs, num_data, num_test, noise_dist=None,
    gate_dist=None, gpu=False, seed=0, batch_size=32,
    return_fidelity=False, add_uncomputation=True
):
    """Run the full experiment with the given parameters."""
    
    # Set random seed for reproducibility
    # MODIFIED: Split into more keys for new model's seed and test data
    jax_prng_keys = jax.random.split(jax.random.PRNGKey(seed), 6).flatten()
    print(f"Using Seed and JAX PRNG Keys: {seed, jax_prng_keys}")
    
    # --- Generate Noise Model (Unchanged) ---
    if noise_dist:
        noise_model = PennylaneNoisyGates(**noise_dist, seed=jax_prng_keys[1])
    else:
        noise_model = PennylaneNoisyGates(seed=jax_prng_keys[1])

    # --- NEW: Perform the one-time setup and circuit compilation ---
    pqc_gate_names = ['rz', 'rx', 'rz'] # Assuming pqc_type='zxz' by default

    random_circuit_ops = generate_random_circuit(
        num_qubits=num_qubits,
        num_gates=num_gates,
        gate_dist=gate_dist,
        seed=seed
    )

    if add_uncomputation: # As of now assume uncomp gates are the same as comp gates. 
        print("Using Uncomputation (U U†)")
        # qiskit_adjoint_circuit = qiskit_random_circuit.inverse()
        # qiskit_uncomp_circuit = qiskit_random_circuit.compose(qiskit_adjoint_circuit)
        random_circuit_ops_rev = random_circuit_ops[::-1]
        uncomp_circuit_ops = random_circuit_ops + random_circuit_ops_rev
    else:
        print("Not using Uncomputation")
        # qiskit_uncomp_circuit = qiskit_random_circuit
        uncomp_circuit_ops = random_circuit_ops


    total_circuit_gates = len(uncomp_circuit_ops)

    static_circuit_executor = create_static_pqc_circuit(
        num_qubits=num_qubits,
        num_gates=total_circuit_gates, # Use the total number of gates
        gate_blocks=gate_blocks,
        pqc_gate_names=pqc_gate_names,
        noise_model=noise_model
    )

    # Generate ideal data
    ideal_train_data = get_input_data(num_qubits, num_data, seed=jax_prng_keys[0])
    train_dataset = JAXStateDataset(ideal_train_data)
    train_dataloader = JAXDataLoader(train_dataset, batch_size=batch_size, shuffle=True, seed=jax_prng_keys[2])

    # Generate random circuit list
    # qiskit_random_circuit = generate_random_circuit(
    #     num_qubits=num_qubits,
    #     num_gates=num_gates,
    #     gate_dist=gate_dist,
    #     seed=seed
    # )


    # uncomp_circuit_ops = tokenize_qiskit_circuit(qiskit_uncomp_circuit)

    # Initialize model
    # model = StateInputModelInterleavedPQCModel(circuit_ops=uncomp_circuit_ops,
    #                                         num_qubits=num_qubits,
    #                                         noise_model=noise_model,
    #                                         pqc_blocks=pqc_blocks,
    #                                         gate_blocks=gate_blocks,
    #                                         seed=jax_prng_keys[4])

    # model = StateInputModelInterleavedQuaternionModel(circuit_ops=uncomp_circuit_ops,
    #                                         num_qubits=num_qubits,
    #                                         noise_model=noise_model,
    #                                         pqc_blocks=pqc_blocks,
    #                                         gate_blocks=gate_blocks,
    #                                         seed=jax_prng_keys[4])

    # --- NEW: Encode the unique circuit into numerical data ---
    circuit_data = numerically_encode_circuit(uncomp_circuit_ops, total_circuit_gates)

    # --- MODIFIED: Instantiate the new LIGHTWEIGHT model ---
    # Note that it no longer needs `circuit_ops` or `noise_model` arguments.
    model = StateInputLightweightInterleavedQuaternionModel(
        num_qubits=num_qubits,
        num_gates=total_circuit_gates, # Pass the total number of gates
        pqc_blocks=pqc_blocks,
        gate_blocks=gate_blocks,
        seed=jax_prng_keys[4]
    )    


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
    
    # # Train the model
    # if add_uncomputation:
    #     train_pqc_model_with_uncomp(model, train_dataloader, optimizer, schedule, epochs=epochs)
    # else:
    #     train_pqc_model_no_uncomp(model, train_dataloader, optimizer, schedule, epochs=epochs)

        # --- MODIFIED: Pass the extra arguments to the training function ---
    
    # Your training function (e.g., train_pqc_model_with_uncomp) must be
    # updated to accept `static_circuit_executor` and `circuit_data`.
    if add_uncomputation:
        train_pqc_model_with_uncomp_optimized(
            model, train_dataloader, optimizer, schedule, epochs=epochs,
            static_circuit_executor=static_circuit_executor,
            circuit_data=circuit_data
        )
    else:
        train_pqc_model_no_uncomp_optimized(
            model, train_dataloader, optimizer, schedule, epochs=epochs,
            static_circuit_executor=static_circuit_executor,
            circuit_data=circuit_data
        )

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

    # print(f'Running PQC model on test data...')
    # pqc_state = model.run_model_batch(ideal_test_input_data)

    # --- MODIFIED: Use the new model call signature for testing ---
    print(f'Running PQC model on test data...')
    pqc_state = model(static_circuit_executor, ideal_test_input_data, circuit_data)
    

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

