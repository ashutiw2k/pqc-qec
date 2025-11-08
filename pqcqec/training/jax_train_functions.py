import jax
import jax.numpy as jnp
import optax
import numpy as np
# import pennylane as qml
# import random
from tqdm import tqdm
# from typing import Callable 

from .jax_loss_functions import jax_mse_complex_loss_aligned, jax_pure_state_fidelity, jax_fidelity_loss, jax_hilbert_schmidt_density_loss
from ..simulate.simulate import run_ideal_circuit, run_circuit_with_noise_model
from ..simulate.jax_statevector import jax_run_many_states, build_jax_circuit
from ..circuits.pqc_circuits import list_LEL_ZZ

from ..noise.simple_noise import PennylaneNoisyGates
from ..noise.builder import add_rotation_noise_to_base_ops

def train_pqc_model_with_uncomp(model, dataloader, optimizer, schedule, main_loss_fn=jax_fidelity_loss, epochs=1):

    @jax.jit
    def update_step(params, opt_state, ideal_data):
        """Perform a single update step for the model parameters."""
        
        def loss_fn(p):
            measured = model(ideal_data, params=p)

            return main_loss_fn(ideal_data, measured)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        # Sanitize gradients to avoid NaN/Inf explosions
        grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        # Keep params finite
        new_params = jax.tree.map(lambda p: jnp.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), new_params)

        # Fidelity after parameter update
        measured = model(ideal_data, params=new_params)
        fidelity = jax_pure_state_fidelity(ideal_data, measured)

        return opt_state, new_params, loss, fidelity

    # Initialize optimizer state once and carry across epochs
    opt_state = optimizer.init(model.get_model_params())
    global_step = 0  # Track global step count across epochs for learning rate schedule
    
    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")
        data_iterator = tqdm(dataloader, desc="Training", total=len(dataloader), leave=False, unit='batch')
        
        # Initialize lists to track metrics for this epoch
        epoch_fidelities = []
        epoch_losses = []

        for i, batch in enumerate(data_iterator):

            # ideal_data = batch  # Assuming the first element is the ideal data
            # print(f'Batch Shape: {batch}')
            ideal_data = batch[0]  # Assuming the first element is the ideal data
            # print(f'Ideal Data Shape: {ideal_data.shape}')
            # print(f'Ideal Data \n: {ideal_data}')

            opt_state, params, loss, fidelity = update_step(model.get_model_params(), opt_state, ideal_data)
            model.set_model_params(params)

            # Track metrics
            epoch_fidelities.append(float(fidelity))
            epoch_losses.append(float(loss))

            # Display current learning rate from schedule (for monitoring only)
            current_lr = schedule(global_step)

            data_iterator.set_postfix_str(f"Fidelity (Ideal, Measured): {fidelity:.4e}, Loss: {loss:.4e}, LR: {current_lr:.4e}")
            
            global_step += 1
        
        # Print mean metrics at the end of each epoch
        mean_fidelity = np.mean(epoch_fidelities)
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {e+1} summary - Mean Fidelity: {mean_fidelity:.4e}, Mean Loss: {mean_loss:.4e}")


def train_pqc_model_no_uncomp(model, dataloader, optimizer, schedule, main_loss_fn=jax_fidelity_loss, epochs=1):
    
    no_noise_model = PennylaneNoisyGates(x_rad=0, z_rad=0, delta_x=0, delta_z=0, seed=0)

    @jax.jit
    def update_step(params, opt_state, ideal_data):
        """Perform a single update step for the model parameters."""
        
        def loss_fn(p):
            measured = model(ideal_data, params=p)
            simulated = run_circuit_with_noise_model(model.circuit_ops, ideal_data, no_noise_model, model.num_qubits)
            return main_loss_fn(simulated, measured)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params = jax.tree.map(lambda p: jnp.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), new_params)

        # Fidelity after parameter update
        measured = model(ideal_data, params=new_params)
        simulated = run_circuit_with_noise_model(model.circuit_ops, ideal_data, no_noise_model, model.num_qubits)

        fidelity = jax_pure_state_fidelity(simulated, measured)

        return opt_state, new_params, loss, fidelity

    opt_state = optimizer.init(model.get_model_params())
    global_step = 0  # Track global step count across epochs for learning rate schedule
    
    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")
        data_iterator = tqdm(dataloader, desc="Training", total=len(dataloader), leave=False, unit='batch')
        
        # Initialize lists to track metrics for this epoch
        epoch_fidelities = []
        epoch_losses = []

        for i, batch in enumerate(data_iterator):

            # ideal_data = batch  # Assuming the first element is the ideal data
            # print(f'Batch Shape: {batch}')
            ideal_data = batch[0]  # Assuming the first element is the ideal data
            # print(f'Ideal Data Shape: {ideal_data.shape}')
            # print(f'Ideal Data \n: {ideal_data}')

            opt_state, params, loss, fidelity = update_step(model.get_model_params(), opt_state, ideal_data)
            model.set_model_params(params)

            # Track metrics
            epoch_fidelities.append(float(fidelity))
            epoch_losses.append(float(loss))

            # Display current learning rate from schedule (for monitoring only)
            current_lr = schedule(global_step)

            data_iterator.set_postfix_str(f"Fidelity (Ideal, Measured): {fidelity:.4e}, Loss: {loss:.4e}, LR: {current_lr:.4e}")
            
            global_step += 1
        
        # Print mean metrics at the end of each epoch
        mean_fidelity = np.mean(epoch_fidelities)
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {e+1} summary - Mean Fidelity: {mean_fidelity:.4e}, Mean Loss: {mean_loss:.4e}")



def train_lel_zz_custom_statevec_with_uncomp(model, dataloader, optimizer, schedule, 
                                              main_loss_fn=jax_fidelity_loss, epochs=1):
    """
    Train LEL-ZZ model with uncomputation using custom statevector backend.
    
    With uncomputation (U U†), the ideal output equals the input, so we train
    the PQC to map input -> input, correcting noise along the way.
    """
    
    @jax.jit
    def update_step(params, opt_state, ideal_data):
        """Perform a single update step for the model parameters."""
        
        def loss_fn(params_tuple):
            # Unpack parameters from tuple
            # pre, theta, post = params_tuple
            
            # Run model with current PQC parameters
            measured = model.run_model_batch(ideal_data, params_tuple)
            # With uncomputation, target is the input itself
            per_state_loss = jax.vmap(main_loss_fn, in_axes=(0, 0))(ideal_data, measured)
            return jnp.mean(per_state_loss)
        
        # Compute loss and gradients w.r.t. all PQC parameters (tuple of 3 arrays)
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Sanitize gradients to avoid NaN/Inf explosions
        grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Keep params finite
        new_params = jax.tree.map(lambda p: jnp.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), new_params)

        # Compute fidelity with updated parameters (unpack tuple)
        measured = model.run_model_batch(ideal_data, new_params)
        fidelity = jax_pure_state_fidelity(ideal_data, measured)

        return opt_state, new_params, loss, fidelity
    
    # Initialize optimizer state
    params = model.get_model_params()
    opt_state = optimizer.init(params)
    
    global_step = 0  # Track global step count across epochs for learning rate schedule
    
    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")
        data_iterator = tqdm(dataloader, desc="Training", total=len(dataloader), leave=False, unit='batch')
        
        # Initialize lists to track metrics for this epoch
        epoch_fidelities = []
        epoch_losses = []
        
        for i, batch in enumerate(data_iterator):
            ideal_data = batch[0]  # Extract input states from batch
            
            # Get current parameters
            params = model.get_model_params()
            
            # Update step
            opt_state, new_params, loss, fidelity = update_step(
                params,
                opt_state, ideal_data
            )
            
            # Update model parameters
            model.set_model_params(new_params)
            
            # Track metrics
            epoch_fidelities.append(float(fidelity))
            epoch_losses.append(float(loss))
            
            # Display current learning rate from schedule (for monitoring only)
            current_lr = schedule(global_step)
            data_iterator.set_postfix_str(
                f"Fidelity (Ideal, Measured): {fidelity:.4e}, Loss: {loss:.4e}, LR: {current_lr:.4e}"
            )
            
            global_step += 1
        
        # Print mean metrics at the end of each epoch
        mean_fidelity = np.mean(epoch_fidelities)
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {e+1} summary - Mean Fidelity: {mean_fidelity:.4e}, Mean Loss: {mean_loss:.4e}")
    
    # Return final mean fidelity from last epoch
    return mean_fidelity


def train_lel_zz_custom_statevec_no_uncomp(model, dataloader, optimizer, schedule,
                                            main_loss_fn=jax_fidelity_loss, epochs=1):
    """
    Train LEL-ZZ model without uncomputation using custom statevector backend.
    
    Without uncomputation, we train PQC to map input -> target_output,
    where target_output is the ideal (noiseless) circuit output.
    """
    
    @jax.jit
    def update_step(params, opt_state, input_data, target_data):
        """Perform a single update step for the model parameters."""

        def loss_fn(params_tuple):
            # Unpack parameters from tuple
            # pre, theta, post = params_tuple
            
            # Run model with current PQC parameters
            measured = model.run_model_batch(input_data, params_tuple)
            # Target is the ideal noiseless output
            per_state_loss = jax.vmap(main_loss_fn, in_axes=(0, 0))(target_data, measured)
            return jnp.mean(per_state_loss)
        
        # Compute loss and gradients w.r.t. all PQC parameters (tuple of 3 arrays)
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Sanitize gradients
        grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Keep params finite
        new_params = jax.tree.map(lambda p: jnp.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), new_params)

        # Compute fidelity with updated parameters (unpack tuple dynamically)
        measured = model.run_model_batch(input_data, new_params)
        fidelity = jax_pure_state_fidelity(target_data, measured)

        return opt_state, new_params, loss, fidelity
    
    # Initialize optimizer state
    params = model.get_model_params()
    opt_state = optimizer.init(params)

    global_step = 0  # Track global step count across epochs for learning rate schedule
    
    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")
        data_iterator = tqdm(dataloader, desc="Training", total=len(dataloader), leave=False, unit='batch')
        
        # Initialize lists to track metrics for this epoch
        epoch_fidelities = []
        epoch_losses = []
        
        for i, batch in enumerate(data_iterator):
            input_data = batch[0]  # Input states
            target_data = batch[1]
            
            # Get current parameters
            params = model.get_model_params()
            
            # Update step
            opt_state, new_params, loss, fidelity = update_step(
                params,
                opt_state, input_data, target_data
            )
            
            # Update model parameters
            model.set_model_params(new_params)
            
            # Track metrics
            epoch_fidelities.append(float(fidelity))
            epoch_losses.append(float(loss))
            
            # Display current learning rate from schedule (for monitoring only)
            current_lr = schedule(global_step)
            data_iterator.set_postfix_str(
                f"Fidelity (Target, Measured): {fidelity:.4e}, Loss: {loss:.4e}, LR: {current_lr:.4e}"
            )
            
            global_step += 1
        
        # Print mean metrics at the end of each epoch
        mean_fidelity = np.mean(epoch_fidelities)
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {e+1} summary - Mean Fidelity: {mean_fidelity:.4e}, Mean Loss: {mean_loss:.4e}")
    
    # Return final mean fidelity from last epoch
    return mean_fidelity


def train_lel_zz_single_block_progressive_no_uncomp(
    model, dataloader, optimizer, schedule, block_idx,
    main_loss_fn=jax_fidelity_loss, epochs=1
):
    """
    Train a single PQC block progressively while keeping previous blocks frozen.
    
    This function is designed for progressive/incremental training where blocks
    are trained one at a time. Previous blocks are frozen using stop_gradient to
    prevent gradient flow, while the current block's parameters are optimized.
    
    Args:
        model: LELZZInterleavedQuaternionCustomStatevecModel instance
        dataloader: JAXDataLoader with (input_states, target_states) batches
        optimizer: Fresh Optax optimizer instance for this block
        schedule: Learning rate schedule function
        block_idx: Which block to train (0-indexed)
        main_loss_fn: Loss function to minimize
        epochs: Number of training epochs for this block
    """
    
    @jax.jit
    def update_step(pre_quats, theta_zz, post_quats, opt_state, input_data, target_data):
        """Single optimization step for current block only."""
        
        def loss_fn(trainable_pre, trainable_theta, trainable_post):
            # Reconstruct full parameter arrays with frozen parts
            if block_idx == 0:
                # First block: no previous blocks to freeze
                full_pre = trainable_pre
                full_theta = trainable_theta
                full_post = trainable_post
            else:
                # Freeze previous blocks with stop_gradient
                frozen_pre = jax.lax.stop_gradient(pre_quats[:block_idx])
                frozen_theta = jax.lax.stop_gradient(theta_zz[:block_idx])
                frozen_post = jax.lax.stop_gradient(post_quats[:block_idx])
                # frozen_pre = pre_quats[:block_idx]
                # frozen_theta = theta_zz[:block_idx]
                # frozen_post = post_quats[:block_idx]
                
                # Concatenate frozen + trainable
                full_pre = jnp.concatenate([frozen_pre, trainable_pre], axis=0)
                full_theta = jnp.concatenate([frozen_theta, trainable_theta], axis=0)
                full_post = jnp.concatenate([frozen_post, trainable_post], axis=0)
            
            # Simulate up to current block
            measured = model.run_model_batch_up_to_block(
                input_data, block_idx, full_pre, full_theta, full_post
            )
            
            return main_loss_fn(target_data, measured)
        
        # Extract trainable parameters (current block only)
        trainable_pre = pre_quats[block_idx:block_idx+1]
        trainable_theta = theta_zz[block_idx:block_idx+1]
        trainable_post = post_quats[block_idx:block_idx+1]
        
        # Compute loss and gradients (only for trainable params)
        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(
            trainable_pre, trainable_theta, trainable_post
        )
        
        # Sanitize gradients
        grads = jax.tree.map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), 
            grads
        )
        
        # Apply optimizer update
        updates, opt_state = optimizer.update(
            grads, opt_state, (trainable_pre, trainable_theta, trainable_post)
        )
        new_pre, new_theta, new_post = optax.apply_updates(
            (trainable_pre, trainable_theta, trainable_post), updates
        )
        
        # Sanitize updated parameters
        new_pre = jnp.nan_to_num(new_pre, nan=0.0, posinf=0.0, neginf=0.0)
        new_theta = jnp.nan_to_num(new_theta, nan=0.0, posinf=0.0, neginf=0.0)
        new_post = jnp.nan_to_num(new_post, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reconstruct full arrays for fidelity computation
        if block_idx == 0:
            full_pre_new = new_pre
            full_theta_new = new_theta
            full_post_new = new_post
        else:
            full_pre_new = jnp.concatenate([pre_quats[:block_idx], new_pre], axis=0)
            full_theta_new = jnp.concatenate([theta_zz[:block_idx], new_theta], axis=0)
            full_post_new = jnp.concatenate([post_quats[:block_idx], new_post], axis=0)
        
        # Compute fidelity with updated parameters
        measured = model.run_model_batch_up_to_block(
            input_data, block_idx, full_pre_new, full_theta_new, full_post_new
        )
        fidelity = jax_pure_state_fidelity(target_data, measured)
        
        return opt_state, new_pre, new_theta, new_post, loss, fidelity
    
    # Initialize optimizer for this block's parameters only
    params = model.get_model_params()
    trainable_params = (
        params['pre_quaternions'][block_idx:block_idx+1],
        params['theta_zz'][block_idx:block_idx+1],
        params['post_quaternions'][block_idx:block_idx+1]
    )
    opt_state = optimizer.init(trainable_params)
    
    global_step = 0
    
    # Training loop
    for e in range(epochs):
        print(f"  Epoch {e + 1}/{epochs}")
        data_iterator = tqdm(
            dataloader, 
            desc=f"  Block {block_idx}", 
            total=len(dataloader), 
            leave=False
        )
        
        epoch_fidelities = []
        epoch_losses = []
        
        for batch in data_iterator:
            input_data, target_data = batch[0], batch[1]
            
            # Get current parameters
            params = model.get_model_params()
            
            # Update step
            opt_state, new_pre, new_theta, new_post, loss, fidelity = update_step(
                params['pre_quaternions'], 
                params['theta_zz'], 
                params['post_quaternions'],
                opt_state, 
                input_data, 
                target_data
            )
            
            # Update model (only current block changes)
            full_pre = params['pre_quaternions'].at[block_idx].set(new_pre[0])
            full_theta = params['theta_zz'].at[block_idx].set(new_theta[0])
            full_post = params['post_quaternions'].at[block_idx].set(new_post[0])
            
            model.set_model_params(full_pre, full_theta, full_post)
            
            # Track metrics
            epoch_fidelities.append(float(fidelity))
            epoch_losses.append(float(loss))

            current_lr = schedule(global_step)
            data_iterator.set_postfix_str(
                f"Fid: {fidelity:.4e}, Loss: {loss:.4e}, LR: {current_lr:.4e}"
            )
            
            global_step += 1
        
        # Epoch summary
        mean_fidelity = np.mean(epoch_fidelities)
        mean_loss = np.mean(epoch_losses)
        print(f"  Block {block_idx+1} Epoch {e+1}: "
              f"Fidelity={mean_fidelity:.4e}, Loss={mean_loss:.4e}")

    blk_start_idx = block_idx * model.num_pqc_layers
    blk_end_idx = (block_idx + 1) * model.num_pqc_layers
    pqc_params = model.get_pqc_params()

    circuit_block_gates = model.base_circuit_ops[blk_start_idx:blk_end_idx]
    circuit_block_gates_noisy = add_rotation_noise_to_base_ops(
        circuit_block_gates, 
        {'x_noise': model.x_noise[blk_start_idx:blk_end_idx], 
         'z_noise': model.z_noise[blk_start_idx:blk_end_idx]}
        )

    
    circuit_block_gates_pqc = circuit_block_gates_noisy + list_LEL_ZZ(
        model.num_qubits,
        pqc_params['pre_angles'][block_idx],
        pqc_params['theta_zz'][block_idx],
        pqc_params['post_angles'][block_idx]
    )

    # print(f"  Fidelity of JUST Block {block_idx}")
    # ideal_out_block = jax_run_many_states(
    #     model.num_qubits,
    #     *build_jax_circuit(circuit_block_gates),
    #     input_data
    # )

    # noisy_out_block = jax_run_many_states(
    #                 model.num_qubits,
    #                 *build_jax_circuit(circuit_block_gates_noisy),
    #                 input_data
    #             )
    # pqc_out_block = jax_run_many_states(
    #                 model.num_qubits,
    #                 *build_jax_circuit(circuit_block_gates_pqc),
    #                 input_data
    #             )

    # block_fidelity_ideal = jax_pure_state_fidelity(ideal_out_block, noisy_out_block)
    # block_fidelity_pqc = jax_pure_state_fidelity(ideal_out_block, pqc_out_block)

    # print(f"    Noisy Block Fidelity (Ideal vs Noisy): {block_fidelity_ideal:.4e}")
    # print(f"    PQC Corrected Block Fidelity (Ideal vs PQC): {block_fidelity_pqc:.4e}")


def train_lel_zz_single_block_individual_no_uncomp(
    model, dataloader, optimizer, schedule, block_idx,
    main_loss_fn=jax_fidelity_loss, epochs=1
):
    """
    Train a single PQC block in isolation (non-cascading).
    
    This function trains each block independently using only its associated gates.
    Unlike progressive training which cascades through previous blocks, this
    simulates ONLY the current block's gates.
    
    Circuit structure: G1, G2, G3, B1, G4, G5, G6, B2, ...
    - Block 0: Trains B1 using only G1-G3
    - Block 1: Trains B2 using only G4-G6
    - etc.
    
    Args:
        model: LELZZInterleavedQuaternionCustomStatevecModel instance
        dataloader: JAXDataLoader with (input_states, target_states) batches
        optimizer: Fresh Optax optimizer instance for this block
        schedule: Learning rate schedule function
        block_idx: Which block to train (0-indexed)
        main_loss_fn: Loss function to minimize
        epochs: Number of training epochs for this block
    """
    
    @jax.jit
    def update_step(pre_quat, theta_zz_val, post_quat, opt_state, input_data, target_data):
        """Single optimization step for isolated block."""
        
        def loss_fn(trainable_pre, trainable_theta, trainable_post):
            # Simulate only this block (no cascading)
            measured = model.run_single_block_batch(
                input_data, block_idx, trainable_pre, trainable_theta, trainable_post
            )
            return main_loss_fn(target_data, measured)
        
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(
            pre_quat, theta_zz_val, post_quat
        )
        
        # Sanitize gradients
        grads = jax.tree.map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), 
            grads
        )
        
        # Apply optimizer update
        updates, opt_state = optimizer.update(
            grads, opt_state, (pre_quat, theta_zz_val, post_quat)
        )
        new_pre, new_theta, new_post = optax.apply_updates(
            (pre_quat, theta_zz_val, post_quat), updates
        )
        
        # Sanitize updated parameters
        new_pre = jnp.nan_to_num(new_pre, nan=0.0, posinf=0.0, neginf=0.0)
        new_theta = jnp.nan_to_num(new_theta, nan=0.0, posinf=0.0, neginf=0.0)
        new_post = jnp.nan_to_num(new_post, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute fidelity with updated parameters
        measured = model.run_single_block_batch(
            input_data, block_idx, new_pre, new_theta, new_post
        )
        fidelity = jax_pure_state_fidelity(target_data, measured)
        
        return opt_state, new_pre, new_theta, new_post, loss, fidelity
    
    # Initialize optimizer for this block's parameters only
    params = model.get_model_params()
    trainable_params = (
        params['pre_quaternions'][block_idx:block_idx+1],
        params['theta_zz'][block_idx:block_idx+1],
        params['post_quaternions'][block_idx:block_idx+1]
    )
    opt_state = optimizer.init(trainable_params)
    
    global_step = 0
    
    # Training loop
    for e in range(epochs):
        print(f"  Epoch {e + 1}/{epochs}")
        data_iterator = tqdm(
            dataloader, 
            desc=f"  Block {block_idx}", 
            total=len(dataloader), 
            leave=False
        )
        
        epoch_fidelities = []
        epoch_losses = []
        
        for batch in data_iterator:
            input_data, target_data = batch[0], batch[1]
            
            # Get current block parameters
            params = model.get_model_params()
            block_pre = params['pre_quaternions'][block_idx:block_idx+1]
            block_theta = params['theta_zz'][block_idx:block_idx+1]
            block_post = params['post_quaternions'][block_idx:block_idx+1]
            
            # Update step
            opt_state, new_pre, new_theta, new_post, loss, fidelity = update_step(
                block_pre, block_theta, block_post,
                opt_state, 
                input_data, 
                target_data
            )
            
            # Update model (only current block changes)
            full_pre = params['pre_quaternions'].at[block_idx].set(new_pre[0])
            full_theta = params['theta_zz'].at[block_idx].set(new_theta[0])
            full_post = params['post_quaternions'].at[block_idx].set(new_post[0])
            
            model.set_model_params(full_pre, full_theta, full_post)
            
            # Track metrics
            epoch_fidelities.append(float(fidelity))
            epoch_losses.append(float(loss))

            current_lr = schedule(global_step)
            data_iterator.set_postfix_str(
                f"Fid: {fidelity:.4e}, Loss: {loss:.4e}, LR: {current_lr:.4e}"
            )
            
            global_step += 1
        
        # Epoch summary
        mean_fidelity = np.mean(epoch_fidelities)
        mean_loss = np.mean(epoch_losses)
        print(f"  Block {block_idx+1} Epoch {e+1}: "
              f"Fidelity={mean_fidelity:.4e}, Loss={mean_loss:.4e}")

    # Diagnostic: Evaluate isolated block performance
    # blk_start_idx = block_idx * model.gate_blocks
    # blk_end_idx = (block_idx + 1) * model.gate_blocks
    # pqc_params = model.get_pqc_params()

    # circuit_block_gates = model.base_circuit_ops[blk_start_idx:blk_end_idx]
    # circuit_block_gates_noisy = add_noise_to_base_ops(
    #     circuit_block_gates, 
    #     model.x_noise[blk_start_idx:blk_end_idx], 
    #     model.z_noise[blk_start_idx:blk_end_idx]
    # )
    
    # circuit_block_gates_pqc = circuit_block_gates_noisy + list_LEL_ZZ(
    #     model.num_qubits,
    #     pqc_params['pre_angles'][block_idx],
    #     pqc_params['theta_zz'][block_idx],
    #     pqc_params['post_angles'][block_idx]
    # )

    # print(f"  Fidelity of JUST Block {block_idx}")
    # ideal_out_block = jax_run_many_states(
    #     model.num_qubits,
    #     *build_jax_circuit(circuit_block_gates),
    #     input_data
    # )

    # noisy_out_block = jax_run_many_states(
    #     model.num_qubits,
    #     *build_jax_circuit(circuit_block_gates_noisy),
    #     input_data
    # )
    
    # pqc_out_block = jax_run_many_states(
    #     model.num_qubits,
    #     *build_jax_circuit(circuit_block_gates_pqc),
    #     input_data
    # )

    # block_fidelity_ideal = jax_pure_state_fidelity(ideal_out_block, noisy_out_block)
    # block_fidelity_pqc = jax_pure_state_fidelity(ideal_out_block, pqc_out_block)

    # print(f"    Noisy Block Fidelity (Ideal vs Noisy): {block_fidelity_ideal:.4e}")
    # print(f"    PQC Corrected Block Fidelity (Ideal vs PQC): {block_fidelity_pqc:.4e}")
