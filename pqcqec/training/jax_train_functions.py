import jax
import jax.numpy as jnp
import optax
import numpy as np
import pennylane as qml
import random
from tqdm import tqdm
from typing import Callable 

from .jax_loss_functions import jax_mse_complex_loss_aligned, jax_pure_state_fidelity, jax_fidelity_loss, jax_hilbert_schmidt_density_loss
from ..simulate.simulate import run_ideal_circuit, run_circuit_with_noise_model
from ..noise.simple_noise import PennylaneNoisyGates

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
    global_step = 0  # Track steps across all epochs
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

            current_lr = schedule(global_step)
            global_step += 1  # Increment global step counter

            data_iterator.set_postfix_str(f"Fidelity (Ideal, Measured): {fidelity:.4e}, Loss: {loss:.4e}, LR: {current_lr:.4e}")
        
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
    global_step = 0  # Track steps across all epochs
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

            current_lr = schedule(global_step)
            global_step += 1  # Increment global step counter

            data_iterator.set_postfix_str(f"Fidelity (Ideal, Measured): {fidelity:.4e}, Loss: {loss:.4e}, LR: {current_lr:.4e}")
        
        # Print mean metrics at the end of each epoch
        mean_fidelity = np.mean(epoch_fidelities)
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {e+1} summary - Mean Fidelity: {mean_fidelity:.4e}, Mean Loss: {mean_loss:.4e}")



def _normalize_quats(q, eps=1e-12):
    """q: (..., 4) -> unit quaternion with w >= 0.
    
    Handles zero-norm quaternions by replacing them with identity quaternion [1,0,0,0].
    """
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
    # Check for zero norm and replace with identity quaternion
    is_zero_norm = norm < eps
    default_q = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    # Normalize if norm is non-zero, otherwise use identity
    q_norm = jnp.where(is_zero_norm, default_q, q / (norm + eps))
    # Enforce w >= 0 to avoid q ~ -q ambiguity
    sign = jnp.where(q_norm[..., :1] < 0, -1.0, 1.0)
    return q_norm * sign

def project_params(params):
    """Project raw params back to valid manifold (unit quats + w>=0)."""
    # params is a dict pytree as returned by model.get_model_params()
    pre_q  = _normalize_quats(params["pre_quaternions"])
    post_q = _normalize_quats(params["post_quaternions"])
    # theta_zz can be left unconstrained; clamp if you want (optional)
    theta_zz = params["theta_zz"]
    return {
        "pre_quaternions":  pre_q,
        "theta_zz":         theta_zz,
        "post_quaternions": post_q,
    }


def train_complex_pqc_model_with_uncomp(
    model,
    dataloader,
    optimizer,
    schedule,
    main_loss_fn=jax_fidelity_loss,        # expects (target, predicted) -> scalar loss
    epochs=1
):
    @jax.jit
    def update_step(params, opt_state, ideal_batch):
        """One optimizer step on a batch of input states."""
        def loss_fn(p):
            # model(...) returns corrected states after noisy circuit + LEL–ZZ PQC
            measured = model(ideal_batch, params=p)          # shape: [B, 2**n]
            # In uncomp mode, target == original ideal input
            return main_loss_fn(ideal_batch, measured)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        # sanitize grads
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params = jax.tree_util.tree_map(lambda p: jnp.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), new_params)
        # **critical**: project back to valid quaternion manifold
        new_params = project_params(new_params)

        # compute batch fidelity (pure states)
        measured = model(ideal_batch, params=new_params)
        fidelity = jax_pure_state_fidelity(ideal_batch, measured)  # mean over batch

        return opt_state, new_params, loss, fidelity

    params = model.get_model_params()  # dict pytree
    opt_state = optimizer.init(params)

    global_step = 0
    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")
        data_iterator = tqdm(dataloader, desc="Training", total=len(dataloader), leave=False, unit='batch')

        epoch_fids, epoch_losses = [], []
        for batch in data_iterator:
            ideal_batch = batch[0]  # [B, 2**n]

            opt_state, params, loss, fid = update_step(params, opt_state, ideal_batch)
            model.set_model_params(params)  # keep model in sync (for outside calls)

            epoch_fids.append(float(fid))
            epoch_losses.append(float(loss))

            lr = schedule(global_step)
            global_step += 1
            data_iterator.set_postfix_str(f"Fid: {fid:.4e} | Loss: {loss:.4e} | LR: {lr:.4e}")

        print(f"Epoch {e+1} summary - Mean Fidelity: {np.mean(epoch_fids):.4e}, Mean Loss: {np.mean(epoch_losses):.4e}")

def train_complex_pqc_model_no_uncomp(
    model,
    dataloader,
    optimizer,
    schedule,
    main_loss_fn=jax_fidelity_loss,        # (target, predicted) -> scalar loss
    epochs=1
):
    # Noise-free simulator to produce the *target* state
    no_noise_model = PennylaneNoisyGates(x_rad=0.0, z_rad=0.0, delta_x=0.0, delta_z=0.0, seed=0)

    @jax.jit
    def update_step(params, opt_state, ideal_batch, target):
        """One optimizer step on a batch; target = noiseless circuit output (precomputed)."""
        def loss_fn(p):
            measured = model(ideal_batch, params=p)  # corrected, noisy+PQC
            return main_loss_fn(target, measured)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params = jax.tree_util.tree_map(lambda p: jnp.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), new_params)
        new_params = project_params(new_params)

        measured = model(ideal_batch, params=new_params)
        fidelity = jax_pure_state_fidelity(target, measured)

        return opt_state, new_params, loss, fidelity

    params = model.get_model_params()
    opt_state = optimizer.init(params)

    global_step = 0
    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")
        data_iterator = tqdm(dataloader, desc="Training", total=len(dataloader), leave=False, unit='batch')

        epoch_fids, epoch_losses = [], []
        for batch in data_iterator:
            ideal_batch = batch[0]  # [B, 2**n]
            
            # Compute target ONCE per batch (outside JIT for efficiency and correctness)
            target = run_circuit_with_noise_model(
                model.circuit_ops, ideal_batch, no_noise_model, model.num_qubits, batched=True
            )  # shape: [B, 2**n]

            opt_state, params, loss, fid = update_step(params, opt_state, ideal_batch, target)
            model.set_model_params(params)

            epoch_fids.append(float(fid))
            epoch_losses.append(float(loss))

            lr = schedule(global_step)
            global_step += 1
            data_iterator.set_postfix_str(f"Fid(target,measured): {fid:.4e} | Loss: {loss:.4e} | LR: {lr:.4e}")

        print(f"Epoch {e+1} summary - Mean Fidelity: {np.mean(epoch_fids):.4e}, Mean Loss: {np.mean(epoch_losses):.4e}")



