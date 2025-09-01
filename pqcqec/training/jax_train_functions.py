import jax
import jax.numpy as jnp
import optax
import numpy as np
import pennylane as qml
import random
from tqdm import tqdm
from typing import Callable 

from .jax_loss_functions import jax_mse_complex_loss, jax_pure_state_fidelity, jax_fidelity_loss
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

            current_lr = schedule(i)

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

            current_lr = schedule(i)

            data_iterator.set_postfix_str(f"Fidelity (Ideal, Measured): {fidelity:.4e}, Loss: {loss:.4e}, LR: {current_lr:.4e}")
        
        # Print mean metrics at the end of each epoch
        mean_fidelity = np.mean(epoch_fidelities)
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {e+1} summary - Mean Fidelity: {mean_fidelity:.4e}, Mean Loss: {mean_loss:.4e}")
