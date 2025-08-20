import jax
import jax.numpy as jnp
import optax
import numpy as np
import pennylane as qml
import random
from tqdm import tqdm
from typing import Callable 

from .jax_loss_functions import jax_mse_complex_loss, jax_pure_state_fidelity

def train_pqc_model(model, dataloader, optimizer, schedule, main_loss_fn=jax_mse_complex_loss, epochs=1):

    @jax.jit
    def update_step(params, opt_state, ideal_data):
        """Perform a single update step for the model parameters."""
        
        def loss_fn(p):
            measured = model(ideal_data, params=p)

            return main_loss_fn(ideal_data, measured)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Fidelity after parameter update
        measured = model(ideal_data, params=new_params)
        fidelity = jax_pure_state_fidelity(ideal_data, measured)

        return opt_state, new_params, loss, fidelity

    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")
        # Reset op state for each epoch    
        opt_state = optimizer.init(model.pqc_params)
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

            opt_state, params, loss, fidelity = update_step(model.pqc_params, opt_state, ideal_data)
            model.pqc_params = params
            
            # Track metrics
            epoch_fidelities.append(float(fidelity))
            epoch_losses.append(float(loss))

            current_lr = schedule(i)

            data_iterator.set_postfix_str(f"Fidelity (Ideal, Measured): {fidelity:.4e}, Loss: {loss:.4e}, LR: {current_lr:.4e}")
        
        # Print mean metrics at the end of each epoch
        mean_fidelity = np.mean(epoch_fidelities)
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {e+1} summary - Mean Fidelity: {mean_fidelity:.4e}, Mean Loss: {mean_loss:.4e}")
