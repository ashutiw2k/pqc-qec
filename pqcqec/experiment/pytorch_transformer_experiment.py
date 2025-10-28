"""
PyTorch Transformer Experiment Runner.

This module provides the main experiment runner for training a transformer
to predict PQC angles for quantum error correction.
"""

import torch
import os
import json
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

from ..training.pytorch_train_transformer import (
    ZZRingAnglePredictorPyTorch,
    train_transformer_progressive,
    train_transformer_individual,
)
from ..utils.pytorch_utils import (
    CircuitDatasetPyTorch,
    collate_circuit_batch,
)


def run_pytorch_transformer_experiment(
    data_path: str,
    n_qubits: int,
    gate_blocks: int,
    k_random: int,
    noise_x_rad: float,
    noise_z_rad: float,
    epochs: int,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[str] = None,
    mode: str = 'progressive',
    seed: int = 0,
    train_split: float = 0.8,
) -> Dict[str, Any]:
    """
    Run transformer training experiment on a dataset of circuits.
    
    Args:
        data_path: Path to JSON/JSONL file with circuits
        n_qubits: Number of qubits (model is specific to this)
        gate_blocks: Number of base gates per PQC block
        k_random: Number of random initial states to use
        noise_x_rad: X rotation noise magnitude
        noise_z_rad: Z rotation noise magnitude
        epochs: Number of training epochs
        batch_size: Batch size (usually 1 for variable-length circuits)
        learning_rate: Learning rate for Adam optimizer
        device: torch device (defaults to CUDA if available)
        checkpoint_dir: Directory to save checkpoints
        mode: 'progressive' or 'individual' training mode
        seed: Random seed
        train_split: Fraction of data for training (rest is test)
    
    Returns:
        results: Dictionary with training metrics and model info
    """
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Training mode: {mode}")
    print(f"Data path: {data_path}")
    print(f"n_qubits: {n_qubits}, gate_blocks: {gate_blocks}")
    print(f"k_random: {k_random}, noise_x: {noise_x_rad}, noise_z: {noise_z_rad}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = CircuitDatasetPyTorch(data_path, n_qubits=n_qubits)
    print(f"Loaded {len(dataset)} circuits for {n_qubits} qubits")
    
    # Split into train/test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Train size: {train_size}, Test size: {test_size}")
    
    # Create dataloaders with custom collate function
    from functools import partial
    
    collate_fn = partial(
        collate_circuit_batch,
        gate_blocks=gate_blocks,
        device=device
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    print("\nInitializing model...")
    model = ZZRingAnglePredictorPyTorch(
        gate_blocks=gate_blocks,
        n_qubits=n_qubits,
        max_blocks=100
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )
    
    # Create checkpoint directory if specified
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Train based on mode
    print(f"\nStarting training in {mode} mode...")
    
    if mode == 'progressive':
        train_transformer_progressive(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            gate_blocks=gate_blocks,
            n_qubits=n_qubits,
            k_random=k_random,
            noise_x_rad=noise_x_rad,
            noise_z_rad=noise_z_rad,
            epochs=epochs,
            device=device,
            seed=seed
        )
    elif mode == 'individual':
        train_transformer_individual(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            gate_blocks=gate_blocks,
            n_qubits=n_qubits,
            k_random=k_random,
            noise_x_rad=noise_x_rad,
            noise_z_rad=noise_z_rad,
            epochs=epochs,
            device=device,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'progressive' or 'individual'")
    
    # Save final model
    if checkpoint_dir is not None:
        final_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'n_qubits': n_qubits,
            'gate_blocks': gate_blocks,
            'mode': mode,
        }, final_path)
        print(f"\nSaved final model to: {final_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_results = evaluate_model(
        model=model,
        dataloader=test_loader,
        gate_blocks=gate_blocks,
        n_qubits=n_qubits,
        k_random=k_random,
        noise_x_rad=noise_x_rad,
        noise_z_rad=noise_z_rad,
        mode=mode,
        device=device,
        seed=seed
    )
    
    print(f"Test set - Mean fidelity: {test_results['mean_fidelity']:.6f}")
    print(f"Test set - Mean loss: {test_results['mean_loss']:.6f}")
    
    # Prepare results dictionary
    results = {
        'n_qubits': n_qubits,
        'gate_blocks': gate_blocks,
        'k_random': k_random,
        'noise_x_rad': noise_x_rad,
        'noise_z_rad': noise_z_rad,
        'epochs': epochs,
        'mode': mode,
        'learning_rate': learning_rate,
        'num_parameters': num_params,
        'train_size': train_size,
        'test_size': test_size,
        'test_mean_fidelity': test_results['mean_fidelity'],
        'test_mean_loss': test_results['mean_loss'],
    }
    
    # Save results
    if checkpoint_dir is not None:
        results_path = os.path.join(checkpoint_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {results_path}")
    
    return results


def evaluate_model(
    model: ZZRingAnglePredictorPyTorch,
    dataloader: DataLoader,
    gate_blocks: int,
    n_qubits: int,
    k_random: int,
    noise_x_rad: float,
    noise_z_rad: float,
    mode: str,
    device: torch.device,
    seed: int = 0
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Returns:
        Dictionary with 'mean_fidelity' and 'mean_loss'
    """
    from ..simulate.pytorch_pqc_simulator import (
        simulate_block_progressive,
        simulate_block_individual,
        compute_target_states_progressive,
        compute_target_states_individual,
        compute_fidelity_loss,
    )
    from ..utils.pytorch_utils import (
        create_circuit_ops_from_data,
        generate_random_initial_states,
        generate_fixed_noise,
    )
    import math
    import numpy as np
    
    model.eval()
    all_losses = []
    all_fidelities = []
    
    with torch.no_grad():
        for batch in dataloader:
            for circuit in batch:
                # Extract circuit data
                circuit_ops = create_circuit_ops_from_data(circuit)
                num_gates = len(circuit_ops)
                
                # Calculate number of blocks
                num_blocks = math.ceil(num_gates / gate_blocks) if num_gates > 0 else 1
                
                # Generate initial states
                input_states = generate_random_initial_states(
                    n_qubits, k_random, device, seed=seed
                )
                
                # Generate fixed noise
                x_noise, z_noise = generate_fixed_noise(
                    num_gates, noise_x_rad, noise_z_rad, seed=seed + circuit['idx']
                )
                x_noise = x_noise.to(device)
                z_noise = z_noise.to(device)
                
                # Previous angles buffer
                PREV_K = 4
                prev_angles_buffer = torch.zeros(
                    (PREV_K, model.angles_per_block), device=device
                )
                
                # Store predicted angles for progressive mode
                all_predicted_angles = []
                
                # Evaluate block by block
                for block_idx in range(num_blocks):
                    # Predict angles
                    predicted_angles = model.forward_single_block(
                        circuit_ops, block_idx, prev_angles_buffer, device
                    )
                    
                    all_predicted_angles.append(predicted_angles)
                    
                    # Reshape to LEL-ZZ format
                    pre_angles = predicted_angles[:3*n_qubits].view(n_qubits, 3)
                    theta_zz = predicted_angles[3*n_qubits:4*n_qubits]
                    post_angles = predicted_angles[4*n_qubits:7*n_qubits].view(n_qubits, 3)
                    
                    if mode == 'progressive':
                        # Collect previous blocks' angles
                        prev_pqc_angles = []
                        for prev_idx in range(block_idx):
                            prev_ang = all_predicted_angles[prev_idx]
                            prev_pre = prev_ang[:3*n_qubits].view(n_qubits, 3)
                            prev_theta = prev_ang[3*n_qubits:4*n_qubits]
                            prev_post = prev_ang[4*n_qubits:7*n_qubits].view(n_qubits, 3)
                            prev_pqc_angles.append((prev_pre, prev_theta, prev_post))
                        
                        # Simulate progressive
                        predicted_states = simulate_block_progressive(
                            input_states, block_idx, gate_blocks, n_qubits,
                            circuit_ops, x_noise, z_noise,
                            prev_pqc_angles,
                            (pre_angles, theta_zz, post_angles),
                            device
                        )
                        
                        # Target states
                        target_states = compute_target_states_progressive(
                            input_states, block_idx, gate_blocks, n_qubits,
                            circuit_ops, device
                        )
                    
                    else:  # individual
                        # Simulate individual
                        predicted_states = simulate_block_individual(
                            input_states, block_idx, gate_blocks, n_qubits,
                            circuit_ops, x_noise, z_noise,
                            (pre_angles, theta_zz, post_angles),
                            device
                        )
                        
                        # Target states
                        target_states = compute_target_states_individual(
                            input_states, block_idx, gate_blocks, n_qubits,
                            circuit_ops, device
                        )
                    
                    # Compute loss
                    loss = compute_fidelity_loss(predicted_states, target_states)
                    fidelity = 1.0 - loss.item()
                    
                    all_losses.append(loss.item())
                    all_fidelities.append(fidelity)
                    
                    # Update previous angles buffer
                    prev_angles_buffer = torch.roll(prev_angles_buffer, shifts=-1, dims=0)
                    prev_angles_buffer[-1] = predicted_angles
    
    return {
        'mean_loss': np.mean(all_losses),
        'mean_fidelity': np.mean(all_fidelities),
    }


def load_trained_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> ZZRingAnglePredictorPyTorch:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: torch device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config
    n_qubits = checkpoint['n_qubits']
    gate_blocks = checkpoint['gate_blocks']
    
    # Create model
    model = ZZRingAnglePredictorPyTorch(
        gate_blocks=gate_blocks,
        n_qubits=n_qubits,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  n_qubits: {n_qubits}, gate_blocks: {gate_blocks}")
    print(f"  mode: {checkpoint.get('mode', 'unknown')}")
    
    return model
