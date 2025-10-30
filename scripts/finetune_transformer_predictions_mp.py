#!/usr/bin/env python3
"""
Fine-tune transformer-predicted PQC angles using multiprocessing.

This script loads transformer-generated circuit predictions, initializes PQC models
with the predicted angles, and fine-tunes them to maximize fidelity. Uses multiprocessing
to parallelize across multiple circuits.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from contextlib import contextmanager

import numpy as np
import jax
import jax.numpy as jnp
import optax

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout and stderr (including tqdm progress bars)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

from pqcqec.noise.simple_noise import PennylaneNoisyGates
from pqcqec.models.pqc_models import ZXZInterleavedAngleCustomStatevecModel
from pqcqec.training.jax_train_functions import train_lel_zz_custom_statevec_no_uncomp
from pqcqec.training.jax_loss_functions import jax_pure_state_fidelity
from pqcqec.simulate.simulate import get_input_data
from pqcqec.simulate.jax_statevector import build_jax_circuit, jax_run_many_states
from pqcqec.utils.jax_utils import JAXStateMeasuredDataset, JAXDataLoader


def load_transformer_predictions(filepath: str) -> List[Tuple[list, list, list]]:
    """
    Load transformer-generated circuit predictions from JSONL file.
    
    Args:
        filepath: Path to JSONL file with transformer predictions
        
    Returns:
        List of (base_circuit, pqc_circuit, init_pqc_angles) tuples
    """
    transformer_circuit_pqc_data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            data = list(json.loads(line)['circuit_tokens'])
            pqc_angles = []
            base_circuit = []
            pqc_circuit = []
            
            for item in data:
                gate_data = item.split(':')
                if len(gate_data) != 2:
                    # Base gate without PQC
                    base_circuit.append((gate_data[0], [0], []))
                    pqc_circuit.append((gate_data[0], [0], []))
                else:
                    # PQC gate with angle
                    pqc_angles.append(float(gate_data[1]))
                    pqc_circuit.append((gate_data[0], [0], [float(gate_data[1])]))
            
            transformer_circuit_pqc_data.append((base_circuit, pqc_circuit, pqc_angles))
    
    return transformer_circuit_pqc_data


def process_single_circuit(args_tuple):
    """
    Process a single circuit: fine-tune PQC angles and evaluate.
    
    Args:
        args_tuple: Tuple of (circuit_idx, circuit_data, hyperparams)
        
    Returns:
        Dictionary with results for this circuit
    """
    circuit_idx, (base_circuit, pqc_circuit, init_pqc_angles), hyperparams = args_tuple
    
    # Unpack hyperparameters
    num_qubits = hyperparams['num_qubits']
    num_gates = hyperparams['num_gates']
    gate_blocks = hyperparams['gate_blocks']
    num_data = hyperparams['num_data']
    num_test = hyperparams['num_test']
    batch_size = hyperparams['batch_size']
    epochs = hyperparams['epochs']
    verbose = hyperparams.get('verbose', False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing Circuit {circuit_idx}")
        print(f"{'='*60}")
    
    # Set random seeds for reproducibility
    jax_prng_keys = jax.random.split(jax.random.PRNGKey(circuit_idx), 6).flatten()
    
    if verbose:
        print(f"Using Seed: {circuit_idx}, JAX PRNG Keys: {jax_prng_keys}")
    
    # Generate ideal training data (input states)
    ideal_train_data = get_input_data(num_qubits, num_data, seed=int(jax_prng_keys[0]))
    
    # Create noise model
    noise_model = PennylaneNoisyGates(seed=int(jax_prng_keys[1]))
    
    # Generate noise arrays
    np.random.seed(circuit_idx)
    x_noise_arr = np.random.uniform(
        noise_model.x_noise_min,
        noise_model.x_noise_max,
        (num_gates,)
    ).astype(np.float32)
    z_noise_arr = np.random.uniform(
        noise_model.z_noise_min,
        noise_model.z_noise_max,
        (num_gates,)
    ).astype(np.float32)
    
    if verbose:
        print(f"X-noise range: [{x_noise_arr.min():.4f}, {x_noise_arr.max():.4f}]")
        print(f"Z-noise range: [{z_noise_arr.min():.4f}, {z_noise_arr.max():.4f}]")
        print(f"Circuit has {len(base_circuit)} operations")
    
    # Initialize model with angle-based PQC
    model = ZXZInterleavedAngleCustomStatevecModel(
        base_circuit_ops=base_circuit,
        num_qubits=num_qubits,
        x_noise=x_noise_arr,
        z_noise=z_noise_arr,
        pqc_blocks=1,
        gate_blocks=gate_blocks,
        seed=int(jax_prng_keys[4]),
        pqc_type='zxz'
    )
    
    # Set initial PQC angles from transformer prediction
    init_angles = jnp.array(init_pqc_angles, dtype=jnp.float32).reshape(1, 1, 3)
    model.set_model_params(new_params={'pre_angles': init_angles})
    
    params = model.get_model_params_to_store()
    total_params = sum([p.size for p in params.values()])
    
    if verbose:
        print(f"Model initialized with {total_params} trainable parameters")
        for key in params:
            print(f"  {key}: {params[key].shape}")
    
    # Generate ideal target states for training
    if verbose:
        print("Generating ideal target states for training...")
    
    base_jax_ops = build_jax_circuit(base_circuit)
    ideal_train_outputs = jax_run_many_states(num_qubits, *base_jax_ops, ideal_train_data)
    
    # Create dataset and dataloader
    train_dataset = JAXStateMeasuredDataset(ideal_train_data, ideal_train_outputs)
    train_dataloader = JAXDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=int(jax_prng_keys[2])
    )
    
    # Define optimizer with learning rate schedule
    total_steps = int(num_data / batch_size)
    warmup_steps = int(0.1 * total_steps)
    restart_period = int(0.25 * total_steps)
    
    init_lr = 1e-4
    peak_lr = 5e-3
    min_lr = 5e-5
    
    # Warmup schedule
    warmup = optax.linear_schedule(
        init_value=init_lr,
        end_value=peak_lr,
        transition_steps=warmup_steps
    )
    
    # Cosine decay with restarts
    def cosine_with_restart_schedule(step):
        step_in_period = step % restart_period
        cosine = 0.5 * (1 + jnp.cos(jnp.pi * step_in_period / restart_period))
        return min_lr + (peak_lr - min_lr) * cosine
    
    # Stitch warmup + cosine
    schedule = optax.join_schedules(
        schedules=[warmup, cosine_with_restart_schedule],
        boundaries=[warmup_steps]
    )
    
    # Optimizer chain
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-8),
        optax.add_decayed_weights(weight_decay=1e-5),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)
    )
    
    if verbose:
        print(f"\nStarting training with {epochs} epochs...")
    
    # Train the model (suppress training output unless verbose)
    if verbose:
        final_train_fidelity = train_lel_zz_custom_statevec_no_uncomp(
            model,
            train_dataloader,
            optimizer,
            schedule,
            epochs=epochs
        )
    else:
        with suppress_stdout():
            final_train_fidelity = train_lel_zz_custom_statevec_no_uncomp(
                model,
                train_dataloader,
                optimizer,
                schedule,
                epochs=epochs
            )
    
    # Test the model
    if verbose:
        print(f"\nGenerating test data...")
    
    ideal_test_input_data = get_input_data(num_qubits, num_test, seed=int(jax_prng_keys[5]))
    
    # Generate noisy outputs
    if verbose:
        print(f'Running circuit with noise on test data...')
    
    noisy_test_ops = []
    for i, op in enumerate(base_circuit):
        noisy_test_ops.append(op)
        gate, qubits, params_list = op
        for q in qubits:
            noisy_test_ops.append(('rx', [q], [float(x_noise_arr[min(i, len(x_noise_arr)-1)])]))
            noisy_test_ops.append(('rz', [q], [float(z_noise_arr[min(i, len(z_noise_arr)-1)])]))
    
    noisy_test_jax_ops = build_jax_circuit(noisy_test_ops)
    noisy_state = jax_run_many_states(num_qubits, *noisy_test_jax_ops, ideal_test_input_data)
    
    # Generate ideal (noiseless) output states
    if verbose:
        print(f'Generating ideal (noiseless) output states...')
    
    base_test_jax_ops = build_jax_circuit(base_circuit)
    ideal_out_state = jax_run_many_states(num_qubits, *base_test_jax_ops, ideal_test_input_data)
    
    # Run PQC model on test data
    if verbose:
        print(f'Running fine-tuned PQC model on test data...')
    
    pqc_state = model.run_model_batch(ideal_test_input_data)
    
    # Compute fidelities
    batched_fidelity = jax.vmap(jax_pure_state_fidelity, in_axes=(0, 0))
    fidelity_ideal_noisy = batched_fidelity(ideal_out_state, noisy_state)
    fidelity_ideal_pqc = batched_fidelity(ideal_out_state, pqc_state)
    
    # Evaluate transformer prediction (before fine-tuning)
    if verbose:
        print(f'Evaluating transformer prediction (before fine-tuning)...')
    
    noisy_transformer_ops = []
    for i, op in enumerate(pqc_circuit):
        noisy_transformer_ops.append(op)
        gate, qubits, params_list = op
        # Skip adding noise after PQC gates (rx, rz)
        if gate in ['rx', 'rz']:
            continue
        for q in qubits:
            noisy_transformer_ops.append(('rx', [q], [float(x_noise_arr[min(i, len(x_noise_arr)-1)])]))
            noisy_transformer_ops.append(('rz', [q], [float(z_noise_arr[min(i, len(z_noise_arr)-1)])]))
    
    noisy_transformer_jax_ops = build_jax_circuit(noisy_transformer_ops)
    noisy_transformer_state = jax_run_many_states(num_qubits, *noisy_transformer_jax_ops, ideal_test_input_data)
    fidelity_ideal_transformer = batched_fidelity(ideal_out_state, noisy_transformer_state)
    
    # Collect results
    results = {
        'circuit_idx': circuit_idx,
        'num_params': total_params,
        'final_train_fidelity': float(final_train_fidelity),
        'test_fidelity_noisy_mean': float(jnp.mean(fidelity_ideal_noisy)),
        'test_fidelity_noisy_std': float(jnp.std(fidelity_ideal_noisy)),
        'test_fidelity_pqc_mean': float(jnp.mean(fidelity_ideal_pqc)),
        'test_fidelity_pqc_std': float(jnp.std(fidelity_ideal_pqc)),
        'test_fidelity_transformer_mean': float(jnp.mean(fidelity_ideal_transformer)),
        'test_fidelity_transformer_std': float(jnp.std(fidelity_ideal_transformer)),
        'init_angles': init_pqc_angles,
        'final_angles': model.get_pqc_params()['pre_angles'].tolist(),
        'x_noise_range': [float(x_noise_arr.min()), float(x_noise_arr.max())],
        'z_noise_range': [float(z_noise_arr.min()), float(z_noise_arr.max())],
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Circuit {circuit_idx} Results:")
        print(f"  Fidelity (Ideal, Noisy):       {results['test_fidelity_noisy_mean']:.4f} ± {results['test_fidelity_noisy_std']:.4f}")
        print(f"  Fidelity (Ideal, Transformer): {results['test_fidelity_transformer_mean']:.4f} ± {results['test_fidelity_transformer_std']:.4f}")
        print(f"  Fidelity (Ideal, Fine-tuned):  {results['test_fidelity_pqc_mean']:.4f} ± {results['test_fidelity_pqc_std']:.4f}")
        print(f"  Improvement over transformer:  {(results['test_fidelity_pqc_mean'] - results['test_fidelity_transformer_mean']):.4f}")
        print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune transformer-predicted PQC angles with multiprocessing'
    )
    
    # Input/Output
    parser.add_argument(
        '-i', '--input-file',
        type=str,
        required=True,
        help='Path to JSONL file with transformer predictions'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='nogit/finetune_results',
        help='Directory to save results (default: nogit/finetune_results)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='finetuned',
        help='Prefix for output files (default: finetuned)'
    )
    
    # Circuit parameters
    parser.add_argument(
        '-q', '--num-qubits',
        type=int,
        default=1,
        help='Number of qubits (default: 1)'
    )
    parser.add_argument(
        '-g', '--num-gates',
        type=int,
        default=10,
        help='Number of gates in circuit (default: 10)'
    )
    parser.add_argument(
        '-k', '--gate-blocks',
        type=int,
        default=10,
        help='Number of gate blocks (default: 10)'
    )
    
    # Training parameters
    parser.add_argument(
        '-n', '--num-data',
        type=int,
        default=1000,
        help='Number of training samples (default: 1000)'
    )
    parser.add_argument(
       '-t', '--num-test',
        type=int,
        default=100,
        help='Number of test samples (default: 100)'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=10,
        help='Batch size (default: 10)'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    
    # Multiprocessing parameters
    parser.add_argument(
        '-p', '--num-processes',
        type=int,
        default=None,
        help='Number of parallel processes (default: half of CPU cores)'
    )
    
    # Other options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed progress for each circuit'
    )
    
    args = parser.parse_args()
    
    # Determine number of processes
    if args.num_processes is None:
        args.num_processes = max(1, mp.cpu_count() // 2)
    
    print(f"\n{'='*60}")
    print("Fine-tune Transformer PQC Predictions")
    print(f"{'='*60}")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Circuit: {args.num_qubits}q, {args.num_gates}g, {args.gate_blocks} blocks")
    print(f"Training: {args.num_data} samples, {args.batch_size} batch, {args.epochs} epochs")
    print(f"Test: {args.num_test} samples")
    print(f"Multiprocessing: {args.num_processes} processes")
    print(f"{'='*60}\n")
    
    # Load transformer predictions
    print("Loading transformer predictions...")
    all_circuit_data = load_transformer_predictions(args.input_file)
    print(f"Loaded {len(all_circuit_data)} circuit predictions")
    
    
    # Prepare hyperparameters
    hyperparams = {
        'num_qubits': args.num_qubits,
        'num_gates': args.num_gates,
        'gate_blocks': args.gate_blocks,
        'num_data': args.num_data,
        'num_test': args.num_test,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'verbose': args.verbose,
    }
    
    # Prepare arguments for multiprocessing
    process_args = [
        (i, data, hyperparams)
        for i, data in enumerate(all_circuit_data)
    ]
    
    # Create output directory and file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_{args.num_qubits}q_{args.num_gates}g_results.jsonl"
    )
    
    # Process circuits in parallel
    num_procs = min(args.num_processes, mp.cpu_count())
    print(f"\nProcessing {len(process_args)} circuits with {num_procs} processes...")
    print(f"Writing results to: {output_file}")
    
    if not args.verbose:
        print("(Training output suppressed. Use -v for verbose mode)")

    results = []
    
    # Open file for writing results as they complete
    with open(output_file, 'w') as f:
        if num_procs == 1:
            # Single process for easier debugging
            for i, arg in enumerate(process_args):
                if not args.verbose:
                    print(f"Progress: {i+1}/{len(process_args)}", end='\r')
                result = process_single_circuit(arg)
                results.append(result)
                # Write immediately
                f.write(json.dumps(result) + '\n')
                f.flush()  # Ensure it's written to disk
            if not args.verbose:
                print()  # New line after progress
        else:
            # Multiprocessing with progress updates
            with mp.Pool(processes=num_procs) as pool:
                for i, result in enumerate(pool.imap(process_single_circuit, process_args)):
                    results.append(result)
                    # Write immediately
                    f.write(json.dumps(result) + '\n')
                    f.flush()  # Ensure it's written to disk
                    if not args.verbose:
                        print(f"Progress: {i+1}/{len(process_args)} circuits completed, fidelity: {result['test_fidelity_pqc_mean']:.4f}", end='\r')
                if not args.verbose:
                    print()  # New line after progress
    
    print(f"\nCompleted processing {len(results)} circuits")
    
    # Compute and display summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    noisy_fids = [r['test_fidelity_noisy_mean'] for r in results]
    transformer_fids = [r['test_fidelity_transformer_mean'] for r in results]
    finetuned_fids = [r['test_fidelity_pqc_mean'] for r in results]
    improvements = [ft - tr for ft, tr in zip(finetuned_fids, transformer_fids)]
    
    print(f"Noisy Circuit Fidelity:       {np.mean(noisy_fids):.4f} ± {np.std(noisy_fids):.4f}")
    print(f"Transformer Fidelity:         {np.mean(transformer_fids):.4f} ± {np.std(transformer_fids):.4f}")
    print(f"Fine-tuned Fidelity:          {np.mean(finetuned_fids):.4f} ± {np.std(finetuned_fids):.4f}")
    print(f"Improvement (fine-tune - transformer): {np.mean(improvements):.4f} ± {np.std(improvements):.4f}")
    print(f"\nCircuits where fine-tuning helped: {sum(1 for imp in improvements if imp > 0.01)}/{len(improvements)}")
    print(f"Circuits where fine-tuning hurt:   {sum(1 for imp in improvements if imp < -0.01)}/{len(improvements)}")
    print(f"{'='*60}\n")
    
    # Save summary statistics
    summary_file = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_{args.num_qubits}q_{args.num_gates}g_summary.json"
    )
    
    summary = {
        'num_circuits': len(results),
        'hyperparameters': hyperparams,
        'noisy_fidelity_mean': float(np.mean(noisy_fids)),
        'noisy_fidelity_std': float(np.std(noisy_fids)),
        'transformer_fidelity_mean': float(np.mean(transformer_fids)),
        'transformer_fidelity_std': float(np.std(transformer_fids)),
        'finetuned_fidelity_mean': float(np.mean(finetuned_fids)),
        'finetuned_fidelity_std': float(np.std(finetuned_fids)),
        'improvement_mean': float(np.mean(improvements)),
        'improvement_std': float(np.std(improvements)),
        'num_improved': sum(1 for imp in improvements if imp > 0.01),
        'num_degraded': sum(1 for imp in improvements if imp < -0.01),
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {summary_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()
