import os
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))  # Makes other scripts and functions discoverable

from pqcqec.experiment.pqc_experiment import pqc_experiment_runner
from pqcqec.utils.args import get_all_valid_args, parse_args

def main():
    # Parse command line arguments
    required_args = ['qubit_range', 'gate_range', 'gate_blocks', 'pqc_blocks', 'epochs', 'num_data', 'num_test', 'gate_dist', 'gpu', 'seed', 'batch', 'figure_output']
    script_description = 'Train an Interleaved PQC model for a specific qubit and gate configuration and plot the test experiment results.'
    args = parse_args(required_args, script_description=script_description)

    config = get_all_valid_args(args, include_args=required_args)
    seed = int(config['seed']) if config['seed'] is not None else 0
    # Run the experiment
    fidelity_noisy, fidelity_pqc = pqc_experiment_runner(
        num_qubits=config['qubits'][0],
        num_gates=config['gates'][0],
        gate_blocks=config['gate_blocks'],
        pqc_blocks=config['pqc_blocks'],
        epochs=config['epochs'],
        num_data=config['num_data'],
        num_test=config['num_test'],
        gate_dist=config['gate_dist'],
        gpu=config['gpu'],
        seed=seed,
        batch_size=config['batch'],
        return_fidelity=True
    )

    print(f"Final Fidelity (Noisy): {np.mean(fidelity_noisy):.4e}")
    print(f"Final Fidelity (PQC): {np.mean(fidelity_pqc):.4e}")

    plt.figure()
    plt.plot(fidelity_noisy, label="Noisy")
    plt.plot(fidelity_pqc, label="PQC")
    plt.xlabel("Experiment Number")
    plt.ylabel("Fidelity")
    plt.title(f"Fidelity {config['gate_blocks']}g, {config['pqc_blocks']}b, {config['num_data']}n, {config['batch']}a")
    plt.legend()
    plt.savefig(os.path.join(config['figure_output'], f"Fidelity_{config['qubits'][0]}q_{config['gates'][0]}g_seed{seed}.png"))

if __name__ == "__main__":
    main()