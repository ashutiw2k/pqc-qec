import sys
import os
import gc

from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))  # Makes other scripts and functions discoverable

from qiskit.qasm2 import dumps

from pqcqec.experiment.pqc_experiment import pqc_experiment_custom_statevec_runner
from pqcqec.circuits.generate import create_qiskit_circuit_from_ops

from pqcqec.utils.args import get_all_valid_args, parse_args
from pqcqec.utils.json_utils import write_json


def main():
    # Parse command line arguments
    required_args = ['qubit_range', 'gate_range', 'gate_blocks', 'pqc_blocks', 'epochs', 'config', 'seed',
                     'num_data', 'num_test', 'gate_dist', 'gpu', 'batch', 'figure_output', 'noise_dist', 
                     'force', 'redo', 'uncomp', 'noise_config']
    script_description = 'Train and Tokenize Circuits with error correcting interleaved PQC up for `seed` number of circuits per qubit, gate configuration.'

    args = parse_args(required_args, script_description=script_description)
    config = get_all_valid_args(args, include_args=required_args)
    
    poor_total = 0
    good_total = 0
    gate_blocks = config['gate_blocks']

    for qubit in config['qubits']:
        for gate in config['gates']:

            config_seed = int(config['seed'])
            num_circs = config_seed + 1

            if not config['redo']:
                print(f"Generating atmost {config_seed} set of tokens for Qubits: {qubit}, Gates: {gate}, Gate Blocks: {gate_blocks}")
            else:
                print(f"Redoing training and tokenization for Qubits: {qubit}, Gates: {gate}, Gate Blocks: {gate_blocks} with seed {config_seed}")

            data_dir = os.path.join(config['figure_output'], f"{qubit}q_{gate}g_{gate_blocks}blk_data")
            os.makedirs(data_dir, exist_ok=True)

            if config['force']:
                for root, dirs, files in os.walk(data_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))

            config_file = os.path.join(data_dir, "config.json")
            # poor_fid_file = os.path.join(data_dir, "poor_fid_params.json")
            # good_fid_file = os.path.join(data_dir, "good_fid_params.json")
            good_fid_dir = os.path.join(data_dir, "good_fidelity")
            poor_fid_dir = os.path.join(data_dir, "poor_fidelity")

            # Save config atomically and minified (exclude gate_sequence_noise_rules with tuple keys)
            config_to_save = {k: v for k, v in config.items() if k != 'gate_sequence_noise_rules'}
            if config.get('noise_config'):
                config_to_save['noise_config'] = config['noise_config']
            write_json(config_file, config_to_save)
            print(f"Config file saved to {config_file}")

            # Ensure output directories exist up front
            os.makedirs(good_fid_dir, exist_ok=True)
            os.makedirs(poor_fid_dir, exist_ok=True)

            for seed in range(num_circs):
                # Paths for outputs
                good_file = os.path.join(good_fid_dir, f"{seed}.json")
                poor_file = os.path.join(poor_fid_dir, f"{seed}.json")

                # Respect redo/force semantics
                if config['redo'] and seed != config.get('seed'):
                    print(f"Skipping seed {seed} (not the specified seed for redo)")
                    continue

                # Check for existing output files directly on the filesystem
                if not config['force'] and not config['redo'] and (os.path.exists(good_file) or os.path.exists(poor_file)):
                    print(f"Seed {seed} already processed (output file exists).")
                    continue

                print(f"Running experiment with Qubits: {qubit}, Gates: {gate}, Seed: {seed}")

                base_circ, pqc_circ, mean_fidelity_ideal_pqc, pqc_params = pqc_experiment_custom_statevec_runner(
                    num_qubits=qubit,
                    num_gates=gate,
                    gate_blocks=gate_blocks,
                    pqc_blocks=config['pqc_blocks'],
                    epochs=config['epochs'],
                    num_data=config['num_data'],
                    num_test=config['num_test'],
                    gate_dist=config['gate_dist'],
                    noise_dist=config['noise_dist'],
                    gpu=config['gpu'],
                    seed=seed,
                    batch_size=config['batch'],
                    add_uncomputation=config['uncomp'],
                    gate_sequence_noise_rules=config.get('gate_sequence_noise_rules', None)
                )
                gc.collect()

                # Handle pqc_params - could be dict or array depending on model
                if isinstance(pqc_params, dict):
                    pqc_params_serializable = pqc_params
                elif hasattr(pqc_params, 'tolist'):
                    pqc_params_serializable = pqc_params.tolist()
                else:
                    pqc_params_serializable = pqc_params

                print(base_circ)
                print(pqc_circ)

                token_data = {
                    'seed': seed,
                    'fidelity': mean_fidelity_ideal_pqc,
                    'pqc_params': pqc_params_serializable,
                    'base_circuit_tokens': base_circ,
                    'pqc_circuit_tokens': pqc_circ,
                    'base_circuit_qasm': dumps(create_qiskit_circuit_from_ops(base_circ, qubit)),
                    'pqc_circuit_qasm': dumps(create_qiskit_circuit_from_ops(pqc_circ, qubit)),
                    'noise_config': config.get('noise_config', None),
                }
                is_good = mean_fidelity_ideal_pqc > 0.95
                out_path = good_file if is_good else poor_file
                write_json(out_path, token_data)

                if is_good:
                    print(f"PQC Circuit Fidelity good for seed {seed} : {mean_fidelity_ideal_pqc}")
                    good_total += 1
                else:
                    print(f"Poor PQC Circuit Fidelity for seed {seed} : {mean_fidelity_ideal_pqc}")
                    poor_total += 1

            print()
            print(f"Configuration complete: Qubits={qubit}, Gates={gate}, Blocks={gate_blocks}")
            print(f"  Good fidelity: {good_total}, Poor fidelity: {poor_total}")

    print("\nAll configurations complete.")

if __name__ == "__main__":
    main()