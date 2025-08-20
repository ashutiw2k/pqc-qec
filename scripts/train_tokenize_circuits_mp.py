import sys
import os
import gc
import multiprocessing

from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))  # Makes other scripts and functions discoverable

import json
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps

from pqcqec.experiment.pqc_experiment import pqc_experiment_runner


from pqcqec.utils.args import get_all_valid_args, parse_args

def create_circuit_from_ops(ops_list, num_qubits):
    """
    Creates a Qiskit QuantumCircuit from a list of operations.

    Args:
        ops_list (list): List of operations in the format (op, params, qubits).
        num_qubits (int): Number of qubits in the circuit.

    Returns:
        QuantumCircuit: The constructed quantum circuit.
    """

    circuit = QuantumCircuit(num_qubits)
    for op, qubits, params in ops_list:
        if hasattr(circuit, op):
            getattr(circuit, op)(*params, *qubits)
        else:
            raise ValueError(f"Unsupported operation: {op}")
    return circuit


def deep_tuple(x):
    """
    Recursively convert nested lists and tuples into tuples.

    This helper walks the input structure and converts every list or tuple it
    encounters into a tuple, preserving the original nesting and leaving all
    non-sequence elements unchanged.

    Args:
        x: A value that may be a list, tuple, or a nested combination of these,
           containing arbitrary objects.

    Returns:
        A tuple mirroring the structure of x if x is a list or tuple; otherwise the
        original value x.

    Examples:
        >>> deep_tuple([1, (2, [3, 4])])
        (1, (2, (3, 4)))
        >>> deep_tuple("abc")
        'abc'
    """
    return (tuple(deep_tuple(i) for i in x)
            if isinstance(x, (list, tuple)) else x)


def process_seed(args):
    """
    Worker function to run the experiment for a single seed.
    This function is designed to be called by a multiprocessing pool.
    """
    # Unpack all the arguments passed by the main process
    seed, qubit, gate, gate_blocks, data_dir, config, poor_fid_seed = args

    # --- 1. Pre-computation Checks (same as your original loop) ---
    output_file = os.path.join(data_dir, f"{seed}.json")
    if config['redo'] and seed != config.get('seed'): # Check if config_seed exists
        return ('skipped', seed, f"Skipping seed {seed} (not the specified seed for redo)")

    if os.path.exists(output_file) and not config['force'] and not config['redo']:
        return ('skipped', seed, f"Seed {seed} file exists.")

    if seed in poor_fid_seed and not config['force'] and not config['redo']:
        return ('skipped', seed, f"Seed {seed} is in poor fidelity seeds.")

    print(f"Running experiment with Qubits: {qubit}, Gates: {gate}, Seed: {seed}")

    # --- 2. Run the main experiment ---
    try:
        base_circ, pqc_circ, mean_fidelity_ideal_pqc, pqc_params = pqc_experiment_runner(
            num_qubits=qubit,
            num_gates=gate,
            gate_blocks=gate_blocks,
            pqc_blocks=config['pqc_blocks'],
            epochs=config['epochs'],
            num_data=config['num_data'],
            num_test=config['num_test'],
            gate_dist=config['gate_dist'],
            gpu=config['gpu'],
            seed=seed,
            batch_size=config['batch'],
            return_fidelity=False
        )
        gc.collect() # Good practice to clear memory within the worker

    except Exception as e:
        return ('error', f"Error on seed {seed}: {e}")

    # --- 3. Process and return the result ---
    if mean_fidelity_ideal_pqc > 0.95:
        token_data = {
            'seed': seed,
            'fidelity': mean_fidelity_ideal_pqc,
            'pqc_params': pqc_params.tolist(),
            'base_circuit_tokens': base_circ,
            'pqc_circuit_tokens': pqc_circ,
            'base_circuit_qasm': dumps(create_circuit_from_ops(base_circ, qubit)),
            'pqc_circuit_qasm': dumps(create_circuit_from_ops(pqc_circ, qubit)),
        }
        # Instead of writing the file here, we return the data to the main process
        return ('success', seed, (output_file, token_data))
    else:
        # Return the data for a poor fidelity run
        val = (qubit, gate, seed, mean_fidelity_ideal_pqc, deep_tuple(pqc_params.tolist()))
        return ('poor_fidelity', seed, val)



def main():
    # Parse command line arguments
    required_args = ['qubit_range', 'gate_range', 'gate_blocks', 'pqc_blocks', 'epochs', 'config', 'seed',
                     'num_data', 'num_test', 'gate_dist', 'gpu', 'batch', 'figure_output', 'noise_dist', 'force', 'redo']
    script_description = 'Train and Tokenize Circuits with error correcting interleaved PQC up for `seed` number of circuits per qubit, gate configuration.'

    args = parse_args(required_args, script_description=script_description)

    mp_stat_out = "mp_stats.json"
    mp_stats = []

    config = get_all_valid_args(args, include_args=required_args)
    poor_fid_params_all = []
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
            poor_fid_file = os.path.join(data_dir, "poor_fid_params.json")
            
            with open(config_file, 'w') as f:
                json.dump(config, f, default=str)
                f.close()
            print(f"Config file saved to {config_file}")

            if os.path.exists(poor_fid_file):
                with open(poor_fid_file, 'r') as f:
                    poor_fid_params = json.load(f)
                    f.close()
            else:
                poor_fid_params = []

            poor_fid_params = [deep_tuple(param) for param in poor_fid_params]  # Convert lists to tuples for easier comparison

            poor_fid_seed = [params[2] for params in poor_fid_params if params[0] == qubit and params[1] == gate]

            # for seed in range(num_circs):
            tasks = []
            for seed in range(num_circs):
                tasks.append((seed, qubit, gate, gate_blocks, data_dir, config, poor_fid_seed))

            # --- Run in Parallel ---
            # Use half of available CPU cores to allow other processes to run smoothly
            num_processes = int(multiprocessing.cpu_count() // 2)
            print(f"\nStarting parallel processing with {num_processes} cores...")

            pool = multiprocessing.Pool(processes=num_processes)
            try:
                # Use pool.imap_unordered for memory efficiency and a nice progress bar
                # It processes tasks in parallel and yields results as they complete
                results = list(pool.imap_unordered(process_seed, tasks))
                pool.close()
                pool.join()
            except Exception as e:
                print(f"Error occurred: {e}")
            finally:
                pool.terminate()  # Ensure the pool is terminated in case of an error

            print("\nProcessing results...")
            for status, seed, data in results:
                if status == 'success':
                    output_file, token_data = data
                    print(f"PQC Circuit Fidelity good for seed {seed} : {token_data['fidelity']}")
                    mp_stats.append({"seed": seed, "fidelity": token_data['fidelity']})

                    with open(output_file, 'w') as f:
                        json.dump(token_data, f, default=str)
                elif status == 'poor_fidelity':
                    print(f"Poor PQC Circuit Fidelity for seed {seed} : {data[3]}")
                    mp_stats.append({"seed": seed, "fidelity": data[3]})

                    poor_fid_params.append(data)
                elif status == 'skipped':
                    # Optionally print skipped messages
                    print(data)
                    pass
                elif status == 'error':
                    print(data) # Print any errors that occurred in the workers


                # Run the experiment
                # if config['redo'] and not seed == config_seed:
                #     print(f"Skipping seed {seed} as it is not the specified seed for redo.")
                #     continue

                # output_file = os.path.join(data_dir, f"{seed}.json")
                # if os.path.exists(output_file) and not config['force'] and not config['redo']:
                #     print(f"Seed {seed} token file {output_file} already exists. Skipping...")
                #     continue
                # elif seed in poor_fid_seed and not config['force'] and not config['redo']:
                #     print(f"Seed {seed} is in poor fidelity seeds. Skipping...")
                #     continue

                # print(f"Running experiment with Qubits: {qubit}, Gates: {gate}, Seed: {seed}")
                # # continue

                # base_circ, pqc_circ, mean_fidelity_ideal_pqc, pqc_params = experiment_runner(
                #     num_qubits=qubit,
                #     num_gates=gate,
                #     gate_blocks=gate_blocks,
                #     pqc_blocks=config['pqc_blocks'],
                #     epochs=config['epochs'],
                #     num_data=config['num_data'],
                #     num_test=config['num_test'],
                #     gate_dist=config['gate_dist'],
                #     gpu=config['gpu'],
                #     seed=seed,
                #     batch_size=config['batch']
                # )

                # gc.collect()  # Clear memory after each experiment

                # # print(f"base circuit tokens: {base_circ}")
                # # print(f"pqc circuit tokens: {pqc_circ}")

                # if mean_fidelity_ideal_pqc > 0.95:
                #     token_data = {
                #         'seed': seed,
                #         'fidelity': mean_fidelity_ideal_pqc,
                #         'pqc_params': pqc_params.tolist(),
                #         'base_circuit_tokens': base_circ,
                #         'pqc_circuit_tokens': pqc_circ,
                #         'base_circuit_qasm': dumps(create_circuit_from_ops(base_circ, qubit)),
                #         'pqc_circuit_qasm': dumps(create_circuit_from_ops(pqc_circ, qubit)),
                #         # For any future use
                #         # 'qubit': qubit,
                #         # 'gate': gate,
                #         # 'gate_block': gate_blocks,
                #     }
                #     with open(output_file, 'w') as f:
                #         json.dump(token_data, f, default=str)
                #         f.close()
                # else:
                #     val = (qubit, gate, seed, mean_fidelity_ideal_pqc, deep_tuple(pqc_params.tolist())) # Since lists are not hashable, convert it all to tuples. 
                #     # print(f"Poor fidelity circuit found: \n{val}")
                #     poor_fid_params.append(val)

            print()
            
            poor_fid_params = sorted(set(poor_fid_params), key=lambda x: (x[0], x[1], x[2], x[3]))  # Remove duplicates
            poor_fid_params_all.extend(poor_fid_params)
            with open(poor_fid_file, 'w') as f:
                json.dump(poor_fid_params, f)
                f.close()

    poor_params_len = len(poor_fid_params_all)
    print(f"{poor_params_len} circuits not saved for the following poor fidelity parameters:")
    for params in poor_fid_params_all:
        print(f" - Qubits: {params[0]}, Gates: {params[1]}, Seed: {params[2]}, PQC Fidelity: {params[3]}")

    if config['force']:
        print("Writing recent fidelity stats to file...")
        with open(mp_stat_out, 'w') as f:
            json.dump(mp_stats, f)
            f.close()

if __name__ == "__main__":
    main()