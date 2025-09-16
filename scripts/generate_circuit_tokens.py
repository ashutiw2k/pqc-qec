import sys
import os
from tqdm.auto import tqdm

from pathlib import Path

from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))  # Makes other scripts and functions discoverable

import json
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps

from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit
from pqcqec.utils.json_utils import write_json

from pqcqec.utils.args import get_all_valid_args, parse_args

def main():
    required_args = ['qubit_range', 'gate_range', 'config', 'seed', 
                     'gate_dist', 'gpu', 'figure_output', 
                     'force', 'mp_cores', 'uncomp']

    script_description = 'Generate and tokenize ``base/pure`` circuits for `seed` number of circuits per qubit, gate configuration.'

    args = parse_args(required_args, script_description=script_description)
    config = get_all_valid_args(args, include_args=required_args)

    for qubit in config['qubits']:
        for gate in config['gates']:

            data_dir = os.path.join(config['figure_output'], f"{qubit}q_{gate}g_circuit_data")
            os.makedirs(data_dir, exist_ok=True)

            config_seed = int(config['seed'])
            num_circs = config_seed + 1

            config_file = os.path.join(data_dir, "config.json")

            # Save config atomically and minified
            write_json(config_file, config)
            print(f"Config file saved to {config_file}")

            circuit_tokens = []


            if config['force']:
                for root, dirs, files in os.walk(data_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))



            for seed in tqdm(range(num_circs), desc=f"Generating circuits for {qubit} qubits and {gate} gates"):
                qc = generate_random_circuit(
                num_qubits=qubit,
                num_gates=gate,
                seed=seed,
                gate_dist=config['gate_dist'],
                )

                if config['uncomp']:
                    inverse_qc = qc.inverse()
                    uncomp_qc = qc.compose(inverse_qc)
                    qc = uncomp_qc

                qasm_str = dumps(qc)
                qc_ops = tokenize_qiskit_circuit(qc)

                circuit_tokens.append({
                    "seed": seed,
                    "base_circuit_tokens": qc_ops,
                    "base_circuit_qasm": qasm_str
                })
                # print(f"Generated and tokenized circuit for seed {seed}")
            
            tokens_file = os.path.join(data_dir, "circuit_tokens.json")
            write_json(tokens_file, circuit_tokens)
            print(f"Circuit tokens saved to {tokens_file}")


if __name__ == "__main__":
    main()