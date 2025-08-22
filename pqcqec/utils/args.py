
import argparse
import sys
import os
import torch
from .constants import *
import json


def load_config(path: str) -> dict:
    """Load a JSON configuration file and return its contents as a dict."""
    if not os.path.isfile(path):
        print(f"Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, 'r') as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"Error loading config file '{path}': {e}", file=sys.stderr)
        sys.exit(1)


def str2bool(v):
    """Convert string representation of truth to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Configuration for all arguments
# Each argument definition can contain the following keys:
# - 'flags': List of command-line flags (e.g., ['--config', '-c'])
# - 'type': Data type for the argument (str, int, float, dict, etc.)
# - 'nargs': Number of arguments ('?', '+', '*', or specific count)
# - 'help': Help text description for the argument
# - 'choices': Valid choices (can be a list or callable returning a list)
# - 'default': Default value if not provided
# - 'action': Special action like 'store_true' for boolean flags
# - 'config_aliases': Alternative names when reading from config file
# - 'required': Whether the argument is mandatory
# - 'const': Constant value when nargs='?' and argument is present without value
ARG_DEFINITIONS = {
    'config': {
        'flags': ['--config', '-c'],
        'type': str,
        'help': 'Path to YAML config file'
    },
    'qubit_range': {
        'flags': ['--qubit_range', '-q'],
        'type': int,
        'nargs': '+',
        'help': 'One or three ints: single qubit count or min, max qubits with step',
        'config_aliases': ['qubits'],
        'required': True
    },
    'gate_range': {
        'flags': ['--gate_range', '-g'],
        'type': int,
        'nargs': '+',
        'help': 'One or three ints: single gate count or min, max gates with step',
        'config_aliases': ['gates'],
        'required': True
    },
    'pqc_function': {
        'flags': ['--pqc_function', '-p'],
        'type': str,
        'help': 'Name of the PQC function to use',
        'choices': lambda: PQC_MAPPINGS.keys(),
        'required': True
    },
    'pqc_blocks': {
        'flags': ['--pqc_blocks', '-b'],
        'type': int,
        'nargs': '?',
        'help': 'Number of blocks of PQC function to append',
        'default': 1
    },
    'model': {
        'flags': ['--model', '-m'],
        'type': str,
        'help': 'Which model (mapping) to use',
        'choices': lambda: PENNYLANE_MODELS.keys(),
        'required': True
    },
    'gate_blocks': {
        'flags': ['--gate_blocks', '-k'],
        'type': int,
        'nargs': '?',
        'help': 'Number of gates after which to interleve PQC Block',
        'default': 1
    },
    'figure_output': {
        'flags': ['--figure_output', '-o'],
        'type': str,
        'nargs': '?',
        'help': 'Output path for generated figures',
        'default': 'plots/'
    },
    'epochs': {
        'flags': ['--epochs', '-e'],
        'type': int,
        'nargs': '?',
        'help': 'Number of training epochs (optional)',
        'default': 5
    },
    'num_data': {
        'flags': ['--num_data', '-n'],
        'type': int,
        'nargs': '?',
        'help': 'Number of training data samples (optional)',
        'default': 5000
    },
    'num_test': {
        'flags': ['--num_test', '-t'],
        'type': int,
        'nargs': '?',
        'help': 'Number of values to test against (optional)',
        'default': 50
    },
    'num_val': {
        'flags': ['--num_val', '-v'],
        'type': int,
        'nargs': '?',
        'help': 'Number of values to validate during training loop (optional)',
        'default': 50
    },
    'learning_rate': {
        'flags': ['--learning_rate', '-l'],
        'type': float,
        'nargs': '?',
        'help': 'Learning Rate for optimizer (optional)',
        'default': 0.005
    },
    'seed': {
        'flags': ['--seed'],
        'nargs': '?',
        'help': 'Seed to initialize randomizer with (optional)'
    },
    'gate_dist': {
        'flags': ['--gate_dist', '-d'],
        'type': dict,
        'nargs': '?',
        'help': 'Gate Distribution in Random Circuit (optional)'
    },
    'noise_dist': {
        'flags': ['--noise_dist', '-z'],
        'type': dict,
        'nargs': '?',
        'help': 'Noise Distribution in Circuit (optional)'
    },
    'gpu': {
        'flags': ['--gpu'],
        'action': 'store_true',
        'help': 'Run on GPU if available',
        'default': False
    },
    'save_circuit': {
        'flags': ['--save_circuit'],
        'action': 'store_true',
        'help': 'Save the circuit to the figure output folder',
        'default': False
    },
    'batch': {
        'flags': ['--batch', '-a'],
        'type': int,
        'nargs': '?',
        'help': 'Number of batches value to pass to the dataloader (optional)',
        'default': 1
    },
    'backend': {
        'flags': ['--backend'],
        'type': str,
        'help': 'Backend to use for circuit generation and simulation (default: pennylane)',
        'default': 'pennylane'
    },
    'force': {
        'flags': ['--force'],
        'nargs': '?',
        'const': True,
        'type': str2bool,
        'help': 'Force deletion and recreation of ALL existing output files (use --force or --force=true/false)',
        'default': False
    },
    'redo': {
        'flags': ['--redo'],
        'nargs': '?',
        'const': True,
        'type': str2bool,
        'help': 'Redo training and recreate the output file for the specified parameters and seed (use --redo or --redo=true/false)',
        'default': False
    },
    'mp_cores': {
        'flags': ['--mp_cores', '--mp-cores'],
        'type': int,
        'nargs': '?',
        'help': 'Number of CPU cores for multiprocessing. 0=auto, -1=all cores (default: 0)',
        'default': 0
    }
}


def create_parser(include_args=None, script_description='Parse arguments for PQC experiment runner.'):
    """Create argument parser based on configuration."""
    parser = argparse.ArgumentParser(description=script_description)
    
    for arg_name, arg_config in ARG_DEFINITIONS.items():
        if include_args is None or arg_name in include_args:
            kwargs = {k: v for k, v in arg_config.items() 
                     if k not in ['flags', 'config_aliases', 'required']}
            
            # Handle callable choices
            if 'choices' in kwargs and callable(kwargs['choices']):
                kwargs['choices'] = kwargs['choices']()
            
            # Remove default to avoid argparse defaults when using config
            if 'default' in kwargs:
                del kwargs['default']
                
            parser.add_argument(*arg_config['flags'], **kwargs)
    
    return parser


def get_arg_value(args, config, arg_name):
    """Get argument value with priority: CLI args > config > default."""
    arg_def = ARG_DEFINITIONS[arg_name]
    
    # Check CLI argument
    cli_val = getattr(args, arg_name, None)
    if cli_val is not None:
        return cli_val
    
    # Check config file (primary name and aliases)
    if arg_name in config:
        return config[arg_name]
    
    for alias in arg_def.get('config_aliases', []):
        if alias in config:
            return config[alias]
    
    # Return default
    return arg_def.get('default')


def normalize_range(val, name: str):
    """Normalize a range specification to a (min, max) or (start, stop, step) tuple."""
    if isinstance(val, int):
        return [val, val]
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            return [val[0], val[0]]
        if len(val) == 2:
            return [val[0], val[1]]
        if len(val) == 3:
            return [val[0], val[1], val[2]]
        if len(val) > 3:
            start = val[0]
            stop = val[-1] + (val[-1] - val[-2]) if len(val) > 1 else val[0] + 1
            step = val[1] - val[0] if len(val) > 1 else 1
            rlist = [start, stop, step]
            if list(range(*rlist)) == list(val):
                return rlist
            print(f"Invalid {name}: cannot infer range parameters from sequence", file=sys.stderr)
            sys.exit(1)
    print(f"Invalid {name}: must be one, two, or three integers, or a sequence matching a range", file=sys.stderr)
    sys.exit(1)


def parse_args(include_args=None, script_description='Parse arguments for PQC experiment runner.'):
    """Parse command line arguments."""
    parser = create_parser(include_args, script_description)
    return parser.parse_args()


def get_all_valid_args(args, include_args=None):
    """Main function to get all validated arguments."""
    config = {}
    
    # Load config if specified
    if (include_args is None or 'config' in include_args) and getattr(args, 'config', None):
        config = load_config(args.config)
        print(f"Loaded base config from {args.config} :\n{config}\n")
    
    # Extract all values
    result = {}
    missing_required = []
    
    for arg_name, arg_def in ARG_DEFINITIONS.items():
        if include_args is None or arg_name in include_args:
            value = get_arg_value(args, config, arg_name)
            
            # Check required arguments
            if arg_def.get('required') and value is None:
                missing_required.append(arg_name)
            
            result[arg_name] = value
    
    if missing_required:
        print(f"Missing required parameters: {', '.join(missing_required)}", file=sys.stderr)
        sys.exit(1)
    
    # Handle special processing
    device = torch.device('cuda' if result['gpu'] and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Normalize ranges
    if result['qubit_range']:
        qubit_range = normalize_range(result['qubit_range'], 'qubit_range')
        result['qubits'] = list(range(qubit_range[0], qubit_range[1] + 1, *qubit_range[2:]))
    
    if result['gate_range']:
        gate_range = normalize_range(result['gate_range'], 'gate_range')
        result['gates'] = list(range(gate_range[0], gate_range[1] + 1, *gate_range[2:]))
    
    # Create output directory
    if result['figure_output']:
        os.makedirs(result['figure_output'], exist_ok=True)
    
    result['device'] = device
    
    print(f"Final Config after parsing arguments:\n{result}\n")
    return result
