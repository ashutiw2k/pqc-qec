import sys
import os
import gc
import multiprocessing
import time
import platform
import resource
import signal
import atexit
from datetime import datetime

from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))  # Makes other scripts and functions discoverable

import json
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps

from pqcqec.experiment.pqc_experiment import pqc_experiment_runner
from pqcqec.circuits.generate import create_qiskit_circuit_from_ops

from pqcqec.utils.args import get_all_valid_args, parse_args
from pqcqec.utils.json_utils import write_json, load_seed_fid_map, save_seed_fid_map

# psutil is optional; use if available for richer stats (child processes)
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _proc_snapshot():
    """Return a snapshot of process resource metrics (best-effort)."""
    snap = {}
    try:
        ru_self = resource.getrusage(resource.RUSAGE_SELF)
        snap.update({
            'ru_utime_sec': ru_self.ru_utime,
            'ru_stime_sec': ru_self.ru_stime,
            # ru_maxrss: Linux: KiB, macOS: bytes. We'll include units below.
            'ru_maxrss': ru_self.ru_maxrss,
            'ru_maxrss_units': 'KiB_on_linux_bytes_on_macos',
            'ru_minflt': ru_self.ru_minflt,
            'ru_majflt': ru_self.ru_majflt,
            'ru_inblock': getattr(ru_self, 'ru_inblock', 0),
            'ru_oublock': getattr(ru_self, 'ru_oublock', 0),
            'ru_nvcsw': getattr(ru_self, 'ru_nvcsw', 0),
            'ru_nivcsw': getattr(ru_self, 'ru_nivcsw', 0),
        })
    except Exception:
        pass

    if psutil is not None:
        try:
            p = psutil.Process(os.getpid())
            with p.oneshot():
                mi = p.memory_info()
                ct = p.cpu_times()
                snap.update({
                    'rss_bytes': getattr(mi, 'rss', None),
                    'vms_bytes': getattr(mi, 'vms', None),
                    'num_threads': p.num_threads(),
                    'cpu_user_time_sec': getattr(ct, 'user', None),
                    'cpu_system_time_sec': getattr(ct, 'system', None),
                })
                try:
                    io = p.io_counters()
                    snap.update({
                        'io_read_bytes': getattr(io, 'read_bytes', None),
                        'io_write_bytes': getattr(io, 'write_bytes', None),
                    })
                except Exception:
                    pass
        except Exception:
            pass

    return snap

def process_seed(args):
    """Run a single seed experiment and persist results.

    Args:
        args: Tuple of
            - seed (int)
            - qubit (int)
            - gate (int)
            - gate_blocks (int)
            - good_dir (str): directory for good-fidelity outputs
            - poor_dir (str): directory for poor-fidelity outputs
            - config (dict)

    Returns:
        Tuple (status, seed, data):
            - status: 'success' | 'poor_fidelity' | 'skipped' | 'error'
            - seed: the seed id
            - data: fidelity (float) for success/poor_fidelity, or message (str) for skipped/error
    """
    # Unpack and name args explicitly for clarity
    seed, qubit, gate, gate_blocks, good_dir, poor_dir, config = args

    # Paths for outputs (dirs are created in main)
    good_file = os.path.join(good_dir, f"{seed}.json")
    poor_file = os.path.join(poor_dir, f"{seed}.json")

    # Respect redo/force semantics
    if config['redo'] and seed != config.get('seed'):
        return {
            'status': 'skipped', 'seed': seed, 'qubits': qubit, 'gates': gate, 'gate_blocks': gate_blocks,
            'message': f"Skipping seed {seed} (not the specified seed for redo)"
        }
    
    # OPTIMIZATION: Check for existing output files directly on the filesystem
    # This avoids loading a potentially massive set of processed seeds into the main process's memory.
    if not config['force'] and not config['redo'] and (os.path.exists(good_file) or os.path.exists(poor_file)):
        return {
            'status': 'skipped', 'seed': seed, 'qubits': qubit, 'gates': gate, 'gate_blocks': gate_blocks,
            'message': f"Seed {seed} already processed (output file exists)."
        }

    print(f"Running experiment with Qubits: {qubit}, Gates: {gate}, Seed: {seed}")

    # Run the main experiment
    # Timing + resource snapshots
    run_start_wall = time.perf_counter()
    snap_before = _proc_snapshot()

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
            noise_dist=config['noise_dist'],
            gpu=config['gpu'],
            seed=seed,
            batch_size=config['batch'],
            add_uncomputation=config['uncomp']
        )
        gc.collect()
    except Exception as e:
        snap_after = _proc_snapshot()
        run_end_wall = time.perf_counter()
        wall_time_sec = run_end_wall - run_start_wall
        return {
            'status': 'error', 'seed': seed, 'qubits': qubit, 'gates': gate, 'gate_blocks': gate_blocks,
            'message': f"Error on seed {seed}: {e}",
            'wall_time_sec': wall_time_sec,
            'resource_before': snap_before, 'resource_after': snap_after,
        }

    # Persist results and return compact status
    token_data = {
        'seed': seed,
        'fidelity': mean_fidelity_ideal_pqc,
        'pqc_params': pqc_params.tolist(),
        'base_circuit_tokens': base_circ,
        'pqc_circuit_tokens': pqc_circ,
        'base_circuit_qasm': dumps(create_qiskit_circuit_from_ops(base_circ, qubit)),
        'pqc_circuit_qasm': dumps(create_qiskit_circuit_from_ops(pqc_circ, qubit)),
    }
    is_good = mean_fidelity_ideal_pqc > 0.95
    out_path = good_file if is_good else poor_file
    write_json(out_path, token_data)

    snap_after = _proc_snapshot()
    run_end_wall = time.perf_counter()
    wall_time_sec = run_end_wall - run_start_wall

    # Compute deltas
    def _delta(key):
        a = snap_after.get(key)
        b = snap_before.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a - b
        return None

    run_stat = {
        'status': 'success' if is_good else 'poor_fidelity',
        'seed': seed,
        'qubits': qubit,
        'gates': gate,
        'gate_blocks': gate_blocks,
        'fidelity': float(mean_fidelity_ideal_pqc),
        'wall_time_sec': wall_time_sec,
        'resource_before': snap_before,
        'resource_after': snap_after,
        'deltas': {
            'cpu_user_time_sec': _delta('cpu_user_time_sec') or _delta('ru_utime_sec'),
            'cpu_system_time_sec': _delta('cpu_system_time_sec') or _delta('ru_stime_sec'),
            'rss_bytes': _delta('rss_bytes'),
            'vms_bytes': _delta('vms_bytes'),
            'io_read_bytes': _delta('io_read_bytes'),
            'io_write_bytes': _delta('io_write_bytes'),
            'ru_inblock': _delta('ru_inblock'),
            'ru_oublock': _delta('ru_oublock'),
            'ru_nvcsw': _delta('ru_nvcsw'),
            'ru_nivcsw': _delta('ru_nivcsw'),
        },
    }
    return run_stat



def main():
    # Parse command line arguments
    required_args = ['qubit_range', 'gate_range', 'gate_blocks', 'pqc_blocks', 'epochs', 'config', 'seed',
                     'num_data', 'num_test', 'gate_dist', 'gpu', 'batch', 'figure_output', 'noise_dist', 
                     'force', 'redo', 'mp_cores', 'uncomp']
    script_description = 'Train and Tokenize Circuits with error correcting interleaved PQC up for `seed` number of circuits per qubit, gate configuration.'

    args = parse_args(required_args, script_description=script_description)

    # OPTIMIZATION: `runs` list is removed to prevent storing all results in memory.
    # Individual run stats will be streamed to a file instead.
    mp_stats = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'host': platform.node(),
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'python': platform.python_version(),
        },
        'config_summary': {},
        'overall': {
            'total_wall_time_sec': 0.0,
            'sum_run_wall_time_sec': 0.0,
            'sum_cpu_user_time_sec': 0.0,
            'sum_cpu_system_time_sec': 0.0,
            'sum_io_read_bytes': 0,
            'sum_io_write_bytes': 0,
            'total_runs': 0,
            'runs_good_fidelity': 0,
            'runs_poor_fidelity': 0,
            'runs_skipped': 0,
            'runs_error': 0,
            'peak_ru_maxrss': 0,
            'peak_ru_maxrss_units': 'KiB_on_linux_bytes_on_macos',
        }
    }

    overall_start = time.perf_counter()
    ru_maxrss_peak = 0

    config = get_all_valid_args(args, include_args=required_args)
    # Keep only compact counters in memory across runs
    poor_total = 0
    good_total = 0
    gate_blocks = config['gate_blocks']

    # -----------------------------
    # Safe-shutdown autosave plumbing
    # -----------------------------
    state = {
        'good_fid_file': None,
        'poor_fid_file': None,
        # Small buffers (seed -> fidelity) to minimize memory
        'good_buf': {},
        'poor_buf': {},
        'current_qubit': None,
        'current_gate': None,
        'last_flush_ts': 0.0,
        'flush_interval_sec': 1200.0,  # also flush periodically
        'processed_since_flush': 0,
        'autosave_every': 1000,       # flush every N processed results
    }

    def _flush_seed_maps(reason="manual"):
        """Merge buffered entries into JSON files and clear buffers (best-effort)."""
        try:
            gfile = state.get('good_fid_file')
            pfile = state.get('poor_fid_file')
            gbuf = state.get('good_buf') or {}
            pbuf = state.get('poor_buf') or {}

            if gfile and gbuf:
                current = load_seed_fid_map(gfile)
                for k, v in list(gbuf.items()):
                    try:
                        current[int(k)] = float(v)
                    except Exception:
                        continue
                save_seed_fid_map(gfile, current)
                state['good_buf'].clear()

            if pfile and pbuf:
                current = load_seed_fid_map(pfile)
                for k, v in list(pbuf.items()):
                    try:
                        current[int(k)] = float(v)
                    except Exception:
                        continue
                save_seed_fid_map(pfile, current)
                state['poor_buf'].clear()

            state['last_flush_ts'] = time.time()
            state['processed_since_flush'] = 0
        except Exception:
            # Avoid raising inside signal/atexit context
            pass

    def _signal_handler(signum, frame):  # pragma: no cover - best-effort
        try:
            print(f"\nReceived signal {signum}; attempting to flush seed maps before exit...")
        except Exception:
            pass
        _flush_seed_maps(reason=f"signal_{signum}")
        # Re-raise KeyboardInterrupt for SIGINT to trigger outer handlers/cleanup
        if signum == getattr(signal, 'SIGINT', None):
            raise KeyboardInterrupt

    # Register atexit and signal handlers once
    atexit.register(lambda: _flush_seed_maps(reason="atexit"))
    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass

    # Shallow config summary for context in the stats
    mp_stats['config_summary'] = {
        'qubits': config.get('qubits'),
        'gates': config.get('gates'),
        'gate_blocks': config.get('gate_blocks'),
        'pqc_blocks': config.get('pqc_blocks'),
        'epochs': config.get('epochs'),
        'num_data': config.get('num_data'),
        'num_test': config.get('num_test'),
        'batch': config.get('batch'),
        'gate_dist': config.get('gate_dist'),
        'noise_dist': config.get('noise_dist'),
        'gpu': config.get('gpu'),
        'seed_max': config.get('seed'),
        'redo': config.get('redo'),
        'force': config.get('force'),
        'uncomp': config.get('uncomp'),
    }

    # Resolve worker count from --mp_cores
    try:
        system_cpu_count = int(multiprocessing.cpu_count())
    except Exception:
        system_cpu_count = 1
    configured_mp_cores = config.get('mp_cores')
    try:
        configured_mp_cores = int(configured_mp_cores) if configured_mp_cores is not None else 0
    except Exception:
        configured_mp_cores = 0
    if configured_mp_cores == -1:
        resolved_workers = max(1, system_cpu_count)
    elif configured_mp_cores == 0:
        # Auto heuristic: leave 1-2 cores free
        resolved_workers = max(1, system_cpu_count - 2 if system_cpu_count >= 4 else system_cpu_count)
    else:
        resolved_workers = min(max(1, configured_mp_cores), system_cpu_count)

    mp_stats['overall'].update({
        'system_cpu_count': system_cpu_count,
        'configured_mp_cores': configured_mp_cores,
        'resolved_workers': resolved_workers,
    })

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
            good_fid_file = os.path.join(data_dir, "good_fid_params.json")
            good_fid_dir = os.path.join(data_dir, "good_fidelity")
            poor_fid_dir = os.path.join(data_dir, "poor_fidelity")

            # Save config atomically and minified
            write_json(config_file, config)
            print(f"Config file saved to {config_file}")
            
            # OPTIMIZATION: Do not load all processed seeds into a set.
            # The check is now done via filesystem in the worker process.

            # Update autosave state for this config
            state.update({
                'good_fid_file': good_fid_file,
                'poor_fid_file': poor_fid_file,
                'current_qubit': qubit,
                'current_gate': gate,
            })

            # Create a generator to avoid holding the whole task list in memory
            # Ensure output directories exist up front
            os.makedirs(good_fid_dir, exist_ok=True)
            os.makedirs(poor_fid_dir, exist_ok=True)

            def task_iter():
                for seed in range(num_circs):
                    # OPTIMIZATION: Do not pass the large processed_seed_set to each worker.
                    yield (seed, qubit, gate, gate_blocks, good_fid_dir, poor_fid_dir, config)

            # Per-config stats containers
            config_start = time.perf_counter()
            cfg_ru_maxrss_peak = 0
            cfg_counts = {
                'total_runs': 0,
                'runs_good_fidelity': 0,
                'runs_poor_fidelity': 0,
                'runs_skipped': 0,
                'runs_error': 0,
                'sum_run_wall_time_sec': 0.0,
                'sum_cpu_user_time_sec': 0.0,
                'sum_cpu_system_time_sec': 0.0,
                'sum_io_read_bytes': 0,
                'sum_io_write_bytes': 0,
            }

            # --- Run in Parallel ---
            num_processes = int(resolved_workers)
            print(f"\nStarting parallel processing with {num_processes} cores (system={system_cpu_count}, configured={configured_mp_cores})...")

            # Use the 'spawn' context for multiprocessing to ensure cross-platform compatibility
            for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'TBB_NUM_THREADS'):
                os.environ.setdefault(_var, '1')
            
            pool = multiprocessing.get_context("spawn").Pool(processes=num_processes)
            
            # OPTIMIZATION: Open a file to stream run results as JSON Lines (JSONL).
            # This avoids accumulating all results in a list in memory.
            config_runs_path = os.path.join(data_dir, 'mp_stats_runs.jsonl')
            
            try:
                with open(config_runs_path, 'w') as f_runs:
                    chunksize = max(1, num_circs // (num_processes * 4) or 1)
                    results_iter = pool.imap_unordered(process_seed, task_iter(), chunksize=chunksize)

                    print("\nProcessing results...")
                    for res in results_iter:
                        status = res.get('status')
                        seed = res.get('seed')
                        fid = res.get('fidelity')

                        if status in ('success', 'poor_fidelity'):
                            # Write the full run stat to the JSONL file instead of appending to a list
                            f_runs.write(json.dumps(res) + '\n')

                            mp_stats['overall']['total_runs'] += 1
                            w = res.get('wall_time_sec')
                            if isinstance(w, (int, float)):
                                mp_stats['overall']['sum_run_wall_time_sec'] += w
                                cfg_counts['sum_run_wall_time_sec'] += w
                                cfg_counts['total_runs'] += 1

                            # Aggregate deltas
                            deltas = res.get('deltas') or {}
                            du, ds, dr, dw = (deltas.get(k) for k in ['cpu_user_time_sec', 'cpu_system_time_sec', 'io_read_bytes', 'io_write_bytes'])
                            if isinstance(du, (int, float)):
                                mp_stats['overall']['sum_cpu_user_time_sec'] += du
                                cfg_counts['sum_cpu_user_time_sec'] += du
                            if isinstance(ds, (int, float)):
                                mp_stats['overall']['sum_cpu_system_time_sec'] += ds
                                cfg_counts['sum_cpu_system_time_sec'] += ds
                            if isinstance(dr, (int, float)):
                                mp_stats['overall']['sum_io_read_bytes'] += int(dr)
                                cfg_counts['sum_io_read_bytes'] += int(dr)
                            if isinstance(dw, (int, float)):
                                mp_stats['overall']['sum_io_write_bytes'] += int(dw)
                                cfg_counts['sum_io_write_bytes'] += int(dw)

                            # Update ru_maxrss peak across children
                            ra = (res.get('resource_after') or {}).get('ru_maxrss')
                            if isinstance(ra, (int, float)):
                                ru_maxrss_peak = max(ru_maxrss_peak, ra)
                                cfg_ru_maxrss_peak = max(cfg_ru_maxrss_peak, ra)

                            if status == 'success':
                                print(f"PQC Circuit Fidelity good for seed {seed} : {fid}")
                                good_total += 1
                                state['good_buf'][int(seed)] = float(fid)
                                state['processed_since_flush'] += 1
                                mp_stats['overall']['runs_good_fidelity'] += 1
                                cfg_counts['runs_good_fidelity'] += 1
                            else:
                                print(f"Poor PQC Circuit Fidelity for seed {seed} : {fid}")
                                poor_total += 1
                                state['poor_buf'][int(seed)] = float(fid)
                                state['processed_since_flush'] += 1
                                mp_stats['overall']['runs_poor_fidelity'] += 1
                                cfg_counts['runs_poor_fidelity'] += 1

                            # Periodic autosave based on count or time interval
                            if state['processed_since_flush'] >= state['autosave_every'] or (time.time() - state['last_flush_ts']) >= state['flush_interval_sec']:
                                _flush_seed_maps(reason="periodic")

                        elif status == 'skipped':
                            print(res.get('message'))
                            mp_stats['overall']['runs_skipped'] += 1
                            cfg_counts['runs_skipped'] += 1
                        elif status == 'error':
                            print(res.get('message'))
                            mp_stats['overall']['runs_error'] += 1
                            cfg_counts['runs_error'] += 1
                    
                pool.close()
            except Exception as e:
                print(f"Error occurred during multiprocessing: {e}")
                pool.terminate()
            finally:
                pool.join()
            
            try:
                _flush_seed_maps(reason="end_of_config")
            except Exception as e:
                print(f"Warning: failed to write seed:fidelity maps: {e}")

            print()

            # Per-config stats file under this data_dir
            config_end = time.perf_counter()
            cfg_stats = {
                'created_at': datetime.now().isoformat() + 'Z',
                'host': platform.node(),
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'python': platform.python_version(),
                },
                'config_summary': {
                    'qubits': qubit,
                    'gates': gate,
                    'gate_blocks': gate_blocks,
                    'pqc_blocks': config.get('pqc_blocks'),
                    'epochs': config.get('epochs'),
                    'num_data': config.get('num_data'),
                    'num_test': config.get('num_test'),
                    'batch': config.get('batch'),
                    'gate_dist': config.get('gate_dist'),
                    'noise_dist': config.get('noise_dist'),
                    'gpu': config.get('gpu'),
                },
                'overall': {
                    'total_wall_time_sec': config_end - config_start,
                    'sum_run_wall_time_sec': cfg_counts['sum_run_wall_time_sec'],
                    'sum_cpu_user_time_sec': cfg_counts['sum_cpu_user_time_sec'],
                    'sum_cpu_system_time_sec': cfg_counts['sum_cpu_system_time_sec'],
                    'sum_io_read_bytes': cfg_counts['sum_io_read_bytes'],
                    'sum_io_write_bytes': cfg_counts['sum_io_write_bytes'],
                    'total_runs': cfg_counts['total_runs'],
                    'runs_good_fidelity': cfg_counts['runs_good_fidelity'],
                    'runs_poor_fidelity': cfg_counts['runs_poor_fidelity'],
                    'runs_skipped': cfg_counts['runs_skipped'],
                    'runs_error': cfg_counts['runs_error'],
                    'peak_ru_maxrss': cfg_ru_maxrss_peak,
                    'peak_ru_maxrss_units': 'KiB_on_linux_bytes_on_macos',
                }
            }
            # Add worker config and observed parallelism to per-config stats
            cfg_total = cfg_stats['overall']['total_wall_time_sec']
            cfg_sum = cfg_stats['overall']['sum_run_wall_time_sec']
            cfg_eff = (cfg_sum / cfg_total) if (isinstance(cfg_sum, (int,float)) and isinstance(cfg_total, (int,float)) and cfg_total > 0) else None
            cfg_stats['overall'].update({
                'system_cpu_count': system_cpu_count,
                'configured_mp_cores': configured_mp_cores,
                'resolved_workers': resolved_workers,
                'observed_effective_parallelism': cfg_eff,
            })
            # This file now only contains the summary, not the individual runs.
            # The individual runs are in the adjacent 'mp_stats_runs.jsonl' file.
            write_json(os.path.join(data_dir, 'mp_stats.json'), cfg_stats)

    # Maintain only counts in memory to minimize overhead
    total_processed = good_total + poor_total
    print(f"\n{total_processed} circuits processed in total.")
    
    print("\n===== Run Summary =====")
    print(f"Total seeds attempted: {mp_stats['overall']['total_runs'] + mp_stats['overall']['runs_skipped'] + mp_stats['overall']['runs_error']}")
    print(f"Good fidelity: {mp_stats['overall']['runs_good_fidelity']}")
    print(f"Poor fidelity: {mp_stats['overall']['runs_poor_fidelity']}")
    print(f"Skipped: {mp_stats['overall']['runs_skipped']}")
    print(f"Errors: {mp_stats['overall']['runs_error']}")

    # Finalize overall stats and always write mp_stats
    overall_end = time.perf_counter()
    mp_stats['overall']['total_wall_time_sec'] = overall_end - overall_start
    mp_stats['overall']['peak_ru_maxrss'] = ru_maxrss_peak

    # Observed effective parallelism
    sum_run = mp_stats['overall'].get('sum_run_wall_time_sec')
    total_elapsed = mp_stats['overall'].get('total_wall_time_sec')
    if isinstance(sum_run, (int, float)) and isinstance(total_elapsed, (int, float)) and total_elapsed > 0:
        mp_stats['overall']['observed_effective_parallelism'] = float(max(1.0, sum_run / total_elapsed))

    # Resolve mp_stats output path under figure_output
    try:
        os.makedirs(config['figure_output'], exist_ok=True)
    except Exception:
        pass
    mp_stat_out = os.path.join(config['figure_output'], 'mp_stats.json')
    
    print(f"Writing final summary stats to {mp_stat_out} ...")
    write_json(mp_stat_out, mp_stats)

if __name__ == "__main__":
    main()