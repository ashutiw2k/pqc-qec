import sys
import os
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
from tqdm.auto import tqdm
from qiskit.qasm2 import dumps

from pqcqec.circuits.generate import generate_random_circuit
from pqcqec.circuits.modify import tokenize_qiskit_circuit

from pqcqec.utils.args import get_all_valid_args, parse_args
from pqcqec.utils.json_utils import write_json

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
    """Generate and tokenize a single random circuit for a seed.

    Args:
        args: Tuple of
            - seed (int)
            - qubit (int)
            - gate (int)
            - out_dir (str): directory to persist per-seed outputs
            - config (dict)

    Returns:
        Dict run_stat with status, timings and resource snapshots.
    """
    seed, qubit, gate, out_dir, config = args

    out_file = os.path.join(out_dir, f"{seed}.json")

    # Respect existing outputs if not forcing regeneration
    if not config['force'] and os.path.exists(out_file):
        return {
            'status': 'skipped',
            'seed': seed,
            'qubits': qubit,
            'gates': gate,
            'message': f"Seed {seed} already processed (output file exists)."
        }

    # print(f"Generating base circuit tokens for Qubits: {qubit}, Gates: {gate}, Seed: {seed}")

    run_start_wall = time.perf_counter()
    snap_before = _proc_snapshot()

    try:
        qc = generate_random_circuit(
            num_qubits=qubit,
            num_gates=gate,
            seed=seed,
            gate_dist=config['gate_dist'],
        )

        if config['uncomp']:
            inverse_qc = qc.inverse()
            qc = qc.compose(inverse_qc)

        qasm_str = dumps(qc)
        qc_ops = tokenize_qiskit_circuit(qc)

        token_data = {
            'seed': seed,
            'base_circuit_tokens': qc_ops,
            'base_circuit_qasm': qasm_str,
        }
        write_json(out_file, token_data)

        snap_after = _proc_snapshot()
        run_end_wall = time.perf_counter()
        wall_time_sec = run_end_wall - run_start_wall

        def _delta(key):
            a = snap_after.get(key)
            b = snap_before.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a - b
            return None

        return {
            'status': 'success',
            'seed': seed,
            'qubits': qubit,
            'gates': gate,
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

    except Exception as e:
        snap_after = _proc_snapshot()
        run_end_wall = time.perf_counter()
        wall_time_sec = run_end_wall - run_start_wall
        return {
            'status': 'error',
            'seed': seed,
            'qubits': qubit,
            'gates': gate,
            'message': f"Error on seed {seed}: {e}",
            'wall_time_sec': wall_time_sec,
            'resource_before': snap_before,
            'resource_after': snap_after,
        }


def main():
    # Parse command line arguments
    required_args = ['qubit_range', 'gate_range', 'config', 'seed',
                     'gate_dist', 'gpu', 'figure_output', 'force', 'mp_cores', 'uncomp']
    script_description = 'Generate and tokenize base/pure circuits in parallel for `seed` number of circuits per qubit/gate configuration.'

    args = parse_args(required_args, script_description=script_description)

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
            'runs_skipped': 0,
            'runs_error': 0,
            'peak_ru_maxrss': 0,
            'peak_ru_maxrss_units': 'KiB_on_linux_bytes_on_macos',
        }
    }

    overall_start = time.perf_counter()
    ru_maxrss_peak = 0

    config = get_all_valid_args(args, include_args=required_args)

    # Set up graceful shutdown (no special autosave needed here)
    def _signal_handler(signum, frame):  # pragma: no cover - best-effort
        try:
            print(f"\nReceived signal {signum}; attempting clean exit...")
        except Exception:
            pass
        # Re-raise KeyboardInterrupt for SIGINT to trigger outer handlers/cleanup
        if signum == getattr(signal, 'SIGINT', None):
            raise KeyboardInterrupt

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
        'gate_dist': config.get('gate_dist'),
        'gpu': config.get('gpu'),
        'seed_max': config.get('seed'),
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

            print(f"Generating up to {config_seed} seeds for Qubits: {qubit}, Gates: {gate}")

            data_dir = os.path.join(config['figure_output'], f"{qubit}q_{gate}g_circuit_data")
            os.makedirs(data_dir, exist_ok=True)

            if config['force']:
                for root, dirs, files in os.walk(data_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))

            config_file = os.path.join(data_dir, "config.json")
            per_seed_dir = os.path.join(data_dir, "per_seed_data")
            os.makedirs(per_seed_dir, exist_ok=True)

            # Save config atomically and minified
            write_json(config_file, config)
            print(f"Config file saved to {config_file}")

            # Per-config stats containers
            config_start = time.perf_counter()
            cfg_ru_maxrss_peak = 0
            cfg_counts = {
                'total_runs': 0,
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
            print(f"\nStarting parallel generation with {num_processes} cores (system={system_cpu_count}, configured={configured_mp_cores})...")

            # Limit threads within each worker to avoid oversubscription
            threads_per_worker = config.get('threads_per_worker') if isinstance(config, dict) else None
            if threads_per_worker is None:
                threads_per_worker = int(os.environ.get("THREADS_PER_WORKER", max(1, system_cpu_count // max(1, num_processes))))
            for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'TBB_NUM_THREADS'):
                os.environ[_var] = str(threads_per_worker)

            def task_iter():
                for seed in range(num_circs):
                    yield (seed, qubit, gate, per_seed_dir, config)

            pool = multiprocessing.get_context("spawn").Pool(processes=num_processes)
            config_runs_path = os.path.join(data_dir, 'mp_stats_runs.jsonl')

            try:
                with open(config_runs_path, 'w') as f_runs:
                    chunksize = max(1, num_circs // (num_processes * 4) or 1)
                    results_iter = pool.imap_unordered(process_seed, task_iter(), chunksize=chunksize)

                    print("\nProcessing results...")
                    pbar = tqdm(total=num_circs, desc=f"Seeds Processed")
                    for res in results_iter:
                        status = res.get('status')
                        seed_id = res.get('seed')

                        if status == 'success':
                            # Stream run stat to JSONL
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

                            ra = (res.get('resource_after') or {}).get('ru_maxrss')
                            if isinstance(ra, (int, float)):
                                ru_maxrss_peak = max(ru_maxrss_peak, ra)
                                cfg_ru_maxrss_peak = max(cfg_ru_maxrss_peak, ra)

                        elif status == 'skipped':
                            print(res.get('message'))
                            mp_stats['overall']['runs_skipped'] += 1
                            cfg_counts['runs_skipped'] += 1
                        elif status == 'error':
                            print(res.get('message'))
                            mp_stats['overall']['runs_error'] += 1
                            cfg_counts['runs_error'] += 1
                        # Progress update for every result
                        pbar.update(1)
                        # Show quick status counts
                        pbar.set_postfix({
                            'ok': cfg_counts['total_runs'],
                            'skip': cfg_counts['runs_skipped'],
                            'err': cfg_counts['runs_error'],
                        })

                pool.close()
            except Exception as e:
                print(f"Error occurred during multiprocessing: {e}")
                pool.terminate()
            finally:
                try:
                    pbar.close()
                except Exception:
                    pass
                pool.join()

            # Build the combined tokens file from per-seed JSONs
            # try:
            #     tokens_file = os.path.join(data_dir, "circuit_tokens.json")
            #     all_entries = []
            #     # Read per-seed files in seed order if possible
            #     for fname in sorted(os.listdir(per_seed_dir), key=lambda x: int(os.path.splitext(x)[0]) if x.endswith('.json') and os.path.splitext(x)[0].isdigit() else float('inf')):
            #         if not fname.endswith('.json'):
            #             continue
            #         fpath = os.path.join(per_seed_dir, fname)
            #         try:
            #             with open(fpath, 'r') as fr:
            #                 all_entries.append(json.load(fr))
            #         except Exception:
            #             continue
            #     write_json(tokens_file, all_entries)
            #     print(f"Circuit tokens saved to {tokens_file}")
            # except Exception as e:
            #     print(f"Warning: failed to write combined circuit_tokens.json: {e}")

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
                    'seed_max': config.get('seed'),
                    'gate_dist': config.get('gate_dist'),
                    'gpu': config.get('gpu'),
                    'uncomp': config.get('uncomp'),
                },
                'overall': {
                    'total_wall_time_sec': config_end - config_start,
                    'sum_run_wall_time_sec': cfg_counts['sum_run_wall_time_sec'],
                    'sum_cpu_user_time_sec': cfg_counts['sum_cpu_user_time_sec'],
                    'sum_cpu_system_time_sec': cfg_counts['sum_cpu_system_time_sec'],
                    'sum_io_read_bytes': cfg_counts['sum_io_read_bytes'],
                    'sum_io_write_bytes': cfg_counts['sum_io_write_bytes'],
                    'total_runs': cfg_counts['total_runs'],
                    'runs_skipped': cfg_counts['runs_skipped'],
                    'runs_error': cfg_counts['runs_error'],
                    'peak_ru_maxrss': cfg_ru_maxrss_peak,
                    'peak_ru_maxrss_units': 'KiB_on_linux_bytes_on_macos',
                }
            }
            cfg_total = cfg_stats['overall']['total_wall_time_sec']
            cfg_sum = cfg_stats['overall']['sum_run_wall_time_sec']
            cfg_eff = (cfg_sum / cfg_total) if (isinstance(cfg_sum, (int,float)) and isinstance(cfg_total, (int,float)) and cfg_total > 0) else None
            cfg_stats['overall'].update({
                'system_cpu_count': system_cpu_count,
                'configured_mp_cores': configured_mp_cores,
                'resolved_workers': resolved_workers,
                'observed_effective_parallelism': cfg_eff,
            })
            write_json(os.path.join(data_dir, 'mp_stats.json'), cfg_stats)

    # Finalize overall stats and always write mp_stats
    overall_end = time.perf_counter()
    mp_stats['overall']['total_wall_time_sec'] = overall_end - overall_start
    mp_stats['overall']['peak_ru_maxrss'] = ru_maxrss_peak

    sum_run = mp_stats['overall'].get('sum_run_wall_time_sec')
    total_elapsed = mp_stats['overall'].get('total_wall_time_sec')
    if isinstance(sum_run, (int, float)) and isinstance(total_elapsed, (int, float)) and total_elapsed > 0:
        mp_stats['overall']['observed_effective_parallelism'] = float(max(1.0, sum_run / total_elapsed))

    try:
        os.makedirs(config['figure_output'], exist_ok=True)
    except Exception:
        pass
    mp_stat_out = os.path.join(config['figure_output'], 'mp_stats.json')
    print(f"Writing final summary stats to {mp_stat_out} ...")
    write_json(mp_stat_out, mp_stats)


if __name__ == "__main__":
    main()
