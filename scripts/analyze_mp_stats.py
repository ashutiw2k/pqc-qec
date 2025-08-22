#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import sys
try:
    import psutil  # optional, for auto-detecting system RAM
except Exception:
    psutil = None
from datetime import datetime
import os


def human_bytes(n):
    try:
        n = float(n)
    except Exception:
        return str(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)} {units[i]}"
    return f"{n:.2f} {units[i]}"


def human_seconds(s):
    try:
        s = float(s)
    except Exception:
        return str(s)
    days, rem = divmod(int(round(s)), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def safe_float(x):
    return x if isinstance(x, (int, float)) else None


def _deep_sizeof(obj, seen=None):
    """Approximate the deep size of a Python object graph.
    Uses sys.getsizeof and recursively sums for containers.
    """
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_sizeof(k, seen)
            size += _deep_sizeof(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i in obj:
            size += _deep_sizeof(i, seen)
    return size


def analyze_modern(data, stats_path=None, parent_mem_safety=1.5, worker_mem_safety=1.3,
                   system_ram_bytes=None, ram_utilization=0.85):
    runs = data.get("runs") or []
    overall = data.get("overall") or {}

    executed = [r for r in runs if isinstance(r.get("wall_time_sec"), (int, float))]
    saved = [r for r in executed if r.get("status") in ("saved", "success")]
    poor = [r for r in executed if r.get("status") in ("poor_fidelity",)]
    skipped = overall.get("runs_skipped") or (overall.get("runs_skipped_existing", 0) + overall.get("runs_skipped_poor_seed", 0))
    errors = overall.get("runs_error", 0)

    # Timing stats
    wall = [r["wall_time_sec"] for r in executed]
    wall_avg = sum(wall) / len(wall) if wall else 0.0
    wall_med = statistics.median(wall) if wall else 0.0

    # CPU deltas
    du = [safe_float(r.get("deltas", {}).get("cpu_user_time_sec")) for r in executed]
    du = [x for x in du if x is not None]
    ds = [safe_float(r.get("deltas", {}).get("cpu_system_time_sec")) for r in executed]
    ds = [x for x in ds if x is not None]

    # Memory deltas (best-effort)
    drss = [safe_float(r.get("deltas", {}).get("rss_bytes")) for r in executed]
    drss = [x for x in drss if x is not None]

    # IO deltas
    dio_r = [safe_float(r.get("deltas", {}).get("io_read_bytes")) for r in executed]
    dio_r = [x for x in dio_r if x is not None]
    dio_w = [safe_float(r.get("deltas", {}).get("io_write_bytes")) for r in executed]
    dio_w = [x for x in dio_w if x is not None]

    # Fidelity summary (only for executed runs that contain fidelity)
    fvals = [safe_float(r.get("fidelity")) for r in executed]
    fvals = [x for x in fvals if x is not None]

    peak = overall.get("peak_ru_maxrss")
    units = overall.get("peak_ru_maxrss_units")
    platform_info = data.get("platform") or {}
    system_name = (platform_info.get("system") or "").strip()

    print("===== MP Stats Analysis =====")
    if data.get("created_at"):
        print("Created:", data.get("created_at"))
    if data.get("host"):
        print("Host:", data.get("host"))
    if data.get("platform"):
        print("Platform:", data.get("platform"))
    cs = data.get("config_summary") or {}
    if cs:
        print("Config summary:", cs)
    print()
    # Show safety factors and RAM context
    print("Safety factors:")
    print(f" - Parent objects: x{parent_mem_safety}")
    print(f" - Worker RSS: x{worker_mem_safety}")
    # Show RAM info if available
    if system_ram_bytes is not None:
        print("RAM context:")
        print(f" - System RAM: {human_bytes(system_ram_bytes)}")
        print(f" - Planning utilization: {int(ram_utilization*100)}%")
    print()

    print("Runs summary:")
    print(" - Executed:", len(executed))
    print(" - Good Fidelity:", len(saved))
    print(" - Poor Fidelity:", len(poor))
    print(" - Skipped:", skipped)
    print(" - Errors:", errors)
    print()

    print("Timing:")
    print(" - Avg wall time/run:", f"{wall_avg:.3f}s", f"({human_seconds(wall_avg)})")
    if wall:
        print(" - Median:", f"{wall_med:.3f}s")
        print(" - Min/Max:", f"{min(wall):.3f}s", "/", f"{max(wall):.3f}s")
    sum_wall_runs = sum(wall)
    print(" - Sum wall time (runs):", f"{sum_wall_runs:.1f}s", f"({human_seconds(sum_wall_runs)})")
    sum_wall_overall = overall.get("sum_run_wall_time_sec")
    if isinstance(sum_wall_overall, (int, float)):
        print(" - Sum wall reported (overall):", f"{sum_wall_overall:.1f}s")
    total_elapsed = overall.get("total_wall_time_sec")
    if isinstance(total_elapsed, (int, float)):
        print(" - Total elapsed (script):", f"{total_elapsed:.1f}s", f"({human_seconds(total_elapsed)})")
    # Workers and observed parallelism
    sys_cnt = overall.get('system_cpu_count')
    cfg_cnt = overall.get('configured_mp_cores')
    res_cnt = overall.get('resolved_workers')
    if any(x is not None for x in (sys_cnt, cfg_cnt, res_cnt)):
        print("Workers:")
        if sys_cnt is not None:
            print(" - System CPU count:", sys_cnt)
        if cfg_cnt is not None:
            print(" - Configured mp_cores:", cfg_cnt)
        if res_cnt is not None:
            print(" - Resolved workers:", res_cnt)
    # Estimate effective parallelism from ratio of summed run time to elapsed
    p_eff = overall.get('observed_effective_parallelism')
    if isinstance(total_elapsed, (int, float)) and total_elapsed > 0:
        if not isinstance(p_eff, (int, float)):
            num = sum_wall_overall if isinstance(sum_wall_overall, (int, float)) else sum_wall_runs
            p_eff = max(1.0, num / total_elapsed)
        print(f" - Observed effective parallelism: {float(p_eff):.2f}×")
    print()

    print("CPU deltas (per executed run):")
    if du:
        print(" - Avg user:", f"{sum(du)/len(du):.3f}s")
    if ds:
        print(" - Avg system:", f"{sum(ds)/len(ds):.3f}s")
    print()

    print("Memory/IO:")
    if peak is not None:
        # Convert peak ru_maxrss to bytes for display when possible
        if units == 'KiB_on_linux_bytes_on_macos':
            # On macOS: ru_maxrss is bytes; on Linux: KiB
            # We will print both interpretations to be explicit
            gib_linux = float(peak) / 1024 / 1024
            gib_macos = float(peak) / 1024 / 1024 / 1024
            print(" - Peak ru_maxrss:", peak, units, f"(~{gib_linux:.2f} GiB if Linux, ~{gib_macos:.2f} GiB if macOS)")
        else:
            print(" - Peak ru_maxrss:", peak, units)
    if drss:
        print(" - Avg RSS delta/run:", human_bytes(sum(drss)/len(drss)))
    if dio_r:
        print(" - Avg IO read delta/run:", human_bytes(sum(dio_r)/len(dio_r)))
    if dio_w:
        print(" - Avg IO write delta/run:", human_bytes(sum(dio_w)/len(dio_w)))
    print()

    if fvals:
        print("Fidelity:")
        print(" - Avg fidelity:", f"{sum(fvals)/len(fvals):.6f}")
        print(" - Median fidelity:", f"{statistics.median(fvals):.6f}")
        print(" - Min/Max fidelity:", f"{min(fvals):.6f}", "/", f"{max(fvals):.6f}")
        print()

    # Estimate Python object memory in parent process from the stats structure
    try:
        meta_copy = dict(data)
        # Replace runs with empty list to isolate metadata footprint
        meta_copy['runs'] = []
        meta_bytes = _deep_sizeof(meta_copy)
        # Sample per-run object sizes; if many runs, sample a subset for speed
        sample_runs = executed[:min(200, len(executed))]
        per_run_sizes = [_deep_sizeof(r) for r in sample_runs]
        per_run_pyobj_avg = (sum(per_run_sizes) / len(per_run_sizes)) if per_run_sizes else None
        current_py_mem = meta_bytes + (per_run_pyobj_avg * len(executed) if per_run_pyobj_avg else 0)
    except Exception:
        meta_bytes = None
        per_run_pyobj_avg = None
        current_py_mem = None
    if meta_bytes is not None:
        print("Python objects (parent process) estimate:")
        print(" - Metadata footprint:", human_bytes(meta_bytes))
        if per_run_pyobj_avg is not None:
            print(" - Avg per-run object size:", human_bytes(per_run_pyobj_avg))
            print(" - Current total objects size:", human_bytes(current_py_mem), f"(safe ~ {human_bytes(current_py_mem*parent_mem_safety)})")
        print()

    # Helper: convert ru_maxrss to bytes based on OS
    def ru_to_bytes(v):
        if not isinstance(v, (int, float)):
            return None
        if units == 'KiB_on_linux_bytes_on_macos':
            if system_name == 'Darwin':
                return float(v)
            # assume Linux -> KiB
            return float(v) * 1024.0
        return float(v)

    # Estimate per-worker peak RSS from worker snapshots
    worker_rss_samples = []
    for r in executed:
        ra = r.get('resource_after') or {}
        rss = ra.get('rss_bytes')
        if isinstance(rss, (int, float)):
            worker_rss_samples.append(float(rss))
            continue
        ru = ra.get('ru_maxrss')
        rb = ru_to_bytes(ru)
        if isinstance(rb, (int, float)):
            worker_rss_samples.append(float(rb))

    per_worker_peak_bytes = max(worker_rss_samples) if worker_rss_samples else None
    per_worker_peak_bytes_safe = (per_worker_peak_bytes * worker_mem_safety) if per_worker_peak_bytes else None

    # Scaling estimates for multiples of the observed/declared seed
    # Determine the base seed from config or infer from runs
    base_seed = None
    cs = data.get("config_summary") or {}
    if cs:
        # seed_max may be a string in some stats
        sm = cs.get("seed_max") or cs.get("seed")
        try:
            base_seed = int(sm)
        except Exception:
            base_seed = None
    if base_seed is None:
        # Infer from the maximum seed observed in runs
        seeds = [r.get("seed") for r in executed if isinstance(r.get("seed"), int)]
        if seeds:
            base_seed = max(seeds)

    if wall_avg > 0 and base_seed is not None and base_seed >= 0:
        print()
        print("Scaling estimates (sequential) based on multiples of seed:")
        print(" - Base seed:", base_seed)
        for mul in (10, 25, 50, 100, 200):
            target_seed = base_seed * mul
            n = target_seed + 1  # script runs 0..seed
            total = wall_avg * n
            mem_note = ""
            if per_worker_peak_bytes is not None:
                raw_peak = per_worker_peak_bytes
                safe_peak = raw_peak * worker_mem_safety
                mem_note = f", est. peak RSS ~ {human_bytes(raw_peak)} (safe ~ {human_bytes(safe_peak)})"
            py_note = ""
            if per_run_pyobj_avg is not None and meta_bytes is not None:
                py_est = meta_bytes + per_run_pyobj_avg * n
                py_note = f", est. parent objects ~ {human_bytes(py_est)} (safe ~ {human_bytes(py_est*parent_mem_safety)})"
            print(f"   seed={target_seed:>7}: {human_seconds(total)} ({total/3600:.2f} h){mem_note}{py_note}")
        if p_eff is not None:
            print()
            print("Scaling estimates (parallel, using observed throughput):")
            # Optional: recommend workers per target seed given RAM
            for mul in (10, 25, 50, 100, 200):
                target_seed = base_seed * mul
                n = target_seed + 1
                total = (wall_avg * n) / p_eff
                mem_note = ""
                if per_worker_peak_bytes is not None:
                    # Estimate concurrent peak memory = per-worker peak * resolved workers
                    workers = overall.get('resolved_workers') or overall.get('configured_mp_cores') or 1
                    try:
                        workers = int(workers)
                    except Exception:
                        workers = 1
                    if workers < 1:
                        workers = 1
                    raw_peak = per_worker_peak_bytes * workers
                    safe_peak = raw_peak * worker_mem_safety
                    mem_note = f", est. peak RSS ~ {human_bytes(raw_peak)} (safe ~ {human_bytes(safe_peak)}) ({workers} procs)"
                py_note = ""
                if per_run_pyobj_avg is not None and meta_bytes is not None:
                    py_est = meta_bytes + per_run_pyobj_avg * n
                    py_note = f", est. parent objects ~ {human_bytes(py_est)} (safe ~ {human_bytes(py_est*parent_mem_safety)})"
                rec_note = ""
                if system_ram_bytes is not None and per_worker_peak_bytes_safe is not None and per_run_pyobj_avg is not None and meta_bytes is not None:
                    budget = system_ram_bytes * ram_utilization
                    parent_safe = (meta_bytes + per_run_pyobj_avg * n) * parent_mem_safety
                    leftover = max(0.0, budget - parent_safe)
                    if per_worker_peak_bytes_safe > 0:
                        max_workers = int(leftover // per_worker_peak_bytes_safe)
                        if max_workers < 1:
                            max_workers = 1
                        # respect CPU count if present
                        sys_cnt = overall.get('system_cpu_count')
                        try:
                            if isinstance(sys_cnt, (int, float)) and sys_cnt > 0:
                                max_workers = min(max_workers, int(sys_cnt))
                        except Exception:
                            pass
                        rec_note = f", rec workers <= {max_workers}"
                print(f"   seed={target_seed:>7}: {human_seconds(total)} ({total/3600:.2f} h) @ ~{p_eff:.1f}×{mem_note}{py_note}{rec_note}")

        # Estimate mp_stats size growth (file size as a proxy)
        try:
            current_file_size = os.path.getsize(stats_path) if stats_path else None
        except Exception:
            current_file_size = None
        if current_file_size and len(executed) > 0:
            bytes_per_run = current_file_size / len(executed)
            print()
            print("Stats file size estimates (linear scale w.r.t runs):")
            for mul in (10, 25, 50, 100, 200):
                target_seed = base_seed * mul
                n = target_seed + 1
                est_size = bytes_per_run * n
                print(f"   seed={target_seed:>7}: ~{human_bytes(est_size)}")


def analyze_legacy(data):
    # Legacy format: list of {seed, fidelity}
    if not isinstance(data, list):
        print("Unrecognized legacy format")
        return
    seeds = [d.get("seed") for d in data if isinstance(d, dict)]
    fids = [d.get("fidelity") for d in data if isinstance(d, dict) and isinstance(d.get("fidelity"), (int, float))]
    print("===== Legacy Stats Analysis =====")
    print("Entries:", len(data))
    print("Unique seeds:", len(set([s for s in seeds if s is not None])))
    if fids:
        print("Avg fidelity:", f"{sum(fids)/len(fids):.6f}")
        print("Median fidelity:", f"{statistics.median(fids):.6f}")
        print("Min/Max fidelity:", f"{min(fids):.6f}", "/", f"{max(fids):.6f}")


def main():
    ap = argparse.ArgumentParser(description="Analyze mp_stats.json file produced by training scripts.")
    ap.add_argument("stats_path", help="Path to mp_stats.json (overall or per-config)")
    ap.add_argument("--parent-mem-safety", type=float, default=1.5,
                    help="Safety factor for parent Python objects memory (default: 1.5)")
    ap.add_argument("--worker-mem-safety", type=float, default=1.3,
                    help="Safety factor for per-worker RSS (default: 1.3)")
    ap.add_argument("--system-ram-gb", type=float, default=None,
                    help="System RAM in GB (if not provided, attempts auto-detect)")
    ap.add_argument("--ram-utilization", type=float, default=0.85,
                    help="Fraction of RAM to plan for (default: 0.85)")
    args = ap.parse_args()

    path = args.stats_path
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON: {e}")
        return

    print(f"Analyzing: {path}")
    print()
    # Resolve system RAM
    sys_ram_bytes = None
    if args.system_ram_gb is not None:
        try:
            sys_ram_bytes = float(args.system_ram_gb) * (1024**3)
        except Exception:
            sys_ram_bytes = None
    elif psutil is not None:
        try:
            sys_ram_bytes = float(psutil.virtual_memory().total)
        except Exception:
            sys_ram_bytes = None

    if isinstance(data, dict) and ("runs" in data or "overall" in data):
        analyze_modern(data, stats_path=path,
                       parent_mem_safety=args.parent_mem_safety,
                       worker_mem_safety=args.worker_mem_safety,
                       system_ram_bytes=sys_ram_bytes,
                       ram_utilization=args.ram_utilization)
    else:
        analyze_legacy(data)


if __name__ == "__main__":
    main()
