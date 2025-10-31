#!/usr/bin/env python3
"""Targeted watchdog for Python jobs.

Given the name of a running Python script or module, this tool watches the
matching processes (including children) and displays a rolling summary of
their resource usage—similar to a per-application ``top`` view. Metrics are
refreshed at a configurable interval until the tracked processes exit or the
user interrupts the program (Ctrl+C).

Example
-------
::

    python scripts/python_watchdog.py --name finetune_transformer_predictions_mp.py

Requirements
------------
This script relies on :mod:`psutil`. Install it once via::

    python -m pip install psutil

"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import psutil  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    sys.stderr.write(
        "psutil is required for scripts/python_watchdog.py.\n"
        "Install it with: python -m pip install psutil\n"
    )
    raise SystemExit(1) from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous resource monitor for a Python script"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Substring to match in the target process command line (case-insensitive)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--show-ports",
        action="store_true",
        help="Display open file/port counts (may require elevated privileges)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print a single snapshot then exit",
    )
    return parser.parse_args(argv)


def _matches_target(proc: psutil.Process, needle: str) -> bool:
    needle = needle.lower()
    name = (proc.info.get("name") or "").lower()
    if needle in name:
        return True
    cmdline = " ".join(proc.info.get("cmdline") or []).lower()
    return needle in cmdline


def find_process_tree(needle: str) -> List[psutil.Process]:
    """Return processes whose command contains *needle* plus their descendants."""

    roots: List[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            if _matches_target(proc, needle):
                roots.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    seen: Dict[int, psutil.Process] = {}
    for root in roots:
        if root.pid in seen:
            continue
        seen[root.pid] = root
        try:
            for child in root.children(recursive=True):
                if child.pid not in seen:
                    seen[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Return sorted by PID for stable output
    return [seen[pid] for pid in sorted(seen)]


def format_bytes(num: float) -> str:
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while num >= 1024 and idx < len(suffixes) - 1:
        num /= 1024.0
        idx += 1
    return f"{num:6.1f}{suffixes[idx]}"


def clear_screen() -> None:
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()


def collect_metrics(
    processes: Iterable[psutil.Process], show_ports: bool
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    totals = {
        "cpu": 0.0,
        "rss": 0.0,
        "threads": 0,
        "ports": 0,
    }

    for proc in processes:
        try:
            with proc.oneshot():
                cpu = proc.cpu_percent(interval=None)
                mem_info = proc.memory_info()
                rss = mem_info.rss
                mem_percent = proc.memory_percent()
                status = proc.status()
                create = proc.create_time()
                threads = proc.num_threads()
                username = proc.username()
                cmdline = " ".join(proc.cmdline()) or proc.name()
                ports = proc.num_fds() if show_ports else None
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        rows.append(
            {
                "pid": proc.pid,
                "ppid": proc.ppid(),
                "cpu": cpu,
                "mem_pct": mem_percent,
                "rss": rss,
                "threads": threads,
                "status": status,
                "uptime": _dt.timedelta(seconds=int(time.time() - create)),
                "user": username,
                "cmd": cmdline,
                "ports": ports,
            }
        )

        totals["cpu"] += cpu
        totals["rss"] += rss
        totals["threads"] += threads
        if show_ports and ports is not None:
            totals["ports"] += ports

    rows.sort(key=lambda r: (-r["cpu"], -r["rss"], r["pid"]))
    return rows, totals


def render_snapshot(
    needle: str,
    rows: List[Dict[str, object]],
    totals: Dict[str, float],
    show_ports: bool,
    interval: float,
    start_time: float,
) -> None:
    clear_screen()
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    runtime = _dt.timedelta(seconds=int(time.time() - start_time))
    sys_cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    header = (
        f"python-watchdog | target='{needle}' | now={now} | runtime={runtime} | "
        f"system CPU={sys_cpu:4.1f}% | RAM used={vm.percent:4.1f}% | "
        f"Swap used={swap.percent:4.1f}% | refresh={interval:.1f}s"
    )
    print(header)
    print("=" * len(header))

    if not rows:
        print("No matching processes found. Waiting...")
        return

    columns = [
        ("PID", "pid", "{pid:>6}"),
        ("PPID", "ppid", "{ppid:>6}"),
        ("CPU%", "cpu", "{cpu:6.1f}"),
        ("MEM%", "mem_pct", "{mem_pct:6.2f}"),
        ("RSS", "rss", "{rss_fmt}"),
        ("THR", "threads", "{threads:>4}"),
        ("STATE", "status", "{status:>8}"),
        ("UPTIME", "uptime", "{uptime}"),
        ("USER", "user", "{user:>10}"),
    ]
    if show_ports:
        columns.append(("FDS", "ports", "{ports:>5}"))
    columns.append(("COMMAND", "cmd", "{cmd}"))

    header_line = "  ".join(title for title, _, _ in columns)
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        row = dict(row)
        row["rss_fmt"] = format_bytes(row["rss"])
        line_parts = []
        for _, key, fmt in columns:
            value = row.get(key, "")
            line_parts.append(fmt.format(**{key: value, "rss_fmt": row.get("rss_fmt"), "cmd": value}))
        print("  ".join(line_parts))

    total_line = (
        f"Totals: cpu={totals['cpu']:6.1f}% | rss={format_bytes(totals['rss'])} | "
        f"threads={int(totals['threads'])}"
    )
    if show_ports:
        total_line += f" | fds={int(totals['ports'])}"
    print("\n" + total_line)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    needle = args.name
    interval = max(0.5, args.interval)
    show_ports = args.show_ports
    start_time = time.time()

    # Prime CPU measurement so first refresh has meaningful values
    for proc in find_process_tree(needle):
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    try:
        while True:
            processes = find_process_tree(needle)
            rows, totals = collect_metrics(processes, show_ports=show_ports)
            render_snapshot(
                needle=needle,
                rows=rows,
                totals=totals,
                show_ports=show_ports,
                interval=interval,
                start_time=start_time,
            )

            if args.once:
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nWatchdog terminated by user.")


if __name__ == "__main__":
    main()
