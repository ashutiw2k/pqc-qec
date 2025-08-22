import json
import os

# ----------------------------
# JSON/IO Helper Functions
# ----------------------------
def read_json(path, default=None):
    """Read JSON from path, returning default on error or file missing."""
    if default is None:
        default = {}
    try:
        if not os.path.exists(path):
            return default
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    """Write JSON to path (atomic best-effort)."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, default=str)
    os.replace(tmp_path, path)


def load_seed_fid_map(path):
    """Load a seed->fidelity map, supporting dict or legacy list formats."""
    data = read_json(path, default={})
    seed_map = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                seed_map[int(k)] = float(v)
            except Exception:
                continue
        return seed_map
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, (list, tuple)):
                if len(entry) >= 2 and isinstance(entry[0], (int, str)):
                    try:
                        seed_map[int(entry[0])] = float(entry[1])
                    except Exception:
                        continue
                elif len(entry) >= 4:
                    # legacy format [qubit, gate, seed, fidelity, ...]
                    try:
                        seed_map[int(entry[2])] = float(entry[3])
                    except Exception:
                        continue
    return seed_map


def save_seed_fid_map(path, mapping):
    """Persist a seed->fidelity map as a sorted dict with string keys."""
    payload = {str(k): float(v) for k, v in sorted(mapping.items())}
    write_json(path, payload)


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


