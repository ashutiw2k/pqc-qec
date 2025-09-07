import time
import jax
import jax.numpy as jnp

from pqcqec.simulate.simulate import (
    run_ideal_circuit,
    run_circuit_with_noise_model,
    _IDEAL_CACHE,
    _NOISY_CACHE,
)
from pqcqec.noise.simple_noise import PennylaneNoisyGates
from pqcqec.utils.jax_utils import JAXStateDataset, JAXDataLoader


def _simple_ops(num_qubits=2):
    # minimal circuit: X on qubit 0, then H on qubit 1
    return [
        ("x", [0], []),
        ("h", [1], []),
    ]


def test_qnode_cache_reuse_and_no_growth_ideal():
    ops = _simple_ops(2)
    key = ("default.qubit", 2, tuple((g, tuple(w)) for g, w, _ in ops))

    # clear any prior entries for this key if test re-runs in same process
    _IDEAL_CACHE.pop(key, None)

    x = jax.random.normal(jax.random.PRNGKey(0), (4, 2 ** 2,)) + 1j * jax.random.normal(
        jax.random.PRNGKey(1), (4, 2 ** 2,)
    )
    x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    # First call constructs and caches
    t0 = time.perf_counter()
    y0 = run_ideal_circuit(ops, x, num_qubits=2, batched=True)
    t1 = time.perf_counter()

    # Subsequent calls should reuse the cache (no new entries)
    cache_size_before = len(_IDEAL_CACHE)
    y_sum = 0.0
    for _ in range(5):
        y = run_ideal_circuit(ops, x, num_qubits=2, batched=True)
        y_sum += jnp.linalg.norm(y)
    cache_size_after = len(_IDEAL_CACHE)
    t2 = time.perf_counter()

    # Basic invariants
    assert y0.shape == x.shape
    assert cache_size_after == cache_size_before == 1

    # Print timing info (no strict assertion to avoid flakiness)
    print(f"Ideal first call: {t1 - t0:.4f}s, subsequent avg: {(t2 - t1)/5:.4f}s")


def test_qnode_cache_reuse_and_no_growth_noisy():
    ops = _simple_ops(2)
    # zero noise to make outputs stable
    nm = PennylaneNoisyGates(x_rad=0.0, z_rad=0.0, delta_x=0.0, delta_z=0.0, seed=0)

    # Build input
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 2 ** 2,)) + 1j * jax.random.normal(
        jax.random.PRNGKey(3), (3, 2 ** 2,)
    )
    x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    # First call constructs and caches
    t0 = time.perf_counter()
    y0 = run_circuit_with_noise_model(ops, x, nm, num_qubits=2, batched=True)
    t1 = time.perf_counter()

    cache_size_before = len(_NOISY_CACHE)
    for _ in range(4):
        _ = run_circuit_with_noise_model(ops, x, nm, num_qubits=2, batched=True)
    cache_size_after = len(_NOISY_CACHE)
    t2 = time.perf_counter()

    assert y0.shape == x.shape
    # Same noise_model instance should reuse the cache
    assert cache_size_after == cache_size_before == 1

    print(f"Noisy first call: {t1 - t0:.4f}s, subsequent avg: {(t2 - t1)/4:.4f}s")


def test_dataloader_shuffle_differs_each_epoch():
    # Create distinct data so we can infer indices from values
    N = 64
    data = jnp.arange(N, dtype=jnp.float32).reshape(N, 1)
    ds = JAXStateDataset(data)
    dl = JAXDataLoader(ds, batch_size=8, shuffle=True, seed=42)

    def collect_order(loader):
        order = []
        for xb, _ in loader:
            # take the scalar id from the first column
            order.extend([int(v) for v in xb[:, 0]])
        return order

    order1 = collect_order(dl)
    order2 = collect_order(dl)

    # With key splitting per reset, permutations should differ deterministically
    assert order1 != order2

