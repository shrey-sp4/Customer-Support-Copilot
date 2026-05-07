"""Latency benchmarking utility."""
import time
from typing import Callable, List


def benchmark_latency(
    executor,
    queries: List[str],
    warmup: int = 2,
) -> List[float]:
    """
    Benchmark end-to-end latency for a list of queries.
    Returns list of latencies in milliseconds.
    """
    latencies = []
    # Warmup
    for q in queries[:warmup]:
        try:
            executor.run(q)
        except Exception:
            pass

    # Benchmark
    for q in queries:
        t0 = time.time()
        try:
            executor.run(q)
        except Exception:
            pass
        latencies.append((time.time() - t0) * 1000)

    return latencies
