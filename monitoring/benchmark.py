"""Latency benchmark against a running instance.

Usage: python -m monitoring.benchmark --url http://localhost:8000 --n 500
"""

from __future__ import annotations

import argparse
import statistics
import time

import httpx

APP = {
    "application_id": "bench",
    "LIMIT_BAL": 50000,
    "EDUCATION": 2,
    "pay_status": [2, 2, 0, 0, 0, 0],
    "bill_amt": [46990, 48233, 49291, 28314, 28959, 29547],
    "pay_amt": [0, 2000, 1500, 1000, 1000, 800],
}


def pct(values: list[float], q: float) -> float:
    return sorted(values)[int(q * (len(values) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    with httpx.Client(base_url=args.url, timeout=30) as c:
        for _ in range(20):  # warm-up
            c.post("/v1/score", json=APP)
        single = []
        for _ in range(args.n):
            t = time.perf_counter()
            c.post("/v1/score", json=APP).raise_for_status()
            single.append((time.perf_counter() - t) * 1000)
        batch = []
        payload = {"applications": [APP | {"application_id": f"b{i}"} for i in range(1000)]}
        for _ in range(10):
            t = time.perf_counter()
            c.post("/v1/score/batch", json=payload).raise_for_status()
            batch.append((time.perf_counter() - t) * 1000)
    p50, p95, p99 = statistics.median(single), pct(single, 0.95), pct(single, 0.99)
    print(f"single  p50={p50:.1f} ms  p95={p95:.1f} ms  p99={p99:.1f} ms")
    print(f"batch(1000) p50={statistics.median(batch):.0f} ms  -> {1000 / statistics.median(batch) * 1000:.0f} apps/s")


if __name__ == "__main__":
    main()
