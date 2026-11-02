"""Population Stability Index (PSI): the standard drift metric in credit risk.

PSI < 0.10 stable · 0.10-0.25 moderate shift · > 0.25 significant shift.
"""

from __future__ import annotations

from collections import deque
from threading import Lock

import numpy as np
import pandas as pd

EPS = 1e-4


def bin_edges(reference: np.ndarray, n_bins: int = 10, max_categories: int = 20) -> np.ndarray:
    """Quantile bins for continuous features, one bin per value for discrete ones.

    Quantile edges collapse on discrete features (e.g. payment status, where
    most values are 0), which would hide a shift from 0 to 1 inside one bin.
    """
    values = np.unique(reference)
    if len(values) <= max_categories:
        mids = (values[:-1] + values[1:]) / 2
        return np.concatenate([[-np.inf], mids, [np.inf]])
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], edges, [np.inf]])


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    edges = bin_edges(np.asarray(reference, float), n_bins)
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / max(len(current), 1)
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_status(value: float) -> str:
    if value < 0.10:
        return "stable"
    return "moderate" if value < 0.25 else "significant"


def psi_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.Series:
    return pd.Series({c: psi(reference[c].to_numpy(), current[c].to_numpy()) for c in reference.columns})


class DriftMonitor:
    """Keeps a rolling window of recent feature vectors and scores, thread-safe."""

    def __init__(self, reference: pd.DataFrame, window: int = 2000, min_samples: int = 200):
        self.reference = reference
        self.min_samples = min_samples
        self._rows: deque[np.ndarray] = deque(maxlen=window)
        self._lock = Lock()

    def observe(self, features: pd.DataFrame) -> None:
        with self._lock:
            self._rows.extend(features[self.reference.columns].to_numpy())

    def report(self) -> dict:
        with self._lock:
            rows = np.array(self._rows)
        n = len(rows)
        if n < self.min_samples:
            cols = list(self.reference.columns)
            return {
                "window_size": n,
                "psi": dict.fromkeys(cols, 0.0),
                "status": dict.fromkeys(cols, "insufficient_data"),
            }
        current = pd.DataFrame(rows, columns=self.reference.columns)
        values = psi_table(self.reference, current)
        return {
            "window_size": n,
            "psi": {k: round(float(v), 4) for k, v in values.items()},
            "status": {k: psi_status(v) for k, v in values.items()},
        }
