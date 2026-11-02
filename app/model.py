"""Model loading and scoring.

Artifacts are framework-native and language-agnostic (LightGBM text model +
JSON metadata + CSV reference sample) instead of a pickle, so they can be
inspected, diffed, and loaded without importing training code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES, REASON_TEXT, build_features


@dataclass
class Scored:
    pd: np.ndarray
    decisions: list[str]
    reasons: list[list[str]]
    features: pd.DataFrame


class ModelService:
    def __init__(self, artifact_dir: str | Path):
        d = Path(artifact_dir)
        self.metadata = json.loads((d / "metadata.json").read_text())
        if self.metadata["features"] != FEATURES:
            raise RuntimeError("Model was trained with a different feature set than the serving code")
        self.booster = lgb.Booster(model_file=str(d / "model.txt"))
        self.reference = pd.read_csv(d / "reference.csv")[FEATURES]
        self.cutoff = float(self.metadata["pd_cutoff"])
        self.version = str(self.metadata["model_version"])

    def score(self, raw: pd.DataFrame, n_reasons: int = 3) -> Scored:
        X = build_features(raw)
        p = self.booster.predict(X)
        contrib = self.booster.predict(X, pred_contrib=True)[:, :-1]
        reasons = []
        for row in contrib:
            order = np.argsort(row)[::-1][:n_reasons]
            reasons.append([REASON_TEXT[FEATURES[j]] for j in order if row[j] > 0])
        decisions = ["approve" if v <= self.cutoff else "decline" for v in p]
        return Scored(pd=p, decisions=decisions, reasons=reasons, features=X)
