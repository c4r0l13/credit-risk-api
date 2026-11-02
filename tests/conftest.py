"""Build a tiny model on synthetic data so tests never depend on the real dataset."""

from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from app.features import BILL, FEATURES, MONOTONE, PAY_AMT, PAY_STATUS, build_features


def synthetic_raw(n: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"LIMIT_BAL": rng.choice([20_000, 50_000, 200_000], n), "EDUCATION": rng.choice([1, 2, 3], n)})
    delay = rng.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1])
    for c in PAY_STATUS:
        df[c] = delay
    for c in BILL:
        df[c] = rng.uniform(0, 1, n) * df["LIMIT_BAL"]
    for c in PAY_AMT:
        df[c] = rng.uniform(0, 0.2, n) * df["LIMIT_BAL"]
    y = (rng.uniform(size=n) < 0.1 + 0.3 * delay).astype(int)
    return df, y


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    raw, y = synthetic_raw(3000)
    X = build_features(raw)
    model = lgb.LGBMClassifier(n_estimators=40, monotone_constraints=[MONOTONE[f] for f in FEATURES], verbose=-1)
    model.fit(X, y)
    model.booster_.save_model(str(d / "model.txt"))
    X.to_csv(d / "reference.csv", index=False)
    meta = {
        "model_version": "test-0001",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "pd_cutoff": 0.3,
        "metrics": {"auc": 0.7},
        "features": FEATURES,
        "data": {"file": "synthetic"},
    }
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


@pytest.fixture(scope="session")
def client(artifact_dir):
    os.environ["ARTIFACT_DIR"] = str(artifact_dir)
    os.environ["DRIFT_WINDOW"] = "500"
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def application() -> dict:
    return {
        "application_id": "app-1",
        "LIMIT_BAL": 50000,
        "EDUCATION": 2,
        "pay_status": [2, 2, 0, 0, 0, 0],
        "bill_amt": [46990, 48233, 49291, 28314, 28959, 29547],
        "pay_amt": [0, 2000, 1500, 1000, 1000, 800],
    }
