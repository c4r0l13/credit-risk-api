"""Train the PD model and write versioned, framework-native artifacts.

The run fails (non-zero exit) if the candidate does not pass the quality
gates, so a bad model can never reach the artifact folder or the image.

Usage: python -m training.train --data data/raw/default_of_credit_card_clients.csv --out artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from app.features import FEATURES, MONOTONE, build_features

TARGET = "default payment next month"
MAX_BAD_RATE = 0.12
GATES = {"auc_min": 0.75, "ks_min": 0.38, "calibration_gap_max": 0.02}
PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def cutoff_for_bad_rate(y: np.ndarray, p: np.ndarray, max_bad_rate: float) -> float:
    order = np.argsort(p)
    bad_rate = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    ok = np.flatnonzero(bad_rate <= max_bad_rate)
    return float(p[order][ok[-1]])


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "ks": round(float(np.max(tpr - fpr)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "calibration_gap": round(float(abs(p.mean() - y.mean())), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/raw/default_of_credit_card_clients.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    data_hash = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()[:12]
    y = df[TARGET].to_numpy()
    X = build_features(df)
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=args.seed)
    Xva, Xte, yva, yte = train_test_split(Xrest, yrest, test_size=0.5, stratify=yrest, random_state=args.seed)

    model = lgb.LGBMClassifier(
        **PARAMS, monotone_constraints=[MONOTONE[f] for f in FEATURES], random_state=args.seed, verbose=-1
    )
    model.fit(Xtr, ytr)
    cutoff = cutoff_for_bad_rate(yva, model.predict_proba(Xva)[:, 1], MAX_BAD_RATE)
    p_te = model.predict_proba(Xte)[:, 1]
    metrics = evaluate(yte, p_te)
    metrics["approval_rate"] = round(float((p_te <= cutoff).mean()), 4)
    metrics["bad_rate_approved"] = round(float(yte[p_te <= cutoff].mean()), 4)

    failed = [
        name
        for name, ok in {
            "auc_min": metrics["auc"] >= GATES["auc_min"],
            "ks_min": metrics["ks"] >= GATES["ks_min"],
            "calibration_gap_max": metrics["calibration_gap"] <= GATES["calibration_gap_max"],
        }.items()
        if not ok
    ]
    print(json.dumps({"metrics": metrics, "gates": GATES, "failed": failed}, indent=2))
    if failed:
        print(f"Quality gates failed: {failed}. Artifacts NOT written.", file=sys.stderr)
        return 1

    version = datetime.now(UTC).strftime("%Y%m%d") + f"-{data_hash[:7]}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / "model.txt"))
    Xva.sample(3000, random_state=0).to_csv(out / "reference.csv", index=False)
    meta = {
        "model_version": version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": FEATURES,
        "monotone_constraints": MONOTONE,
        "params": PARAMS,
        "pd_cutoff": round(cutoff, 6),
        "policy": {"max_bad_rate": MAX_BAD_RATE},
        "metrics": metrics,
        "gates": GATES,
        "data": {"file": Path(args.data).name, "sha256_12": data_hash, "rows": len(df)},
        "lightgbm_version": lgb.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Artifacts written to {out.resolve()} (version {version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
