"""Offline drift study: what does monitoring show when the economy turns?

Compares the training reference against
  1. the untouched hold-out set (should be stable), and
  2. a stress scenario: a downturn where 30% of customers fall one month
     further behind, balances grow 15% and payments shrink 20%.

Writes reports/drift.md and reports/figures/drift_psi.png.
Usage: python -m monitoring.drift_report --data data/raw/default_of_credit_card_clients.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.drift import psi, psi_table
from app.features import BILL, PAY_AMT, PAY_STATUS
from app.model import ModelService

from . import style

TARGET = "default payment next month"


def stress(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    hit = rng.uniform(size=len(df)) < 0.30
    for c in PAY_STATUS:
        out.loc[hit, c] = np.clip(out.loc[hit, c].clip(lower=0) + 1, -2, 9)
    out[BILL] = out[BILL] * 1.15
    out[PAY_AMT] = out[PAY_AMT] * 0.80
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/raw/default_of_credit_card_clients.csv")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    style.apply_style()

    svc = ModelService(args.artifacts)
    df = pd.read_csv(args.data)
    _, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=42)
    _, holdout = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=42)

    scenarios = {"Hold-out (no drift)": holdout, "Downturn stress test": stress(holdout)}
    psis, summary = {}, []
    ref_scores = svc.booster.predict(svc.reference)
    for name, data in scenarios.items():
        scored = svc.score(data)
        psis[name] = psi_table(svc.reference, scored.features)
        summary.append(
            {
                "scenario": name,
                "max_feature_psi": round(float(psis[name].max()), 3),
                "score_psi": round(psi(ref_scores, scored.pd), 3),
                "mean_pd": round(float(scored.pd.mean()), 3),
                "approval_rate": round(float(np.mean(np.array(scored.decisions) == "approve")), 3),
            }
        )
    table = pd.DataFrame(psis).sort_values("Downturn stress test")
    summary_df = pd.DataFrame(summary)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(table))
    h = 0.38
    for i, name in enumerate(table.columns):
        ax.barh(y + (0.5 - i) * h, table[name], h - 0.04, color=style.SERIES[i], label=name)
    for x, lbl, ha in ((0.10, "moderate 0.10 ", "right"), (0.25, " significant 0.25", "left")):
        ax.axvline(x, color=style.INK_2, ls="--", lw=1)
        ax.text(x, len(table) - 0.4, lbl, fontsize=8, color=style.INK_2, va="bottom", ha=ha)
    ax.set_yticks(y, table.index)
    ax.set_xlabel("Population Stability Index vs training reference")
    ax.set_title("Drift monitoring catches the downturn before defaults arrive")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "figures" / "drift_psi.png")
    plt.close(fig)

    md = [
        "# Drift study",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## PSI per feature",
        "",
        table.round(3).to_markdown(),
    ]
    (out / "drift.md").write_text("\n".join(md) + "\n")
    print(summary_df.to_string(index=False))
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
