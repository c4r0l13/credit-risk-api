"""Feature engineering shared by training AND serving.

Keeping one implementation for both paths is the simplest guard against
training/serving skew, the most common silent failure of deployed models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_COLUMNS = ["LIMIT_BAL", "EDUCATION", *PAY_STATUS, *BILL, *PAY_AMT]

# feature -> monotonic direction of risk (+1 more risk as it grows, -1 less, 0 free)
MONOTONE = {
    "PAY_0": 1,
    "max_delinquency_6m": 1,
    "months_delinquent_6m": 1,
    "LIMIT_BAL": -1,
    "utilization_last": 1,
    "utilization_avg_6m": 1,
    "payment_ratio_last": -1,
    "payment_ratio_avg_6m": -1,
    "bill_trend_6m": 1,
    "EDUCATION": 0,
    "pay_amt_last": -1,
    "PAY_2": 1,
}
FEATURES = list(MONOTONE)

REASON_TEXT = {
    "PAY_0": "Recent payment delay on the account",
    "PAY_2": "Payment delay two months ago",
    "max_delinquency_6m": "Severity of past payment delays",
    "months_delinquent_6m": "Number of months with late payments",
    "LIMIT_BAL": "Low credit limit",
    "utilization_last": "High utilization of the credit limit",
    "utilization_avg_6m": "High average utilization over 6 months",
    "payment_ratio_last": "Low share of the last bill paid",
    "payment_ratio_avg_6m": "Low share of bills paid over 6 months",
    "bill_trend_6m": "Growing balance over the last 6 months",
    "EDUCATION": "Education level",
    "pay_amt_last": "Small last payment amount",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw applicant history -> model features. Pure, deterministic, vectorized."""
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    out = pd.DataFrame(index=df.index)
    limit = df["LIMIT_BAL"].clip(lower=1)
    pay_status = df[PAY_STATUS].clip(lower=0)
    bills = df[BILL].to_numpy(dtype=float)
    pays = df[PAY_AMT].to_numpy(dtype=float)
    education = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4)

    out["PAY_0"] = pay_status["PAY_0"]
    out["PAY_2"] = pay_status["PAY_2"]
    out["max_delinquency_6m"] = pay_status.max(axis=1)
    out["months_delinquent_6m"] = (pay_status > 0).sum(axis=1)
    out["LIMIT_BAL"] = df["LIMIT_BAL"].astype(float)
    out["utilization_last"] = (df["BILL_AMT1"] / limit).clip(-1, 3)
    out["utilization_avg_6m"] = (bills.mean(axis=1) / limit).clip(-1, 3)
    prev_bills = np.clip(bills[:, 1:], 1, None)
    ratios = np.clip(pays[:, :-1] / prev_bills, 0, 2)
    out["payment_ratio_last"] = ratios[:, 0]
    out["payment_ratio_avg_6m"] = ratios.mean(axis=1)
    out["bill_trend_6m"] = ((bills[:, 0] - bills[:, -1]) / limit).clip(-3, 3)
    out["EDUCATION"] = education
    out["pay_amt_last"] = df["PAY_AMT1"].astype(float)
    return out[FEATURES]
