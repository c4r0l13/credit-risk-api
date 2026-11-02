# Model card: credit-card probability of default

## Overview

| | |
|---|---|
| **Task** | Probability that an existing credit-card customer defaults on next month's payment |
| **Model** | LightGBM binary classifier with monotonic constraints on 11 of 12 features |
| **Output** | PD in [0, 1], approve/decline decision at a fixed cutoff, top-3 risk reasons for declines |
| **Version** | See `artifacts/metadata.json` (`model_version` = training date + data hash) |
| **Owner** | Carolina Lievanos |

## Intended use

Portfolio project that demonstrates how a PD model is built, governed and served. **It is not intended for real credit decisions.** The data is from 2005, from one market, and contains no bureau or income information.

## Data

- UCI *Default of Credit Card Clients* (Yeh & Lien, 2009): 30,000 customers in Taiwan, April–September 2005.
- Split: stratified 60% train / 20% validation (cutoff) / 20% test.
- Base default rate: 22.1%.

## Features

Payment delay (last month, two months ago, worst in 6 months, number of months late), credit limit, utilization (last month, 6-month average), share of bill paid (last month, 6-month average), balance trend, last payment amount, education level.

**Excluded on purpose:** sex, age and marital status. They are used only to audit outcomes (see the fairness section of the research repo). Including them did not improve AUC (0.779 with vs 0.781 without).

## Performance (test set, n = 6,000)

| Metric | Value |
|---|---:|
| AUC | 0.781 |
| Gini | 0.561 |
| KS | 0.432 |
| Brier score | 0.133 |
| Mean PD vs observed default rate | within 0.1 pt |
| Approval rate at cutoff | 66.9% |
| Default rate among approved | 11.3% |

## Fairness (at the production cutoff)

All groups pass the four-fifths rule (adverse impact ratio ≥ 0.80): female 1.00, male 0.95, age < 30 0.89, 30–45 1.00, > 45 0.92. To monitor: applicants under 30 who would repay are approved 8 points less often than those aged 30–45, and risk for applicants over 45 is under-predicted by about 3 points.

## Monitoring and retraining triggers

- Any feature PSI > 0.25, or score PSI > 0.10 → investigate.
- Score PSI > 0.25 sustained for a week → retrain candidate.
- Realized default rate in any score decile deviates > 5 pts from predicted once labels mature → retrain candidate.
- Every candidate must pass the CI quality gates: AUC ≥ 0.75, KS ≥ 0.38, |mean PD − default rate| ≤ 2 pts.

## Limitations

- Accepted-only population: the model never saw rejected applicants.
- Payment-status codes −2/−1/0 are poorly documented and are treated as "no delay".
- Old, single-country data. The method transfers, the parameters do not.
