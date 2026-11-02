# Drift study

| scenario             |   max_feature_psi |   score_psi |   mean_pd |   approval_rate |
|:---------------------|------------------:|------------:|----------:|----------------:|
| Hold-out (no drift)  |             0.006 |       0.005 |     0.221 |           0.669 |
| Downturn stress test |             1.836 |       0.307 |     0.304 |           0.446 |

## PSI per feature

|                      |   Hold-out (no drift) |   Downturn stress test |
|:---------------------|----------------------:|-----------------------:|
| EDUCATION            |                 0.001 |                  0.001 |
| LIMIT_BAL            |                 0.002 |                  0.002 |
| bill_trend_6m        |                 0.002 |                  0.008 |
| utilization_avg_6m   |                 0.005 |                  0.065 |
| pay_amt_last         |                 0.002 |                  0.11  |
| utilization_last     |                 0.002 |                  0.182 |
| PAY_0                |                 0.005 |                  0.314 |
| max_delinquency_6m   |                 0.006 |                  0.472 |
| payment_ratio_avg_6m |                 0.004 |                  0.678 |
| months_delinquent_6m |                 0.005 |                  0.709 |
| payment_ratio_last   |                 0.005 |                  1.135 |
| PAY_2                |                 0.003 |                  1.836 |
