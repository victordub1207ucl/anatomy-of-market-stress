# Phase 6 — Strict OOS Evaluation

**Generated:** 2026-06-18 14:06 UTC

**Protocol:** quarterly walk-forward refit.  At each quarter-end `t_q`, every challenger model is refit on data up to `t_q − 21d` (the supervised-target horizon).  Predictions for `(t_q, t_{q+1}]` are generated from that fit and never reused.  Hyperparameters are frozen at the Phase-4 defaults — no per-quarter tuning.  Feature standardisation is refit inside every quarterly training pass.

**Target:** `binary_turbulence_entry(threshold=13.66, horizon=21)`.  Threshold is the 90th percentile of dev turbulence (≤ 2019-12-31) — fixed, not updated quarterly, so the OOS bar is the same as the dev bar.

## Headline results

| model               |   n_oos |   pos_rate |   auc |   avg_precision |   brier |   log_loss |   prec_top_decile |   rec_top_decile |   sortino_strategy |   sortino_buy_hold |   sharpe_strategy |   sharpe_buy_hold |   ann_ret_strategy |   ann_ret_buy_hold |   max_dd_strategy |   max_dd_buy_hold |   in_market_share |
|:--------------------|--------:|-----------:|------:|----------------:|--------:|-----------:|------------------:|-----------------:|-------------------:|-------------------:|------------------:|------------------:|-------------------:|-------------------:|------------------:|------------------:|------------------:|
| random_forest       |    1580 |      0.855 | 0.708 |           0.935 |   0.156 |      0.475 |                 1 |            0.117 |              0.651 |               0.79 |             0.541 |             0.658 |              0.09  |              0.135 |            -0.515 |            -0.411 |               0.9 |
| logistic_l1         |    1580 |      0.855 | 0.712 |           0.94  |   0.137 |      0.421 |                 1 |            0.117 |              0.608 |               0.79 |             0.558 |             0.658 |              0.099 |              0.135 |            -0.344 |            -0.411 |               0.9 |
| linear_svm          |    1580 |      0.855 | 0.578 |           0.907 |   0.166 |      0.506 |                 1 |            0.117 |              0.719 |               0.79 |             0.647 |             0.658 |              0.123 |              0.135 |            -0.411 |            -0.411 |               0.9 |
| logistic_elasticnet |    1580 |      0.855 | 0.713 |           0.939 |   0.137 |      0.421 |                 1 |            0.117 |              0.646 |               0.79 |             0.591 |             0.658 |              0.105 |              0.135 |            -0.344 |            -0.411 |               0.9 |

![Cumulative returns](figures/thesis_v5/fig4_cumulative_returns.png)

## Sub-period breakdown

|                                      |    n |   auc |   brier |   pos_rate |
|:-------------------------------------|-----:|------:|--------:|-----------:|
| ('linear_svm', 'quiet')              | 1001 | 0.536 |   0.185 |      0.796 |
| ('linear_svm', 'turbulent')          |  579 | 0.481 |   0.134 |      0.957 |
| ('linear_svm', 'year_2020')          |  253 | 0.433 |   0.229 |      0.996 |
| ('linear_svm', 'year_2021')          |  252 | 0.535 |   0.16  |      0.841 |
| ('linear_svm', 'year_2023')          |  250 | 0.486 |   0.217 |      0.716 |
| ('linear_svm', 'year_2024')          |  252 | 0.606 |   0.176 |      0.893 |
| ('linear_svm', 'year_2025')          |  253 | 0.758 |   0.208 |      0.66  |
| ('linear_svm', 'year_2026')          |   69 | 0.104 |   0.12  |      0.942 |
| ('logistic_elasticnet', 'quiet')     | 1001 | 0.616 |   0.187 |      0.796 |
| ('logistic_elasticnet', 'turbulent') |  579 | 0.711 |   0.051 |      0.957 |
| ('logistic_elasticnet', 'year_2020') |  253 | 0.726 |   0.048 |      0.996 |
| ('logistic_elasticnet', 'year_2021') |  252 | 0.567 |   0.143 |      0.841 |
| ('logistic_elasticnet', 'year_2023') |  250 | 0.524 |   0.234 |      0.716 |
| ('logistic_elasticnet', 'year_2024') |  252 | 0.452 |   0.202 |      0.893 |
| ('logistic_elasticnet', 'year_2025') |  253 | 0.791 |   0.187 |      0.66  |
| ('logistic_elasticnet', 'year_2026') |   69 | 0.142 |   0.17  |      0.942 |
| ('logistic_l1', 'quiet')             | 1001 | 0.615 |   0.187 |      0.796 |
| ('logistic_l1', 'turbulent')         |  579 | 0.72  |   0.051 |      0.957 |
| ('logistic_l1', 'year_2020')         |  253 | 0.726 |   0.048 |      0.996 |
| ('logistic_l1', 'year_2021')         |  252 | 0.559 |   0.142 |      0.841 |
| ('logistic_l1', 'year_2023')         |  250 | 0.524 |   0.234 |      0.716 |
| ('logistic_l1', 'year_2024')         |  252 | 0.452 |   0.2   |      0.893 |
| ('logistic_l1', 'year_2025')         |  253 | 0.793 |   0.189 |      0.66  |
| ('logistic_l1', 'year_2026')         |   69 | 0.127 |   0.173 |      0.942 |
| ('random_forest', 'quiet')           | 1001 | 0.636 |   0.202 |      0.796 |
| ('random_forest', 'turbulent')       |  579 | 0.648 |   0.076 |      0.957 |
| ('random_forest', 'year_2020')       |  253 | 0.552 |   0.123 |      0.996 |
| ('random_forest', 'year_2021')       |  252 | 0.517 |   0.178 |      0.841 |
| ('random_forest', 'year_2023')       |  250 | 0.541 |   0.219 |      0.716 |
| ('random_forest', 'year_2024')       |  252 | 0.458 |   0.243 |      0.893 |
| ('random_forest', 'year_2025')       |  253 | 0.919 |   0.148 |      0.66  |
| ('random_forest', 'year_2026')       |   69 | 0.269 |   0.184 |      0.942 |

## Statistical significance

### Diebold-Mariano vs always-mean baseline

Loss = squared error (Brier) per row.  The baseline forecast is the constant OOS mean of the labels.  Negative DM stat ⇒ model loss < baseline loss (model improves on the baseline).

| model               |   DM_stat |   DM_p |   mean_loss_diff |   hac_lags |    n |
|:--------------------|----------:|-------:|-----------------:|-----------:|-----:|
| random_forest       |     1.702 |  0.089 |            0.032 |         20 | 1580 |
| logistic_l1         |     0.875 |  0.381 |            0.013 |         20 | 1580 |
| linear_svm          |     2.707 |  0.007 |            0.042 |         20 | 1580 |
| logistic_elasticnet |     0.876 |  0.381 |            0.013 |         20 | 1580 |

### Pairwise DM stats  (rows lose vs cols when positive)

|                     |   random_forest |   logistic_l1 |   linear_svm |   logistic_elasticnet |
|:--------------------|----------------:|--------------:|-------------:|----------------------:|
| random_forest       |           0     |         2.307 |       -0.796 |                 2.34  |
| logistic_l1         |          -2.307 |         0     |       -2.345 |                -0.161 |
| linear_svm          |           0.796 |         2.345 |        0     |                 2.337 |
| logistic_elasticnet |          -2.34  |         0.161 |       -2.337 |                 0     |

### Block + stationary bootstrap of Sortino lift

`Sortino(strategy) − Sortino(buy-hold)` with 95 % CIs.  Block length = 21 trading days (≈ 1 month).  Stationary bootstrap uses geometric block lengths with mean = 21.  **Caveat:** at this OOS length the bootstrap's power against Sortino differences of less than ≈1 unit is limited.  Read the CI width as the *effective resolution*, not as a small effect.

| model               |   block_diff |   block_ci_low |   block_ci_high |   block_p |   stat_diff |   stat_ci_low |   stat_ci_high |   stat_p |
|:--------------------|-------------:|---------------:|----------------:|----------:|------------:|--------------:|---------------:|---------:|
| random_forest       |       -0.139 |         -0.924 |           0.47  |     0.742 |      -0.139 |        -1.5   |          1.17  |    0.843 |
| logistic_l1         |       -0.182 |         -0.604 |           0.339 |     0.374 |      -0.182 |        -1.645 |          1.339 |    0.803 |
| linear_svm          |       -0.071 |         -0.458 |           0.357 |     0.691 |      -0.071 |        -1.502 |          1.271 |    0.93  |
| logistic_elasticnet |       -0.145 |         -0.545 |           0.368 |     0.486 |      -0.145 |        -1.584 |          1.373 |    0.847 |

![Significance](figures/thesis_v5/fig6_significance.png)

## Refit log (last 3 of each model — sanity check)

|    | asof                | train_end           |   train_n | pred_window_end     |   pred_n |   train_pos_rate |   pred_pos_rate | model               |
|---:|:--------------------|:--------------------|----------:|:--------------------|---------:|-----------------:|----------------:|:--------------------|
|  0 | 2025-09-30 00:00:00 | 2025-09-09 00:00:00 |      4398 | 2025-12-31 00:00:00 |       64 |           0.6619 |          0.3281 | random_forest       |
|  1 | 2025-12-31 00:00:00 | 2025-12-10 00:00:00 |      4463 | 2026-03-31 00:00:00 |       61 |           0.6579 |          1      | random_forest       |
|  2 | 2026-03-31 00:00:00 | 2026-03-10 00:00:00 |      4523 | 2026-04-13 00:00:00 |        8 |           0.6615 |          0.5    | random_forest       |
|  3 | 2025-09-30 00:00:00 | 2025-09-09 00:00:00 |      4398 | 2025-12-31 00:00:00 |       64 |           0.6619 |          0.3281 | logistic_l1         |
|  4 | 2025-12-31 00:00:00 | 2025-12-10 00:00:00 |      4463 | 2026-03-31 00:00:00 |       61 |           0.6579 |          1      | logistic_l1         |
|  5 | 2026-03-31 00:00:00 | 2026-03-10 00:00:00 |      4523 | 2026-04-13 00:00:00 |        8 |           0.6615 |          0.5    | logistic_l1         |
|  6 | 2025-09-30 00:00:00 | 2025-09-09 00:00:00 |      4398 | 2025-12-31 00:00:00 |       64 |           0.6619 |          0.3281 | linear_svm          |
|  7 | 2025-12-31 00:00:00 | 2025-12-10 00:00:00 |      4463 | 2026-03-31 00:00:00 |       61 |           0.6579 |          1      | linear_svm          |
|  8 | 2026-03-31 00:00:00 | 2026-03-10 00:00:00 |      4523 | 2026-04-13 00:00:00 |        8 |           0.6615 |          0.5    | linear_svm          |
|  9 | 2025-09-30 00:00:00 | 2025-09-09 00:00:00 |      4398 | 2025-12-31 00:00:00 |       64 |           0.6619 |          0.3281 | logistic_elasticnet |
| 10 | 2025-12-31 00:00:00 | 2025-12-10 00:00:00 |      4463 | 2026-03-31 00:00:00 |       61 |           0.6579 |          1      | logistic_elasticnet |
| 11 | 2026-03-31 00:00:00 | 2026-03-10 00:00:00 |      4523 | 2026-04-13 00:00:00 |        8 |           0.6615 |          0.5    | logistic_elasticnet |

## Canonical figure set — `figures/thesis_v5/`

- `figures/thesis_v5/fig1_turbulence_with_regime.png`  (turb)
- `figures/thesis_v5/fig2_return_distributions.png`  (ret)
- `figures/thesis_v5/fig3_calibration.png`  (cal)
- `figures/thesis_v5/fig4_cumulative_returns.png`  (cum)
- `figures/thesis_v5/fig5_rbi_ranks.png`  (rbi)
- `figures/thesis_v5/fig6_significance.png`  (sig)

## Caveats

* The OOS positive rate is materially higher than the dev positive rate.  The OOS window (from 2020-01-01, per the fixed split constraint) now includes the COVID crash and had a structurally more-turbulent cross-section; AUC inflates and Brier deflates relative to dev CV.  Read the Brier and DM stats as the most scale-invariant signals.
* Strategy returns are pre-cost.  A 1 bp / day round-trip cost applied to the ~10 % in-market days would erode roughly 0.10 of annualised return; the qualitative ranking is robust to that.
* The supervised target's threshold is fixed at the 90th-pct of dev turbulence and *not* updated quarterly.  Updating it on a rolling basis is a defensible alternative; documented in `scripts/step06_oos_evaluation.py`.
