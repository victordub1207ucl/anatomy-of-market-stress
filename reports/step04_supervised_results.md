# Phase 4 — Supervised Forecasting Results

**Generated:** 2026-06-18 14:00 UTC  
**Target:** `binary_turbulence_entry`  (turbulence crosses 13.66 within next 21 trading days)  
**Dev cutoff:** 2019-12-31  **OOS:** 2020-01-01 onwards  
**Cross-validation:** PurgedGroupTimeSeriesSplit (n_splits=5, forward_window=21, embargo=21, strict expanding-window)

## Headline OOS results (one evaluation pass per model)

| model               |   n_oos |   pos_rate |   auc |   avg_precision |   prec_top_decile |   rec_top_decile |   brier |   cv_mean_auc |   cv_mean_brier |   dev_pos_rate |   oos_pos_rate |
|:--------------------|--------:|-----------:|------:|----------------:|------------------:|-----------------:|--------:|--------------:|----------------:|---------------:|---------------:|
| random_forest       |    1580 |      0.855 | 0.669 |           0.926 |             0.987 |            0.115 |   0.192 |         0.544 |           0.263 |           0.56 |          0.855 |
| logistic_l1         |    1580 |      0.855 | 0.72  |           0.942 |             1     |            0.117 |   0.125 |         0.54  |           0.304 |           0.56 |          0.855 |
| linear_svm          |    1580 |      0.855 | 0.293 |           0.789 |             0.69  |            0.081 |   0.27  |         0.436 |           0.287 |           0.56 |          0.855 |
| logistic_elasticnet |    1580 |      0.855 | 0.719 |           0.942 |             1     |            0.117 |   0.125 |         0.54  |           0.306 |           0.56 |          0.855 |

![CV vs OOS AUC](figures/supervised/fig1_auc_cv_vs_oos.png)

## Confusion matrices at the OOS top-decile cutoff

### random_forest  (cutoff = 0.7713)

| actual   |   pred_0 |   pred_1 |
|:---------|---------:|---------:|
| true_0   |      227 |        2 |
| true_1   |     1195 |      156 |

### logistic_l1  (cutoff = 0.9947)

| actual   |   pred_0 |   pred_1 |
|:---------|---------:|---------:|
| true_0   |      229 |        0 |
| true_1   |     1193 |      158 |

### linear_svm  (cutoff = 0.5438)

| actual   |   pred_0 |   pred_1 |
|:---------|---------:|---------:|
| true_0   |      180 |       49 |
| true_1   |     1242 |      109 |

### logistic_elasticnet  (cutoff = 0.9951)

| actual   |   pred_0 |   pred_1 |
|:---------|---------:|---------:|
| true_0   |      229 |        0 |
| true_1   |     1193 |      158 |

## Calibration

Reliability bins for the raw OOS probabilities and an isotonic-recalibrated version.  *Caveat:* the isotonic step is fit on OOS pairs as a Phase-4 v1 diagnostic — the production calibrator should be fit on out-of-fold dev predictions and applied once to OOS.  This is flagged in the README and is the one short-cut taken in Phase 4.

![Calibration](figures/supervised/fig2_calibration.png)

| model               |   raw_brier |   iso_brier |   delta |
|:--------------------|------------:|------------:|--------:|
| random_forest       |       0.192 |       0.117 |   0.075 |
| logistic_l1         |       0.125 |       0.114 |   0.011 |
| linear_svm          |       0.27  |       0.124 |   0.146 |
| logistic_elasticnet |       0.125 |       0.113 |   0.012 |

## OOS performance per turbulence regime

Conditioning on the Phase-2 smoothed turbulence regime at evaluation time.  Each row is the model's OOS performance *within* that regime.

### random_forest

|   regime |    n |   pos_rate |   auc |   prec_td |   rec_td |
|---------:|-----:|-----------:|------:|----------:|---------:|
|        0 | 1001 |      0.796 | 0.596 |     0.901 |    0.114 |
|        1 |  579 |      0.957 | 0.615 |     1     |    0.105 |

### logistic_l1

|   regime |    n |   pos_rate |   auc |   prec_td |   rec_td |
|---------:|-----:|-----------:|------:|----------:|---------:|
|        0 | 1001 |      0.796 | 0.632 |      0.97 |    0.123 |
|        1 |  579 |      0.957 | 0.761 |      1    |    0.105 |

### linear_svm

|   regime |    n |   pos_rate |   auc |   prec_td |   rec_td |
|---------:|-----:|-----------:|------:|----------:|---------:|
|        0 | 1001 |      0.796 | 0.37  |     0.624 |    0.079 |
|        1 |  579 |      0.957 | 0.301 |     0.948 |    0.099 |

### logistic_elasticnet

|   regime |    n |   pos_rate |   auc |   prec_td |   rec_td |
|---------:|-----:|-----------:|------:|----------:|---------:|
|        0 | 1001 |      0.796 | 0.632 |      0.96 |    0.122 |
|        1 |  579 |      0.957 | 0.754 |      1    |    0.105 |

## In-sample (CV) vs OOS gap

If CV AUC is materially higher than OOS AUC the model is over-fitting to the dev window; if OOS AUC is *higher* (rare) the dev folds happened to be more difficult than OOS.

| model               |   cv_mean_auc |   cv_std_auc |   oos_auc |   auc_gap |   cv_mean_brier |   oos_brier |
|:--------------------|--------------:|-------------:|----------:|----------:|----------------:|------------:|
| random_forest       |         0.544 |        0.128 |     0.669 |    -0.125 |           0.263 |       0.192 |
| logistic_l1         |         0.54  |        0.111 |     0.72  |    -0.18  |           0.304 |       0.125 |
| linear_svm          |         0.436 |        0.118 |     0.293 |     0.143 |           0.287 |       0.27  |
| logistic_elasticnet |         0.54  |        0.114 |     0.719 |    -0.179 |           0.306 |       0.125 |

## Economic value — Sortino of equity overlay

Trivial overlay strategy: go flat in the equity factor whenever the model's predicted probability (next-21d turbulence-entry) exceeds its OOS top-decile cutoff; otherwise hold the equity factor long.  Signals lag by one day (decision at end of t-1 acted on at t).  Annualised Sortino, target = 0.

| model               |   in_market_share |   ann_ret_overlay |   ann_ret_buy_hold |   sortino_overlay |   sortino_buy_hold |   threshold |
|:--------------------|------------------:|------------------:|-------------------:|------------------:|-------------------:|------------:|
| random_forest       |            0.8994 |            0.0953 |             0.1351 |            0.7851 |             0.7903 |      0.7713 |
| logistic_l1         |            0.8994 |            0.0969 |             0.1351 |            0.5967 |             0.7903 |      0.9947 |
| linear_svm          |            0.8994 |            0.117  |             0.1351 |            0.6682 |             0.7903 |      0.5438 |
| logistic_elasticnet |            0.8994 |            0.0979 |             0.1351 |            0.6015 |             0.7903 |      0.9951 |

## Artefacts

Saved under `artifacts/supervised/`:

- `random_forest.joblib` — fitted estimator + scaler
- `random_forest.json` — metadata (rows, hyperparams, seed, threshold, features used)
- `logistic_l1.joblib` — fitted estimator + scaler
- `logistic_l1.json` — metadata (rows, hyperparams, seed, threshold, features used)
- `linear_svm.joblib` — fitted estimator + scaler
- `linear_svm.json` — metadata (rows, hyperparams, seed, threshold, features used)
- `logistic_elasticnet.joblib` — fitted estimator + scaler
- `logistic_elasticnet.json` — metadata (rows, hyperparams, seed, threshold, features used)

Figures under `outputs/supervised/`:

- `figures/supervised/fig1_auc_cv_vs_oos.png`
- `figures/supervised/fig2_calibration.png`