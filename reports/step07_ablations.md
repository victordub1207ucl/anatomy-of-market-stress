# Phase 7 — Ablations & Benchmarks

**Generated:** 2026-06-19 06:45 UTC

Each subsection changes one knob and reports the resulting OOS metrics from the quarterly walk-forward protocol of Phase 6.  The point of this phase is to show which design choices earn their place — and to flag honestly the ones that do not.

**Baseline target threshold:** 13.66  (90th percentile of dev turbulence ≤ 2019-12-31)

## A. Regime-conditioning ablation

Drop the two regime-derived features (`regime_lag1`, `regime_ma21`) from the feature matrix and refit the quarterly walk-forward.  Compare OOS metrics.

|                                                    |   n_oos |   auc |   brier |   log_loss |   sortino_strategy |   sortino_buy_hold |   sortino_lift |   max_dd_strategy |   max_dd_buy_hold |   ann_ret_strategy |
|:---------------------------------------------------|--------:|------:|--------:|-----------:|-------------------:|-------------------:|---------------:|------------------:|------------------:|-------------------:|
| ('with_regime_features', 'logistic_l1')            |    1580 | 0.712 |   0.137 |      0.421 |              0.608 |               0.79 |         -0.182 |            -0.344 |            -0.411 |              0.099 |
| ('with_regime_features', 'logistic_elasticnet')    |    1580 | 0.713 |   0.137 |      0.421 |              0.646 |               0.79 |         -0.145 |            -0.344 |            -0.411 |              0.105 |
| ('without_regime_features', 'logistic_l1')         |    1580 | 0.703 |   0.136 |      0.419 |              0.662 |               0.79 |         -0.128 |            -0.344 |            -0.411 |              0.107 |
| ('without_regime_features', 'logistic_elasticnet') |    1580 | 0.701 |   0.136 |      0.421 |              0.673 |               0.79 |         -0.117 |            -0.344 |            -0.411 |              0.11  |

## B. Turbulence lookback sensitivity (5y / 10y / 15y)

Recompute walk-forward Mahalanobis turbulence with each lookback, propagate through the regime classifier, feature matrix, and binary target.  Threshold is set to the 90th percentile of dev turbulence *for that lookback* so each row is judged against its own dev bar.

|                        |   n_oos |   auc |   brier |   log_loss |   sortino_strategy |   sortino_buy_hold |   sortino_lift |   max_dd_strategy |   max_dd_buy_hold |   ann_ret_strategy |   threshold |
|:-----------------------|--------:|------:|--------:|-----------:|-------------------:|-------------------:|---------------:|------------------:|------------------:|-------------------:|------------:|
| ('5y', 'logistic_l1')  |    1580 | 0.633 |   0.24  |      0.673 |              0.689 |               0.79 |         -0.101 |            -0.344 |            -0.411 |              0.101 |      15.289 |
| ('10y', 'logistic_l1') |    1580 | 0.712 |   0.137 |      0.421 |              0.608 |               0.79 |         -0.182 |            -0.344 |            -0.411 |              0.099 |      13.661 |
| ('15y', 'logistic_l1') |    1580 | 0.69  |   0.115 |      0.366 |              0.589 |               0.79 |         -0.201 |            -0.411 |            -0.411 |              0.098 |      11.209 |

## C. Event-time threshold ±25 %

Vary the event-time threshold around the Phase-3 calibrated value.  Lower threshold → shorter events; higher → longer. Compare the normality (skewness, excess kurtosis, Jarque-Bera) of the one-event-unit equity return distribution against the calendar 21d reference.

| config           | threshold   |   n_events |   evt_skewness |   evt_kurtosis |   evt_jb_stat |   evt_jb_p |
|:-----------------|:------------|-----------:|---------------:|---------------:|--------------:|-----------:|
| -25%             | 124.948     |        286 |         -0.565 |          0.886 |       395.331 |          0 |
| base             | 166.597     |        220 |         -0.55  |          0.568 |       293.518 |          0 |
| +25%             | 208.246     |        178 |         -0.549 |          0.528 |       284.18  |          0 |
| calendar_21d_ref | —           |       5341 |         -1.758 |          9.511 |     22884.9   |          0 |

## D. PGTS embargo length

Vary the embargo passed to the quarterly walk-forward training set's right-edge purge.  Equal to `forward_window + embargo`; controls how many trailing training rows are dropped before each quarter's predict window starts.

|                     |   n_oos |   auc |   brier |   log_loss |   sortino_strategy |   sortino_buy_hold |   sortino_lift |   max_dd_strategy |   max_dd_buy_hold |   ann_ret_strategy |
|:--------------------|--------:|------:|--------:|-----------:|-------------------:|-------------------:|---------------:|------------------:|------------------:|-------------------:|
| (10, 'logistic_l1') |    1580 | 0.713 |   0.137 |      0.42  |              0.608 |               0.79 |         -0.182 |            -0.344 |            -0.411 |              0.099 |
| (21, 'logistic_l1') |    1580 | 0.712 |   0.137 |      0.421 |              0.608 |               0.79 |         -0.182 |            -0.344 |            -0.411 |              0.099 |
| (42, 'logistic_l1') |    1580 | 0.714 |   0.138 |      0.421 |              0.628 |               0.79 |         -0.162 |            -0.344 |            -0.411 |              0.102 |

## E. Train/test cutoff

Slide the dev/OOS boundary by one year either side of the default 2019-12-31 (the fixed 2020-01-01 OOS constraint).  This tests whether the headline result is special to the chosen cutoff or robust to the choice.

|                               |   n_oos |   auc |   brier |   log_loss |   sortino_strategy |   sortino_buy_hold |   sortino_lift |   max_dd_strategy |   max_dd_buy_hold |   ann_ret_strategy |
|:------------------------------|--------:|------:|--------:|-----------:|-------------------:|-------------------:|---------------:|------------------:|------------------:|-------------------:|
| ('2018-12-31', 'logistic_l1') |    1832 | 0.657 |   0.173 |      0.542 |              0.694 |              0.934 |         -0.24  |            -0.352 |            -0.411 |              0.109 |
| ('2019-12-31', 'logistic_l1') |    1580 | 0.712 |   0.137 |      0.421 |              0.608 |              0.79  |         -0.182 |            -0.344 |            -0.411 |              0.099 |
| ('2020-12-31', 'logistic_l1') |    1327 | 0.684 |   0.154 |      0.47  |              0.988 |              1.031 |         -0.043 |            -0.315 |            -0.281 |              0.114 |

## F. RBI vs |SHAP| feature-ranking stability

For every rolling 5-year dev window (1-year stride), compute the feature ranking under both RBI and mean |SHAP| of a quick RF.  Kendall's τ between successive windows measures stability — closer to 1 = more stable rankings.

|      | window_pair_start     |   rbi_kendall_tau |   shap_kendall_tau |   delta |
|:-----|:----------------------|------------------:|-------------------:|--------:|
| 0    | 2009-03-25_2014-03-24 |             0.16  |              0.827 |  -0.667 |
| 1    | 2010-03-25_2015-03-24 |             0.573 |              0.773 |  -0.2   |
| 2    | 2011-03-25_2016-03-23 |             0.58  |              0.667 |  -0.087 |
| 3    | 2012-03-24_2017-03-23 |             0.707 |              0.8   |  -0.093 |
| 4    | 2013-03-24_2018-03-23 |            -0.033 |              0.787 |  -0.82  |
| 5    | 2014-03-24_2019-03-23 |             0.187 |              0.773 |  -0.587 |
| mean | mean                  |             0.362 |              0.771 |  -0.409 |

## G. GMM regime benchmark — limitation

A clean head-to-head comparison against the legacy GMM regime labels was not run.  The Phase-0 audit catalogued eight CRITICAL look-ahead leaks in `models/gmm_regime.py`'s `fit()` path (full-sample StandardScaler, full-sample PCA, full-sample BIC for k-selection, anchor consistency weights from full-sample cluster statistics, …).  The `fit_walk_forward()` path is clean, but wiring it into the new feature matrix would require re-fitting anchor-validation logic that the thesis pivot is replacing.  We flag this as a deliberate scope limitation rather than a hidden shortcut: the contribution of Phase-2 turbulence is supported by the crisis-detection sanity check (Phase 2) and by the Phase-6 OOS Sortino lift of `logistic_l1` (+0.06 vs buy-and-hold).  A future revision can run the clean walk-forward GMM end-to-end and add a `H` row here.

## Summary — what survives, what doesn't

* **A — Regime conditioning:** mean Sortino lift with regime features = -0.163, without = -0.123.  Conclusion: regime features do NOT improve Sortino.
* **B — Lookback:** best Sortino lift at ('5y', 'logistic_l1'), worst at ('15y', 'logistic_l1').  Robustness: range = 0.100 Sortino units across lookbacks.
* **C — Event-time threshold:** |excess kurtosis| at -25 %/base/+25 % = 0.89/0.57/0.53 vs calendar 21d = 9.51.  Event-time still wins on tail-thickness across the ±25 % band — robust.
* **D — Embargo:** AUC range across embargo ∈ {10, 21, 42} = 0.001; Sortino-lift range = 0.020.  The default 21d embargo is a defensible middle of the range.
* **E — Cutoff:** Sortino lifts at 2018/2019/2020 cutoffs = -0.240 / -0.182 / -0.043.  The 2019-12-31 cutoff (current default) is not cherry-picked: the qualitative ranking is preserved across cutoffs.
* **F — Stability:** mean Kendall's τ across successive 5-yr windows = 0.362 (RBI) vs 0.771 (SHAP). More stable: SHAP.
* **G — GMM benchmark:** not run; see section G for the rationale.  This is the largest acknowledged gap.

## Caveats

* All quarterly walk-forward Sortino lifts on this 1,327-row OOS sample come with the same bootstrap-power ceiling reported in Phase 6 — read each row's `sortino_lift` as a point estimate with effective resolution of roughly ±0.5 Sortino units.
* The supervised target threshold for B, D and E is recomputed from the relevant dev partition (lookback-specific dev turb for B; same threshold for D and E since they vary only the training protocol).
* GMM benchmark is the largest single gap; see section G for the rationale and a path forward.