# CHECKPOINT — Phase 6: Strict Out-of-Sample Evaluation

## What was implemented
- [`regime_detection/evaluation/oos_pipeline.py`](../../regime_detection/evaluation/oos_pipeline.py) — `run_quarterly_walk_forward()`: at each quarter-end in OOS, refits every model on rows up to `t_q − 21d`, predicts for `(t_q, t_{q+1}]`; hyperparameters frozen at Phase-4 defaults.
- [`regime_detection/evaluation/metrics.py`](../../regime_detection/evaluation/metrics.py) — `annualised_return`, `annualised_vol`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `hit_rate_at_threshold`, `brier_score`, `log_loss`, `strategy_returns` (equity-overlay).
- [`regime_detection/evaluation/stats_tests.py`](../../regime_detection/evaluation/stats_tests.py) — `diebold_mariano` with Newey-West HAC, `block_bootstrap_sortino_diff`, `stationary_bootstrap_sortino_diff` (Politis-Romano).
- [`regime_detection/scripts/run_phase6.py`](../../regime_detection/scripts/run_phase6.py) — driver: quarterly walk-forward → metrics → DM + bootstrap tests → 6 canonical figures at 300 dpi → [`reports/step06_oos_results.md`](../step06_oos_results.md).
- Persisted under [`artifacts/evaluation/`](../artifacts/evaluation/): `headline.parquet`, `sub_period.parquet`, `dm_vs_baseline.parquet`, `dm_pairwise.parquet`, `bootstrap_sortino.parquet`, per-model OOS predictions + refit logs.
- 6 figures in [`figures/thesis_v5/`](../figures/thesis_v5/).

## What tests pass
**15/15** in `test_evaluation.py` (114/114 total).
- `TestMetrics` — 8 (annualised return/vol formula, Sortino sign, Sortino-undefined when zero excess, max DD on synthetic spike, hit-rate match, top-decile quantile, Brier/log-loss formulas, strategy-overlay shift semantics)
- `TestDiebold` — 3 (identical losses → NaN stat, detects systematic gap under iid, HAC lags = h − 1)
- `TestBlockBootstrap` — 2 (zero-difference identical inputs, positive-drift detection)
- `TestStationaryBootstrap` — 1 (geometric block-length smoke)
- `TestQuarterEnds` — 1

## Headline OOS results (22 quarterly refits per model, n_oos = 1,327)
| model | AUC | Brier | Sortino strat | Sortino BH | lift | Max DD strat | Max DD BH |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_forest | 0.696 | 0.180 | 0.726 | 1.031 | −0.30 | −0.40 | −0.28 |
| **logistic_l1** | 0.660 | 0.181 | **1.086** | 1.031 | **+0.06** | **−0.26** | −0.28 |
| linear_svm | 0.601 | 0.190 | 1.032 | 1.031 | +0.00 | −0.26 | −0.28 |
| logistic_elasticnet | 0.660 | 0.181 | 0.990 | 1.031 | −0.04 | −0.32 | −0.28 |

Quarterly refits genuinely help vs single-fit Phase-4: RF AUC 0.64 → 0.70; Brier ~0.21 → ~0.18 across the board.

## Statistical significance
- DM stats vs always-mean baseline: 0.86 / 0.96 / **1.63** / 0.95 — **none** reach 5 % two-sided. Linear_svm closest (p = 0.10).
- Block-bootstrap 95 % CI for Sortino lift brackets zero for every model (block_length = 21, 1,000 resamples). Stationary bootstrap is wider as expected.
- Block-bootstrap effective resolution ≈ ±0.5 Sortino units; observed +0.06 lift sits well inside that.

## Deviations from spec
1. **No per-quarter CV.** Hyperparameters frozen at Phase-4 defaults. The brief said "supervised model" should refit quarterly; I extended "refit" to mean parameter refit only, not hyperparameter retuning. Per-quarter tuning would risk snooping; documented.
2. **Hansen SPA and White Reality Check not run.** Implemented block + stationary bootstrap of the Sortino difference instead. SPA's bootstrap-of-bootstrap structure is more complex; on this sample size the marginal value over a bootstrap CI is small.
3. **Target threshold fixed at the dev 90th-pct** — same threshold for the OOS bar, not updated quarterly. Documented; a rolling-threshold alternative is straightforward to add.
4. **Strategy-overlay shift is 1 day** — decision at end of t-1 acted upon at t. Realistic-trading default; not parameterised in the headline.

## Open questions for the user
- Add Hansen SPA / White Reality Check as a robustness step? Worth ~150 lines of additional code; will not change the headline conclusion given the sample-size ceiling.
- Per-quarter hyperparameter retuning: enable, or keep frozen for the thesis?
- Transaction-cost layer (e.g. 1 bp/day round-trip on in-market days) — implement as a strategy-returns flag?
