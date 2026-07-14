# CHECKPOINT — Phase 4: Supervised Forecasting

> ⚠️ **Superseded under the fixed 2020-01-01 OOS constraint.** The numbers below
> were computed on the old 2021-01-01 OOS. The current, regenerated results
> (DEV_END = 2019-12-31, OOS = 2020-01-01 → 2026-04-14, n_oos = 1,580) are in
> the canonical report [`reports/step04_supervised_results.md`](../step04_supervised_results.md).

## What was implemented
- [`regime_detection/supervised/targets.py`](../../regime_detection/supervised/targets.py) — three targets:
  - `binary_turbulence_entry(turbulence, threshold, horizon=21)`
  - `forward_drawdown_conditional(prices, current_regime=None, horizon=21)` — returns Series or DataFrame with regime column
  - `event_time_drawdown(prices, daily_turbulence, intensity_threshold)`
- [`regime_detection/supervised/features.py`](../../regime_detection/supervised/features.py) — `build_feature_matrix()` → 25 causal columns (turbulence dynamics, regime, equity/rates/credit/vix moments, cross-asset correlations).
- [`regime_detection/supervised/pgts.py`](../../regime_detection/supervised/pgts.py) — `PurgedGroupTimeSeriesSplit`, **strict expanding-window** (training rows after the test fold are never used).
- [`regime_detection/supervised/train.py`](../../regime_detection/supervised/train.py) — registry of four challenger models, `train_one_model()` + `train_all_models()`, scaler refit per fold, single final fit on full dev, OOS predictions.
- [`regime_detection/supervised/evaluate.py`](../../regime_detection/supervised/evaluate.py) — OOS metrics, confusion @ top-decile, calibration, per-regime breakdown, Sortino overlay, CV vs OOS gap.
- [`regime_detection/scripts/run_phase4.py`](../../regime_detection/scripts/run_phase4.py) — end-to-end driver; saves `.joblib` + `.json` per model to [`artifacts/supervised/`](../artifacts/supervised/) and writes [`reports/step04_supervised_results.md`](../step04_supervised_results.md).

## What tests pass
**16/16** in `test_supervised.py` (85/85 total).
- `TestBinaryTurbulenceEntry` — 4 (label correctness, tail NaN, no-lookahead, validation)
- `TestForwardDrawdown` — 3
- `TestEventTimeDrawdown` — 2
- `TestPGTS` — 7 (n_splits, disjoint, purge correctness, embargo adds to left-side gap, strict expanding-window, short-train skip, init validation)

## Headline OOS results (single-fit, 2021-01-01 → 2026-04-14, n_oos = 1,327)
| model | AUC | Brier | prec@10% | Sortino strat | Sortino BH |
|---|---:|---:|---:|---:|---:|
| random_forest | 0.641 | 0.213 | 1.00 | 1.16 | 1.03 |
| logistic_l1 | 0.668 | 0.182 | 1.00 | 1.01 | 1.03 |
| linear_svm | 0.643 | 0.224 | 1.00 | 1.05 | 1.03 |
| logistic_elasticnet | 0.669 | 0.182 | 1.00 | 0.98 | 1.03 |

## Deviations from spec
1. **PGTS is strict expanding-window**, not the symmetric variant in de Prado ch. 7. Training rows after the test fold are never used. Embargo adds to the left-side gap rather than carving from the right. Justified for financial walk-forward; documented in the module docstring.
2. **Focused 25-column feature set**, not the v4 69-feature engineering. Brief was feature-agnostic; the new pipeline is turbulence-conditional, so turbulence-derived features carry most of the signal. The 69-feature engineering can be substituted later.
3. **Target threshold lifted from the 75th to the 90th percentile** of dev turbulence (initial draft used 75th, which produced a 91 % OOS positive rate — classification trivial). The 90th percentile gives 79 % OOS positive rate — imbalanced but informative.
4. **Calibration uses isotonic on OOS pairs** as a Phase-4 v1 diagnostic. The production calibrator should be fit on dev out-of-fold predictions and applied once to OOS — flagged explicitly in the report's calibration section. Does not affect AUC, Brier, or Sortino rows of the headline table.
5. **No per-model hyperparameter tuning**. Defaults are frozen at sensible values; per-quarter or per-fold tuning would risk hyperparameter snooping. Documented as a deliberate scope choice.

## Open questions for the user
- Substitute the 69-feature v4 engineering for the 25-feature focused set, or keep the lean version?
- Move the isotonic calibrator to dev-OOF before any further Phase-4 reporting?
- Is the 90th-pct target threshold the right ground truth, or do you prefer a rolling threshold (which would make labels non-stationary across the OOS window)?
