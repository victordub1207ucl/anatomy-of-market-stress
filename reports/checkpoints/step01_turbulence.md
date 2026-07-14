# CHECKPOINT — Phase 2: Mahalanobis Turbulence Index

## What was implemented
- [`regime_detection/models/turbulence.py`](../../regime_detection/models/turbulence.py) — `TurbulenceIndex` class:
  - `lookback_days=2520, min_periods=504, shrinkage="ledoit-wolf", refit_every=1` (defaults match Skulls paper)
  - `.fit_transform(returns)` returns daily `d_t` series + populates `.shrinkage_intensities_`
  - Cached `(mu, Omega^{-1}, intensity)` per refit so step > 1 is cheap
- [`regime_detection/models/turbulence_regimes.py`](../../regime_detection/models/turbulence_regimes.py):
  - `classify_regimes(turb, threshold_quantile, rolling_quantile_window, min_periods)` — rolling-quantile threshold, fully causal
  - `smooth_min_duration(regime, min_duration=5)` — **causal** variant (accepts transition only after `min_duration` consecutive past days)
- [`regime_detection/scripts/sanity_check_turbulence.py`](../../regime_detection/scripts/sanity_check_turbulence.py) — runs on real factor prices; verifies every crisis window registers top-decile.
- [`regime_detection/scripts/build_diagnostics_notebook.py`](../../regime_detection/scripts/build_diagnostics_notebook.py) — assembles + executes [`notebooks/step01_turbulence_diagnostics.ipynb`](../../notebooks/step01_turbulence_diagnostics.ipynb).
- Persisted: [`outputs/turbulence/turbulence_series.parquet`](../outputs/turbulence/turbulence_series.parquet) (5,124 rows × 4 cols) + 4 figures at 120 dpi.

## What tests pass
**19/19** in `test_turbulence.py` (48/48 total).
- `TestTurbulenceIndex` — 8 tests (manual Mahalanobis match with/without LW, no-lookahead, warm-up NaN, shrinkage intensities in [0,1], step-block freezing, known-shift detection, input validation)
- `TestClassifyRegimes` — 5 tests
- `TestSmoothMinDuration` — 6 tests (causal property verified by truncation)

**Sanity check passes all 7 crisis windows** (2008 GFC, 2011 Euro/US dg, 2015 oil/China, 2018 Vol-mageddon, 2018 Q4, 2020 COVID, 2022 inflation). Top-20 list reads like a textbook crisis chronology.

## Deviations from spec
1. **10-factor stable subset** (drops `em_credit, momentum, short_vol, low_risk`) for the sanity check. The cached parquet is raw Yahoo data without ETF-inception splicing, so the full 14-factor all-finite slice only starts 2011-10-21 — which loses the 2008 GFC after the 504-day warm-up. With the 10-factor subset, all-finite starts 2005-12-08 and the first finite turbulence value is 2007-12-06. The brief asked for 2008 to be flagged; the subset achieves that. Documented in the sanity-check script header.
2. **Notebook uses `classify_regimes(min_periods=63)`** instead of the class default `504`. The turbulence series is already 504-day-warmed-up, so the chained warmup pushed regime label coverage past the 2008 GFC. Lowered the notebook value so 2008 gets a regime label.
3. **Notebook is built programmatically via `nbformat` + `nbclient`** (not hand-written JSON). The `.ipynb` is executed once at build time so it ships with rendered output and figures embedded.

## Open questions for the user
- Should the v4 loader's pre-inception splicing be ported so the full 14-factor set covers 2008? Currently the 10-factor compromise is the only known way to keep 2008 coverage with strict walk-forward turbulence.
- Carry the `min_periods=63` notebook override into the `classify_regimes` default, or leave the conservative `504` default and require callers to opt in to faster bootstrapping?
