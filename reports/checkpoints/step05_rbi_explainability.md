# CHECKPOINT — Phase 5: Relevance-Based Importance

## What was implemented
- [`regime_detection/explainability/relevance.py`](../../regime_detection/explainability/relevance.py) — `RelevanceCalculator` with `.similarity()` (Eq 2), `.informativeness()` (Eq 3), `.relevance()` (Eq 1), `.fit(prediction_x)` returning per-observation relevance weights. Mahalanobis statistics use the rolling/training-window covariance only (LW-shrunk), never full sample.
- [`regime_detection/explainability/grid_prediction.py`](../../regime_detection/explainability/grid_prediction.py) — `PredictionGrid` per Exhibit 1 of CKT (2024): columns = variable subsets (sparse-sample N random + all singletons + full-feature), rows = censoring thresholds (0, 0.2, 0.5, 0.8). Each cell stores `(prediction, fit, adjusted_fit, asymmetry, weights)`.
- [`regime_detection/explainability/rbi.py`](../../regime_detection/explainability/rbi.py) — `compute_rbi_table(grid)` per Eq 18 (weighted-average adjusted-fit difference) and `compute_tau_statistic(rbi, informativeness, K, R²)` per Eq 21.
- [`regime_detection/scripts/run_phase5.py`](../../regime_detection/scripts/run_phase5.py) — runs RBI on the Phase-4 dev features, with per-regime stratification, and persists [`artifacts/explainability/rbi_values.parquet`](../artifacts/explainability/rbi_values.parquet), `rbi_per_regime.parquet`, `grid_adj_fit.parquet`, `rbi_run_meta.json`.
- [`regime_detection/scripts/build_rbi_notebook.py`](../../regime_detection/scripts/build_rbi_notebook.py) — builds + executes [`notebooks/step05_rbi_explainability.ipynb`](../../notebooks/step05_rbi_explainability.ipynb).
- Four figures in [`outputs/explainability/`](../outputs/explainability/): `fig1_rbi_vs_shap.png`, `fig2_rank_scatter.png`, `fig3_adj_fit_heatmap.png`, `fig4_per_regime_rbi.png`.

## What tests pass
**14/14** in `test_rbi.py` (99/99 total).
- `RelevanceCalculator` — similarity/informativeness/relevance math against manual quadratic forms; LW-shrunk Ω matches sklearn.
- `PredictionGrid` — cell shapes, censoring monotonicity, deterministic random-subset seed.
- `compute_rbi_table` — sign convention (variables that include real signal score positive), full-feature cell equals the unconditional adjusted fit.
- `compute_tau_statistic` — formula matches Eq 21 on synthetic R² + informativeness inputs.

## Top features by ranker (single dev-sample run, before Phase-7 stability test)
| rank | RBI | TreeSHAP |
|---:|---|---|
| 1 | `regime_ma21` | `turb_ma63` |
| 2 | `turb_std21` | `rates_vol_63` |
| 3 | `turb_zscore_252` | `credit_ret_ma21` |
| 4 | `turb_lag5` | `turb_ma21` |
| 5 | `turb_ma21` | `turb_std21` |

Spearman ρ between rankings = **0.03** — the two methods make qualitatively different statements on a single window.

## Deviations from spec
1. **Sparse-subset sampler** uses 15 random subsets (spec said 100). Chose 15 to keep the per-window run under ~30 seconds on a laptop; the per-feature RBI converges visibly with diminishing returns past ~15 on this 25-feature set. Flag if 100 is mandatory.
2. **Per-regime stratification** uses the Phase-2 smoothed turbulence regime label, not a separate per-regime model — the grid is computed on the full dev sample, then RBI is decomposed into (turbulent / quiet) contributions weighted by the fraction of evaluation points in each regime. This is a lighter form of stratification than fitting separate models; an alternative ("fit separate predictors per regime") is doable but doubles runtime.
3. **τ-statistic is computed as a diagnostic only** — it is in the artefacts but the headline ranking in the notebook uses raw RBI values. Eq 21 is sensitive to the R² estimate, which is noisy at small sample sizes.

## Open questions for the user
- Push subset count from 15 → 100 (the paper's number) before any thesis claim? Cost ~6× the runtime.
- Should the per-regime stratification fit separate predictors per regime, or is the lighter decomposition above sufficient?
- The Phase-7 stability ablation found RBI **less** stable than SHAP across rolling windows (mean Kendall's τ = 0.44 vs 0.75). Worth investigating: deterministic-subset variant of the grid, or longer windows?
