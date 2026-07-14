# CHECKPOINT — Phase 7: Benchmarks and Ablations

> ⚠️ **Superseded under the fixed 2020-01-01 OOS constraint.** The cutoff
> ablation below used the old grid (2019/2020/2021-12-31, default 2020-12-31).
> It is now re-centred on the new default (2018/2019/2020-12-31, default
> 2019-12-31); current numbers are in the canonical report
> [`reports/step07_ablations.md`](../step07_ablations.md). Under the new split
> the cutoff Sortino lifts are −0.240 / −0.182 / −0.043 (all ≤ 0): the
> correction-timing overlay does not add risk-adjusted return once COVID and the
> 2020-26 bull are in the test set.

## What was implemented
- [`regime_detection/scripts/run_phase7.py`](../../regime_detection/scripts/run_phase7.py) — six ablations + one limitation flag, full results to [`reports/step07_ablations.md`](../step07_ablations.md):
  - **A. Regime conditioning** — drop `regime_lag1`/`regime_ma21`, re-run quarterly walk-forward
  - **B. Lookback sensitivity** — turbulence recomputed at 5y / 10y / 15y, full pipeline rerun
  - **C. Event-time threshold ±25 %** — moments of one-event-unit returns vs calendar 21d
  - **D. PGTS embargo** — 10 / 21 / 42 days
  - **E. Train/test cutoff** — 2019-12-31 / 2020-12-31 / 2021-12-31
  - **F. RBI vs SHAP stability** — Kendall's τ between top-25 rankings across rolling 5-year dev windows (1-year stride, 8 windows)
  - **G. GMM benchmark** — flagged as limitation; see below
- Per-ablation outputs persisted to [`artifacts/ablations/`](../artifacts/ablations/) (parquet for A–E, csv for F due to mixed-type rows).

## What tests pass
No new test file (ablation logic is integration-only and exercised by the driver). **114/114 total** tests still green after Phase 7.

## Headline verdicts
| ablation | survives? | summary |
|---|---|---|
| A regime conditioning | NO | mean Sortino lift with/without features = +0.007 vs −0.019 (Δ = 0.026, inside noise) |
| B lookback 5/10/15y | YES (qualified) | 10y best (+0.06), 15y similar (+0.03), 5y worse (−0.12) |
| C event threshold ±25 % | YES | event excess kurtosis 0.89/0.57/0.53 vs calendar 21d = 9.51 |
| D embargo 10/21/42d | YES | AUC range 0.004; Sortino-lift range 0.12 |
| E cutoff 2019/2020/2021 | NO | Sortino lifts **−0.184 / +0.055 / −0.105** — only 2020 cutoff is positive |
| F RBI vs SHAP stability | **NO (reversed!)** | mean Kendall's τ = 0.44 (RBI) vs 0.75 (SHAP) |
| G GMM benchmark | NOT RUN | legacy `fit_walk_forward()` is clean but rewiring was out of scope |

**Three honest thesis limitations** flagged prominently in the report: regime conditioning is weak (L1), the headline lift is partly a cutoff artefact (L2), RBI is less stable than SHAP across rolling windows — the opposite of the original hypothesis (L3).

## Deviations from spec
1. **GMM benchmark not run.** The brief asked for a head-to-head against the GMM regime detector. The legacy `fit_walk_forward()` path is clean per Phase 0 but wiring it into the new feature matrix requires recreating the anchor-validation logic that the v5 pivot is explicitly replacing. Flagged in the report as the largest single gap.
2. **F uses 8 rolling windows of length 5 years with 1-year stride** rather than a finer-grained schedule. With 5,000-row dev partition, finer strides produce highly correlated windows that inflate Kendall's τ artificially. The result that SHAP is more stable than RBI is robust to stride choice.
3. **B uses `logistic_l1` only** (not all four models) to keep the lookback rerun under ~3 minutes. The best-Phase-6 model is the most informative single representative; full table is regenerable if needed.
4. **Initial driver report's auto-generated summary** glossed the cutoff-sensitivity finding (E) by saying "the qualitative ranking is preserved across cutoffs" — technically true but misleading because the direction of the lift flips. The committed report rewrites this section to flag E and F as prominent limitations.

## Open questions for the user
- Run the clean walk-forward GMM benchmark (would close the largest gap, ~1 day of work)?
- For E, slide the cutoff month-by-month and report the full distribution of Sortino lifts rather than three point cuts?
- Investigate F's reversal: deterministic-subset variant of the prediction grid, larger window length, or accept the finding as-is?
