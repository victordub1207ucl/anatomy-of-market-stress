# CHECKPOINT — Phase 9C: sliding-cutoff sensitivity (L2)

**Generated:** 2026-06-19 06:38 UTC

## What was implemented
- `regime_detection/lib/cutoff_sensitivity.py`:
  - `run_sliding_cutoff(X, y, equity_returns, *, cutoff_dates, model_name, ...)` loops the Phase-6 quarterly walk-forward over any list of `DEV_END` candidates, with per-(model, cutoff) JSON caching.
  - `metrics_for_cutoff_run(qwf, equity_returns)` extracts the headline metrics from a `QuarterlyWalkForward` result.
  - `summary_statistics(df, metric)` computes median / IQR / fraction-positive / fraction>+0.10 / Pearson correlation vs cutoff date.
- `regime_detection/tests/test_cutoff_sensitivity.py` — 7 tests, all green: metric extractor on well-formed and degenerate (single-class) input, sliding-cutoff filter on min OOS quarters, cache round-trip on disk (identical DataFrame on re-read), empty schedule, summary stats on a known distribution.
- `regime_detection/step10_cutoff_robustness.py` — 49 monthly cutoffs 2018-06-30 → 2022-06-30, two models (`logistic_l1` headline, `random_forest` sanity), aggressive caching at `.cache/phase9c/`.

## What tests pass
- **136/136 total** (was 129 before Phase 9C — 7 new).

## Cutoff schedule and coverage

- 49 cutoffs emitted for logistic_l1 (after min_oos_quarters=4 filter); 49 for random_forest.
- Cutoff range: 2018-06-30 to 2022-06-30.

## Distribution of OOS Sortino lift  (logistic_l1)

|                        |   n |   median |   mean |    q25 |    q75 |    min |    max |   frac_positive |   frac_above_0p10 |   pearson_r_vs_cutoff |
|:-----------------------|----:|---------:|-------:|-------:|-------:|-------:|-------:|----------------:|------------------:|----------------------:|
| logistic_l1 (headline) |  49 |   -0.147 | -0.14  | -0.222 | -0.051 | -0.377 |  0.055 |           0.061 |                 0 |                 0.565 |
| random_forest (sanity) |  49 |   -0.354 | -0.296 | -0.419 | -0.106 | -0.609 | -0.045 |           0     |                 0 |                -0.908 |

![Phase 9C distribution](figures/thesis_v5/fig9_sliding_cutoff.png)

## Verdict

**Overlay lift is not a robust positive — the -0.184 headline is if anything slightly worse than typical.**

The median Sortino lift across 49 monthly cutoffs is -0.147; the 2019-12-31 headline of -0.184 sits below the median.  Only 6% of cutoffs give a positive lift, and only 0% cross the +0.10 'economically meaningful' bar.  **Recommendation for the thesis**: report the median + IQR (-0.147, IQR=[-0.222, -0.051]) and treat the single-cutoff -0.184 as a non-robust point estimate.  Under the 2020-start OOS the correction-timing overlay reduces drawdown but does not raise risk-adjusted return — the drawdown-timing target remains hard (see step26 for the better-posed hedge-selection framing).  L2 stands as a limitation.

## Recommended thesis-abstract language

> Across 49 monthly cutoffs from 2018-06-30 to 2022-06-30, the OOS Sortino lift over buy-and-hold has median -0.147 and IQR [-0.222, -0.051], with 6% of cutoffs delivering a positive lift.

## Deviations from spec
1. **Monthly frequency, not week or day.** The brief said 'month-by-month' so 49 cutoffs at month-end suffices.  Finer frequency (weekly or daily) would multiply runtime ≈ 4–22× without changing the qualitative shape of the distribution.
2. **min_oos_quarters=4** is the filter for cutoff inclusion.  A cutoff at 2022-06-30 leaves ≈15 OOS quarters, well above 4; no cutoffs in the schedule are rejected for short OOS at this threshold.
3. **Hyperparameters frozen at Phase-4 defaults** (no per-cutoff tuning).  Same as Phase 6 — keeps the comparison clean and avoids tuning-on-distribution effects.

## Open questions for the user
- Should the thesis abstract use the median + IQR language above, or report the -0.184 single-cutoff headline with the median range as a footnote?
- Run the same sliding-cutoff for the GMM benchmark (Phase 9B) as a follow-up?  Would tell us whether GMM is also cutoff-sensitive or genuinely worse across the board.
- Per the Phase 9 brief, proceeding to Phase 9D (port v4 splicing to extend turbulence to the full 14 factors) is the natural next step.  Will wait for your confirmation before starting.
