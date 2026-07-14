# CHECKPOINT — Phase 1: Walk-Forward Infrastructure

## What was implemented
- New package skeleton: `regime_detection/{__init__.py, utils/, tests/}`.
- [`regime_detection/lib/splits.py`](../../regime_detection/lib/splits.py) — `DEV_END = 2019-12-31`, `OOS_START = 2020-01-01` (fixed-split constraint, 2026-06: OOS starts 1 Jan 2020 so COVID is held out), `get_dev_test_split()`, `assert_no_oos_contamination()`.
- [`regime_detection/utils/walk_forward.py`](../../regime_detection/utils/walk_forward.py) — three classes with a unified `.fit_transform_walk_forward(data, window, step)` interface:
  - `RollingStandardizer` (rolling or expanding, step ≥ 1, ddof configurable)
  - `RollingPCA` (per-block refit, optional sign-flip noted but not corrected)
  - `RollingCovariance` (Ledoit-Wolf / empirical / constant-α shrinkage; returns `dict[Timestamp, DataFrame]`; `.cov_at(date)` helper)
- [`regime_detection/tests/test_walk_forward.py`](../../regime_detection/tests/test_walk_forward.py) — 5 test classes.

## What tests pass
**29/29** in `test_walk_forward.py`:
- `TestSplits` — 6 tests (cutoff invariant, disjoint slices, non-datetime rejection, OOS-contamination guard)
- `TestRollingStandardizer` — 9 tests (manual stats match, warm-up NaN, no-lookahead under perturbation, block mode, expanding mode, constant column, series input, init validation)
- `TestRollingPCA` — 5 tests (manual fit per block, no-lookahead, warm-up NaN, history populated, column naming)
- `TestRollingCovariance` — 7 tests (refit timestamps, LW intensity in [0,1], no-lookahead, `cov_at`, constant shrinkage, empirical, init validation)
- `TestNoNaNLeakage` — 2 tests (body has no NaN, warm-up rows ARE NaN)

The strongest test runs each class twice on the same input with a `1e6` shock injected into the tail of the second run; every row whose fit window predates the shock must be byte-identical between the two runs.

## Deviations from spec
1. **Asked the user** where to put the new code (new package vs extend existing `utils/`). Confirmed: create new `regime_detection/` package.
2. **Removed `assert_no_oos_contamination` from inside the walk-forward classes** — initial draft called it at every refit, which is overkill (slicing is correct by construction). The helper now belongs to the pipeline-entry seam, as documented in `splits.py`.
3. **One bug caught by test**: my first version of the no-lookahead-when-future-is-perturbed test expected `iloc[:modify_from + 1]` to match between two runs. Wrong — the row at `modify_from` has its own raw value perturbed (not just its fit window), so the standardised output differs there. The test was the bug, not the standardiser; fixed the test.

## Open questions for the user
- Sign-flip correction across PCA refits: deliberately deferred to downstream consumer; noted in module docstring. Flag if you want the package to handle it.
