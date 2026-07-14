# CHECKPOINT — Phase 3: Event-Time Conversion

## What was implemented
- [`regime_detection/models/event_time.py`](../../regime_detection/models/event_time.py):
  - `EventTimeConverter(intensity_threshold)` — `.fit_transform(daily_turbulence)` returns `DataFrame(event_id, start_date, end_date, calendar_days_in_event, trading_days_in_event, cumulative_intensity)`
  - `EventTimeConverter.calibrate(turbulence, target_mean_trading_days=21.0)` — bisection on threshold; returns `(converter, diagnostics)`
  - `get_overlapping_event_returns(prices, turbulence, threshold, log_returns=True)` — companion utility for the supervised target
- [`regime_detection/scripts/build_event_time_notebook.py`](../../regime_detection/scripts/build_event_time_notebook.py) — builds + executes [`notebooks/step03_event_time_diagnostics.ipynb`](../../notebooks/step03_event_time_diagnostics.ipynb).
- Persisted: [`outputs/event_time/`](../outputs/event_time/) — `events.parquet`, `returns_by_time_base.parquet`, `moments_summary.csv`, `crisis_drawdown_z.csv`, 3 figures.

## Calibrated threshold
`intensity_threshold = 166.60` (10 bisection iterations) → mean event length 20.955 trading days (target 21.0, tolerance 0.10). 220 disjoint events; median 20 trading days; range 1–64.

## What tests pass
**21/21** in `test_event_time.py` (69/69 total).
- `TestEventTimeConverter` — 10 tests (boundary correctness, spike absorbed into single-day event, NaN warmup, no-lookahead via truncation, empty input, input validation)
- `TestCalibrate` — 4 tests (constant turb → target threshold, exponential converges, short-series rejection)
- `TestOverlappingEventReturns` — 7 tests (constant-turb fixed window, tail NaN, log vs arithmetic, no-lookahead under tail mutation, validation)

## Headline empirical result
| metric | calendar 21d | event time | reduction |
|---|---:|---:|---:|
| skewness | -1.74 | -0.55 | 68 % |
| excess kurtosis | 8.81 | 0.57 | **94 %** |
| Jarque-Bera | 17 204 | 294 | **98 %** |
| KL vs N(μ,σ) | 0.164 | 0.030 | 82 % |

Crisis drawdown |z|: 2008 GFC 7.54 → 4.24 (0.56×); 2020 COVID 8.20 → 3.43 (0.42×); 2022 infl 2.94 → 2.58 (0.88×). All three crises look less extreme under event time.

## Deviations from spec
1. **Absorb-overflow semantics, not carry.** The spec said "reset cumulative to 0 (or to the overflow amount)" — both are valid. I chose absorb (each event contiguous, intensity ≥ threshold, no spurious zero-turbulence "events" on quiet days after spikes). Documented in module docstring; consistent with `get_overlapping_event_returns`. Trade-off: each event's cumulative_intensity can exceed threshold for single-day spike days (e.g. 2018-02-06 with d_t ≈ 11,746), but the event window remains a meaningful calendar bracket.
2. **Calibration target is 21 *trading* days** (not 21 *calendar* days as the spec read literally). Rationale: matches the 21-day forward-drawdown horizon the supervised layer uses; calendar 21 days ≈ 15 trading days, which would mismatch.
3. **`get_overlapping_event_returns` signature** uses `(prices, daily_turbulence, intensity_threshold)` not `(prices, threshold, asset_col)`. The signature lets the caller pass any column; an `asset_col` argument is redundant when the user just passes `prices['equity']`.

## Open questions for the user
- Carry-overflow semantics: would you prefer the spec's other reading ("each event has exactly threshold of weight, can fire on quiet days") for closer fidelity to the paper's "equal informational weight" language? Easy to add as a flag.
- Use `event_time_drawdown` as the Phase-4 supervised target in v2 (currently `binary_turbulence_entry` is the headline). Worth a head-to-head?
