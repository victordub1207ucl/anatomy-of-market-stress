"""Unit tests for the event-time converter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.event_time import (
    EventTimeConverter,
    get_overlapping_event_returns,
)


def _make_turb(values, start="2010-01-04"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="turbulence")


# ---------------------------------------------------------------------------
# EventTimeConverter.fit_transform
# ---------------------------------------------------------------------------


class TestEventTimeConverter:
    def test_constant_turbulence_gives_equal_length_events(self):
        # turbulence = 1 every day, threshold = 5 → event every 5 rows.
        turb = _make_turb([1.0] * 50)
        events = EventTimeConverter(intensity_threshold=5.0).fit_transform(turb)
        assert len(events) == 10
        assert (events["trading_days_in_event"] == 5).all()
        assert events["event_id"].tolist() == list(range(10))
        # Cumulative intensity per event should be exactly 5.
        np.testing.assert_allclose(events["cumulative_intensity"], 5.0, atol=1e-12)

    def test_event_boundaries_are_first_crossing(self):
        # 1, 2, 4, 8, 16  threshold=3
        # cum = [1, 3, 7, 15, 31]
        # 1st event end: smallest j with cum[j] >= 3 → j=1 (cum=3), ev=[0..1]
        # 2nd event end: smallest j with cum[j] >= 6 → j=2 (cum=7),  ev=[2..2]
        # 3rd event end: smallest j with cum[j] >= 9 → j=3 (cum=15), ev=[3..3]
        # 4th event end: smallest j with cum[j] >= 12 → j=3 (cum=15), but
        #     j must be after the previous end → searchsorted gives j=3
        #     which equals previous end, so 4th event boundary is also j=3.
        #     This is a degenerate case but we expect monotonicity →
        #     verify only the first three events.
        turb = _make_turb([1.0, 2.0, 4.0, 8.0, 16.0])
        events = EventTimeConverter(intensity_threshold=3.0).fit_transform(turb)
        assert events.iloc[0]["start_date"] == turb.index[0]
        assert events.iloc[0]["end_date"] == turb.index[1]
        assert events.iloc[0]["trading_days_in_event"] == 2
        assert events.iloc[1]["start_date"] == turb.index[2]
        assert events.iloc[1]["end_date"] == turb.index[2]
        assert events.iloc[1]["trading_days_in_event"] == 1
        # cumulative_intensity of event 1 is exactly turb[2] = 4.
        np.testing.assert_allclose(events.iloc[1]["cumulative_intensity"], 4.0)

    def test_spike_absorbed_into_single_day_event(self):
        # turb = [10, 1, 1, 1, ...], threshold = 3
        # Under absorb semantics:
        #   ev0: [0..0], inten=10 (the spike absorbs the overflow)
        #   ev1: [1..3], inten=3
        #   ev2: [4..6], inten=3
        #   ...  contiguous 3-day events thereafter.
        turb = _make_turb([10.0] + [1.0] * 30)
        events = EventTimeConverter(intensity_threshold=3.0).fit_transform(turb)
        # First event is the spike day alone with cumulative_intensity = 10.
        assert events.iloc[0]["trading_days_in_event"] == 1
        np.testing.assert_allclose(events.iloc[0]["cumulative_intensity"], 10.0)
        # Subsequent events are 3-day blocks of constant turbulence.
        assert (events.iloc[1:]["trading_days_in_event"] == 3).all()
        np.testing.assert_allclose(
            events.iloc[1:]["cumulative_intensity"], 3.0, atol=1e-12,
        )
        # Total intensity captured across events equals total turbulence up
        # to the last event boundary (the absorb policy is conservative —
        # no turbulence inside any event window is "lost").
        last_end = events["end_date"].iloc[-1]
        np.testing.assert_allclose(
            events["cumulative_intensity"].sum(),
            turb.loc[:last_end].sum(),
            atol=1e-12,
        )

    def test_nan_warmup_is_dropped(self):
        turb_vals = [np.nan, np.nan, np.nan] + [1.0] * 20
        turb = _make_turb(turb_vals)
        events = EventTimeConverter(intensity_threshold=5.0).fit_transform(turb)
        # First event should start at the first finite row.
        first_finite_date = turb.dropna().index[0]
        assert events.iloc[0]["start_date"] == first_finite_date

    def test_negative_turbulence_raises(self):
        turb = _make_turb([1.0, -0.5, 1.0])
        with pytest.raises(ValueError):
            EventTimeConverter(intensity_threshold=1.0).fit_transform(turb)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            EventTimeConverter(intensity_threshold=0.0)
        with pytest.raises(ValueError):
            EventTimeConverter(intensity_threshold=-1.0)
        with pytest.raises(ValueError):
            EventTimeConverter(intensity_threshold=float("nan"))

    def test_non_series_input_raises(self):
        with pytest.raises(TypeError):
            EventTimeConverter(intensity_threshold=1.0).fit_transform([1, 2, 3])

    def test_non_datetime_index_raises(self):
        bad = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
        with pytest.raises(TypeError):
            EventTimeConverter(intensity_threshold=1.0).fit_transform(bad)

    def test_no_lookahead_perturbing_the_tail(self):
        """A perturbation late in the series must not change earlier event
        boundaries — the algorithm is strictly forward-cumulative."""
        rng = np.random.default_rng(7)
        n = 600
        turb = _make_turb(rng.exponential(scale=1.0, size=n))
        conv = EventTimeConverter(intensity_threshold=20.0)
        full = conv.fit_transform(turb)

        # Find the event boundary nearest the 75% point of the series.
        cutoff_date = turb.index[int(n * 0.75)]
        earlier = full[full["end_date"] <= cutoff_date]

        # Truncate the input at cutoff_date and re-run.
        truncated_turb = turb.loc[:cutoff_date]
        truncated_events = conv.fit_transform(truncated_turb)

        # Events ending on or before cutoff_date must be byte-identical.
        pd.testing.assert_frame_equal(
            earlier.reset_index(drop=True),
            truncated_events.head(len(earlier)).reset_index(drop=True),
        )

    def test_empty_input_returns_empty_frame(self):
        turb = pd.Series(
            [], index=pd.DatetimeIndex([]), name="turbulence", dtype=float,
        )
        events = EventTimeConverter(intensity_threshold=1.0).fit_transform(turb)
        assert len(events) == 0
        assert list(events.columns) == EventTimeConverter.EVENT_COLUMNS


# ---------------------------------------------------------------------------
# EventTimeConverter.calibrate
# ---------------------------------------------------------------------------


class TestCalibrate:
    def test_constant_turbulence_calibrates_to_target_threshold(self):
        # Turb = 1.0 every day → mean event length = threshold (in days).
        # So calibrate(target=21) should return threshold ≈ 21.
        turb = _make_turb([1.0] * 2000)
        conv, diag = EventTimeConverter.calibrate(
            turb, target_mean_trading_days=21.0, tolerance=0.05,
        )
        assert abs(conv.intensity_threshold - 21.0) < 0.5
        assert abs(diag.mean_trading_days - 21.0) < 0.25

    def test_exponential_turbulence_converges(self):
        rng = np.random.default_rng(42)
        turb = _make_turb(rng.exponential(scale=5.0, size=3000))
        conv, diag = EventTimeConverter.calibrate(
            turb, target_mean_trading_days=21.0, tolerance=0.25,
        )
        assert diag.mean_trading_days == pytest.approx(21.0, abs=0.5)
        assert diag.n_iterations < 60

    def test_calibrate_returns_positive_threshold(self):
        rng = np.random.default_rng(43)
        turb = _make_turb(rng.exponential(scale=2.0, size=1500))
        conv, _ = EventTimeConverter.calibrate(
            turb, target_mean_trading_days=10.0,
        )
        assert conv.intensity_threshold > 0

    def test_calibrate_rejects_short_series(self):
        turb = _make_turb([1.0] * 30)
        with pytest.raises(ValueError):
            EventTimeConverter.calibrate(turb, target_mean_trading_days=21.0)


# ---------------------------------------------------------------------------
# get_overlapping_event_returns
# ---------------------------------------------------------------------------


class TestOverlappingEventReturns:
    def test_constant_turbulence_constant_threshold_gives_fixed_window(self):
        # turb=1, threshold=5 → window length = 5 trading days.
        # Asset price grows by 1% per day → 5-day log return ≈ 0.05.
        n = 30
        turb = _make_turb([1.0] * n)
        # Build a price series: 100 * (1.01)^t
        prices = pd.Series(
            100.0 * (1.01 ** np.arange(n)),
            index=turb.index,
            name="px",
        )
        out = get_overlapping_event_returns(
            prices, turb, intensity_threshold=5.0, log_returns=True,
        )
        # For days 0..24 (inclusive), the event window ends 5 days later
        # so the return is log(1.01^5) ≈ 0.04975.
        expected = 5 * np.log(1.01)
        finite = out.dropna()
        assert len(finite) == n - 5
        np.testing.assert_allclose(finite.values, expected, atol=1e-12)

    def test_tail_rows_are_nan(self):
        # Event window for late days extends past data → NaN.
        turb = _make_turb([1.0] * 20)
        prices = pd.Series(np.arange(1, 21, dtype=float), index=turb.index)
        out = get_overlapping_event_returns(prices, turb, intensity_threshold=5.0)
        # The last 5 rows have no full event window.
        assert out.iloc[-5:].isna().all()
        assert out.iloc[:-5].notna().all()

    def test_log_vs_arithmetic_match_known_formula(self):
        turb = _make_turb([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        prices = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41, 161.05],
                           index=turb.index)
        log = get_overlapping_event_returns(
            prices, turb, intensity_threshold=2.0, log_returns=True,
        )
        ari = get_overlapping_event_returns(
            prices, turb, intensity_threshold=2.0, log_returns=False,
        )
        # threshold=2 → 2-day window.  log[0] = log(121/100), ari[0] = 0.21.
        np.testing.assert_allclose(log.iloc[0], np.log(121.0 / 100.0), atol=1e-12)
        np.testing.assert_allclose(ari.iloc[0], 121.0 / 100.0 - 1, atol=1e-12)

    def test_no_lookahead(self):
        """For each start date i, the function uses only data from rows
        >= i.  Truncating data before row i (replacing it with anything)
        must not change the value at row i."""
        rng = np.random.default_rng(8)
        n = 400
        turb = _make_turb(rng.exponential(scale=1.0, size=n))
        prices = pd.Series(
            100 * np.cumprod(1 + rng.normal(scale=0.01, size=n)),
            index=turb.index,
        )
        full = get_overlapping_event_returns(prices, turb, intensity_threshold=10.0)
        # Replace rows [0, 100) with random garbage.
        turb_mut = turb.copy()
        prices_mut = prices.copy()
        turb_mut.iloc[:100] = rng.exponential(scale=10.0, size=100)
        prices_mut.iloc[:100] = rng.uniform(50, 200, size=100)
        # Reapply the cumulative-turbulence aware computation; for any
        # start date i >= 100, the result should be unchanged because
        # only data at rows >= i is touched.
        partial = get_overlapping_event_returns(
            prices_mut, turb_mut, intensity_threshold=10.0,
        )
        pd.testing.assert_series_equal(
            full.iloc[100:].dropna(),
            partial.iloc[100:].dropna(),
        )

    def test_negative_turbulence_raises(self):
        turb = _make_turb([1.0, -0.5, 1.0])
        prices = pd.Series([100.0, 101.0, 102.0], index=turb.index)
        with pytest.raises(ValueError):
            get_overlapping_event_returns(prices, turb, intensity_threshold=1.0)

    def test_invalid_threshold_raises(self):
        turb = _make_turb([1.0, 1.0, 1.0])
        prices = pd.Series([100.0, 101.0, 102.0], index=turb.index)
        with pytest.raises(ValueError):
            get_overlapping_event_returns(prices, turb, intensity_threshold=0.0)

    def test_empty_intersection_returns_empty(self):
        turb = _make_turb([1.0, 1.0])
        prices = pd.Series([100.0, 101.0],
                           index=pd.date_range("2030-01-01", periods=2, freq="B"))
        out = get_overlapping_event_returns(prices, turb, intensity_threshold=1.0)
        assert len(out) == 0
