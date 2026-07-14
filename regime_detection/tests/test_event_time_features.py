"""Tests for event-time features (step28)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.event_time_features import (
    build_event_time_features, univariate_turbulence,
)


def _idx(n):
    return pd.date_range("2010-01-04", periods=n, freq="B")


class TestEventTimeFeatures:
    def test_columns_and_index(self):
        idx = _idx(50)
        turb = pd.Series(1.0, index=idx)
        px = pd.Series(np.linspace(100, 110, 50), index=idx)
        f = build_event_time_features(turb, px, threshold=5.0)
        assert list(f.columns) == ["et_intensity_to_date", "et_days_since_start",
                                   "et_velocity", "et_turb_per_day", "et_momentum"]
        assert f.index.equals(idx)

    def test_intensity_resets_at_threshold(self):
        idx = _idx(10)
        turb = pd.Series(1.0, index=idx)  # 1/day, threshold 3 → resets every 3 days
        px = pd.Series(100.0, index=idx)
        f = build_event_time_features(turb, px, threshold=3.0)
        # day0:1/3, day1:2/3, day2:3/3(reset), day3:1/3 ...
        np.testing.assert_allclose(f["et_intensity_to_date"].iloc[:4].values,
                                   [1/3, 2/3, 1.0, 1/3])
        # days_since_start resets after a completed event
        assert f["et_days_since_start"].iloc[3] == 0

    def test_velocity_counts_completed_events(self):
        idx = _idx(12)
        turb = pd.Series(1.0, index=idx)
        px = pd.Series(100.0, index=idx)
        f = build_event_time_features(turb, px, threshold=3.0, tempo_window=6)
        # by day 5, two events completed (days 2 and 5) within trailing 6 days
        assert f["et_velocity"].iloc[5] == pytest.approx(2.0)

    def test_causal_future_shock_does_not_change_past(self):
        idx = _idx(40)
        rng = np.random.default_rng(0)
        turb = pd.Series(np.abs(rng.normal(5, 1, 40)), index=idx)
        px = pd.Series(100 + np.cumsum(rng.normal(0, 1, 40)), index=idx)
        f1 = build_event_time_features(turb, px, threshold=10.0)
        turb2 = turb.copy(); turb2.iloc[-1] = 1e6
        px2 = px.copy(); px2.iloc[-1] = 1e6
        f2 = build_event_time_features(turb2, px2, threshold=10.0)
        # every row before the last must be identical
        pd.testing.assert_frame_equal(f1.iloc[:-1], f2.iloc[:-1])

    def test_rejects_bad_threshold(self):
        idx = _idx(5)
        with pytest.raises(ValueError):
            build_event_time_features(pd.Series(1.0, index=idx),
                                      pd.Series(100.0, index=idx), threshold=0.0)


class TestUnivariateTurbulence:
    def test_nonnegative_and_causal(self):
        idx = _idx(400)
        rng = np.random.default_rng(3)
        px = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)), index=idx)
        u = univariate_turbulence(px, window=60, min_periods=20)
        assert (u.dropna() >= 0).all()                      # squared z-score
        # future shock must not change any past value (causal)
        px2 = px.copy(); px2.iloc[-1] = 1e6
        u2 = univariate_turbulence(px2, window=60, min_periods=20)
        pd.testing.assert_series_equal(u.iloc[:-1], u2.iloc[:-1])

    def test_rejects_bad_window(self):
        with pytest.raises(ValueError):
            univariate_turbulence(pd.Series([1.0, 2.0], index=_idx(2)), window=1)
