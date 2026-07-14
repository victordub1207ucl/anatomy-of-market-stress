"""Tests for event-clock interpretation (step27)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.event_clock_interpretation import (
    event_windows, predictability_from_runup, resample_to_event_grid,
    runup_composition_calendar, runup_composition_eventtime,
)


def _idx(n, start="2010-01-04"):
    return pd.date_range(start, periods=n, freq="B")


def _contrib(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    return pd.DataFrame(np.abs(rng.normal(size=(n, 3))),
                        index=idx, columns=["a", "b", "c"])


class TestCalendarRunup:
    def test_window_is_pre_episode_and_normalised(self):
        contrib = _contrib(60)
        start = contrib.index[40]
        out = runup_composition_calendar(contrib, [start], lead_window=10)
        assert out.shape == (1, 3)
        # rows sum to 1 (composition)
        assert out.iloc[0].sum() == pytest.approx(1.0)
        # equals the normalised mean of rows 30..39 (strictly before row 40)
        expected = contrib.iloc[30:40].mean(axis=0)
        expected = expected / expected.sum()
        np.testing.assert_allclose(out.iloc[0].values, expected.values)


class TestEventtimeRunup:
    def test_accumulates_threshold_of_turbulence(self):
        contrib = _contrib(60)
        # turbulence = 1.0/day → 1 event-unit of 5.0 spans exactly 5 days
        turb = pd.Series(1.0, index=contrib.index)
        start = contrib.index[40]
        out = runup_composition_eventtime(contrib, turb, [start],
                                          threshold=5.0, event_units=1.0)
        assert out.attrs["mean_runup_days"] == pytest.approx(5.0)
        assert out.iloc[0].sum() == pytest.approx(1.0)

    def test_high_turbulence_shortens_window(self):
        contrib = _contrib(60)
        turb = pd.Series(1.0, index=contrib.index)
        turb.iloc[35:40] = 10.0  # spike just before the episode
        start = contrib.index[40]
        out = runup_composition_eventtime(contrib, turb, [start],
                                          threshold=20.0, event_units=1.0)
        # 20 units accumulate in ~2 recent high-turb days, not 20 calendar days
        assert out.attrs["mean_runup_days"] <= 3

    def test_rejects_bad_threshold(self):
        contrib = _contrib(10)
        with pytest.raises(ValueError):
            runup_composition_eventtime(contrib, pd.Series(1.0, index=contrib.index),
                                        [contrib.index[5]], threshold=0.0)


class TestPredictability:
    def test_perfect_match_high_cosine_low_p(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2010-01-01") + pd.Timedelta(days=i)
                                for i in range(12)], name="start")
        rng = np.random.default_rng(1)
        comp = pd.DataFrame(np.abs(rng.normal(size=(12, 3))), index=idx,
                            columns=["a", "b", "c"])
        comp = comp.div(comp.sum(axis=1), axis=0)
        # run-up == composition → cosine 1.0, p small
        res = predictability_from_runup(comp.copy(), comp, n_perm=500, random_state=0)
        assert res["n_episodes"] == 12
        assert res["mean_cosine_observed"] == pytest.approx(1.0, abs=1e-9)
        assert res["p_value"] < 0.05

    def test_empty_overlap_returns_nan(self):
        a = pd.DataFrame([[1.0, 0.0]], index=pd.DatetimeIndex(["2010-01-01"]),
                         columns=["a", "b"])
        b = pd.DataFrame([[0.0, 1.0]], index=pd.DatetimeIndex(["2020-01-01"]),
                         columns=["a", "b"])
        res = predictability_from_runup(a, b, n_perm=10)
        assert res["n_episodes"] == 0
        assert np.isnan(res["p_value"])


class TestEventGrid:
    def test_event_windows_cut_at_threshold(self):
        turb = pd.Series(1.0, index=_idx(10))
        w = event_windows(turb, threshold=3.0)
        # 10 days, 3.0 per window → windows of length 3: (0,2),(3,5),(6,8)
        assert list(w["start_pos"]) == [0, 3, 6]
        assert list(w["end_pos"]) == [2, 5, 8]

    def test_resample_sums_returns_means_turbulence(self):
        idx = _idx(6)
        rets = pd.DataFrame({"x": [0.01] * 6, "y": [-0.02] * 6}, index=idx)
        turb = pd.Series(1.0, index=idx)
        out = resample_to_event_grid(rets, turb, threshold=3.0)
        # two events of 3 days each
        assert len(out["returns"]) == 2
        assert out["returns"]["x"].iloc[0] == pytest.approx(0.03)
        assert out["turbulence"].iloc[0] == pytest.approx(1.0)
