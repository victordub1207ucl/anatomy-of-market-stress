"""Tests for the typology-conditioned hedge-selection layer (step26)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.typology_hedge import (
    BONDS, CASH, EQUITY, GOLD,
    performance_summary, rates_pressure_share, route_hedge,
    signed_duration_trend, strategy_returns,
)


def _idx(n, start="2010-01-04"):
    return pd.date_range(start, periods=n, freq="B")


class TestRatesPressureShare:
    def test_share_in_unit_interval_and_matches_definition(self):
        idx = _idx(4)
        contrib = pd.DataFrame(
            {"rates": [1.0, 2.0, 0.0, -1.0],
             "inflation": [1.0, 0.0, 0.0, 0.0],
             "equity": [2.0, 2.0, 1.0, 5.0]},
            index=idx,
        )
        s = rates_pressure_share(contrib)
        assert ((s.dropna() >= 0) & (s.dropna() <= 1)).all()
        # row0: (1+1)/(1+1+2) = 0.5
        assert s.iloc[0] == pytest.approx(0.5)
        # row3: negative rates clipped to 0 -> 0/(0+0+5) = 0
        assert s.iloc[3] == pytest.approx(0.0)

    def test_raises_when_no_rates_columns(self):
        contrib = pd.DataFrame({"equity": [1.0], "vix": [2.0]}, index=_idx(1))
        with pytest.raises(ValueError):
            rates_pressure_share(contrib)


class TestSignedDurationTrend:
    def test_trailing_sum_and_sign(self):
        r = pd.Series([0.01, 0.01, -0.05, -0.05], index=_idx(4))
        t = signed_duration_trend(r, window=2)
        assert np.isnan(t.iloc[0])
        assert t.iloc[1] == pytest.approx(0.02)   # rising -> positive
        assert t.iloc[3] == pytest.approx(-0.10)  # selling off -> negative

    def test_rejects_bad_window(self):
        with pytest.raises(ValueError):
            signed_duration_trend(pd.Series([0.0], index=_idx(1)), window=0)


class TestRouteHedge:
    def test_three_way_routing(self):
        idx = _idx(4)
        gate = pd.Series([False, True, True, False], index=idx)
        shock = pd.Series([False, False, True, True], index=idx)
        out = route_hedge(gate, shock)
        assert list(out) == [EQUITY, BONDS, GOLD, EQUITY]

    def test_nan_shock_treated_as_no_shock(self):
        idx = _idx(2)
        gate = pd.Series([True, True], index=idx)
        shock = pd.Series([np.nan, True], index=idx)
        out = route_hedge(gate, shock)
        assert list(out) == [BONDS, GOLD]


class TestStrategyReturns:
    def test_picks_held_label_return(self):
        idx = _idx(3)
        positions = pd.Series([EQUITY, BONDS, EQUITY], index=idx)
        rets = {
            EQUITY: pd.Series([0.01, -0.02, 0.03], index=idx),
            BONDS: pd.Series([0.0, 0.04, 0.0], index=idx),
        }
        out = strategy_returns(positions, rets, cost_bps=0.0)
        # day0 equity 0.01, day1 bonds 0.04, day2 equity 0.03
        assert out.iloc[0] == pytest.approx(0.01)
        assert out.iloc[1] == pytest.approx(0.04)
        assert out.iloc[2] == pytest.approx(0.03)

    def test_cash_label_is_zero_when_absent(self):
        idx = _idx(2)
        positions = pd.Series([CASH, CASH], index=idx)
        out = strategy_returns(positions, {}, cost_bps=0.0)
        assert (out == 0.0).all()

    def test_switching_cost_charged_on_changes_only(self):
        idx = _idx(3)
        positions = pd.Series([EQUITY, CASH, CASH], index=idx)
        rets = {EQUITY: pd.Series([0.0, 0.0, 0.0], index=idx)}
        out = strategy_returns(positions, rets, cost_bps=10.0)
        # day0: no prior -> no cost; day1: switch -> -0.001; day2: no switch
        assert out.iloc[0] == pytest.approx(0.0)
        assert out.iloc[1] == pytest.approx(-0.001)
        assert out.iloc[2] == pytest.approx(0.0)

    def test_missing_return_series_raises(self):
        idx = _idx(1)
        positions = pd.Series([GOLD], index=idx)
        with pytest.raises(KeyError):
            strategy_returns(positions, {EQUITY: pd.Series([0.0], index=idx)})


class TestPerformanceSummary:
    def test_keys_and_positive_drift(self):
        idx = _idx(252)
        r = pd.Series(0.0005, index=idx)  # steady positive drift
        s = performance_summary(r)
        assert set(s) == {"cagr", "vol", "sharpe", "sortino",
                          "max_drawdown", "total_return"}
        assert s["total_return"] > 0
        assert s["max_drawdown"] <= 0

    def test_empty_returns_all_nan(self):
        s = performance_summary(pd.Series([], dtype=float))
        assert all(np.isnan(v) for v in s.values())
