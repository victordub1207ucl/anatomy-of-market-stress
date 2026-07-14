"""Tests for the regime-conditional Granger causality port.

These are descriptive-statistics functions (no walk-forward contract), so
the tests check structural correctness and that a known lead-lag is
recovered, rather than a no-look-ahead invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_detection.lib.causality import (
    granger_causality_matrix, regime_conditional_granger,
    granger_drivers_of_turbulence,
)


def _xy_with_x_leading_y(n=500, seed=0):
    """x is white noise; y_t depends on x_{t-1} -> x Granger-causes y."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    x = rng.normal(0, 1, n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.6 * x[t - 1] + 0.2 * rng.normal()
    return pd.DataFrame({"x": x, "y": y}, index=idx)


def test_granger_matrix_recovers_known_direction():
    df = _xy_with_x_leading_y()
    M = granger_causality_matrix(df, max_lag=3)
    # x -> y should be far more significant than y -> x.
    assert M.loc["y", "x"] < 0.01
    assert M.loc["y", "x"] < M.loc["x", "y"]
    assert np.isnan(M.loc["x", "x"])


def test_regime_conditional_returns_per_regime_frames():
    df = _xy_with_x_leading_y(n=800)
    # Two interleaved regimes by calendar month parity.
    regime = pd.Series(np.where(df.index.month % 2 == 0, "a", "b"),
                       index=df.index)
    out = regime_conditional_granger(df, regime, max_lag=2)
    assert set(out) <= {"a", "b"}
    for frame in out.values():
        assert {"causing_factor", "caused_factor", "p_value_raw",
                "p_value_adj", "significant"} <= set(frame.columns)


def test_drivers_of_turbulence_structure():
    df = _xy_with_x_leading_y(n=600)
    turb = pd.Series(np.abs(df["x"].cumsum()) + 5.0, index=df.index)
    out = granger_drivers_of_turbulence(df, turb, max_lag=3)
    assert "full_sample" in set(out["scope"])
    assert {"factor", "scope", "p_value_raw", "p_value_adj",
            "significant"} <= set(out.columns)
