"""Unit tests for supervised.targets and supervised.pgts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.pgts import PurgedGroupTimeSeriesSplit
from regime_detection.lib.targets import (
    binary_turbulence_entry,
    event_time_drawdown,
    forward_drawdown_conditional,
)


def _series(values, start="2010-01-04", name="s"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, name=name)


# ---------------------------------------------------------------------------
# binary_turbulence_entry
# ---------------------------------------------------------------------------


class TestBinaryTurbulenceEntry:
    def test_label_is_one_when_spike_in_window(self):
        # turbulence = [1, 1, 1, 1, 5, 1, 1, 1, ...]
        # threshold = 4, horizon = 3
        # row 1: forward window = [t+1,t+3] = [1,1,5] → max=5 >= 4 → label 1
        # row 4: forward = [1,1,1] → max=1 < 4 → label 0
        turb = _series([1, 1, 1, 1, 5, 1, 1, 1, 1, 1])
        y = binary_turbulence_entry(turb, threshold=4, horizon=3)
        # Row 1: forward [2,3,4] = [1,1,5] → label 1
        assert y.iloc[1] == 1.0
        # Row 2: forward [3,4,5] = [1,5,1] → label 1
        assert y.iloc[2] == 1.0
        # Row 3: forward [4,5,6] = [5,1,1] → label 1
        assert y.iloc[3] == 1.0
        # Row 4: forward [5,6,7] = [1,1,1] → label 0
        assert y.iloc[4] == 0.0
        # Last row (no forward data) → NaN
        assert np.isnan(y.iloc[-1])

    def test_tail_rows_are_nan(self):
        turb = _series(list(range(10)))
        y = binary_turbulence_entry(turb, threshold=100, horizon=3)
        # Last 3 rows have insufficient forward data — but at least one
        # forward value exists for rows 6..7, so those are 0 (not NaN).
        # Only the very last row has no forward data at all → NaN.
        assert np.isnan(y.iloc[-1])
        # Row n-2 has forward [n-1] = [9], finite → label 0
        assert y.iloc[-2] == 0.0

    def test_no_lookahead_only_uses_strictly_future_data(self):
        # If we modify the value AT row t, label at row t should be
        # unchanged (it depends on rows t+1..t+h, not row t).
        rng = np.random.default_rng(0)
        turb = _series(rng.exponential(size=200))
        baseline = binary_turbulence_entry(turb, threshold=2.0, horizon=5)
        # Mutate row 50 only.
        mutated = turb.copy()
        mutated.iloc[50] = 1e6
        y = binary_turbulence_entry(mutated, threshold=2.0, horizon=5)
        assert y.iloc[50] == baseline.iloc[50]

    def test_invalid_inputs(self):
        with pytest.raises(TypeError):
            binary_turbulence_entry([1, 2, 3], threshold=1.0)
        with pytest.raises(ValueError):
            binary_turbulence_entry(_series([1, 2]), threshold=1.0, horizon=0)
        with pytest.raises(ValueError):
            binary_turbulence_entry(_series([1, 2]), threshold=float("nan"))


# ---------------------------------------------------------------------------
# forward_drawdown_conditional
# ---------------------------------------------------------------------------


class TestForwardDrawdown:
    def test_matches_manual_log_return_min(self):
        prices = _series([100, 110, 100, 90, 95, 105], name="px")
        y = forward_drawdown_conditional(prices, horizon=3)
        # Row 1: forward log returns to t+1..t+3 from p[1]=110
        #   log(100/110), log(90/110), log(95/110)
        #   min ≈ log(90/110) = -0.2007
        expected = np.log(90.0 / 110.0)
        np.testing.assert_allclose(y.iloc[1], expected, atol=1e-10)

    def test_returns_dataframe_with_regime(self):
        prices = _series([100, 110, 100, 90, 95], name="px")
        regime = _series([0, 0, 1, 1, 0], name="regime")
        out = forward_drawdown_conditional(prices, current_regime=regime, horizon=2)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["forward_drawdown", "current_regime"]
        # regime alignment: row 2 in regime is 1
        assert out["current_regime"].iloc[2] == 1.0

    def test_tail_rows_are_nan(self):
        prices = _series([100, 101, 102, 103, 104])
        y = forward_drawdown_conditional(prices, horizon=3)
        # Last row has no forward data
        assert np.isnan(y.iloc[-1])


# ---------------------------------------------------------------------------
# event_time_drawdown
# ---------------------------------------------------------------------------


class TestEventTimeDrawdown:
    def test_matches_minimum_log_return_inside_event_window(self):
        # Constant turbulence = 1, threshold = 3 → window length = 3 rows
        prices = _series([100, 105, 99, 98, 102, 108, 95, 96],
                         name="px")
        turb = _series([1.0] * 8, name="turb")
        y = event_time_drawdown(prices, turb, intensity_threshold=3.0)
        # At i=0, event window = (0, 3] = rows 1..3.
        # log returns from p[0]=100: log(105/100), log(99/100), log(98/100)
        #   min = log(98/100) = -0.0202
        np.testing.assert_allclose(y.iloc[0], np.log(98.0 / 100.0), atol=1e-10)

    def test_no_lookahead_only_uses_future_inside_window(self):
        rng = np.random.default_rng(11)
        n = 200
        prices = _series(100 * np.cumprod(1 + rng.normal(scale=0.01, size=n)),
                         name="px")
        turb = _series(rng.exponential(size=n), name="turb")
        y_full = event_time_drawdown(prices, turb, intensity_threshold=5.0)
        # Mutate rows BEFORE row 100 to be wildly different — the target at
        # rows >= 100 must not change.
        prices2 = prices.copy()
        turb2 = turb.copy()
        prices2.iloc[:100] *= rng.uniform(0.5, 1.5, size=100)
        turb2.iloc[:100] = rng.exponential(scale=10.0, size=100)
        y_after = event_time_drawdown(prices2, turb2, intensity_threshold=5.0)
        pd.testing.assert_series_equal(
            y_full.iloc[100:].dropna(), y_after.iloc[100:].dropna(),
        )


# ---------------------------------------------------------------------------
# PurgedGroupTimeSeriesSplit
# ---------------------------------------------------------------------------


class TestPGTS:
    def _dummy_X(self, n: int) -> pd.DataFrame:
        idx = pd.date_range("2010-01-04", periods=n, freq="B")
        return pd.DataFrame({"a": np.arange(n)}, index=idx)

    def test_yields_n_splits(self):
        # n=500, n_splits=5, fold_size=83.  Need fold k=0 to have enough
        # training rows after purge (test_start=83 - fw=21 - embargo=21 = 41
        # rows).  Lower min_train_size below 41 so every fold is emitted.
        X = self._dummy_X(500)
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=5, forward_window=21, embargo=21, min_train_size=30,
        )
        folds = list(cv.split(X))
        assert len(folds) == 5

    def test_train_and_test_are_disjoint(self):
        X = self._dummy_X(500)
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=4, forward_window=21, embargo=21, min_train_size=50,
        )
        for tr, te in cv.split(X):
            assert len(np.intersect1d(tr, te)) == 0

    def test_purge_removes_training_rows_in_horizon(self):
        n = 500
        X = self._dummy_X(n)
        fw = 21
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=4, forward_window=fw, embargo=0, min_train_size=50,
        )
        for tr, te in cv.split(X):
            test_start = te[0]
            # No training row may have index >= test_start - forward_window.
            assert tr.max() < test_start - fw + 1, (
                f"Purge violated: train max={tr.max()}, "
                f"test_start={test_start}, fw={fw}"
            )

    def test_embargo_adds_to_left_side_gap(self):
        n = 500
        X = self._dummy_X(n)
        forward_window = 10
        embargo = 17
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=4, forward_window=forward_window, embargo=embargo,
            min_train_size=50,
        )
        for tr, te in cv.split(X):
            test_start = te[0]
            # Total gap = forward_window + embargo.  No training row may
            # have index >= test_start - forward_window - embargo.
            max_allowed = test_start - forward_window - embargo - 1
            assert tr.max() <= max_allowed, (
                f"Embargo violated: train max={tr.max()}, "
                f"max allowed={max_allowed} "
                f"(test_start={test_start}, fw={forward_window}, "
                f"embargo={embargo})"
            )

    def test_every_fold_is_strictly_expanding_window(self):
        """Critical property for financial walk-forward: training rows
        are always BEFORE the test fold, never after."""
        X = self._dummy_X(500)
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=4, forward_window=21, embargo=21, min_train_size=50,
        )
        for tr, te in cv.split(X):
            assert tr.max() < te[0], (
                "Training rows must lie strictly before the test fold; "
                "found train.max() >= test[0]"
            )

    def test_skips_folds_with_short_training_set(self):
        X = self._dummy_X(120)
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=5, forward_window=21, embargo=21, min_train_size=200,
        )
        folds = list(cv.split(X))
        # With only 120 rows and min_train_size=200, every fold should be
        # skipped.
        assert folds == []

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesSplit(n_splits=1)
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesSplit(forward_window=-1)
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesSplit(embargo=-1)
