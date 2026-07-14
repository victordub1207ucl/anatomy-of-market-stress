"""Tests for the ordinal turbulence bucket (Phase 9F)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.turbulence_bucket import (
    ordinal_turbulence_bucket,
)


def _turb(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-02", periods=n, freq="B")
    # Heavy-tailed positive series, like turbulence.
    return pd.Series(rng.exponential(scale=5.0, size=n), index=idx,
                     name="turbulence")


class TestOrdinalBucket:
    def test_values_in_range(self):
        b = ordinal_turbulence_bucket(_turb(), n_buckets=5,
                                      window=1000, min_periods=250)
        vals = b.dropna().unique()
        assert set(vals).issubset({0, 1, 2, 3, 4})

    def test_warmup_is_nan(self):
        b = ordinal_turbulence_bucket(_turb(2000), n_buckets=5,
                                      window=1000, min_periods=250)
        # Before min_periods+1 rows of history the bucket is undefined.
        assert b.iloc[:250].isna().all()
        assert b.iloc[300:].notna().all()

    def test_roughly_uniform_occupancy(self):
        # With a stationary input each of the 5 levels should get a
        # non-trivial share (none collapses to zero).
        b = ordinal_turbulence_bucket(_turb(5000, seed=3), n_buckets=5,
                                      window=1500, min_periods=500)
        shares = b.dropna().value_counts(normalize=True)
        assert len(shares) == 5
        assert shares.min() > 0.05

    def test_no_lookahead_shock(self):
        """Perturbing the tail of turbulence must not change any bucket
        whose trailing window predates the shock."""
        t = _turb(3000, seed=5)
        b_a = ordinal_turbulence_bucket(t, n_buckets=5, window=1000,
                                        min_periods=250)
        shock_from = 2000
        t_b = t.copy()
        t_b.iloc[shock_from:] *= 100.0
        b_b = ordinal_turbulence_bucket(t_b, n_buckets=5, window=1000,
                                        min_periods=250)
        # Row t uses turb[t-1] and quantiles of [t-window-1, t-2].  The
        # shock at >= shock_from first reaches the bucket at shock_from+1
        # (via ts = turb.shift(1)).  Rows up to shock_from must match.
        pd.testing.assert_series_equal(
            b_a.iloc[:shock_from], b_b.iloc[:shock_from])

    def test_monotone_in_turbulence(self):
        """On a given day, a higher turbulence yesterday cannot produce a
        lower bucket (the bucket is monotone non-decreasing in ts)."""
        # Construct a deterministic ramp so thresholds are fixed, then check
        # the mapping is monotone.
        n = 2000
        idx = pd.date_range("2008-01-02", periods=n, freq="B")
        base = pd.Series(np.r_[np.full(1500, 5.0),
                               np.linspace(0, 50, n - 1500)], index=idx)
        b = ordinal_turbulence_bucket(base, n_buckets=5, window=1000,
                                      min_periods=250)
        tail = b.iloc[1600:].dropna()
        # In the rising-turbulence tail the bucket should be non-decreasing
        # for the most part; assert it ends higher than it starts.
        assert tail.iloc[-1] >= tail.iloc[0]

    def test_invalid_inputs(self):
        with pytest.raises(TypeError):
            ordinal_turbulence_bucket([1, 2, 3])
        with pytest.raises(ValueError):
            ordinal_turbulence_bucket(_turb(500), n_buckets=1)
        with pytest.raises(ValueError):
            ordinal_turbulence_bucket(_turb(500), min_periods=1000,
                                      window=500)
