"""Tests for the decomposed-turbulence supervised features.

The composition features must be strictly causal: a shock injected at the
end of the inputs must not change any earlier feature row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_detection.lib.decomposition_features import (
    build_decomposition_features,
)


def _synthetic(n: int = 600, k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    cols = [f"f{i}" for i in range(k)]
    contrib = pd.DataFrame(rng.normal(0, 1, size=(n, k)), index=idx, columns=cols)
    turb = contrib.sum(axis=1).abs() + 1.0  # strictly positive denominator
    return contrib, turb


def test_columns_and_alignment():
    contrib, turb = _synthetic()
    X = build_decomposition_features(contrib, turb, windows=(21, 63))
    expected = {f"decshare_{c}_{w}" for c in contrib.columns for w in (21, 63)}
    expected |= {"decconc_21", "decconc_63"}
    assert set(X.columns) == expected
    assert X.index.equals(contrib.index)


def test_no_lookahead_under_future_shock():
    contrib, turb = _synthetic()
    base = build_decomposition_features(contrib, turb)

    c2, t2 = contrib.copy(), turb.copy()
    c2.iloc[-1] = c2.iloc[-1] + 1e6
    t2.iloc[-1] = t2.iloc[-1] + 1e6
    after = build_decomposition_features(c2, t2)

    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def test_no_inf_when_denominator_positive():
    contrib, turb = _synthetic()
    X = build_decomposition_features(contrib, turb)
    finite = X.dropna()
    assert np.isfinite(finite.values).all()
