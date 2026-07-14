"""Tests for the absorption ratio (systemic-fragility signal)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_detection.lib.absorption import AbsorptionRatio, absorption_shift


def _returns(n=700, k=6, rho=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    cov = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
    data = rng.multivariate_normal(np.zeros(k), cov, n) * 0.01
    return pd.DataFrame(data, index=idx, columns=[f"f{i}" for i in range(k)])


PARAMS = dict(lookback_days=252, min_periods=120, n_components=1)


def test_ar_in_unit_interval():
    ar = AbsorptionRatio(**PARAMS).fit_transform(_returns()).dropna()
    assert len(ar) > 0
    assert (ar > 0).all() and (ar <= 1.0 + 1e-9).all()


def test_higher_coupling_gives_higher_ar():
    low = AbsorptionRatio(**PARAMS).fit_transform(_returns(rho=0.1)).dropna().mean()
    high = AbsorptionRatio(**PARAMS).fit_transform(_returns(rho=0.7)).dropna().mean()
    assert high > low   # more correlated factors -> top PC absorbs more variance


def test_no_lookahead_under_mid_shock():
    rets = _returns()
    base = AbsorptionRatio(**PARAMS).fit_transform(rets)
    m = len(rets) // 2
    shocked = rets.copy()
    shocked.iloc[m] = shocked.iloc[m] + 1e6
    after = AbsorptionRatio(**PARAMS).fit_transform(shocked)
    # AR at rows <= m uses covariance windows ending at <= m (excluding m),
    # so every value through row m must be byte-identical.
    pd.testing.assert_series_equal(base.iloc[: m + 1], after.iloc[: m + 1])


def test_absorption_shift_is_causal():
    ar = AbsorptionRatio(**PARAMS).fit_transform(_returns())
    base = absorption_shift(ar)
    ar2 = ar.copy(); ar2.iloc[-1] = ar2.iloc[-1] + 1e6
    after = absorption_shift(ar2)
    pd.testing.assert_series_equal(base.iloc[:-1], after.iloc[:-1])
