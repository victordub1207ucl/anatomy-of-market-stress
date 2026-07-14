"""Tests for the per-factor turbulence decomposition.

The decomposition must (a) reconstruct ``d_t`` exactly and (b) inherit the
same causality contract as :class:`TurbulenceIndex` — the row at ``t``
depends only on data strictly before ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_detection.lib.turbulence import TurbulenceIndex
from regime_detection.lib.turbulence_decomposition import TurbulenceDecomposer


def _synthetic_returns(n: int = 900, k: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    cov = np.eye(k) + 0.3
    data = rng.multivariate_normal(np.zeros(k), cov, size=n) * 0.01
    cols = [f"f{i}" for i in range(k)]
    return pd.DataFrame(data, index=idx, columns=cols)


PARAMS = dict(lookback_days=252, min_periods=120, shrinkage="ledoit-wolf",
              refit_every=1)


def test_contributions_reconstruct_turbulence():
    rets = _synthetic_returns()
    d_t = TurbulenceIndex(**PARAMS).fit_transform(rets)
    contrib = TurbulenceDecomposer(**PARAMS).fit_transform(rets)

    recon = contrib.sum(axis=1)
    both = pd.concat([d_t, recon], axis=1).dropna()
    assert len(both) > 0
    np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-9)


def test_warmup_rows_are_nan():
    rets = _synthetic_returns()
    contrib = TurbulenceDecomposer(**PARAMS).fit_transform(rets)
    # The first min_periods rows can never have a valid fit window.
    assert contrib.iloc[: PARAMS["min_periods"]].isna().all().all()


def test_decomposition_is_causal_under_future_shock():
    """Injecting a shock into the tail must not change any earlier row."""
    rets = _synthetic_returns()
    base = TurbulenceDecomposer(**PARAMS).fit_transform(rets)

    shocked = rets.copy()
    shocked.iloc[-1] = shocked.iloc[-1] + 1e6
    after = TurbulenceDecomposer(**PARAMS).fit_transform(shocked)

    # Every row except the shocked last one must be byte-identical.
    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def test_refit_every_matches_index_path():
    """Decomposition reconstructs d_t for monthly refits too."""
    params = {**PARAMS, "refit_every": 21}
    rets = _synthetic_returns()
    d_t = TurbulenceIndex(**params).fit_transform(rets)
    contrib = TurbulenceDecomposer(**params).fit_transform(rets)
    both = pd.concat([d_t, contrib.sum(axis=1)], axis=1).dropna()
    np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-9)
