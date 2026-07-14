"""Tests for directional (signed) turbulence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_detection.lib.turbulence import TurbulenceIndex
from regime_detection.lib.turbulence_signed import SignedTurbulence


def _synthetic(n=900, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    cov = np.eye(k) + 0.3
    cols = [f"f{i}" for i in range(k - 1)] + ["vix"]
    return pd.DataFrame(rng.multivariate_normal(np.zeros(k), cov, n) * 0.01,
                        index=idx, columns=cols)


PARAMS = dict(lookback_days=252, min_periods=120)


def test_partition_reconstructs_total():
    rets = _synthetic()
    signed = SignedTurbulence(**PARAMS).fit_transform(rets).dropna()
    np.testing.assert_allclose(
        signed["turb_loss"] + signed["turb_gain"],
        signed["turbulence"], atol=1e-9)
    # And the total must equal the canonical TurbulenceIndex output.
    d = TurbulenceIndex(**PARAMS).fit_transform(rets)
    both = pd.concat([d, signed["turbulence"]], axis=1).dropna()
    np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-9)


def test_causal_under_future_shock():
    rets = _synthetic()
    base = SignedTurbulence(**PARAMS).fit_transform(rets)
    shocked = rets.copy()
    shocked.iloc[-1] += 1e6
    after = SignedTurbulence(**PARAMS).fit_transform(shocked)
    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def test_vix_flip_routes_vol_spike_to_loss_side():
    """A large UPWARD vix deviation must land on the loss side when
    flipped, and on the gain side when not."""
    rets = _synthetic()
    t = len(rets) - 1
    rets.iloc[t] = 0.0
    rets.iloc[t, rets.columns.get_loc("vix")] = 0.25  # huge vix up-move

    flipped = SignedTurbulence(**PARAMS, flip=("vix",)).fit_transform(rets)
    plain = SignedTurbulence(**PARAMS, flip=()).fit_transform(rets)
    # vix dominates d_t on that day; with the flip its contribution is
    # loss-side, without it gain-side.
    assert flipped["turb_loss"].iloc[t] > flipped["turb_gain"].iloc[t]
    assert plain["turb_gain"].iloc[t] > plain["turb_loss"].iloc[t]
