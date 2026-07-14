"""Tests for the crisis-typology episode extraction and composition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_detection.lib.crisis_typology import (
    cluster_episodes, composition_predictability, episode_composition,
    extract_episodes,
)


def _turb_with_planted_spikes(n=2000, seed=0):
    """Low-level noise with two long planted high-turbulence blocks."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    base = pd.Series(rng.gamma(2.0, 5.0, n), index=idx)
    base.iloc[1200:1220] *= 30.0
    base.iloc[1600:1615] *= 30.0
    return base


def test_extract_finds_planted_episodes():
    turb = _turb_with_planted_spikes()
    eps = extract_episodes(turb, window=504, min_periods=200,
                           min_len=5, max_gap=5)
    assert len(eps) >= 2
    starts = pd.DatetimeIndex(eps["start"])
    # both planted blocks are detected (within a few days of onset;
    # threshold flags use d_{t-1} so allow +/- 5 days)
    for loc in (1200, 1600):
        onset = turb.index[loc]
        assert (abs((starts - onset).days) <= 7).any()


def test_composition_rows_sum_to_one():
    turb = _turb_with_planted_spikes()
    rng = np.random.default_rng(1)
    contrib = pd.DataFrame(
        rng.dirichlet(np.ones(4), len(turb)) * turb.values[:, None],
        index=turb.index, columns=list("abcd"))
    eps = extract_episodes(turb, window=504, min_periods=200)
    comp = episode_composition(contrib, eps)
    assert len(comp) == len(eps)
    np.testing.assert_allclose(comp.sum(axis=1), 1.0, atol=1e-9)


def test_cluster_and_predictability_run():
    rng = np.random.default_rng(2)
    # two synthetic composition archetypes
    a = rng.dirichlet([10, 1, 1, 1], 12)
    b = rng.dirichlet([1, 1, 10, 1], 12)
    comp = pd.DataFrame(np.vstack([a, b]), columns=list("wxyz"),
                        index=pd.bdate_range("2015-01-01", periods=24))
    labels, k, Z, names = cluster_episodes(comp)
    assert k == 2 and len(set(labels)) == 2
    assert all(n.endswith("-led") for n in names.values())

    # predictability harness runs and returns a sane dict
    turb = pd.Series(10.0, index=pd.bdate_range("2014-06-01", periods=600))
    contrib = pd.DataFrame(
        rng.dirichlet(np.ones(4), 600) * 10.0,
        index=turb.index, columns=list("wxyz"))
    eps = pd.DataFrame({
        "start": comp.index[:5], "end": comp.index[:5] + pd.Timedelta(days=7),
    })
    out = composition_predictability(contrib, eps, comp.iloc[:5], n_perm=200)
    assert 0.0 <= out["p_value"] <= 1.0
    assert out["n_episodes"] > 0
