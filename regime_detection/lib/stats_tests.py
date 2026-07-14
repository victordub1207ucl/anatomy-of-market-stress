"""Statistical tests for OOS forecast and strategy comparison.

Three workhorses are provided:

* :func:`diebold_mariano` — DM test on a loss-differential series with a
  Newey-West HAC standard error.
* :func:`block_bootstrap_sortino_diff` — fixed-length non-overlapping
  block bootstrap of the Sortino difference between two return series.
* :func:`stationary_bootstrap_sortino_diff` — Politis-Romano stationary
  bootstrap with geometric block lengths (mean = ``block_length``) — a
  more robust variant when the block structure is not known.

Both bootstraps return a confidence interval and a (two-sided) bootstrap
p-value for ``H0 : Sortino(A) - Sortino(B) = 0``.

The interpretation caveat from the original thesis (sample-size ceiling
on power) is documented next to each function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from regime_detection.lib.metrics import sortino_ratio


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


@dataclass
class DMResult:
    statistic: float
    p_value: float
    mean_diff: float
    hac_lags: int
    n: int


def _newey_west_var(d: np.ndarray, lags: int) -> float:
    """Newey-West HAC variance estimator with Bartlett weights."""
    n = len(d)
    if n < 2:
        return float("nan")
    d_demean = d - d.mean()
    gamma0 = float((d_demean ** 2).sum() / n)
    var = gamma0
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = float((d_demean[lag:] * d_demean[:-lag]).sum() / n)
        var += 2.0 * w * gamma
    return var


def diebold_mariano(
    loss_a: pd.Series,
    loss_b: pd.Series,
    h: int = 1,
) -> DMResult:
    """Diebold-Mariano (1995) test.

    ``H0`` : ``E[loss_a - loss_b] = 0``.  Two-sided.  Loss series are
    aligned on their common index; rows with any NaN are dropped.

    ``h`` is the forecast horizon (used to set the Newey-West truncation
    to ``h - 1``).  Pass the supervised target horizon (here, 21).

    The DM statistic is asymptotically standard-normal; we return a
    two-sided p-value from the standard normal.
    """
    common = loss_a.index.intersection(loss_b.index)
    d = (loss_a.reindex(common) - loss_b.reindex(common)).dropna().values
    n = len(d)
    if n < 2:
        return DMResult(float("nan"), float("nan"), float("nan"), 0, n)
    mean_d = float(d.mean())
    hac_lags = max(0, h - 1)
    var = _newey_west_var(d, hac_lags)
    if var <= 0.0 or not np.isfinite(var):
        return DMResult(float("nan"), float("nan"), mean_d, hac_lags, n)
    se = np.sqrt(var / n)
    stat = mean_d / se
    # Two-sided p-value from standard normal.
    from math import erf, sqrt
    p_two = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(stat) / sqrt(2.0))))
    return DMResult(float(stat), float(p_two), mean_d, hac_lags, n)


# ---------------------------------------------------------------------------
# Block bootstrap of Sortino difference
# ---------------------------------------------------------------------------


@dataclass
class BootstrapDiffResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n_resamples: int
    block_length: int
    method: str


def _resample_concat(values: np.ndarray, n_blocks: int,
                     block_length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    starts = rng.integers(0, max(1, n - block_length + 1), size=n_blocks)
    chunks = [values[s:s + block_length] for s in starts]
    return np.concatenate(chunks)[:n]


def block_bootstrap_sortino_diff(
    returns_a: pd.Series,
    returns_b: pd.Series,
    *,
    block_length: int = 21,
    n_resamples: int = 2_000,
    alpha: float = 0.05,
    target: float = 0.0,
    random_state: int = 0,
) -> BootstrapDiffResult:
    """Fixed-length non-overlapping block bootstrap of
    ``Sortino(A) - Sortino(B)``.

    The block length defaults to 21 trading days — roughly one month,
    sized to absorb most volatility-cluster autocorrelation.

    Interpretation caveat: with OOS sample lengths of a few hundred
    blocks, the bootstrap's *power* against modest Sortino differences
    is bounded.  Read the CI width as the effective resolution.
    """
    common = returns_a.index.intersection(returns_b.index)
    a = returns_a.reindex(common).dropna()
    b = returns_b.reindex(common).dropna()
    common2 = a.index.intersection(b.index)
    a, b = a.reindex(common2).values, b.reindex(common2).values
    n = len(a)
    if n < block_length * 2:
        return BootstrapDiffResult(
            mean_diff=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
            p_value=float("nan"), n_resamples=0, block_length=block_length,
            method="block",
        )
    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(n / block_length))

    obs_a = sortino_ratio(pd.Series(a), target=target)
    obs_b = sortino_ratio(pd.Series(b), target=target)
    obs_diff = obs_a - obs_b

    diffs = np.empty(n_resamples)
    for k in range(n_resamples):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        a_resamp = np.concatenate([a[s:s + block_length] for s in starts])[:n]
        b_resamp = np.concatenate([b[s:s + block_length] for s in starts])[:n]
        diffs[k] = (
            sortino_ratio(pd.Series(a_resamp), target=target)
            - sortino_ratio(pd.Series(b_resamp), target=target)
        )
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 100:
        return BootstrapDiffResult(
            mean_diff=float(obs_diff), ci_low=float("nan"),
            ci_high=float("nan"), p_value=float("nan"),
            n_resamples=int(len(diffs)), block_length=block_length,
            method="block",
        )
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2))
    # Two-sided bootstrap p-value: fraction of resamples whose
    # |diff| >= |obs_diff| under the recentred null distribution.
    centred = diffs - diffs.mean()
    p_two = float(np.mean(np.abs(centred) >= abs(obs_diff)))
    return BootstrapDiffResult(
        mean_diff=float(obs_diff), ci_low=lo, ci_high=hi,
        p_value=p_two, n_resamples=int(len(diffs)),
        block_length=block_length, method="block",
    )


def stationary_bootstrap_sortino_diff(
    returns_a: pd.Series,
    returns_b: pd.Series,
    *,
    mean_block_length: int = 21,
    n_resamples: int = 2_000,
    alpha: float = 0.05,
    target: float = 0.0,
    random_state: int = 0,
) -> BootstrapDiffResult:
    """Politis-Romano stationary bootstrap.  Block lengths follow a
    geometric distribution with mean ``mean_block_length``."""
    common = returns_a.index.intersection(returns_b.index)
    a = returns_a.reindex(common).dropna()
    b = returns_b.reindex(common).dropna()
    common2 = a.index.intersection(b.index)
    a, b = a.reindex(common2).values, b.reindex(common2).values
    n = len(a)
    if n < mean_block_length * 2:
        return BootstrapDiffResult(
            mean_diff=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
            p_value=float("nan"), n_resamples=0,
            block_length=mean_block_length, method="stationary",
        )

    rng = np.random.default_rng(random_state)
    p_geom = 1.0 / float(mean_block_length)
    obs_diff = (
        sortino_ratio(pd.Series(a), target=target)
        - sortino_ratio(pd.Series(b), target=target)
    )

    def _one_sample(series: np.ndarray) -> np.ndarray:
        out = np.empty(n)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            geom_len = rng.geometric(p_geom)
            geom_len = max(1, min(geom_len, n - i))
            for j in range(geom_len):
                out[i + j] = series[(start + j) % n]
            i += geom_len
        return out

    diffs = np.empty(n_resamples)
    for k in range(n_resamples):
        diffs[k] = (
            sortino_ratio(pd.Series(_one_sample(a)), target=target)
            - sortino_ratio(pd.Series(_one_sample(b)), target=target)
        )
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 100:
        return BootstrapDiffResult(
            mean_diff=float(obs_diff), ci_low=float("nan"),
            ci_high=float("nan"), p_value=float("nan"),
            n_resamples=int(len(diffs)),
            block_length=mean_block_length, method="stationary",
        )
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2))
    centred = diffs - diffs.mean()
    p_two = float(np.mean(np.abs(centred) >= abs(obs_diff)))
    return BootstrapDiffResult(
        mean_diff=float(obs_diff), ci_low=lo, ci_high=hi,
        p_value=p_two, n_resamples=int(len(diffs)),
        block_length=mean_block_length, method="stationary",
    )
