"""Ordinal turbulence bucket — a causal n-level severity binning of turbulence.

Phase 9F.  The Phase-9 "middle path" experiment (turbulence into a cleaned
walk-forward GMM) found that the GMM adds nothing, but surfaced a genuine,
cheap improvement to v5: replacing the *binary* turbulent/quiet regime
feature with an *ordinal* n-level turbulence bucket gives a markedly less
cutoff-sensitive supervised signal.

This module productionises that bucket as a proper, tested feature and a
drop-in feature-matrix builder.  At row ``t`` the bucket compares
*yesterday's* turbulence to the ``[1/n, 2/n, ..., (n-1)/n]`` quantiles of
the trailing ``window`` days ending at ``t-1`` — strictly causal, no
look-ahead.

The bucket replaces v5's two binary-regime columns (``regime_lag1``,
``regime_ma21``) with:

* ``turbbucket_lag1`` — the ordinal level 0..n-1
* ``turbbucket_ma21`` — its 21-day rolling mean
* ``tb_<k>``          — one-hot dummies for n-1 of the n levels (one
                        dropped as reference)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from regime_detection.lib.features import build_feature_matrix


def ordinal_turbulence_bucket(
    turbulence: pd.Series,
    n_buckets: int = 5,
    window: int = 2520,
    min_periods: int = 504,
) -> pd.Series:
    """Causal ordinal bucket (0..n_buckets-1) of turbulence severity.

    At row ``t`` the value is the number of trailing-window quantile
    thresholds that *yesterday's* turbulence exceeds.  All inputs are
    shifted by one row so the bucket at ``t`` uses only data with index
    ``<= t-1``.

    Parameters
    ----------
    turbulence :
        Daily Mahalanobis turbulence.
    n_buckets :
        Number of ordinal levels (default 5).
    window, min_periods :
        Trailing window for the quantile thresholds.

    Returns
    -------
    pd.Series of float
        Values in ``{0, 1, ..., n_buckets-1}`` where defined, ``NaN``
        during the warm-up.
    """
    if not isinstance(turbulence, pd.Series):
        raise TypeError("turbulence must be a pandas Series")
    if n_buckets < 2:
        raise ValueError(f"n_buckets must be >= 2, got {n_buckets}")
    if min_periods < 2 or min_periods > window:
        raise ValueError("require 2 <= min_periods <= window")

    ts = turbulence.shift(1)  # yesterday's turbulence -> causal at row t
    quantile_levels = [i / n_buckets for i in range(1, n_buckets)]

    bucket = pd.Series(0.0, index=turbulence.index)
    first_threshold_na = None
    for q in quantile_levels:
        thr = (turbulence.rolling(window, min_periods=min_periods)
               .quantile(q).shift(1))
        bucket = bucket + (ts > thr).astype(float)
        if first_threshold_na is None:
            first_threshold_na = thr.isna()
    # Undefined wherever the (lowest) threshold is undefined or ts is NaN.
    undefined = first_threshold_na | ts.isna()
    bucket[undefined.values] = np.nan
    return bucket.rename("turbbucket")


def build_feature_matrix_with_bucket(
    prices: pd.DataFrame,
    turbulence: pd.Series,
    binary_regime: pd.Series,
    *,
    factors,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """v5 feature matrix with the binary regime swapped for the ordinal
    turbulence bucket.

    Everything else (turbulence dynamics, equity/rates/credit/vix moments,
    cross-asset correlations) is identical to v5's
    :func:`regime_detection.lib.features.build_feature_matrix`, so a
    head-to-head against v5 isolates exactly the regime representation:
    binary threshold vs ordinal bucket.
    """
    base = build_feature_matrix(
        prices, turbulence, regime_smoothed=binary_regime, factors=factors)
    base = base.drop(columns=[c for c in ("regime_lag1", "regime_ma21")
                              if c in base.columns])

    bucket = ordinal_turbulence_bucket(turbulence, n_buckets=n_buckets)
    base["turbbucket_lag1"] = bucket
    base["turbbucket_ma21"] = bucket.rolling(21, min_periods=10).mean()
    one_hot = pd.get_dummies(bucket, prefix="tb", dtype=float)
    if len(one_hot.columns) > 1:
        one_hot = one_hot.drop(columns=[one_hot.columns[0]])  # reference level
    out = pd.concat([base, one_hot], axis=1)
    return out[sorted(out.columns)]
