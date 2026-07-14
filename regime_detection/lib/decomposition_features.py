"""Decomposed-turbulence features for the supervised layer.

Hypothesis under test: *which* factor is driving turbulence carries
predictive information about imminent stress that the scalar turbulence
level alone does not.  The Phase-2 turbulence index ``d_t`` is a single
number; :class:`~regime_detection.lib.turbulence_decomposition.TurbulenceDecomposer`
splits it into per-factor contributions ``c_{i,t}`` that sum to ``d_t``.

This module turns those contributions into a compact, causal feature
block that is *added on top of* v5's baseline feature matrix, so a
head-to-head isolates exactly the marginal value of composition
information:

* ``decshare_<factor>_<w>`` — factor ``i``'s trailing weighted share of
  turbulence over a ``w``-day window:
  ``sum_{w} c_{i} / sum_{w} d``, shifted by one row.  Weighting by the
  rolling sum (rather than averaging daily shares) makes the share
  dominated by the high-turbulence days that matter, and the denominator
  (a 21+ day sum of a strictly-positive ``d_t``) is safely far from zero.
* ``decconc_<w>`` — a Herfindahl concentration of those shares
  (``sum_i share_i^2``): high when a single factor dominates turbulence,
  low when stress is broad-based.

Every column is trailing-window and ``.shift(1)``-ed, so the value at row
``t`` uses only data with index ``<= t-1`` — the same causality contract
as the rest of the supervised layer.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from regime_detection.lib.features import build_feature_matrix


def build_decomposition_features(
    contributions: pd.DataFrame,
    turbulence: pd.Series,
    windows: Sequence[int] = (21, 63),
) -> pd.DataFrame:
    """Causal per-factor turbulence-share + concentration features.

    Parameters
    ----------
    contributions :
        Per-factor turbulence contributions (output of
        :meth:`TurbulenceDecomposer.fit_transform`); each row sums to
        ``d_t``.
    turbulence :
        The scalar turbulence series ``d_t`` (used as the share
        denominator).  Aligned to ``contributions`` by index.
    windows :
        Trailing windows (in trading days) for the weighted shares /
        concentration.

    Returns
    -------
    pd.DataFrame
        Indexed like ``contributions``; one ``decshare_*`` column per
        (factor, window) plus one ``decconc_<w>`` per window.
    """
    if not isinstance(contributions, pd.DataFrame):
        raise TypeError("contributions must be a pandas DataFrame")
    if not isinstance(turbulence, pd.Series):
        raise TypeError("turbulence must be a pandas Series")

    d = turbulence.reindex(contributions.index)
    out = {}
    for w in windows:
        mp = max(5, w // 2)
        d_sum = d.rolling(w, min_periods=mp).sum()
        share_cols = {}
        for f in contributions.columns:
            c_sum = contributions[f].rolling(w, min_periods=mp).sum()
            share = c_sum / d_sum
            share_cols[f] = share
            out[f"decshare_{f}_{w}"] = share.shift(1)
        # Herfindahl concentration of the (signed) shares.
        share_df = pd.DataFrame(share_cols)
        conc = (share_df ** 2).sum(axis=1)
        # Mask rows where no share is defined yet.
        conc[share_df.isna().all(axis=1)] = np.nan
        out[f"decconc_{w}"] = conc.shift(1)

    X = pd.DataFrame(out, index=contributions.index)
    return X[sorted(X.columns)]


def build_feature_matrix_with_decomposition(
    prices: pd.DataFrame,
    turbulence: pd.Series,
    contributions: pd.DataFrame,
    regime_smoothed: Optional[pd.Series] = None,
    *,
    factors,
    windows: Sequence[int] = (21, 63),
) -> pd.DataFrame:
    """v5 baseline feature matrix **plus** decomposed-turbulence features.

    Everything in v5's
    :func:`regime_detection.lib.features.build_feature_matrix`
    (turbulence dynamics, equity/rates/credit/vix moments, cross-asset
    correlations, binary regime) is retained unchanged, so a head-to-head
    against the baseline isolates exactly the marginal value of the
    per-factor composition features.
    """
    base = build_feature_matrix(
        prices, turbulence, regime_smoothed=regime_smoothed, factors=factors)
    dec = build_decomposition_features(contributions, turbulence, windows=windows)
    out = pd.concat([base, dec], axis=1)
    return out[sorted(out.columns)]
