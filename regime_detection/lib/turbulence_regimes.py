"""Binary turbulent / quiet regime labels from the Mahalanobis turbulence
index, plus a causal minimum-duration smoother.

Both helpers operate on a single :class:`pandas.Series` and respect the
walk-forward causality contract: the value at row ``t`` depends only on
data with index strictly less than ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_TURBULENT = 1
REGIME_QUIET = 0


def classify_regimes(
    turbulence: pd.Series,
    threshold_quantile: float = 0.75,
    rolling_quantile_window: int = 2520,
    min_periods: int = 504,
) -> pd.Series:
    """Convert a turbulence series into a binary regime label.

    The threshold at row ``t`` is the ``threshold_quantile`` of past
    turbulence values over the trailing ``rolling_quantile_window`` rows
    ending at ``t-1``.  This makes the regime definition itself causal:
    today's label depends only on yesterday-and-earlier turbulence.

    Parameters
    ----------
    turbulence :
        Turbulence series from
        :meth:`regime_detection.lib.TurbulenceIndex.fit_transform`.
    threshold_quantile :
        Quantile (``0 < q < 1``) used to mark a day as turbulent.  ``0.75``
        means the top quartile of trailing turbulence is flagged as
        ``REGIME_TURBULENT``.
    rolling_quantile_window :
        Trailing window over which the quantile is computed.
    min_periods :
        Minimum number of past observations required for the threshold to
        be defined.  Rows before this point return ``NaN``.

    Returns
    -------
    pd.Series of float
        ``0`` (quiet), ``1`` (turbulent), or ``NaN`` (insufficient
        history).  Float rather than Int8 so the same series can carry
        NaNs and still arithmetic-mix with downstream code.
    """
    if not isinstance(turbulence, pd.Series):
        raise TypeError(
            f"turbulence must be a pandas Series, got {type(turbulence).__name__}"
        )
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError(
            f"threshold_quantile must lie in (0, 1); got {threshold_quantile}"
        )
    if rolling_quantile_window < 2:
        raise ValueError(
            f"rolling_quantile_window must be >= 2, got {rolling_quantile_window}"
        )
    if min_periods < 2 or min_periods > rolling_quantile_window:
        raise ValueError(
            f"min_periods must lie in [2, rolling_quantile_window]; "
            f"got {min_periods}"
        )

    # Trailing rolling quantile, shifted by one row so row t uses
    # only data with index <= t-1.
    threshold = (
        turbulence.rolling(
            window=rolling_quantile_window,
            min_periods=min_periods,
        )
        .quantile(threshold_quantile)
        .shift(1)
    )

    mask = threshold.notna() & turbulence.notna()
    out = np.full(len(turbulence), np.nan)
    out[mask.values] = (
        turbulence[mask].values > threshold[mask].values
    ).astype(float)
    return pd.Series(out, index=turbulence.index, name="regime")


def smooth_min_duration(
    regime: pd.Series,
    min_duration: int = 5,
) -> pd.Series:
    """Enforce a minimum regime-duration constraint causally.

    A transition into a new regime is accepted only once the new label has
    persisted for ``min_duration`` consecutive non-NaN days.  Until then
    the previous accepted regime is carried forward.  Equivalently: the
    smoothed label at row ``t`` is the last regime value that has held
    uninterrupted for ``min_duration`` days at or before ``t``.

    This is the causal variant — it never inspects regime values past
    ``t``.  Real-time use therefore has no embedded lag beyond the
    persistence requirement itself.

    Parameters
    ----------
    regime :
        Binary regime labels (``0``/``1``/``NaN``) produced by
        :func:`classify_regimes`.
    min_duration :
        Number of consecutive non-NaN days the new label must hold before
        the smoother switches.  Must be ``>= 1``.

    Returns
    -------
    pd.Series of float
        Smoothed regime labels with the same index and name.  NaN rows in
        the input remain NaN (no extrapolation across gaps).
    """
    if not isinstance(regime, pd.Series):
        raise TypeError(
            f"regime must be a pandas Series, got {type(regime).__name__}"
        )
    if min_duration < 1:
        raise ValueError(f"min_duration must be >= 1, got {min_duration}")

    arr = regime.values.astype(float)
    n = len(arr)
    out = np.full(n, np.nan)

    current = np.nan
    candidate = np.nan
    streak = 0

    for i in range(n):
        v = arr[i]
        if np.isnan(v):
            # Carry forward NaN — we do not extrapolate the smoothed label
            # across a gap.  Reset the candidate streak so a transition
            # before the gap doesn't "leak" past it.
            out[i] = np.nan
            candidate = np.nan
            streak = 0
            continue

        if np.isnan(current):
            current = v
            candidate = v
            streak = 1
            out[i] = current
            continue

        if v == current:
            candidate = current
            streak = 1
            out[i] = current
            continue

        # v != current — building up a candidate transition.
        if v == candidate:
            streak += 1
        else:
            candidate = v
            streak = 1

        if streak >= min_duration:
            current = candidate
            # After accepting, treat the candidate as the new current so
            # subsequent same-value rows simply maintain it.
            out[i] = current
        else:
            out[i] = current

    return pd.Series(out, index=regime.index, name=regime.name or "regime")
