"""Supervised targets for the turbulence pipeline.

Three target options:

* :func:`binary_turbulence_entry` — predict imminent crisis entry.
* :func:`forward_drawdown_conditional` — predict the worst forward
  drawdown over a fixed calendar horizon, optionally stratified by the
  current regime label.
* :func:`event_time_drawdown` — predict the worst drawdown over the next
  event window of variable length (event-time native).

Every target inspects future data by construction — that is the point of
a supervised target.  The companion training pipeline aligns features at
time ``t`` with the target value at ``t`` (which describes the future)
and enforces a purge of training rows whose forward window overlaps the
test fold (see :class:`PurgedGroupTimeSeriesSplit`).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def binary_turbulence_entry(
    turbulence: pd.Series,
    threshold: float,
    horizon: int = 21,
) -> pd.Series:
    """Label = 1 if turbulence crosses ``threshold`` within the next
    ``horizon`` trading days.

    Parameters
    ----------
    turbulence :
        Daily Mahalanobis turbulence (output of
        :class:`regime_detection.lib.TurbulenceIndex`).
    threshold :
        Turbulence level that defines a "crisis entry".  A natural choice
        is the rolling-quantile threshold used in
        :func:`regime_detection.lib.classify_regimes`, evaluated at the
        end of the dev window so that the OOS test set sees a fixed
        threshold.
    horizon :
        Forward window length in trading days.  Defaults to ``21`` to
        match the existing 21-day forward-drawdown horizon.

    Returns
    -------
    pd.Series of float
        ``1.0`` if any value of ``turbulence[t+1..t+horizon]`` is finite
        and meets the threshold, ``0.0`` if the window is fully observed
        and contains no such value, ``NaN`` if the entire forward window
        is missing (end of sample).
    """
    if not isinstance(turbulence, pd.Series):
        raise TypeError("turbulence must be a pandas Series")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold!r}")

    forward = np.column_stack([
        turbulence.shift(-k).values for k in range(1, horizon + 1)
    ])  # shape (n, horizon); column k is turbulence at t+k+1.
    has_data = np.isfinite(forward).any(axis=1)
    forward_max = np.full(len(turbulence), np.nan)
    if has_data.any():
        # nanmax only over rows with at least one finite value (silences
        # the all-NaN-slice warning).
        forward_max[has_data] = np.nanmax(forward[has_data], axis=1)
    label = (forward_max >= threshold).astype(float)
    label[~has_data] = np.nan
    return pd.Series(label, index=turbulence.index, name="binary_turbulence_entry")


def forward_drawdown_conditional(
    prices: pd.Series,
    current_regime: Optional[pd.Series] = None,
    horizon: int = 21,
) -> Union[pd.Series, pd.DataFrame]:
    """Worst log-return realized over the next ``horizon`` trading days,
    measured from today's price.

    For each date ``t`` the target is

        min_{1 <= h <= horizon} log(p[t+h] / p[t])

    i.e. the most negative cumulative log return on any day in the forward
    window.  If you instead want the classical peak-to-trough drawdown
    inside the window, see the note below.

    Parameters
    ----------
    prices :
        Asset price level (e.g. SPY-proxy equity factor).
    current_regime :
        Optional regime label at the *start* of the forward window
        (smoothed turbulence regime is the natural choice).  If provided,
        the function returns a DataFrame with both the drawdown target
        and the regime column so the caller can stratify (e.g. fit
        separate models per regime).
    horizon :
        Forward window length in trading days.

    Returns
    -------
    pd.Series or pd.DataFrame
        Series of forward drawdowns when ``current_regime`` is ``None``;
        DataFrame with ``["forward_drawdown", "current_regime"]`` columns
        otherwise.

    Notes
    -----
    This formulation matches the existing 21-day forward-drawdown target
    used by the legacy GMM pipeline.  A "true" peak-to-trough drawdown
    (using a running peak within the window) is a slight generalisation;
    add a separate helper if needed.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    log_p = np.log(prices.values.astype(float))
    n = len(prices)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(log_p[i]):
            continue
        end = min(i + 1 + horizon, n)
        window = log_p[i + 1 : end]
        finite = window[np.isfinite(window)]
        if len(finite) == 0:
            continue
        out[i] = float(finite.min() - log_p[i])

    target = pd.Series(out, index=prices.index, name="forward_drawdown")
    if current_regime is None:
        return target
    if not isinstance(current_regime, pd.Series):
        raise TypeError("current_regime must be a pandas Series")
    regime_aligned = current_regime.reindex(prices.index)
    return pd.DataFrame({
        "forward_drawdown": target,
        "current_regime": regime_aligned,
    })


def event_time_drawdown(
    prices: pd.Series,
    daily_turbulence: pd.Series,
    intensity_threshold: float,
) -> pd.Series:
    """Worst log-return over the event window starting at each calendar day.

    For each date ``t`` the target is

        min_{t < j <= j*}  log(p[j] / p[t])

    where ``j*`` is the smallest index such that turbulence accumulated
    over ``(t, j*]`` reaches ``intensity_threshold``.  This is the
    event-time analogue of :func:`forward_drawdown_conditional` — the
    horizon is variable, with mean ≈ 21 trading days when the threshold
    has been calibrated.

    Parameters
    ----------
    prices :
        Asset price level.
    daily_turbulence :
        Daily Mahalanobis turbulence.  ``NaN`` rows are skipped.
    intensity_threshold :
        Cumulative turbulence per event unit (use
        :meth:`regime_detection.lib.EventTimeConverter.calibrate`).

    Returns
    -------
    pd.Series
        Indexed by the intersection of the two series.  ``NaN`` where
        the event window extends past the end of the data.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if not isinstance(daily_turbulence, pd.Series):
        raise TypeError("daily_turbulence must be a pandas Series")
    threshold = float(intensity_threshold)
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError(
            f"intensity_threshold must be positive and finite, got {threshold!r}"
        )

    s = daily_turbulence.dropna()
    common = prices.index.intersection(s.index)
    if len(common) == 0:
        return pd.Series([], dtype=float, name="event_time_drawdown")
    p = prices.reindex(common).values.astype(float)
    t = s.reindex(common).values.astype(float)
    if (t < 0).any():
        raise ValueError("daily_turbulence contains negative values")
    cum = np.cumsum(t)
    log_p = np.log(p)
    n = len(common)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(log_p[i]):
            continue
        target = cum[i] + threshold
        j = int(np.searchsorted(cum, target, side="left"))
        if j >= n or j <= i:
            continue
        window = log_p[i + 1 : j + 1]
        finite = window[np.isfinite(window)]
        if len(finite) == 0:
            continue
        out[i] = float(finite.min() - log_p[i])
    return pd.Series(out, index=common, name="event_time_drawdown")
