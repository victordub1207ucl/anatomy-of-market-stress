"""Event-time *features* for correction forecasting (step28).

Earlier steps used the event clock as a return-distribution transform (step03),
a prediction *target* (step21), and a re-clocking of interpretation (step27) — but
the supervised model never received event-time quantities as predictive *inputs*.
This module builds those inputs: causal features that encode **where the market
is in the flow of information**, not just how unusual today is.

All features are strictly causal — the value at day ``t`` uses only turbulence /
returns with index ``<= t`` — and the running event accumulator resets forward,
so injecting a future shock cannot change any past row (verified by the suite's
no-look-ahead philosophy and this module's tests).

Features (each a column):
    et_intensity_to_date  cumulative turbulence in the currently-open event,
                          as a fraction of one event-unit (how 'full' this event is)
    et_days_since_start   calendar days since the current event began
    et_velocity           number of completed event-units in the trailing
                          ``tempo_window`` days (how compressed time is now)
    et_turb_per_day       trailing 5-day mean turbulence / threshold (local tempo)
    et_momentum           equity log-return accumulated over the trailing window
                          holding one event-unit of turbulence (return *per event*)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def univariate_turbulence(
    price: pd.Series,
    *,
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    """Per-asset (univariate) turbulence: the squared z-score of the asset's own
    log return against its trailing mean/std — the 1-D Mahalanobis distance.

    Used to build a *per-market* event clock (each market paced by its own stress
    intensity) as opposed to the single global factor-turbulence clock.  Causal:
    the mean/std use returns strictly before t (``.shift(1)``); the value at t
    uses today's return against those trailing statistics, mirroring how the
    multivariate turbulence index treats the current observation.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    r = np.log(price / price.shift(1))
    mu = r.rolling(window, min_periods=min_periods).mean().shift(1)
    sd = r.rolling(window, min_periods=min_periods).std().shift(1)
    z = (r - mu) / sd.replace(0.0, np.nan)
    return (z ** 2).rename("uni_turbulence")


def _eventtime_trailing_return(
    eq_ret: np.ndarray, turb: np.ndarray, threshold: float
) -> np.ndarray:
    """For each t, sum equity returns over the trailing window whose cumulative
    turbulence (walking back from t) first reaches ``threshold``."""
    n = len(turb)
    out = np.full(n, np.nan)
    for i in range(n):
        acc = 0.0
        j = i
        s = 0.0
        while j >= 0 and acc < threshold:
            tj = turb[j]
            if np.isfinite(tj):
                acc += tj
            rj = eq_ret[j]
            if np.isfinite(rj):
                s += rj
            j -= 1
        out[i] = s if acc >= threshold else np.nan
    return out


def build_event_time_features(
    turbulence: pd.Series,
    equity_prices: pd.Series,
    threshold: float,
    *,
    tempo_window: int = 21,
) -> pd.DataFrame:
    """Build the causal event-time feature block (see module docstring).

    Parameters
    ----------
    turbulence :
        Daily Mahalanobis turbulence (causal series).
    equity_prices :
        Price level of the equity factor / benchmark (for the per-event return).
    threshold :
        Cumulative turbulence defining one event-unit — calibrate on dev data.
    tempo_window :
        Trailing window (calendar days) for the event-completion velocity.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    turb = turbulence.astype(float)
    idx = turb.index
    t = turb.to_numpy(dtype=float)
    n = len(t)

    intensity = np.full(n, np.nan)
    days_since = np.full(n, np.nan)
    boundary = np.zeros(n, dtype=float)
    acc, start_i = 0.0, 0
    for i in range(n):
        ti = t[i] if np.isfinite(t[i]) else 0.0
        acc += ti
        intensity[i] = acc / threshold
        days_since[i] = i - start_i
        if acc >= threshold:
            boundary[i] = 1.0
            acc, start_i = 0.0, i + 1

    velocity = pd.Series(boundary, index=idx).rolling(tempo_window).sum()
    turb_per_day = (turb.rolling(5).mean() / threshold)

    eq_ret = np.log(equity_prices / equity_prices.shift(1)).reindex(idx).to_numpy(dtype=float)
    momentum = _eventtime_trailing_return(eq_ret, t, threshold)

    return pd.DataFrame(
        {
            "et_intensity_to_date": intensity,
            "et_days_since_start": days_since,
            "et_velocity": velocity.to_numpy(dtype=float),
            "et_turb_per_day": turb_per_day.to_numpy(dtype=float),
            "et_momentum": momentum,
        },
        index=idx,
    )
