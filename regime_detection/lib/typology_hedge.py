"""Typology-conditioned hedge selection (composition-aware de-risking).

This module is an *additive* layer on top of the turbulence detector: it does
not touch the turbulence index, the decomposition, or the supervised models.
It answers a different, better-posed question than "time the drawdown":

    Given that the turbulence *regime* already tells us *when* to de-risk,
    *where* should we park the capital?

The economic premise is that the right hedge depends on the *kind* of stress:

    * equity / credit / vix-led stress  -> classic flight-to-safety; long
      duration (Treasuries) rallies as equities fall.
    * rates / inflation-led stress       -> Treasuries fall *with* equities
      (e.g. 2022); a real-asset / cash hedge (gold, T-bills) is required.

A key, deliberately documented finding (see step26) is that the *magnitude*
composition of turbulence — the crisis-typology fingerprint of step17 — is
**direction-blind**: both the 2020 (equity-led, bonds rally) and 2022
(rates-led, bonds crash) episodes carry a large rates/inflation contribution,
so the magnitude share cannot separate them.  Routing therefore needs a
*signed* signal (cf. step18 signed turbulence), provided here by the trailing
trend of the duration leg.

All functions are pure and causal: every signal is computed from information
available at the decision time, and the realised return is taken on the
following bar.  Threshold calibration is the caller's responsibility and must
use development data only (index <= DEV_END).
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

EQUITY = "equity"
CASH = "cash"
BONDS = "bonds"
GOLD = "gold"


def rates_pressure_share(
    contrib: pd.DataFrame,
    rates_cols: Iterable[str] = ("rates", "inflation"),
) -> pd.Series:
    """Magnitude composition: share of (positive) turbulence contribution that
    comes from the rates/inflation block.

    This is the *direction-blind* typology signal.  It is exported so that
    step26 can demonstrate, honestly, that it does **not** discriminate the
    flight-to-safety vs inflation-shock hedge regimes (both carry a high
    rates share).  Use :func:`signed_duration_trend` for actual routing.

    Parameters
    ----------
    contrib :
        Per-factor turbulence contributions (one column per factor, each row
        summing to ``d_t``); the output of ``TurbulenceDecomposer``.
    rates_cols :
        Columns treated as the rates/inflation block.

    Returns
    -------
    pd.Series
        Share in ``[0, 1]`` (NaN where the row total is non-positive / NaN).
    """
    cols = [c for c in rates_cols if c in contrib.columns]
    if not cols:
        raise ValueError(f"none of {tuple(rates_cols)} in contributions columns")
    pos = contrib.clip(lower=0.0)
    total = pos.sum(axis=1)
    share = pos[cols].sum(axis=1) / total.replace(0.0, np.nan)
    return share.clip(0.0, 1.0)


def signed_duration_trend(
    duration_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Trailing cumulative return of the duration (bond) leg — a *signed*
    proxy for the rates regime.

    A negative value means bonds have been selling off (a rising-rate / 2022-
    style regime) and therefore make a poor de-risking destination; a positive
    value means bonds are in an uptrend (flight-to-safety still working).

    Parameters
    ----------
    duration_returns :
        Daily log returns of the duration hedge (e.g. long Treasuries).
    window :
        Look-back length in trading days.

    Returns
    -------
    pd.Series
        Trailing cumulative log return; the value at ``t`` uses returns up to
        and including ``t`` (caller must lag it before trading).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    return duration_returns.rolling(window).sum()


def route_hedge(
    gate: pd.Series,
    rates_shock: pd.Series,
    *,
    defensive: str = BONDS,
    inflationary: str = GOLD,
    risk_on: str = EQUITY,
) -> pd.Series:
    """Map (de-risk gate, rates-shock flag) to the asset to hold.

    * ``gate`` False  -> ``risk_on`` (hold equity).
    * ``gate`` True and ``rates_shock`` True  -> ``inflationary`` (gold/cash).
    * ``gate`` True and ``rates_shock`` False -> ``defensive``   (bonds).

    Both inputs are boolean Series aligned on the same index.  The result is a
    label Series; the caller is responsible for lagging it by one bar before
    realising returns.
    """
    g = gate.astype(bool)
    s = rates_shock.reindex(g.index) == True  # noqa: E712 — NaN -> False, stays bool
    out = pd.Series(risk_on, index=g.index, dtype=object)
    out[g & ~s] = defensive
    out[g & s] = inflationary
    return out


def strategy_returns(
    positions: pd.Series,
    asset_returns: Mapping[str, pd.Series],
    cost_bps: float = 5.0,
) -> pd.Series:
    """Realise a label strategy into a daily log-return series.

    Parameters
    ----------
    positions :
        Label per day giving the asset *held during that day* (already lagged
        by the caller, i.e. decided at the prior close).
    asset_returns :
        Mapping from label -> daily log-return Series.  Must contain every
        label that appears in ``positions``.  ``"cash"`` may be omitted and is
        then treated as a zero-return asset.
    cost_bps :
        One-way transaction cost in basis points, charged on the days the held
        label changes.

    Returns
    -------
    pd.Series
        Daily log returns of the strategy, indexed like ``positions``.
    """
    idx = positions.index
    labels = set(positions.dropna().unique())
    ret_df = {}
    for lab in labels:
        if lab == CASH and lab not in asset_returns:
            ret_df[lab] = pd.Series(0.0, index=idx)
        else:
            if lab not in asset_returns:
                raise KeyError(f"asset_returns missing return series for {lab!r}")
            ret_df[lab] = asset_returns[lab].reindex(idx)
    ret_df = pd.DataFrame(ret_df)

    # Pick, for each day, the return of the held label.
    held = positions.to_numpy()
    out = pd.Series(np.nan, index=idx, dtype=float)
    for lab in labels:
        mask = positions == lab
        out[mask] = ret_df.loc[mask, lab]

    # Switching cost: charge when today's held label differs from yesterday's.
    switched = positions != positions.shift(1)
    switched.iloc[0] = False
    out = out - switched.astype(float) * (cost_bps / 1e4)
    return out


def performance_summary(
    daily_log_ret: pd.Series,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Standard performance statistics for a daily log-return series."""
    r = daily_log_ret.dropna()
    if len(r) == 0:
        return {k: float("nan") for k in
                ("cagr", "vol", "sharpe", "sortino", "max_drawdown", "total_return")}
    simple = np.expm1(r)
    n = len(r)
    total_log = r.sum()
    cagr = float(np.expm1(total_log * periods_per_year / n))
    vol = float(simple.std(ddof=1) * np.sqrt(periods_per_year))
    mean_ann = float(simple.mean() * periods_per_year)
    sharpe = mean_ann / vol if vol > 0 else float("nan")
    downside = simple[simple < 0]
    dvol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else float("nan")
    sortino = mean_ann / dvol if dvol and dvol > 0 else float("nan")
    equity = r.cumsum()
    drawdown = float((equity - equity.cummax()).min())  # log-space worst peak-to-trough
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(np.expm1(drawdown)),
        "total_return": float(np.expm1(total_log)),
    }
