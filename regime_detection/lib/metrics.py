"""Headline OOS metrics used by the Phase-6 evaluation pipeline.

Every function takes a strictly causal return / prediction series and
returns a scalar.  No look-ahead — all rolling stats are trailing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def annualised_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float(r.mean() * periods_per_year)


def annualised_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r = returns.dropna()
    if len(r) < 30:
        return float("nan")
    mu = float(r.mean()) - target / periods_per_year
    sd = float(r.std(ddof=0))
    if sd == 0:
        return float("nan")
    return (mu / sd) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sortino against ``target`` (annualised return target,
    converted to per-period inside)."""
    r = returns.dropna()
    if len(r) < 30:
        return float("nan")
    target_per_period = target / periods_per_year
    excess = r - target_per_period
    downside = excess[excess < 0]
    if len(downside) < 5:
        return float("nan")
    ann_mean = float(excess.mean() * periods_per_year)
    ann_dvol = float(downside.std(ddof=0) * np.sqrt(periods_per_year))
    if ann_dvol == 0:
        return float("nan")
    return ann_mean / ann_dvol


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of the cumulative log return series.  Returns a
    non-positive number (e.g. -0.35 = -35% drawdown)."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    eq_curve = r.cumsum()  # cumulative log return
    running_peak = eq_curve.cummax()
    dd = eq_curve - running_peak
    return float(dd.min())


def hit_rate_at_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    decile: float = 0.10,
) -> dict:
    """Precision and recall when classifying the top ``decile`` of
    predicted scores as positive."""
    df = pd.concat([y_true.rename("y"), y_score.rename("p")], axis=1).dropna()
    if len(df) == 0:
        return {"precision": float("nan"), "recall": float("nan"),
                "threshold": float("nan"), "n_predicted_pos": 0}
    cutoff = float(np.quantile(df["p"].values, 1.0 - decile))
    pred_pos = df["p"] >= cutoff
    true_pos = df["y"] > 0.5
    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & ~true_pos).sum())
    fn = int((~pred_pos & true_pos).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {
        "precision":         prec,
        "recall":            rec,
        "threshold":         cutoff,
        "n_predicted_pos":   int(pred_pos.sum()),
    }


def top_decile_threshold(y_score: pd.Series, decile: float = 0.10) -> float:
    s = y_score.dropna()
    if len(s) == 0:
        return float("nan")
    return float(np.quantile(s.values, 1.0 - decile))


def brier_score(y_true: pd.Series, y_score: pd.Series) -> float:
    df = pd.concat([y_true.rename("y"), y_score.rename("p")], axis=1).dropna()
    if len(df) == 0:
        return float("nan")
    return float(np.mean((df["p"].values - df["y"].values) ** 2))


def log_loss(y_true: pd.Series, y_score: pd.Series, eps: float = 1e-12) -> float:
    df = pd.concat([y_true.rename("y"), y_score.rename("p")], axis=1).dropna()
    if len(df) == 0:
        return float("nan")
    p = np.clip(df["p"].values, eps, 1.0 - eps)
    y = df["y"].values
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def strategy_returns(
    asset_log_returns: pd.Series,
    signal: pd.Series,
    flat_when_signal_is_high: bool = True,
    shift: int = 1,
) -> pd.Series:
    """Build the overlay-strategy return series.

    Parameters
    ----------
    asset_log_returns :
        Daily log returns of the underlying asset (e.g. equity factor).
    signal :
        Predicted probability of "imminent turbulence".  Used to decide
        whether to hold the asset on each day.
    flat_when_signal_is_high :
        If True, the strategy goes flat (return 0) when the signal
        exceeds the top-decile threshold of the signal series itself;
        otherwise stays long the asset.
    shift :
        Number of trading days to lag the signal (default 1: decision at
        end of t-1 acted upon at t).

    Returns
    -------
    pd.Series
        Strategy daily log returns aligned with ``asset_log_returns``.
    """
    common = asset_log_returns.index.intersection(signal.index)
    asset = asset_log_returns.reindex(common)
    sig = signal.reindex(common)
    if flat_when_signal_is_high:
        thr = top_decile_threshold(sig)
        in_market = (sig < thr).astype(float)
    else:
        in_market = (sig >= top_decile_threshold(sig)).astype(float)
    return asset * in_market.shift(shift).fillna(0.0)
