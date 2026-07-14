"""Causal feature builder for the Phase-4 supervised layer.

Every column produced here is computable at date ``t`` using only data
with index strictly less than ``t``.  All rolling statistics are
explicitly trailing-window and then shifted by 1 to enforce the rule.

The feature set is deliberately compact (~24 columns).  It is focused on
turbulence-regime dynamics rather than the 69-feature factor lens of the
legacy pipeline; if the 69-feature engineering is to be reused, replace
:func:`build_feature_matrix` here with a call to the legacy
``features.engineering.build_feature_matrix`` once that path is verified
clean of look-ahead by the Phase-0 audit (it is, per the Phase-0 report —
only one MINOR finding at engineering.py:504).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


_DEFAULT_FACTORS = (
    "equity", "rates", "credit", "commodities", "em_equity",
    "fx_usd", "inflation", "value", "quality", "vix",
)


def _rolling_lag_features(s: pd.Series, lags=(1, 5, 21)) -> pd.DataFrame:
    """Plain lag features.  Each column is ``s`` shifted forward by ``k``
    so the value at row ``t`` reflects ``s[t-k]``."""
    return pd.DataFrame({
        f"{s.name}_lag{k}": s.shift(k) for k in lags
    })


def _trailing_mean_std(
    s: pd.Series, windows, min_periods_frac: float = 0.5, label: Optional[str] = None,
) -> pd.DataFrame:
    """Trailing means and stds, shifted by 1 so row ``t`` only uses
    ``s[t-window..t-1]``."""
    name = label or s.name
    out = {}
    for w in windows:
        mp = max(2, int(min_periods_frac * w))
        roll = s.rolling(w, min_periods=mp)
        out[f"{name}_ma{w}"] = roll.mean().shift(1)
        out[f"{name}_std{w}"] = roll.std(ddof=0).shift(1)
    return pd.DataFrame(out)


def _rolling_corr(
    a: pd.Series, b: pd.Series, window: int, name: str,
) -> pd.Series:
    """Trailing correlation of two daily log-return series, shifted by 1."""
    mp = max(5, window // 2)
    return a.rolling(window, min_periods=mp).corr(b).shift(1).rename(name)


def build_feature_matrix(
    prices: pd.DataFrame,
    turbulence: pd.Series,
    regime_smoothed: Optional[pd.Series] = None,
    factors=_DEFAULT_FACTORS,
) -> pd.DataFrame:
    """Produce the daily feature matrix consumed by the Phase-4 trainer.

    Parameters
    ----------
    prices :
        Wide DataFrame of factor price levels (the cached
        ``data/cache/factor_prices.parquet``).  Must contain at least the
        factors listed in ``factors``.
    turbulence :
        Daily Mahalanobis turbulence (Phase 2 output).  Indexed by trading
        date; ``NaN`` warm-up rows are kept as ``NaN`` in the output.
    regime_smoothed :
        Optional 5-day min-duration smoothed turbulence regime label
        (Phase 2 output).  When provided, two regime-based features are
        added (``regime_lag1`` and a 21-day rolling share).
    factors :
        Factor columns to consume from ``prices``.  Defaults to the
        Phase-2 stable 10-factor set.

    Returns
    -------
    pd.DataFrame
        Indexed by trading date, columns are the feature set.  Rows are
        ``NaN`` until enough trailing data exists for every column.
        Callers should drop NaNs (or rely on the downstream trainer to
        align and drop).
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if not isinstance(turbulence, pd.Series):
        raise TypeError("turbulence must be a pandas Series")
    missing = [f for f in factors if f not in prices.columns]
    if missing:
        raise KeyError(f"prices is missing required factors: {missing}")

    prices = prices[list(factors)].sort_index()
    log_p = np.log(prices)
    rets = log_p.diff()  # daily log returns

    pieces = []

    # ---- Turbulence dynamics ---------------------------------------
    pieces.append(_rolling_lag_features(
        turbulence.rename("turb"), lags=(1, 5, 21),
    ))
    pieces.append(_trailing_mean_std(
        turbulence.rename("turb"), windows=(5, 21, 63), label="turb",
    ))
    # 252-day rolling z-score of yesterday's turbulence.
    turb_mean_252 = turbulence.rolling(252, min_periods=63).mean().shift(1)
    turb_std_252 = turbulence.rolling(252, min_periods=63).std(ddof=0).shift(1)
    pieces.append(pd.DataFrame({
        "turb_zscore_252": (turbulence.shift(1) - turb_mean_252) / turb_std_252,
    }))

    # ---- Equity dynamics -------------------------------------------
    eq_ret = rets["equity"]
    pieces.append(pd.DataFrame({
        "eq_ret_lag1": eq_ret.shift(1),
        "eq_ret_ma5":  eq_ret.rolling(5, min_periods=3).sum().shift(1),
        "eq_ret_ma21": eq_ret.rolling(21, min_periods=10).sum().shift(1),
        "eq_vol_21":   eq_ret.rolling(21, min_periods=10).std(ddof=0).shift(1),
        "eq_vol_63":   eq_ret.rolling(63, min_periods=30).std(ddof=0).shift(1),
    }))
    # Drawdown from rolling 63-day high (uses yesterday's data only).
    roll_high_63 = log_p["equity"].rolling(63, min_periods=20).max().shift(1)
    pieces.append(pd.DataFrame({
        "eq_drawdown_63": log_p["equity"].shift(1) - roll_high_63,
    }))

    # ---- Rates / credit / VIX dynamics -----------------------------
    rates_ret = rets["rates"]
    pieces.append(pd.DataFrame({
        "rates_ret_ma21": rates_ret.rolling(21, min_periods=10).sum().shift(1),
        "rates_vol_63":   rates_ret.rolling(63, min_periods=30).std(ddof=0).shift(1),
    }))
    if "credit" in prices.columns:
        credit_ret = rets["credit"]
        pieces.append(pd.DataFrame({
            "credit_ret_ma21": credit_ret.rolling(21, min_periods=10).sum().shift(1),
        }))
    if "vix" in prices.columns:
        vix_log = log_p["vix"]
        pieces.append(pd.DataFrame({
            "vix_log_lag1": vix_log.shift(1),
            "vix_ma21": vix_log.rolling(21, min_periods=10).mean().shift(1),
        }))

    # ---- Cross-asset correlations ----------------------------------
    if "rates" in prices.columns:
        pieces.append(_rolling_corr(
            eq_ret, rates_ret, window=63, name="eq_rates_corr_63",
        ).to_frame())
    if "credit" in prices.columns:
        pieces.append(_rolling_corr(
            eq_ret, rets["credit"], window=63, name="eq_credit_corr_63",
        ).to_frame())

    # ---- Regime features -------------------------------------------
    if regime_smoothed is not None:
        if not isinstance(regime_smoothed, pd.Series):
            raise TypeError("regime_smoothed must be a pandas Series")
        regime = regime_smoothed.rename("regime")
        pieces.append(pd.DataFrame({
            "regime_lag1": regime.shift(1),
            "regime_ma21": regime.rolling(21, min_periods=10).mean().shift(1),
        }))

    X = pd.concat(pieces, axis=1)
    # Sort columns deterministically.
    X = X[sorted(X.columns)]
    X.index.name = prices.index.name
    return X
