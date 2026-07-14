"""Absorption ratio — a systemic-fragility signal complementary to turbulence.

Reference
---------
Kritzman, M., Li, Y., Page, S. and Rigobon, R. (2011), "Principal Components
as a Measure of Systemic Risk", *Journal of Portfolio Management* 37(4).

Intuition
---------
Turbulence measures *how statistically unusual* today is. The **absorption
ratio** measures something different — *how fragile the system is*: the
fraction of total variance captured by the top few principal components of
the factor covariance. A high AR means most of the variance loads on a few
directions — markets are tightly coupled, so a shock propagates everywhere
(fragile). A low AR means risk is spread across many independent
directions (resilient).

    AR_t = ( sum of the top `n` eigenvalues ) / ( sum of all eigenvalues )

of the covariance estimated on the trailing window ending at ``t-1``. With
``k`` factors KLPR use ``n ≈ k/5``.

The predictive object in KLPR is not the level but the **standardised
shift** — a fast rise in AR (coupling spiking up) tends to precede
drawdowns. :func:`absorption_shift` computes it causally.

Causality
---------
AR_t depends only on data with index ``< t`` (the covariance window ends at
``t-1``); this mirrors the no-look-ahead contract of
:class:`regime_detection.lib.TurbulenceIndex` and is verified by the same
shock test.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class AbsorptionRatio:
    """Walk-forward absorption ratio.

    Parameters
    ----------
    n_components :
        Number of leading eigenvalues in the numerator. ``None`` → ``max(1,
        round(k / 5))`` (the KLPR convention; 2 for the 10-factor set).
    lookback_days :
        Trailing covariance window. Defaults to ``504`` (≈2 years) — shorter
        than turbulence's 10-year window because AR is meant to track
        *current* coupling responsively (KLPR use ≈500 days).
    min_periods :
        Minimum finite rows in the window for an AR value to be emitted.
    shrinkage :
        ``"ledoit-wolf"`` (default) or ``None`` (sample covariance).
    refit_every :
        Rows between covariance refits (``1`` = every row).
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        lookback_days: int = 504,
        min_periods: int = 252,
        shrinkage: Optional[str] = "ledoit-wolf",
        refit_every: int = 1,
    ) -> None:
        if n_components is not None and n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if min_periods < 2 or min_periods > lookback_days:
            raise ValueError("require 2 <= min_periods <= lookback_days")
        if shrinkage not in (None, "ledoit-wolf"):
            raise ValueError("shrinkage must be None or 'ledoit-wolf'")
        if refit_every < 1:
            raise ValueError("refit_every must be >= 1")
        self.n_components = n_components
        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.shrinkage = shrinkage
        self.refit_every = refit_every

    def _ar_of_window(self, fit_slice: np.ndarray) -> float:
        # Standardise each column within the window (-> correlation matrix), so
        # AR reflects genuine *coupling* and not scale.  With raw covariance a
        # single high-variance factor (e.g. VIX) dominates the top PC and AR
        # degenerates to ~constant; KLPR use standardised returns for exactly
        # this reason on heterogeneous panels.
        sd = fit_slice.std(axis=0, ddof=0)
        sd = np.where(sd == 0.0, 1.0, sd)
        z = (fit_slice - fit_slice.mean(axis=0)) / sd
        if self.shrinkage == "ledoit-wolf":
            cov = LedoitWolf().fit(z).covariance_
        else:
            cov = np.cov(z, rowvar=False, ddof=0)
        eig = np.linalg.eigvalsh(cov)          # ascending, real (cov is symmetric)
        eig = np.clip(eig[::-1], 0.0, None)    # descending, non-negative
        total = eig.sum()
        if total <= 0:
            return float("nan")
        k = fit_slice.shape[1]
        n = self.n_components or max(1, int(round(k / 5)))
        n = min(n, k)
        return float(eig[:n].sum() / total)

    def fit_transform(self, returns: pd.DataFrame) -> pd.Series:
        """Compute the daily absorption ratio (in (0, 1]); causal."""
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("returns must have a DatetimeIndex")
        if returns.shape[1] < 2:
            raise ValueError("AbsorptionRatio requires at least 2 columns")

        arr = returns.values.astype(float)
        n = len(arr)
        out = np.full(n, np.nan)
        cached_r: Optional[int] = None
        cached_ar = np.nan
        for t in range(n):
            r = (t // self.refit_every) * self.refit_every
            if r != cached_r:
                cached_r = r
                cached_ar = np.nan
                fit_start, fit_end = max(0, r - self.lookback_days), r
                if fit_end > fit_start:
                    sl = arr[fit_start:fit_end]
                    sl = sl[np.isfinite(sl).all(axis=1)]
                    if len(sl) >= self.min_periods:
                        cached_ar = self._ar_of_window(sl)
            out[t] = cached_ar
        return pd.Series(out, index=returns.index, name="absorption_ratio")


def absorption_shift(
    ar: pd.Series, short_span: int = 15, long_window: int = 252,
) -> pd.Series:
    """Standardised absorption-ratio shift (KLPR's early-warning signal).

    ``(EWMA_short(AR) − MA_long(AR)) / std_long(AR)``, every term shifted by
    one row so the value at ``t`` uses only AR through ``t-1``. A large
    positive value means coupling is rising fast relative to its own recent
    history — the configuration that tends to precede drawdowns.
    """
    if not isinstance(ar, pd.Series):
        raise TypeError("ar must be a pandas Series")
    short = ar.ewm(span=short_span, min_periods=short_span).mean()
    long_mean = ar.rolling(long_window, min_periods=long_window // 2).mean()
    long_std = ar.rolling(long_window, min_periods=long_window // 2).std(ddof=0)
    shift = (short - long_mean) / long_std.replace(0.0, np.nan)
    return shift.shift(1).rename("absorption_shift")
