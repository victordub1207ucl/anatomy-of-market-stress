"""Kritzman-Li turbulence index (walk-forward, Ledoit-Wolf-shrunk).

Reference
---------
Kritzman, M. and Li, Y. (2010), "Skulls, Financial Turbulence, and Risk
Management", *Financial Analysts Journal* 66(5).

For each date ``t`` the index measures the statistical unusualness of the
cross-section of factor returns ``y_t`` relative to the trailing
``lookback_days`` window ending at ``t-1``::

    d_t = (y_t - mu_t)' Omega_t^{-1} (y_t - mu_t)

where ``mu_t`` and ``Omega_t`` are the mean and covariance estimated on the
trailing window.  ``Omega_t`` is shrunk via Ledoit-Wolf to remain
invertible in the 14-factor / 2 520-day regime (k * (k+1) / 2 = 105 free
parameters vs ~2 500 observations — the empirical covariance is full rank
in this regime, but shrinkage stabilises noisy off-diagonal entries and
matters more when the user shortens the lookback or widens the factor
set).

This module enforces the causality contract documented in
``regime_detection.lib.walk_forward``: the row at ``t`` depends only on
data with index strictly less than ``t``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class TurbulenceIndex:
    """Walk-forward Mahalanobis turbulence index.

    Parameters
    ----------
    lookback_days :
        Length of the trailing window used to estimate ``mu`` and
        ``Omega``.  Defaults to ``2520`` (≈10 years of business days),
        matching Kritzman & Li (2010).
    min_periods :
        Minimum number of finite rows required in the trailing window for
        a turbulence value to be emitted; otherwise ``NaN``.  Defaults to
        ``504`` (≈2 years).
    shrinkage :
        ``"ledoit-wolf"`` (default) applies sklearn's ``LedoitWolf``
        shrinkage to ``Omega``.  ``None`` uses the empirical covariance.
    refit_every :
        Number of rows between successive refits of ``(mu, Omega)``.
        ``1`` (default) refits at every row — the most rigorous choice.
        Set to ``21`` for monthly refits if compute is constrained; the
        causality contract is preserved either way.

    Attributes (populated after ``fit_transform``)
    ----------------------------------------------
    shrinkage_intensities_ :
        Series of Ledoit-Wolf shrinkage intensities at each row (or 0.0
        when shrinkage is disabled).  ``NaN`` rows correspond to the
        warm-up period.
    """

    def __init__(
        self,
        lookback_days: int = 2520,
        min_periods: int = 504,
        shrinkage: Optional[str] = "ledoit-wolf",
        refit_every: int = 1,
    ) -> None:
        if lookback_days < 2:
            raise ValueError(f"lookback_days must be >= 2, got {lookback_days}")
        if min_periods < 2:
            raise ValueError(f"min_periods must be >= 2, got {min_periods}")
        if min_periods > lookback_days:
            raise ValueError(
                f"min_periods ({min_periods}) cannot exceed lookback_days "
                f"({lookback_days})"
            )
        if shrinkage not in (None, "ledoit-wolf"):
            raise ValueError(
                f"shrinkage must be None or 'ledoit-wolf', got {shrinkage!r}"
            )
        if refit_every < 1:
            raise ValueError(f"refit_every must be >= 1, got {refit_every}")
        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.shrinkage = shrinkage
        self.refit_every = refit_every
        self.shrinkage_intensities_: Optional[pd.Series] = None

    def fit_transform(self, returns: pd.DataFrame) -> pd.Series:
        """Compute the turbulence series.

        Parameters
        ----------
        returns :
            DataFrame of factor returns with a ``DatetimeIndex``.  Rows
            containing any NaN are skipped both as fit observations and as
            evaluation targets (the corresponding output is ``NaN``).

        Returns
        -------
        pd.Series
            Turbulence ``d_t`` indexed by the input index.  Rows whose
            trailing window has fewer than ``min_periods`` finite
            observations are ``NaN``.
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError(
                f"returns must be a pandas DataFrame, got {type(returns).__name__}"
            )
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError(
                f"returns must have a DatetimeIndex, got "
                f"{type(returns.index).__name__}"
            )
        if returns.shape[1] < 2:
            raise ValueError(
                "TurbulenceIndex requires at least 2 factor columns; got "
                f"{returns.shape[1]}"
            )

        n = len(returns)
        arr = returns.values.astype(float)
        d = np.full(n, np.nan)
        intensities = np.full(n, np.nan)

        cached_r: Optional[int] = None
        cached_mu: Optional[np.ndarray] = None
        cached_omega_inv: Optional[np.ndarray] = None
        cached_intensity: float = np.nan
        cached_valid_fit: bool = False

        for t in range(n):
            r = (t // self.refit_every) * self.refit_every

            if r != cached_r:
                cached_r = r
                cached_valid_fit = False
                cached_mu = None
                cached_omega_inv = None
                cached_intensity = np.nan

                fit_start = max(0, r - self.lookback_days)
                fit_end = r
                if fit_end > fit_start:
                    fit_slice = arr[fit_start:fit_end]
                    valid = np.isfinite(fit_slice).all(axis=1)
                    fit_clean = fit_slice[valid]
                    if len(fit_clean) >= self.min_periods:
                        mu, omega_inv, intensity = self._fit_window(fit_clean)
                        if omega_inv is not None:
                            cached_mu = mu
                            cached_omega_inv = omega_inv
                            cached_intensity = intensity
                            cached_valid_fit = True

            if not cached_valid_fit:
                continue

            row = arr[t]
            if not np.isfinite(row).all():
                continue

            diff = row - cached_mu
            d_t = float(diff @ cached_omega_inv @ diff)
            d[t] = d_t
            intensities[t] = cached_intensity

        result = pd.Series(d, index=returns.index, name="turbulence")
        self.shrinkage_intensities_ = pd.Series(
            intensities, index=returns.index, name="shrinkage_intensity",
        )
        return result

    def _fit_window(self, fit_slice: np.ndarray):
        """Estimate ``(mu, Omega^{-1}, shrinkage_intensity)`` on ``fit_slice``.

        Returns ``(None, None, nan)`` if the covariance is too degenerate to
        invert even after shrinkage (extremely rare with Ledoit-Wolf).
        """
        mu = fit_slice.mean(axis=0)
        if self.shrinkage == "ledoit-wolf":
            lw = LedoitWolf().fit(fit_slice)
            omega = lw.covariance_
            intensity = float(lw.shrinkage_)
        else:
            omega = np.cov(fit_slice, rowvar=False, ddof=0)
            intensity = 0.0
        try:
            omega_inv = np.linalg.inv(omega)
        except np.linalg.LinAlgError:
            try:
                omega_inv = np.linalg.pinv(omega)
            except np.linalg.LinAlgError:
                return None, None, np.nan
        return mu, omega_inv, intensity
