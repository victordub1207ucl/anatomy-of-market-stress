"""Per-factor decomposition of the Kritzman-Li turbulence index.

Reference
---------
Kritzman, M. and Li, Y. (2010), "Skulls, Financial Turbulence, and Risk
Management", *Financial Analysts Journal* 66(5).  Section "The Sources of
Turbulence" notes that the Mahalanobis distance admits an exact additive
decomposition into per-asset contributions.

The turbulence index at date ``t`` is::

    d_t = (y_t - mu_t)' Omega_t^{-1} (y_t - mu_t)

Let ``diff = y_t - mu_t`` and ``z = Omega_t^{-1} diff``.  Then::

    d_t = diff' z = sum_i diff_i * z_i

so the contribution of factor ``i`` to today's turbulence is::

    c_{i,t} = diff_i * z_i = diff_i * (Omega_t^{-1} diff)_i

These contributions sum **exactly** to ``d_t`` (verified at runtime against
:class:`~regime_detection.lib.turbulence.TurbulenceIndex`).  A
contribution can be negative: a factor whose move is *consistent* with the
prevailing correlation structure can reduce the joint statistical
unusualness even while moving.  The decomposition answers "which factors
drive today's turbulence", which a single scalar cannot.

This module reuses :meth:`TurbulenceIndex._fit_window` and replicates its
walk-forward loop verbatim, so it inherits the same causality contract:
the row at ``t`` depends only on data with index strictly less than ``t``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from regime_detection.lib.turbulence import TurbulenceIndex


class TurbulenceDecomposer:
    """Walk-forward per-factor decomposition of Mahalanobis turbulence.

    The constructor parameters mirror
    :class:`~regime_detection.lib.turbulence.TurbulenceIndex` exactly so
    a decomposition can be produced for an existing turbulence series simply
    by reusing the same arguments.  See that class for parameter semantics.
    """

    def __init__(
        self,
        lookback_days: int = 2520,
        min_periods: int = 504,
        shrinkage: Optional[str] = "ledoit-wolf",
        refit_every: int = 1,
    ) -> None:
        # Delegate validation + window fitting to TurbulenceIndex so the two
        # paths can never diverge.
        self._index = TurbulenceIndex(
            lookback_days=lookback_days,
            min_periods=min_periods,
            shrinkage=shrinkage,
            refit_every=refit_every,
        )
        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.shrinkage = shrinkage
        self.refit_every = refit_every

    def fit_transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute per-factor turbulence contributions.

        Parameters
        ----------
        returns :
            DataFrame of factor returns with a ``DatetimeIndex`` (same input
            as :meth:`TurbulenceIndex.fit_transform`).

        Returns
        -------
        pd.DataFrame
            One column per input factor, indexed by the input index.  Each
            row sums to ``d_t`` for that date; warm-up / NaN-input rows are
            all-``NaN``.
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
                "TurbulenceDecomposer requires at least 2 factor columns; got "
                f"{returns.shape[1]}"
            )

        n, k = returns.shape
        arr = returns.values.astype(float)
        contrib = np.full((n, k), np.nan)

        cached_r: Optional[int] = None
        cached_mu: Optional[np.ndarray] = None
        cached_omega_inv: Optional[np.ndarray] = None
        cached_valid_fit: bool = False

        for t in range(n):
            r = (t // self.refit_every) * self.refit_every

            if r != cached_r:
                cached_r = r
                cached_valid_fit = False
                cached_mu = None
                cached_omega_inv = None

                fit_start = max(0, r - self.lookback_days)
                fit_end = r
                if fit_end > fit_start:
                    fit_slice = arr[fit_start:fit_end]
                    valid = np.isfinite(fit_slice).all(axis=1)
                    fit_clean = fit_slice[valid]
                    if len(fit_clean) >= self.min_periods:
                        mu, omega_inv, _ = self._index._fit_window(fit_clean)
                        if omega_inv is not None:
                            cached_mu = mu
                            cached_omega_inv = omega_inv
                            cached_valid_fit = True

            if not cached_valid_fit:
                continue

            row = arr[t]
            if not np.isfinite(row).all():
                continue

            diff = row - cached_mu
            z = cached_omega_inv @ diff
            contrib[t] = diff * z  # elementwise; sums to diff' Omega^-1 diff

        return pd.DataFrame(contrib, index=returns.index, columns=returns.columns)
