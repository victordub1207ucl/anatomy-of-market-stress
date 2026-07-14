"""Directional (signed) turbulence — a semivariance analogue for d_t.

The Mahalanobis turbulence index is sign-blind: a +5-sigma rally and a
-5-sigma crash produce identical ``d_t``.  This module splits ``d_t``
**exactly** into a loss-driven and a gain-driven component, mirroring the
realized-semivariance idea (Barndorff-Nielsen, Kinnebrock & Shephard) at
the cross-sectional level.

With ``diff = y_t - mu_t``, ``z = Omega_t^{-1} diff`` and per-factor
contributions ``c_i = diff_i * z_i`` (which sum to ``d_t``), partition the
factors by the *direction* of their deviation:

    d_loss = sum of c_i over factors deviating in their STRESS direction
    d_gain = sum of the rest                    (d_loss + d_gain == d_t)

By default a factor's stress direction is a below-mean return
(``diff_i < 0``).  Factors listed in ``flip`` (default: ``("vix",)``) have
the convention inverted — an *above*-mean VIX move is the stressful one.

Note ``c_i`` can be negative (a move consistent with the prevailing
correlation structure reduces joint unusualness), so ``d_loss`` is not
guaranteed positive — it is an exact accounting split, not a quadratic
form in its own right.

Causality contract: identical to ``TurbulenceIndex`` (same walk-forward
loop, same ``_fit_window``); the row at ``t`` uses only data before ``t``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from regime_detection.lib.turbulence import TurbulenceIndex


class SignedTurbulence:
    """Walk-forward loss/gain split of Mahalanobis turbulence.

    Parameters mirror :class:`TurbulenceIndex`; ``flip`` lists factors
    whose stress direction is an above-mean (not below-mean) deviation.
    """

    def __init__(
        self,
        lookback_days: int = 2520,
        min_periods: int = 504,
        shrinkage: Optional[str] = "ledoit-wolf",
        refit_every: int = 1,
        flip: Sequence[str] = ("vix",),
    ) -> None:
        self._index = TurbulenceIndex(
            lookback_days=lookback_days, min_periods=min_periods,
            shrinkage=shrinkage, refit_every=refit_every,
        )
        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.shrinkage = shrinkage
        self.refit_every = refit_every
        self.flip = tuple(flip)

    def fit_transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute ``turbulence`` (total), ``turb_loss`` and ``turb_gain``.

        Returns a DataFrame indexed like ``returns``;
        ``turb_loss + turb_gain == turbulence`` exactly on finite rows.
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("returns must have a DatetimeIndex")
        if returns.shape[1] < 2:
            raise ValueError("SignedTurbulence requires at least 2 columns")

        n, k = returns.shape
        arr = returns.values.astype(float)
        # +1 for normal factors (stress = diff < 0); -1 for flipped ones.
        sign = np.ones(k)
        for j, col in enumerate(returns.columns):
            if col in self.flip:
                sign[j] = -1.0

        total = np.full(n, np.nan)
        loss = np.full(n, np.nan)
        gain = np.full(n, np.nan)

        cached_r: Optional[int] = None
        cached_mu = cached_omega_inv = None
        cached_valid = False

        for t in range(n):
            r = (t // self.refit_every) * self.refit_every
            if r != cached_r:
                cached_r, cached_valid = r, False
                cached_mu = cached_omega_inv = None
                fit_start, fit_end = max(0, r - self.lookback_days), r
                if fit_end > fit_start:
                    fit_slice = arr[fit_start:fit_end]
                    fit_clean = fit_slice[np.isfinite(fit_slice).all(axis=1)]
                    if len(fit_clean) >= self.min_periods:
                        mu, omega_inv, _ = self._index._fit_window(fit_clean)
                        if omega_inv is not None:
                            cached_mu, cached_omega_inv = mu, omega_inv
                            cached_valid = True
            if not cached_valid:
                continue
            row = arr[t]
            if not np.isfinite(row).all():
                continue
            diff = row - cached_mu
            c = diff * (cached_omega_inv @ diff)
            stress_side = (sign * diff) < 0
            total[t] = c.sum()
            loss[t] = c[stress_side].sum()
            gain[t] = c[~stress_side].sum()

        return pd.DataFrame(
            {"turbulence": total, "turb_loss": loss, "turb_gain": gain},
            index=returns.index,
        )
