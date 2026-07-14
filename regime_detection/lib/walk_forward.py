"""Walk-forward rolling-statistics utilities.

These classes enforce a strict causality contract: the transform at row ``t``
depends only on data with index strictly less than ``t`` (or, when
``step > 1``, strictly less than the most recent refit position).

The three classes share the same method signature

    .fit_transform_walk_forward(data, window=None, step=1)

so they can be composed in feature pipelines without leaking full-sample
statistics into out-of-sample evaluation.

Boundary behaviour:
* Rows whose fit window contains fewer than ``min_periods`` finite
  observations return ``NaN``.  This is by design — NaN is the only
  honest answer when there is not enough history.
* When ``step > 1`` (block refit), rows inside a block are transformed
  using statistics fit at the start of that block (refit position).

Design notes for the new Mahalanobis-turbulence pipeline:
* ``RollingStandardizer`` replaces every ``StandardScaler().fit_transform(X)``
  call audited as CRITICAL in Phase 0.
* ``RollingCovariance`` (with Ledoit-Wolf shrinkage) is the substrate for
  the Mahalanobis distance estimator that supersedes the GMM.
* ``RollingPCA`` is included for completeness — the new pipeline avoids
  PCA on the turbulence signal itself, but it is needed for the
  orthogonalisation tiers in the feature pipeline.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.decomposition import PCA

Frame = Union[pd.DataFrame, pd.Series]


def _validate_data(data: Frame, name: str) -> pd.DataFrame:
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(
            f"{name}.fit_transform_walk_forward expects a pandas DataFrame or "
            f"Series; got {type(data).__name__}."
        )
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(
            f"{name}.fit_transform_walk_forward expects a DatetimeIndex; "
            f"got {type(data.index).__name__}."
        )
    if isinstance(data, pd.Series):
        return data.to_frame()
    return data


def _refit_positions(n: int, step: int) -> np.ndarray:
    """Vector mapping each row index 0..n-1 to its refit position
    ``(i // step) * step``."""
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    return (np.arange(n) // step) * step


# ---------------------------------------------------------------------------
# RollingStandardizer
# ---------------------------------------------------------------------------


class RollingStandardizer:
    """Walk-forward standardizer.

    At row ``t`` the per-column mean and standard deviation are computed
    on the trailing ``window`` rows ending at ``t-1`` (or, when
    ``step > 1``, the trailing ``window`` rows ending just before the most
    recent refit position).

    Parameters
    ----------
    window :
        Trailing window length in rows.
    min_periods :
        Minimum non-NaN observations required to emit a transformed row.
        Rows whose window has fewer than this many observations return NaN.
    mode :
        ``"rolling"`` for a fixed-length trailing window or ``"expanding"``
        for everything from the start of the sample up to ``t-1``.
    ddof :
        Delta degrees of freedom for the standard deviation.  Defaults to
        ``0`` to match ``sklearn.preprocessing.StandardScaler``.
    """

    def __init__(
        self,
        window: int = 252,
        min_periods: int = 63,
        mode: str = "rolling",
        ddof: int = 0,
    ) -> None:
        if mode not in ("rolling", "expanding"):
            raise ValueError(f"mode must be 'rolling' or 'expanding', got {mode!r}")
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if min_periods < 2:
            raise ValueError(f"min_periods must be >= 2, got {min_periods}")
        if min_periods > window:
            raise ValueError(
                f"min_periods ({min_periods}) cannot exceed window ({window})"
            )
        if ddof not in (0, 1):
            raise ValueError(f"ddof must be 0 or 1, got {ddof}")
        self.window = window
        self.min_periods = min_periods
        self.mode = mode
        self.ddof = ddof

    def fit_transform_walk_forward(
        self,
        data: Frame,
        window: Optional[int] = None,
        step: int = 1,
    ) -> Frame:
        was_series = isinstance(data, pd.Series)
        df = _validate_data(data, type(self).__name__)
        W = window if window is not None else self.window

        means, stds = self._compute_stats(df, W, step)
        # Avoid division by zero — constant columns produce NaN, not inf.
        safe_std = np.where(np.isfinite(stds) & (stds > 0), stds, np.nan)
        out = (df.values - means) / safe_std

        result = pd.DataFrame(out, index=df.index, columns=df.columns)
        if was_series:
            return result.iloc[:, 0]
        return result

    def _compute_stats(self, df: pd.DataFrame, window: int, step: int):
        n = len(df)
        arr = df.values.astype(float)
        means = np.full_like(arr, np.nan, dtype=float)
        stds = np.full_like(arr, np.nan, dtype=float)

        if step == 1:
            # Fast path: per-row trailing stats via pandas rolling, shifted
            # by one so row t uses [t-W, t-1] rather than [t-W+1, t].
            if self.mode == "rolling":
                roll = df.rolling(window=window, min_periods=self.min_periods)
            else:
                roll = df.expanding(min_periods=self.min_periods)
            means_df = roll.mean().shift(1)
            stds_df = roll.std(ddof=self.ddof).shift(1)
            means = means_df.values
            stds = stds_df.values
            return means, stds

        # Block refit: stats are frozen within each block of length `step`.
        refit = _refit_positions(n, step)
        for r in np.unique(refit):
            fit_start = max(0, r - window) if self.mode == "rolling" else 0
            fit_end = int(r)
            if fit_end - fit_start < self.min_periods:
                continue
            fit_slice = arr[fit_start:fit_end]
            with np.errstate(invalid="ignore"):
                m = np.nanmean(fit_slice, axis=0)
                s = np.nanstd(fit_slice, axis=0, ddof=self.ddof)
            block_mask = refit == r
            means[block_mask] = m
            stds[block_mask] = s
        return means, stds


# ---------------------------------------------------------------------------
# RollingPCA
# ---------------------------------------------------------------------------


class RollingPCA:
    """Walk-forward PCA with periodic refit.

    Components at row ``t`` are fit on the trailing ``window`` rows ending
    at the most recent refit position strictly before ``t``.  Inside a
    refit block the components are frozen, so multiple consecutive rows
    share a basis (useful for stability).

    Parameters
    ----------
    n_components :
        Number of principal components to retain.
    window :
        Trailing window length in rows for each refit.
    min_periods :
        Minimum non-NaN rows required in the fit window for a refit to
        succeed.  Blocks whose fit window is too short return NaN.
    random_state :
        Forwarded to :class:`sklearn.decomposition.PCA`.

    Notes
    -----
    Sign-flipping between refits is **not** corrected here.  The downstream
    consumer should align signs against the previous fit (e.g. by checking
    the inner product of consecutive components) if temporal continuity
    matters.
    """

    def __init__(
        self,
        n_components: int,
        window: int = 252,
        min_periods: int = 63,
        random_state: Optional[int] = None,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if window < n_components:
            raise ValueError(
                f"window ({window}) must be >= n_components ({n_components})"
            )
        if min_periods < n_components:
            raise ValueError(
                f"min_periods ({min_periods}) must be >= n_components "
                f"({n_components})"
            )
        if min_periods > window:
            raise ValueError(
                f"min_periods ({min_periods}) cannot exceed window ({window})"
            )
        self.n_components = n_components
        self.window = window
        self.min_periods = min_periods
        self.random_state = random_state
        self.components_history_: Dict[pd.Timestamp, PCA] = {}

    def fit_transform_walk_forward(
        self,
        data: Frame,
        window: Optional[int] = None,
        step: int = 21,
    ) -> pd.DataFrame:
        df = _validate_data(data, type(self).__name__)
        W = window if window is not None else self.window
        n = len(df)
        arr = df.values.astype(float)
        out = np.full((n, self.n_components), np.nan)
        history: Dict[pd.Timestamp, PCA] = {}

        refit = _refit_positions(n, step)
        for r in np.unique(refit):
            fit_start = max(0, int(r) - W)
            fit_end = int(r)
            fit_slice = arr[fit_start:fit_end]
            valid_rows = np.isfinite(fit_slice).all(axis=1)
            fit_clean = fit_slice[valid_rows]
            if len(fit_clean) < self.min_periods:
                continue
            pca = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
            )
            pca.fit(fit_clean)
            history[df.index[int(r)]] = pca

            block_rows = np.where(refit == r)[0]
            block_data = arr[block_rows]
            finite = np.isfinite(block_data).all(axis=1)
            if finite.any():
                out[block_rows[finite]] = pca.transform(block_data[finite])

        self.components_history_ = history
        columns = [f"PC{i + 1}" for i in range(self.n_components)]
        return pd.DataFrame(out, index=df.index, columns=columns)


# ---------------------------------------------------------------------------
# RollingCovariance
# ---------------------------------------------------------------------------


class RollingCovariance:
    """Walk-forward covariance estimator with optional Ledoit-Wolf shrinkage.

    Each refit produces a covariance matrix valid from the refit timestamp
    until the next refit.  The covariance fitted at refit position ``r``
    uses rows ``[r - window, r)`` — strictly past data relative to the
    block of rows ``[r, r + step)`` to which it applies.

    Parameters
    ----------
    window :
        Trailing window length in rows for each refit.
    min_periods :
        Minimum non-NaN rows required in the fit window for a refit to
        succeed.
    shrinkage :
        One of:

        * ``"ledoit-wolf"`` — :class:`sklearn.covariance.LedoitWolf` (data-
          driven shrinkage intensity).
        * ``None`` — empirical covariance, no shrinkage.
        * A float in ``[0, 1]`` — constant shrinkage toward
          ``(tr(S)/k) * I``.

    Returns
    -------
    dict[pd.Timestamp, pd.DataFrame]
        A mapping from refit timestamp to the symmetric ``k x k``
        covariance matrix that becomes active at that timestamp.  Use
        :meth:`cov_at` to look up the active matrix for an arbitrary date.
    """

    def __init__(
        self,
        window: int = 252,
        min_periods: int = 63,
        shrinkage: Union[str, float, None] = "ledoit-wolf",
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if min_periods < 2:
            raise ValueError(f"min_periods must be >= 2, got {min_periods}")
        if min_periods > window:
            raise ValueError(
                f"min_periods ({min_periods}) cannot exceed window ({window})"
            )
        if isinstance(shrinkage, str):
            if shrinkage != "ledoit-wolf":
                raise ValueError(
                    f"shrinkage string must be 'ledoit-wolf', got {shrinkage!r}"
                )
        elif shrinkage is not None:
            if not (0.0 <= float(shrinkage) <= 1.0):
                raise ValueError(
                    f"shrinkage must be None, 'ledoit-wolf', or in [0,1]; "
                    f"got {shrinkage!r}"
                )
        self.window = window
        self.min_periods = min_periods
        self.shrinkage = shrinkage
        self.covariances_: Dict[pd.Timestamp, pd.DataFrame] = {}
        self.shrinkage_intensities_: Dict[pd.Timestamp, float] = {}

    def fit_transform_walk_forward(
        self,
        data: Frame,
        window: Optional[int] = None,
        step: int = 21,
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        df = _validate_data(data, type(self).__name__)
        W = window if window is not None else self.window
        n = len(df)
        arr = df.values.astype(float)
        columns = list(df.columns)

        covariances: Dict[pd.Timestamp, pd.DataFrame] = {}
        intensities: Dict[pd.Timestamp, float] = {}

        refit = _refit_positions(n, step)
        for r in np.unique(refit):
            fit_start = max(0, int(r) - W)
            fit_end = int(r)
            fit_slice = arr[fit_start:fit_end]
            valid_rows = np.isfinite(fit_slice).all(axis=1)
            fit_clean = fit_slice[valid_rows]
            if len(fit_clean) < self.min_periods:
                continue

            cov_mat, intensity = self._fit_one(fit_clean)
            ts = df.index[int(r)]
            covariances[ts] = pd.DataFrame(cov_mat, index=columns, columns=columns)
            intensities[ts] = intensity

        self.covariances_ = covariances
        self.shrinkage_intensities_ = intensities
        return covariances

    def _fit_one(self, fit_slice: np.ndarray):
        if self.shrinkage == "ledoit-wolf":
            est = LedoitWolf().fit(fit_slice)
            return est.covariance_, float(est.shrinkage_)
        if self.shrinkage is None:
            est = EmpiricalCovariance().fit(fit_slice)
            return est.covariance_, 0.0
        # Constant shrinkage toward scaled identity.
        alpha = float(self.shrinkage)
        emp = np.cov(fit_slice, rowvar=False, ddof=0)
        k = emp.shape[0]
        target = np.eye(k) * (np.trace(emp) / max(k, 1))
        return (1.0 - alpha) * emp + alpha * target, alpha

    def cov_at(self, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Return the covariance matrix active at ``date`` (or ``None`` if
        no refit has happened yet)."""
        if not self.covariances_:
            return None
        keys = pd.Index(sorted(self.covariances_.keys()))
        target = pd.Timestamp(date)
        active = keys[keys <= target]
        if len(active) == 0:
            return None
        return self.covariances_[active[-1]]
