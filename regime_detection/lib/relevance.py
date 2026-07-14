"""Relevance-based weighting for non-parametric prediction.

Reference
---------
Czasonis, M., Kritzman, M. and Turkington, D. (2024), "Relevance-Based
Importance: A Comprehensive Measure of Variable Importance in
Prediction."

Key identities (paper notation)::

    similarity(x_i, x_t) = -1/2 * (x_i - x_t)' Omega^-1 (x_i - x_t)    [Eq 2]
    informativeness(x_i, x_bar) = 1/2 * (x_i - x_bar)' Omega^-1 (x_i - x_bar)   [Eq 3]
    relevance(x_i, x_t) = similarity(x_i, x_t) + informativeness(x_i, x_bar)   [Eq 1]

An observation is *relevant* if it both resembles the prediction point
``x_t`` and is distinctive from the training mean ``x_bar`` — these are
the two ingredients of a useful analogy.

``Omega`` is the covariance estimated **on the training window only**
with Ledoit-Wolf shrinkage; full-sample statistics are never used (this
was a CRITICAL audit finding in the Phase-0 review of the legacy
pipeline).

For non-parametric prediction at ``x_t``, the historical outcomes
``y_i`` are weighted by relevance and averaged to form ``y_hat(x_t)``.
Censoring keeps only the top ``(1 - censor)`` fraction of observations
by relevance — analogous to using only the nearest neighbours.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class RelevanceCalculator:
    """Per-observation relevance for kernel-weighted prediction.

    Parameters
    ----------
    features :
        Training feature matrix ``X`` (n × p).  Mahalanobis statistics
        are fit on this and these statistics only.
    outcomes :
        Training outcomes ``y`` (n,).  Used by :meth:`predict` and the
        downstream grid; not used in the relevance math itself.
    shrinkage :
        ``"ledoit-wolf"`` (default) or ``None`` (empirical covariance).

    Attributes (populated on construction)
    --------------------------------------
    mean_, cov_, cov_inv_ : np.ndarray
        Training-window mean, covariance and its inverse.
    shrinkage_intensity_ : float
        Ledoit-Wolf shrinkage intensity in [0, 1].
    """

    def __init__(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
        shrinkage: Optional[str] = "ledoit-wolf",
    ) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if not isinstance(outcomes, pd.Series):
            raise TypeError("outcomes must be a pandas Series")
        if shrinkage not in (None, "ledoit-wolf"):
            raise ValueError(
                f"shrinkage must be None or 'ledoit-wolf', got {shrinkage!r}"
            )
        # Align on common index, drop NaN rows so the Mahalanobis math is
        # well-defined.
        joined = features.join(outcomes.rename("__y"), how="inner").dropna()
        self.X = joined[features.columns].copy()
        self.y = joined["__y"].copy()
        self.columns_ = list(self.X.columns)

        arr = self.X.values.astype(float)
        self.mean_ = arr.mean(axis=0)
        if shrinkage == "ledoit-wolf":
            lw = LedoitWolf().fit(arr)
            self.cov_ = lw.covariance_
            self.shrinkage_intensity_ = float(lw.shrinkage_)
        else:
            self.cov_ = np.cov(arr, rowvar=False, ddof=0)
            self.shrinkage_intensity_ = 0.0
        try:
            self.cov_inv_ = np.linalg.inv(self.cov_)
        except np.linalg.LinAlgError:
            self.cov_inv_ = np.linalg.pinv(self.cov_)

    # ------------------------------------------------------------------
    # Single-point math (vectorised internally)
    # ------------------------------------------------------------------

    def _coerce_x(self, x: Union[pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(x, pd.Series):
            return x.reindex(self.columns_).values.astype(float)
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self.columns_),):
            raise ValueError(
                f"x must have shape ({len(self.columns_)},); got {x.shape}"
            )
        return x

    def similarity(self, x_i, x_t) -> float:
        """Eq 2 of the paper."""
        d = self._coerce_x(x_i) - self._coerce_x(x_t)
        return float(-0.5 * d @ self.cov_inv_ @ d)

    def informativeness(self, x_i, x_bar=None) -> float:
        """Eq 3 of the paper.  ``x_bar`` defaults to the fitted mean."""
        bar = self.mean_ if x_bar is None else self._coerce_x(x_bar)
        d = self._coerce_x(x_i) - bar
        return float(0.5 * d @ self.cov_inv_ @ d)

    def relevance(self, x_i, x_t) -> float:
        """Eq 1: similarity(x_i, x_t) + informativeness(x_i, x_bar)."""
        return self.similarity(x_i, x_t) + self.informativeness(x_i)

    # ------------------------------------------------------------------
    # Vectorised relevance + prediction
    # ------------------------------------------------------------------

    def _select_subset(self, variables: Optional[Sequence[str]]):
        if variables is None:
            return list(self.columns_), self.cov_inv_, self.mean_, self.X.values
        cols = list(variables)
        unknown = [c for c in cols if c not in self.columns_]
        if unknown:
            raise KeyError(f"unknown feature columns: {unknown}")
        idx = [self.columns_.index(c) for c in cols]
        cov_sub = self.cov_[np.ix_(idx, idx)]
        try:
            inv_sub = np.linalg.inv(cov_sub)
        except np.linalg.LinAlgError:
            inv_sub = np.linalg.pinv(cov_sub)
        mean_sub = self.mean_[idx]
        X_sub = self.X.values[:, idx]
        return cols, inv_sub, mean_sub, X_sub

    def relevance_weights(
        self,
        x_t: Union[pd.Series, np.ndarray],
        variables: Optional[Sequence[str]] = None,
        censor: float = 0.0,
    ) -> pd.Series:
        """Per-historical-row relevance for predicting at ``x_t``.

        ``censor`` ∈ [0, 1) sets the quantile cutoff: rows with
        relevance below this quantile are dropped (NaN in the output).
        """
        if not 0.0 <= censor < 1.0:
            raise ValueError(f"censor must lie in [0, 1); got {censor}")
        cols, inv_sub, mean_sub, X_sub = self._select_subset(variables)
        if isinstance(x_t, pd.Series):
            x_arr = x_t.reindex(cols).values.astype(float)
        else:
            x_arr = np.asarray(x_t, dtype=float)
            if x_arr.shape != (len(cols),):
                raise ValueError(
                    f"x_t must have shape ({len(cols)},); got {x_arr.shape}"
                )

        diff_t = X_sub - x_arr
        sim = -0.5 * np.einsum("ij,jk,ik->i", diff_t, inv_sub, diff_t)
        diff_bar = X_sub - mean_sub
        inf = 0.5 * np.einsum("ij,jk,ik->i", diff_bar, inv_sub, diff_bar)
        rel = sim + inf

        if censor > 0.0:
            cutoff = np.quantile(rel, censor)
            keep = rel >= cutoff
            rel = np.where(keep, rel, np.nan)

        return pd.Series(rel, index=self.X.index, name="relevance")

    def predict(
        self,
        x_t: Union[pd.Series, np.ndarray],
        variables: Optional[Sequence[str]] = None,
        censor: float = 0.0,
        weighting: str = "linear",
    ) -> float:
        """Kernel-weighted prediction at ``x_t``.

        Parameters
        ----------
        weighting :
            ``"linear"`` — weights are ``max(relevance, 0)`` (i.e. negative
            relevance contributes zero).  ``"exp"`` — softmax-style
            ``exp(relevance - max(relevance))`` (numerically stable).
        """
        rel = self.relevance_weights(x_t, variables=variables, censor=censor)
        valid = rel.dropna()
        if len(valid) == 0:
            return float("nan")
        y_valid = self.y.reindex(valid.index)
        if weighting == "linear":
            w = np.clip(valid.values, 0.0, None)
        elif weighting == "exp":
            r = valid.values - valid.values.max()
            w = np.exp(r)
        else:
            raise ValueError(
                f"weighting must be 'linear' or 'exp', got {weighting!r}"
            )
        total = float(w.sum())
        if total <= 0.0:
            return float(y_valid.mean())
        return float((w * y_valid.values).sum() / total)

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        X_t: pd.DataFrame,
        variables: Optional[Sequence[str]] = None,
        censor: float = 0.0,
        weighting: str = "linear",
    ) -> pd.Series:
        """Compute :meth:`predict` for every row of ``X_t``."""
        out = np.full(len(X_t), np.nan)
        for i, (idx, row) in enumerate(X_t.iterrows()):
            out[i] = self.predict(row, variables=variables, censor=censor,
                                  weighting=weighting)
        return pd.Series(out, index=X_t.index, name="rbi_prediction")
