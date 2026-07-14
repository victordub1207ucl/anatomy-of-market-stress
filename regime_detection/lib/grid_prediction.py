"""Prediction grid for Relevance-Based Importance.

Exhibit 1 of Czasonis-Kritzman-Turkington (2024) lays out a grid::

    rows    = censoring thresholds       (e.g. 0.0, 0.2, 0.5, 0.8)
    columns = variable subsets           (singletons + random subsets + full)

For each (subset, censor) cell, the grid stores:

* ``adjusted_fit`` — a goodness-of-fit measure (here: squared Pearson
  correlation between leave-one-out kernel-weighted predictions and the
  true outcomes, evaluated on a sampled subset of the training data).
* ``mean_prediction`` — average ``y_hat`` across the sample.
* ``asymmetry`` — mean of (rel(x_+) - rel(x_-))^2 style upside-vs-downside
  weight imbalance (a placeholder summary, see the paper for the full
  definition).

The grid is the substrate for :func:`regime_detection.lib.rbi.compute_rbi`,
which contrasts cells that include variable ``k`` against cells that do
not.

For simplicity this Phase-5 v1 implementation:

* uses a uniform random sample of ``n_eval`` prediction points (default
  256) rather than all training rows — the per-cell cost is then linear
  in ``n_eval * n_train``;
* uses linear-relevance weighting (``max(rel, 0)``) by default — the
  weighted-OLS extension is straightforward but not needed for ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from regime_detection.lib.relevance import RelevanceCalculator


# ---------------------------------------------------------------------------
# Cell container
# ---------------------------------------------------------------------------


@dataclass
class GridCell:
    """One cell of the prediction grid."""

    subset: Tuple[str, ...]
    censor: float
    adjusted_fit: float
    mean_prediction: float
    mean_weight_share: float
    n_eval: int


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------


def _generate_random_subsets(
    columns: Sequence[str],
    n_random: int,
    subset_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[str, ...]]:
    rng = rng or np.random.default_rng(0)
    p = len(columns)
    if subset_size is None:
        subset_size = max(2, p // 3)
    out = []
    for _ in range(n_random):
        size = rng.integers(2, max(3, p - 1) + 1)
        size = min(int(size), p - 1)
        idx = rng.choice(p, size=size, replace=False)
        out.append(tuple(columns[i] for i in sorted(idx)))
    return out


def _deterministic_subsets(
    columns: Sequence[str],
    n_target: int,
    min_size: int = 2,
) -> List[Tuple[str, ...]]:
    """Enumerate variable subsets in fixed lexicographic order, starting at
    ``min_size``.  Returns up to ``n_target`` subsets, no randomness used.

    Used by the Phase 9A "deterministic" stability variant to remove the
    random-subset sampler as a possible source of cross-window noise.
    Singletons (size 1) are skipped by default because the framework
    always adds them separately.
    """
    cols = list(columns)
    p = len(cols)
    if n_target <= 0:
        return []
    out: List[Tuple[str, ...]] = []
    for size in range(max(1, min_size), p):
        for combo in combinations(range(p), size):
            out.append(tuple(cols[i] for i in combo))
            if len(out) >= n_target:
                return out
    return out


def _build_subsets(
    columns: Sequence[str],
    n_random: int = 20,
    include_singletons: bool = True,
    include_full: bool = True,
    deterministic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[str, ...]]:
    cols = list(columns)
    out: List[Tuple[str, ...]] = []
    if include_singletons:
        out.extend((c,) for c in cols)
    if n_random > 0:
        if deterministic:
            out.extend(_deterministic_subsets(cols, n_random, min_size=2))
        else:
            out.extend(_generate_random_subsets(cols, n_random, rng=rng))
    if include_full:
        out.append(tuple(cols))
    # De-duplicate while preserving order.
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


class PredictionGrid:
    """Build and evaluate the RBI prediction grid.

    Parameters
    ----------
    features, outcomes :
        Training data — passed straight to :class:`RelevanceCalculator`.
        Both must be aligned and free of NaNs (any NaN rows are dropped
        on construction).
    variable_subsets :
        Iterable of column-name tuples.  Each tuple defines one column of
        the grid (a subset of features).  Pass ``None`` to auto-generate
        a default grid: singletons + ``n_random`` random subsets + full
        feature set.
    censoring_thresholds :
        Iterable of floats in ``[0, 1)``.  Each defines one row of the
        grid.  Default: ``(0.0, 0.2, 0.5, 0.8)``.
    n_eval :
        Number of prediction points to sample for the leave-one-out fit
        estimate.  Lower = faster, less stable; higher = slower.
    n_random :
        Number of additional random variable subsets to draw beyond the
        singletons and the full-feature set.  Retained for back-compat;
        prefer ``subset_count`` in new code.
    subset_count :
        Preferred name for ``n_random``.  If both are passed,
        ``subset_count`` wins.  Default ``None`` falls back to
        ``n_random``.  Used by Phase 9A to push the count to ``100``
        (the CKT 2024 paper's value) without breaking existing call
        sites that still pass ``n_random=15``.
    deterministic_subsets :
        If ``True``, the additional subsets are enumerated in
        lexicographic order starting from size 2 (so they do not depend
        on ``random_state``).  Phase 9A robustness variant — isolates
        whether RBI's cross-window instability is driven by random
        subset sampling or by something more fundamental.
    random_state :
        Seed for both the subset sampler (when
        ``deterministic_subsets=False``) and the prediction-point
        sampler.  Always controls the prediction-point sampler.
    shrinkage :
        Forwarded to :class:`RelevanceCalculator`.

    Attributes (populated after ``fit``)
    -------------------------------------
    cells_ : list[GridCell]
        Flat list of fitted grid cells (subset × censor).
    adj_fit_ : pd.DataFrame
        ``censor`` × ``subset`` table of adjusted-fit values.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
        variable_subsets: Optional[Sequence[Tuple[str, ...]]] = None,
        censoring_thresholds: Sequence[float] = (0.0, 0.2, 0.5, 0.8),
        n_eval: int = 256,
        n_random: int = 20,
        subset_count: Optional[int] = None,
        deterministic_subsets: bool = False,
        random_state: int = 0,
        shrinkage: Optional[str] = "ledoit-wolf",
    ) -> None:
        # Resolve effective extra-subset count.  ``subset_count`` is the
        # preferred parameter; ``n_random`` is kept for back-compat with
        # Phase-5 and Phase-7 call sites.
        effective_count = int(subset_count) if subset_count is not None \
            else int(n_random)
        if effective_count < 0:
            raise ValueError(
                f"subset_count / n_random must be >= 0; got {effective_count}"
            )
        self.subset_count_ = effective_count
        self.deterministic_subsets_ = bool(deterministic_subsets)

        rng = np.random.default_rng(random_state)
        self.calculator = RelevanceCalculator(features, outcomes, shrinkage=shrinkage)
        self.columns_ = list(self.calculator.columns_)
        if variable_subsets is None:
            self.subsets_ = _build_subsets(
                self.columns_,
                n_random=effective_count,
                deterministic=self.deterministic_subsets_,
                rng=rng,
            )
        else:
            self.subsets_ = [tuple(s) for s in variable_subsets]
        for s in self.subsets_:
            unknown = [c for c in s if c not in self.columns_]
            if unknown:
                raise KeyError(f"subset references unknown columns: {unknown}")
        self.censoring_thresholds_ = tuple(float(c) for c in censoring_thresholds)
        for c in self.censoring_thresholds_:
            if not 0.0 <= c < 1.0:
                raise ValueError(f"censor must lie in [0, 1); got {c}")
        self.n_eval = int(n_eval)
        self.random_state = int(random_state)

        # Pre-sample the evaluation rows once and reuse across all cells.
        n = len(self.calculator.X)
        if self.n_eval >= n:
            self.eval_idx_ = np.arange(n)
        else:
            self.eval_idx_ = np.sort(rng.choice(n, size=self.n_eval, replace=False))

        # Cached after fit().
        self.cells_: Optional[List[GridCell]] = None
        self.adj_fit_: Optional[pd.DataFrame] = None
        self._predictions: Optional[pd.DataFrame] = None

    def fit(self) -> "PredictionGrid":
        """Compute adjusted-fit for every cell.

        Returns ``self`` for chaining.
        """
        X_arr = self.calculator.X.values
        y_arr = self.calculator.y.values.astype(float)
        n = X_arr.shape[0]
        cols_to_idx = {c: self.columns_.index(c) for c in self.columns_}

        # For each subset, pre-compute the projected design matrix and
        # the corresponding inverse covariance.  Then loop over eval rows
        # (the slow axis) computing a relevance vector once and reusing
        # it across all censoring rows.
        n_subsets = len(self.subsets_)
        n_censors = len(self.censoring_thresholds_)
        n_eval = len(self.eval_idx_)
        # We store one prediction per (subset, censor, eval_row).
        predictions = np.full((n_subsets, n_censors, n_eval), np.nan)

        for s_idx, subset in enumerate(self.subsets_):
            idx = np.array([cols_to_idx[c] for c in subset])
            cov_sub = self.calculator.cov_[np.ix_(idx, idx)]
            try:
                inv_sub = np.linalg.inv(cov_sub)
            except np.linalg.LinAlgError:
                inv_sub = np.linalg.pinv(cov_sub)
            mean_sub = self.calculator.mean_[idx]
            X_sub = X_arr[:, idx]

            # Pre-compute informativeness of every row (depends only on
            # subset, not on x_t).
            diff_bar = X_sub - mean_sub
            inf_all = 0.5 * np.einsum("ij,jk,ik->i", diff_bar, inv_sub, diff_bar)

            for e, eval_i in enumerate(self.eval_idx_):
                x_t = X_sub[eval_i]
                diff_t = X_sub - x_t
                sim_all = -0.5 * np.einsum("ij,jk,ik->i", diff_t, inv_sub, diff_t)
                rel = sim_all + inf_all
                # Leave-one-out: drop eval_i from the weighting set so the
                # prediction does not see its own outcome.
                rel[eval_i] = -np.inf

                for c_idx, censor in enumerate(self.censoring_thresholds_):
                    if censor > 0.0:
                        finite = np.isfinite(rel)
                        if finite.sum() < 2:
                            continue
                        cutoff = np.quantile(rel[finite], censor)
                        keep = rel >= cutoff
                    else:
                        keep = np.isfinite(rel)
                    w = np.where(keep, rel, np.nan)
                    pos = np.clip(w, 0.0, None)
                    pos = np.where(np.isfinite(pos), pos, 0.0)
                    s_w = pos.sum()
                    if s_w <= 0.0:
                        # Fall back to unweighted mean over kept rows.
                        kept_y = y_arr[keep]
                        if len(kept_y) == 0:
                            continue
                        predictions[s_idx, c_idx, e] = float(kept_y.mean())
                    else:
                        predictions[s_idx, c_idx, e] = float(
                            (pos * y_arr).sum() / s_w
                        )

        self._predictions = pd.DataFrame(
            predictions.reshape(n_subsets * n_censors, n_eval),
            index=pd.MultiIndex.from_product(
                [range(n_subsets), self.censoring_thresholds_],
                names=["subset_id", "censor"],
            ),
        )

        y_eval = y_arr[self.eval_idx_]
        cells: List[GridCell] = []
        adj_fit_rows = []
        for s_idx, subset in enumerate(self.subsets_):
            row = {}
            for c_idx, censor in enumerate(self.censoring_thresholds_):
                pred = predictions[s_idx, c_idx]
                finite = np.isfinite(pred)
                if finite.sum() < 5 or np.std(pred[finite]) < 1e-12 \
                        or np.std(y_eval[finite]) < 1e-12:
                    adj_fit = float("nan")
                else:
                    r = float(np.corrcoef(pred[finite], y_eval[finite])[0, 1])
                    adj_fit = r * r if np.isfinite(r) else float("nan")
                mean_pred = float(np.nanmean(pred))
                cells.append(GridCell(
                    subset=subset,
                    censor=float(censor),
                    adjusted_fit=adj_fit,
                    mean_prediction=mean_pred,
                    mean_weight_share=float(1.0 - censor),
                    n_eval=int(finite.sum()),
                ))
                row[censor] = adj_fit
            adj_fit_rows.append(row)

        self.cells_ = cells
        self.adj_fit_ = pd.DataFrame(
            adj_fit_rows,
            index=pd.Index([",".join(s) for s in self.subsets_], name="subset"),
            columns=pd.Index(self.censoring_thresholds_, name="censor"),
        )
        return self
