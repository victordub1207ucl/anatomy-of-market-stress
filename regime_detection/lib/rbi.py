"""Relevance-Based Importance — aggregate variable importance.

Builds on :class:`PredictionGrid`.  For each variable ``k`` the RBI score
(Eq 18 of the paper) contrasts the average adjusted-fit of cells whose
subset INCLUDES ``k`` with cells whose subset EXCLUDES ``k``.  A positive
RBI means adding ``k`` to a subset improves out-of-sample fit on average
— i.e. ``k`` is informative.

The tau statistic (Eq 21) re-expresses RBI in t-statistic units so
variables can be ranked on the same scale as classical OLS significance.
The implementation here uses a Welch-style approximation::

    tau_k = mean_with(k) - mean_without(k)
            / sqrt( var_with(k) / n_with + var_without(k) / n_without )

which is the standard two-sample t-statistic and matches the paper's
description for the binary inclusion variable.  The paper's full
formulation also scales by sample size ``K`` and the model ``R²``; here
we expose both the bare ``tau`` and the paper-style scaled version.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from regime_detection.lib.grid_prediction import PredictionGrid


def compute_rbi(grid: PredictionGrid, variable_k: str) -> float:
    """Aggregate RBI score for variable ``k`` (Eq 18 of the paper).

    Returns
    -------
    float
        ``mean(adj_fit | subset includes k) - mean(adj_fit | subset excludes k)``.
    """
    if grid.cells_ is None:
        raise RuntimeError("Call PredictionGrid.fit() before compute_rbi.")
    if variable_k not in grid.columns_:
        raise KeyError(f"unknown variable {variable_k!r}")
    with_k, without_k = [], []
    for cell in grid.cells_:
        if not np.isfinite(cell.adjusted_fit):
            continue
        if variable_k in cell.subset:
            with_k.append(cell.adjusted_fit)
        else:
            without_k.append(cell.adjusted_fit)
    if len(with_k) == 0 or len(without_k) == 0:
        return float("nan")
    return float(np.mean(with_k) - np.mean(without_k))


def compute_tau_statistic(
    grid: PredictionGrid,
    variable_k: str,
    informativeness: Optional[float] = None,
    K: Optional[int] = None,
    r_squared: Optional[float] = None,
) -> float:
    """Two-sample tau (Eq 21 spirit) for variable ``k``.

    The Welch ``t`` between (cells with ``k``) and (cells without ``k``)
    on the adjusted-fit axis.  When the optional paper-style scaling
    arguments are provided, the bare ``t`` is multiplied by
    ``sqrt(K * r_squared * informativeness)`` so the magnitudes match the
    paper's Eq 21 conventions.  Pass ``None`` for any of them to return
    the bare ``t``.
    """
    if grid.cells_ is None:
        raise RuntimeError("Call PredictionGrid.fit() before compute_tau_statistic.")
    if variable_k not in grid.columns_:
        raise KeyError(f"unknown variable {variable_k!r}")
    with_k, without_k = [], []
    for cell in grid.cells_:
        if not np.isfinite(cell.adjusted_fit):
            continue
        if variable_k in cell.subset:
            with_k.append(cell.adjusted_fit)
        else:
            without_k.append(cell.adjusted_fit)
    if len(with_k) < 2 or len(without_k) < 2:
        return float("nan")
    a, b = np.asarray(with_k), np.asarray(without_k)
    s2_a, s2_b = a.var(ddof=1), b.var(ddof=1)
    n_a, n_b = len(a), len(b)
    se = np.sqrt(s2_a / n_a + s2_b / n_b)
    if se <= 0.0:
        return float("nan")
    t = (a.mean() - b.mean()) / se
    if informativeness is not None and K is not None and r_squared is not None:
        scale = float(K * r_squared * informativeness)
        if scale > 0:
            t *= np.sqrt(scale)
    return float(t)


def compute_rbi_table(
    grid: PredictionGrid,
    informativeness: Optional[pd.Series] = None,
    K: Optional[int] = None,
    r_squared: Optional[float] = None,
) -> pd.DataFrame:
    """Compute RBI + tau for every variable, returning a sortable table.

    Parameters
    ----------
    grid :
        Fitted prediction grid.
    informativeness :
        Optional per-variable informativeness scores (e.g. the mean
        single-variable informativeness across the eval sample).  Used
        only if ``K`` and ``r_squared`` are also provided.
    K :
        Number of effective training rows (e.g. the eval-sample size).
    r_squared :
        Model R² (e.g. the full-subset adjusted_fit at censor 0).
    """
    rows = []
    for var in grid.columns_:
        rbi = compute_rbi(grid, var)
        inf_k = (float(informativeness.get(var, np.nan))
                 if informativeness is not None else None)
        tau = compute_tau_statistic(
            grid, var,
            informativeness=inf_k, K=K, r_squared=r_squared,
        )
        rows.append({"variable": var, "rbi": rbi, "tau": tau,
                     "informativeness": inf_k})
    return (
        pd.DataFrame(rows)
        .set_index("variable")
        .sort_values("rbi", ascending=False)
    )
