"""Regime-conditional Granger causality for the v5 (causal) framework.

This is a **descriptive characterisation** of the sample: it asks *how the
lead-lag / Granger-causal structure among the factors differs between
turbulent and calm markets*, and *which factors Granger-cause turbulence
escalation*.  It is an interpretation/causality contribution (a stated project
interest, "what drives regime changes"), **not** a predictive feature
feeding the out-of-sample supervised model.

Look-ahead contract
-------------------
A full-sample Granger test is the standard way to report a stylised fact
("equity led credit historically"); it carries no *predictive* look-ahead
because it is never used to make an out-of-sample decision.  The single
real leakage risk in a *regime-conditional* version is the regime labels:
v4 conditioned on a full-sample GMM (a leak).  This module conditions only
on v5's **causal** turbulence regime (rolling-threshold, computed in
``models.turbulence_regimes``), so the conditioning itself is leak-free.
The inputs are factor **log returns** (stationary) and stationary
transforms of turbulence.

Ported and cleaned from ``prevversion/explainability/causal.py``.  The
valuable idea retained verbatim is the contiguous-episode segmentation
with Fisher p-value combination: regime observations are interleaved over
time, so a naive Granger test on scattered dates treats calendar gaps as
unit lags and destroys the autocorrelation structure.  Instead each regime
is split into contiguous episodes, a Granger test is run per episode, and
the per-episode p-values are combined with Fisher's method.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as _stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests


def _granger_min_p(
    pair: pd.DataFrame, max_lag: int, test: str = "ssr_chi2test",
) -> float:
    """Min Granger p-value over lags 1..max_lag for ``pair[:,1] -> pair[:,0]``.

    ``pair`` must have exactly two columns ``[caused, causing]``.  Returns
    ``nan`` on failure.  stdout and the statsmodels ``verbose`` deprecation
    warning are suppressed.
    """
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            res = grangercausalitytests(pair, maxlag=max_lag)
        return float(min(res[l][0][test][1] for l in range(1, max_lag + 1)))
    except Exception:
        return float("nan")


def granger_causality_matrix(
    df: pd.DataFrame, max_lag: int = 5, test: str = "ssr_chi2test",
) -> pd.DataFrame:
    """Pairwise Granger p-value matrix (rows = caused, cols = causing).

    ``M.loc[y, x]`` is the min-over-lags p-value for "x Granger-causes y".
    Diagonal is ``nan``.
    """
    cols = list(df.columns)
    M = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            pair = df[[caused, causing]].dropna()
            if len(pair) < max_lag * 5:
                continue
            M.loc[caused, causing] = _granger_min_p(pair, max_lag, test)
    return M


def _contiguous_episodes(
    dates: pd.DatetimeIndex, min_len: int, gap_days: int = 5,
) -> List[pd.DatetimeIndex]:
    """Split ``dates`` into runs of consecutive trading days (gap <= gap_days),
    keeping only runs of length >= ``min_len``."""
    if len(dates) == 0:
        return []
    s = dates.sort_values()
    diffs = s.to_series().diff().dt.days.fillna(1)
    splits = [0] + list((diffs > gap_days).values.nonzero()[0]) + [len(s)]
    eps = [s[splits[j]:splits[j + 1]] for j in range(len(splits) - 1)]
    return [e for e in eps if len(e) >= min_len]


def _fisher_combine(pvals: Sequence[float]) -> float:
    p = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1.0)
    if len(p) == 1:
        return float(p[0])
    chi2 = -2.0 * np.sum(np.log(p))
    return float(_stats.chi2.sf(chi2, 2 * len(p)))


def regime_conditional_granger(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    *,
    max_lag: int = 3,
    fdr_q: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Granger causality estimated separately within each (causal) regime.

    For each regime the dates are segmented into contiguous episodes; a
    Granger matrix is computed per episode; per-pair p-values are combined
    across episodes with Fisher's method, then BH-FDR corrected.

    Parameters
    ----------
    returns :
        Stationary factor series (log returns), date-indexed.
    regime_labels :
        Causal regime label per date (e.g. v5 binary turbulent/quiet).
    max_lag, fdr_q :
        Granger lag order and Benjamini-Hochberg q-level.

    Returns
    -------
    dict ``regime_name -> DataFrame`` with columns ``causing_factor``,
    ``caused_factor``, ``p_value_raw`` (Fisher-combined), ``p_value_adj``
    (BH), ``n_episodes``, ``significant``.
    """
    reg = regime_labels.dropna()
    common = returns.index.intersection(reg.index)
    returns, reg = returns.loc[common], reg.loc[common]
    cols = list(returns.columns)
    min_len = max_lag * 5

    out: Dict[str, pd.DataFrame] = {}
    for regime in sorted(reg.unique(), key=str):
        dates = reg.index[reg == regime]
        episodes = _contiguous_episodes(dates, min_len)
        if not episodes:
            continue

        pair_pvals: Dict[Tuple[str, str], List[float]] = {}
        for ep in episodes:
            ep_df = returns.loc[ep].dropna()
            if len(ep_df) < max_lag + 2:
                continue
            M = granger_causality_matrix(ep_df, max_lag=max_lag)
            for caused in cols:
                for causing in cols:
                    if caused == causing:
                        continue
                    p = M.loc[caused, causing]
                    if pd.notna(p):
                        pair_pvals.setdefault((causing, caused), []).append(p)

        if not pair_pvals:
            continue
        rows = [{
            "causing_factor": causing, "caused_factor": caused,
            "p_value_raw": _fisher_combine(pl), "n_episodes": len(pl),
        } for (causing, caused), pl in pair_pvals.items()]
        df = pd.DataFrame(rows)
        rej, padj, _, _ = multipletests(df["p_value_raw"].values,
                                        alpha=fdr_q, method="fdr_bh")
        df["p_value_adj"] = padj
        df["significant"] = rej
        out[str(regime)] = df.sort_values("p_value_adj").reset_index(drop=True)
    return out


def granger_drivers_of_turbulence(
    returns: pd.DataFrame,
    turbulence: pd.Series,
    *,
    max_lag: int = 5,
    fdr_q: float = 0.05,
    regime_labels: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Which factor returns Granger-cause turbulence *escalation*.

    The "caused" variable is the change in log-turbulence
    ``d log d_t`` (stationary; turbulence's own persistence is absorbed by
    the Granger test's own-lag terms).  This directly characterises *what
    drives regime changes*.

    If ``regime_labels`` is given, the test is additionally run within each
    regime (Fisher-combined over contiguous episodes); otherwise it is a
    single full-sample test.

    Returns a tidy DataFrame: one row per (factor, scope) with the
    Granger p-value (BH-FDR adjusted within scope).
    """
    dlog_turb = np.log(turbulence.replace(0, np.nan)).diff().rename("dlog_turb")

    def _one_scope(rets: pd.DataFrame, y: pd.Series, scope: str) -> pd.DataFrame:
        data = pd.concat([y, rets], axis=1).dropna()
        if len(data) < max_lag * 5:
            return pd.DataFrame()
        rows = []
        for f in rets.columns:
            pair = data[["dlog_turb", f]]
            rows.append({"factor": f, "scope": scope,
                         "p_value_raw": _granger_min_p(pair, max_lag)})
        df = pd.DataFrame(rows).dropna(subset=["p_value_raw"])
        if df.empty:
            return df
        rej, padj, _, _ = multipletests(df["p_value_raw"].values,
                                        alpha=fdr_q, method="fdr_bh")
        df["p_value_adj"] = padj
        df["significant"] = rej
        return df

    frames = [_one_scope(returns, dlog_turb, "full_sample")]
    if regime_labels is not None:
        reg = regime_labels.dropna()
        common = returns.index.intersection(reg.index)
        for regime in sorted(reg.unique(), key=str):
            mask = reg.loc[common] == regime
            dates = common[mask.values]
            frames.append(_one_scope(returns.loc[dates], dlog_turb.loc[dates],
                                     f"regime_{regime}"))
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)
