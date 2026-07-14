"""An empirical taxonomy of market-stress episodes from decomposed turbulence.

Kritzman-Li (2010) note that the Mahalanobis turbulence index decomposes
into per-asset contributions; ``models.turbulence_decomposition`` computes
that split exactly.  This module takes the next step: it extracts **all**
high-turbulence episodes (not just famous crises), characterises each by
its per-factor *composition* vector, clusters the episodes into an
empirical taxonomy of stress types (credit-led, vol-led, rates-led, ...),
and tests whether the composition of a spike is *predictable* from the
composition of incipient turbulence just before it — the "what kind of
stress is coming" question (actionable: it tells you *which hedge*).

All inputs are causal v5 outputs; the threshold defining an episode is a
trailing rolling quantile shifted by one day, the same convention as
``supervised.turbulence_bucket``.  The taxonomy itself is a descriptive,
in-sample characterisation (like the causality module), not a predictor.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score


def extract_episodes(
    turbulence: pd.Series,
    *,
    quantile: float = 0.90,
    window: int = 2520,
    min_periods: int = 504,
    min_len: int = 5,
    max_gap: int = 5,
) -> pd.DataFrame:
    """Contiguous high-turbulence episodes under a causal threshold.

    A day is "hot" if ``d_{t-1}`` exceeds the trailing ``quantile`` of the
    preceding ``window`` days (both shifted, so the flag at ``t`` uses only
    data through ``t-1``).  Hot runs separated by ``<= max_gap`` business
    days are merged; runs shorter than ``min_len`` days are dropped.

    Returns one row per episode: ``start``, ``end``, ``n_days``,
    ``peak_turb``, ``mean_turb``.
    """
    thr = (turbulence.rolling(window, min_periods=min_periods)
           .quantile(quantile).shift(1))
    hot = (turbulence.shift(1) > thr) & thr.notna()
    dates = turbulence.index[hot.values]
    if len(dates) == 0:
        return pd.DataFrame(columns=["start", "end", "n_days",
                                     "peak_turb", "mean_turb"])

    runs: list[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = prev = dates[0]
    for d in dates[1:]:
        # business-day gap between consecutive hot days
        gap = np.busday_count(prev.date(), d.date())
        if gap > max_gap:
            runs.append((start, prev))
            start = d
        prev = d
    runs.append((start, prev))

    rows = []
    for s, e in runs:
        seg = turbulence.loc[s:e].dropna()
        if len(seg) < min_len:
            continue
        rows.append({"start": s, "end": e, "n_days": len(seg),
                     "peak_turb": float(seg.max()),
                     "mean_turb": float(seg.mean())})
    return pd.DataFrame(rows)


def episode_composition(
    contributions: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Per-episode composition: each factor's share of the episode's total
    turbulence (rows sum to 1).  Indexed by episode start date."""
    rows, idx = [], []
    for _, ep in episodes.iterrows():
        seg = contributions.loc[ep["start"]:ep["end"]].dropna(how="all")
        if seg.empty:
            continue
        mean_c = seg.mean(axis=0)
        total = mean_c.sum()
        if total <= 0:
            continue
        rows.append(mean_c / total)
        idx.append(ep["start"])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="start"))


def cluster_episodes(
    composition: pd.DataFrame,
    k_range: Sequence[int] = (2, 3, 4, 5, 6),
    random_state: int = 0,
) -> Tuple[pd.Series, int, np.ndarray, Dict[int, str]]:
    """Ward hierarchical clustering of composition vectors.

    ``k`` is chosen by silhouette score over ``k_range``.  Returns
    ``(labels, k, linkage_matrix, names)`` where ``names`` maps cluster id
    to a label built from its centroid's dominant factors
    (e.g. ``"vix-led"``, ``"credit+equity-led"``).
    """
    X = composition.values
    Z = linkage(X, method="ward")
    best_k, best_score, best_labels = None, -np.inf, None
    for k in k_range:
        if k >= len(composition):
            continue
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    labels = pd.Series(best_labels, index=composition.index, name="cluster")

    names: Dict[int, str] = {}
    for c in np.unique(best_labels):
        centroid = composition[labels == c].mean(axis=0).sort_values(
            ascending=False)
        top = [centroid.index[0]]
        if len(centroid) > 1 and centroid.iloc[1] > 0.6 * centroid.iloc[0]:
            top.append(centroid.index[1])
        names[int(c)] = "+".join(top) + "-led"
    return labels, int(best_k), Z, names


def composition_predictability(
    contributions: pd.DataFrame,
    episodes: pd.DataFrame,
    composition: pd.DataFrame,
    *,
    lead_window: int = 21,
    n_perm: int = 10000,
    random_state: int = 0,
) -> Dict[str, float]:
    """Is the composition of a spike predictable from incipient turbulence?

    For each episode, the *pre-vector* is the normalised mean contribution
    over the ``lead_window`` days ending the day before the episode starts
    (strictly pre-episode, hence causal).  We measure the cosine
    similarity between pre-vector and realised episode composition, and
    compare the mean against a permutation null in which realised
    compositions are shuffled across episodes.

    Returns mean observed cosine, mean null cosine, and the permutation
    p-value for "pre-spike composition predicts spike composition".
    """
    rng = np.random.default_rng(random_state)
    pre, real = [], []
    for start in composition.index:
        loc = contributions.index.searchsorted(start)
        seg = contributions.iloc[max(0, loc - lead_window):loc].dropna(how="all")
        if seg.empty:
            continue
        v = seg.mean(axis=0)
        if v.sum() <= 0:
            continue
        pre.append((v / v.sum()).values)
        real.append(composition.loc[start].values)
    P, R = np.asarray(pre), np.asarray(real)

    def _cos(a, b):
        na = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return (a * b).sum(axis=1) / np.where(na == 0, np.nan, na)

    obs = float(np.nanmean(_cos(P, R)))
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.nanmean(_cos(P, R[rng.permutation(len(R))]))
    return {
        "n_episodes": int(len(P)),
        "mean_cosine_observed": obs,
        "mean_cosine_null": float(null.mean()),
        "p_value": float((null >= obs).mean()),
    }
