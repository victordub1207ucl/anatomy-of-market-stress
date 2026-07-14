"""Hungarian-algorithm label alignment across GMM refits.

GMM cluster indices are arbitrary: refit at time T may label what used to
be "regime 0" as "regime 3", purely because of EM initialisation order.
Without alignment, any feature built on the integer label series
(``regime_lag1``, one-hot dummies) carries spurious changes at every
refit boundary.

This module implements a greedy across-time alignment: at each refit, the
new cluster means are matched to the previous refit's means by minimising
total Euclidean distance via the Hungarian algorithm.  The permutation is
applied to the new window's labels so the same regime keeps the same
integer over time.

Anchor strategy: greedy-across-time (each refit aligns to its immediate
predecessor) rather than to a single canonical first window.  This is
robust to slow regime-mean drift across the 20-year sample — anchoring to
window 1 would accumulate error.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def hungarian_permutation(
    prev_means: np.ndarray,
    new_means: np.ndarray,
) -> np.ndarray:
    """Permutation that maps each row of ``new_means`` to the closest row
    of ``prev_means`` under the Hungarian (assignment) algorithm.

    Returns
    -------
    np.ndarray of length ``k``
        ``perm[j]`` is the row index of ``prev_means`` that corresponds to
        new cluster ``j``.  Equivalently, applying the permutation
        ``new_means[perm.argsort()]`` returns rows in the order that
        matches ``prev_means``.
    """
    if prev_means.shape != new_means.shape:
        raise ValueError(
            f"prev_means {prev_means.shape} != new_means {new_means.shape}"
        )
    k = prev_means.shape[0]
    # Cost matrix: cost[i, j] = dist(prev_i, new_j)
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost[i, j] = float(np.linalg.norm(prev_means[i] - new_means[j]))
    # linear_sum_assignment returns (row_ind, col_ind) such that
    # cost[row_ind, col_ind].sum() is minimal.  We set perm[col_ind[i]] =
    # row_ind[i] so that "new cluster col" maps to "prev cluster row".
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty(k, dtype=int)
    for r, c in zip(row_ind, col_ind):
        perm[c] = r
    return perm


def align_labels_across_refits(
    label_chunks: List[pd.Series],
    cluster_means: List[np.ndarray],
) -> Tuple[List[pd.Series], List[Dict[int, int]]]:
    """Permute each refit's labels so the same cluster centroid carries
    the same integer label across refits.

    Parameters
    ----------
    label_chunks :
        List of per-refit label series.  ``label_chunks[w]`` covers the
        OOS window for refit ``w``.  Each series has integer labels in
        ``range(k)``.
    cluster_means :
        List of cluster-mean arrays (one per refit), shape ``(k, d)``.
        ``cluster_means[w][j]`` is the mean of cluster ``j`` in refit
        ``w``'s feature space.

    Returns
    -------
    (aligned_chunks, permutations) :
        ``aligned_chunks`` is the same length as ``label_chunks`` with
        labels remapped under the greedy across-time alignment.
        ``permutations[w]`` is the dict ``{old_label: new_label}``
        applied at refit ``w`` (identity at ``w=0``).
    """
    if len(label_chunks) != len(cluster_means):
        raise ValueError(
            f"label_chunks ({len(label_chunks)}) and cluster_means "
            f"({len(cluster_means)}) length mismatch"
        )
    if len(label_chunks) == 0:
        return [], []

    k = cluster_means[0].shape[0]
    for w, m in enumerate(cluster_means):
        if m.shape[0] != k:
            raise ValueError(
                f"cluster_means[{w}] has k={m.shape[0]} but expected k={k}"
            )

    aligned_chunks: List[pd.Series] = []
    permutations: List[Dict[int, int]] = []

    # Window 0: identity.
    aligned_chunks.append(label_chunks[0].copy())
    permutations.append({j: j for j in range(k)})

    # Track the canonical means after window 0's identity permutation.
    canonical_means = cluster_means[0].copy()

    for w in range(1, len(label_chunks)):
        new_means = cluster_means[w]
        perm = hungarian_permutation(canonical_means, new_means)
        # perm[new_j] = canonical_index.  Build {old_label -> new_label}.
        mapping = {int(j): int(perm[j]) for j in range(k)}
        permutations.append(mapping)

        aligned = label_chunks[w].copy()
        aligned = aligned.map(mapping).astype(int)
        aligned_chunks.append(aligned)

        # Update canonical means with the aligned new means (helps with
        # gradual drift — the canonical means are always the most recently
        # aligned set, not a frozen first-window snapshot).
        aligned_new_means = np.empty_like(canonical_means)
        for old_j, new_j in mapping.items():
            aligned_new_means[new_j] = new_means[old_j]
        canonical_means = aligned_new_means

    return aligned_chunks, permutations
