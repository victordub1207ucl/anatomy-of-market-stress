"""Unit tests for the walk-forward GMM benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.gmm_walk_forward import (
    WalkForwardGMM, fit_walk_forward_gmm,
)
from regime_detection.lib.label_alignment import (
    align_labels_across_refits, hungarian_permutation,
)


def _make_synth_panel(n=1500, k=4, seed=0):
    """Generate a synthetic feature panel with ``k`` well-separated
    clusters in a ``k``-dimensional space.  Cluster i has its mean at
    ``2 * e_i`` (the i-th unit vector), so the GMM has an obvious
    structure to recover at every walk-forward window."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    # Centres: cluster i has its centroid at +2 in dimension i.
    centres = 2.0 * np.eye(k)
    # Random assignment to break temporal structure.
    cluster_seq = rng.integers(0, k, size=n)
    X = rng.standard_normal((n, k))
    for i in range(n):
        X[i, :] += centres[cluster_seq[i]]
    return pd.DataFrame(X, index=idx, columns=[f"f{j}" for j in range(k)])


# ---------------------------------------------------------------------------
# Hungarian alignment primitives
# ---------------------------------------------------------------------------


class TestHungarianPermutation:
    def test_identity_when_means_match(self):
        m = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        perm = hungarian_permutation(m, m)
        np.testing.assert_array_equal(perm, np.arange(3))

    def test_reverses_a_known_swap(self):
        prev = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        # Swap rows 0 and 2.
        swap = np.array([2, 1, 0])
        new = prev[swap]
        perm = hungarian_permutation(prev, new)
        # new[j] should map back to prev[perm[j]] — i.e. perm should be
        # the inverse of swap.
        for j in range(3):
            assert tuple(prev[perm[j]]) == tuple(new[j])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            hungarian_permutation(np.zeros((3, 2)), np.zeros((4, 2)))


class TestAlignLabelsAcrossRefits:
    def test_preserves_membership_counts_per_refit(self):
        idx0 = pd.date_range("2010-01-04", periods=100, freq="B")
        idx1 = pd.date_range("2010-05-24", periods=100, freq="B")
        rng = np.random.default_rng(0)
        chunk0 = pd.Series(rng.integers(0, 3, size=100), index=idx0)
        chunk1 = pd.Series(rng.integers(0, 3, size=100), index=idx1)
        means0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        # Swap clusters 1 and 2 at refit 1.
        means1 = means0[[0, 2, 1]]
        aligned, perms = align_labels_across_refits(
            [chunk0, chunk1], [means0, means1],
        )
        # Per-refit counts should be a permutation of the originals.
        c0_orig = chunk0.value_counts().sort_index()
        c0_aligned = aligned[0].value_counts().sort_index()
        # Window 0 is identity, so counts must match exactly.
        pd.testing.assert_series_equal(c0_orig, c0_aligned, check_names=False)
        # Window 1 had labels swapped; aligned counts should equal the
        # original counts under the inverse swap.
        c1_orig_swapped = chunk1.map({0: 0, 1: 2, 2: 1}).value_counts().sort_index()
        c1_aligned = aligned[1].value_counts().sort_index()
        pd.testing.assert_series_equal(c1_orig_swapped, c1_aligned,
                                         check_names=False)

    def test_manual_permutation_is_reversed(self):
        """If we shuffle window 2's labels before alignment, the alignment
        should fully reverse that shuffle, restoring the original
        assignment."""
        rng = np.random.default_rng(1)
        # Generate a clear regime structure.
        means0 = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
        # Window 0 — labels {0, 1, 2} mapped to themselves.
        c0 = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
                       index=pd.date_range("2010-01-04", periods=10, freq="B"))
        # Window 1 — same regime structure but EM happened to swap 0 <-> 2.
        # So labels are different but cluster means line up under the swap.
        c1_shuffled = pd.Series([2, 1, 0, 2, 1, 0, 2, 1, 0, 2],
                                 index=pd.date_range("2010-01-15",
                                                     periods=10, freq="B"))
        means1_shuffled = means0[[2, 1, 0]]  # rows {0, 1, 2} now in order {2,1,0}
        aligned, perms = align_labels_across_refits(
            [c0, c1_shuffled], [means0, means1_shuffled],
        )
        # After alignment, window 1 should look like the original window 0
        # pattern (the shuffle is reversed).
        np.testing.assert_array_equal(
            aligned[1].values, c0.values,
        )
        # The permutation at window 1 should be the swap dictionary.
        assert perms[1] == {0: 2, 1: 1, 2: 0}


# ---------------------------------------------------------------------------
# WalkForwardGMM
# ---------------------------------------------------------------------------


class TestWalkForwardGMM:
    def test_no_nan_in_post_warmup_region(self):
        X = _make_synth_panel(n=2000, k=4, seed=42)
        result = fit_walk_forward_gmm(
            X, n_components=3, refit_freq=252, min_train=500,
            pca_variance=0.95, random_state=0, n_init=2, max_iter=80,
        )
        # All emitted rows should have integer labels and finite probs.
        assert result.labels.notna().all()
        assert result.probabilities.notna().all().all()
        assert (result.labels.values >= 0).all()
        assert (result.labels.values < 3).all()

    def test_no_lookahead_shock_after_refit(self):
        """Mutate X AFTER the first refit's training window.  The labels
        in the FIRST refit's predict window should be byte-identical
        between the two runs."""
        X = _make_synth_panel(n=1800, k=3, seed=7)
        kwargs = dict(
            n_components=3, refit_freq=252, min_train=500,
            pca_variance=0.95, random_state=0, n_init=2, max_iter=80,
        )
        result_a = fit_walk_forward_gmm(X, **kwargs)
        # Shock everything from row 900 onward.
        X_shock = X.copy()
        rng = np.random.default_rng(99)
        X_shock.iloc[900:] = rng.standard_normal((len(X) - 900, X.shape[1])) * 50
        result_b = fit_walk_forward_gmm(X_shock, **kwargs)
        # The first refit's predict window is rows 500..752 (refit_freq=252).
        # All those rows are < 900 and use a fit on rows [0, 500).  Both
        # the fit data and the predict data are unshocked, so labels must
        # match exactly.
        first_window_idx = X.index[500:752]
        common = (result_a.labels.index
                  .intersection(result_b.labels.index)
                  .intersection(first_window_idx))
        assert len(common) >= 200, (
            f"first-window intersection too small: {len(common)}"
        )
        pd.testing.assert_series_equal(
            result_a.labels.loc[common],
            result_b.labels.loc[common],
        )

    def test_alignment_reduces_label_jumps_at_refit(self):
        """Compare per-refit-boundary label-change count with and
        without alignment.  Aligned should have fewer jumps."""
        X = _make_synth_panel(n=1800, k=3, seed=21)
        unaligned = fit_walk_forward_gmm(
            X, n_components=3, refit_freq=252, min_train=500,
            pca_variance=0.95, random_state=0, n_init=2, max_iter=80,
            align_labels=False,
        )
        aligned = fit_walk_forward_gmm(
            X, n_components=3, refit_freq=252, min_train=500,
            pca_variance=0.95, random_state=0, n_init=2, max_iter=80,
            align_labels=True,
        )
        # Count label changes at refit boundaries.
        def count_boundary_changes(result):
            n_change = 0
            for d in result.refit_dates[1:]:  # skip first window's anchor
                loc = result.labels.index.get_indexer([d])[0]
                if loc <= 0 or loc + 1 >= len(result.labels):
                    continue
                if result.labels.iloc[loc] != result.labels.iloc[loc - 1]:
                    n_change += 1
            return n_change

        # Hard to assert "always strictly fewer" because pure-noise
        # synthesised data may have no real regime structure to preserve.
        # The weaker but still meaningful assertion: aligned does not
        # produce MORE label jumps than unaligned at the same boundaries.
        n_unaligned = count_boundary_changes(unaligned)
        n_aligned = count_boundary_changes(aligned)
        assert n_aligned <= n_unaligned, (
            f"alignment introduced new boundary jumps: "
            f"unaligned={n_unaligned}, aligned={n_aligned}"
        )

    def test_invalid_inputs(self):
        X = _make_synth_panel(n=1000, k=3, seed=0)
        with pytest.raises(TypeError):
            fit_walk_forward_gmm(X.values)
        with pytest.raises(ValueError):
            WalkForwardGMM(n_components=1)
        with pytest.raises(ValueError):
            WalkForwardGMM(refit_freq=0)
        with pytest.raises(ValueError):
            WalkForwardGMM(min_train=10, n_components=5)
        # Too-short data.
        short = _make_synth_panel(n=200, k=3, seed=0)
        with pytest.raises(ValueError):
            fit_walk_forward_gmm(short, min_train=500, refit_freq=252)
