"""Unit tests for the RBI explainability layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.grid_prediction import PredictionGrid
from regime_detection.lib.rbi import (
    compute_rbi, compute_rbi_table, compute_tau_statistic,
)
from regime_detection.lib.relevance import RelevanceCalculator


def _gaussian_panel(n: int, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    return pd.DataFrame(
        rng.standard_normal((n, k)),
        index=idx,
        columns=[f"f{j}" for j in range(k)],
    )


# ---------------------------------------------------------------------------
# RelevanceCalculator
# ---------------------------------------------------------------------------


class TestRelevanceCalculator:
    def test_similarity_zero_at_self(self):
        X = _gaussian_panel(200, 3, seed=0)
        y = pd.Series(np.zeros(200), index=X.index, name="y")
        rc = RelevanceCalculator(X, y, shrinkage="ledoit-wolf")
        # At x_t = x_i, similarity = -0.5 * 0' Omega^-1 0 = 0
        x_i = X.iloc[10].values
        assert rc.similarity(x_i, x_i) == pytest.approx(0.0, abs=1e-12)

    def test_informativeness_at_mean_is_zero(self):
        X = _gaussian_panel(200, 3, seed=1)
        y = pd.Series(np.zeros(200), index=X.index)
        rc = RelevanceCalculator(X, y)
        assert rc.informativeness(rc.mean_) == pytest.approx(0.0, abs=1e-12)

    def test_relevance_is_similarity_plus_informativeness(self):
        X = _gaussian_panel(300, 4, seed=2)
        y = pd.Series(np.zeros(300), index=X.index)
        rc = RelevanceCalculator(X, y)
        x_i = X.iloc[5].values
        x_t = X.iloc[150].values
        sim = rc.similarity(x_i, x_t)
        inf = rc.informativeness(x_i)
        rel = rc.relevance(x_i, x_t)
        assert rel == pytest.approx(sim + inf, abs=1e-12)

    def test_relevance_weights_subset_uses_only_selected_columns(self):
        X = _gaussian_panel(200, 4, seed=3)
        y = pd.Series(np.zeros(200), index=X.index)
        rc = RelevanceCalculator(X, y)
        full = rc.relevance_weights(X.iloc[100], variables=list(X.columns))
        sub = rc.relevance_weights(X.iloc[100], variables=["f0", "f1"])
        # Different subsets should produce different relevance values
        # (with very high probability).
        assert not np.allclose(full.dropna().values, sub.dropna().values)

    def test_relevance_weights_no_lookahead_in_construction(self):
        """The RelevanceCalculator must not use the prediction row in
        fitting Omega — verify by checking that adding/removing the
        prediction row from training does not change the mean and cov
        much (it must be a *training* fit only)."""
        X = _gaussian_panel(300, 3, seed=4)
        y = pd.Series(np.zeros(300), index=X.index)
        rc_full = RelevanceCalculator(X, y)
        rc_minus_one = RelevanceCalculator(X.iloc[:-1], y.iloc[:-1])
        # Mean differences should be O(1/n), not zero — that's expected.
        diff = np.max(np.abs(rc_full.mean_ - rc_minus_one.mean_))
        assert diff < 0.05

    def test_censoring_drops_low_relevance(self):
        X = _gaussian_panel(400, 3, seed=5)
        y = pd.Series(np.zeros(400), index=X.index)
        rc = RelevanceCalculator(X, y)
        x_t = X.iloc[200].values
        full = rc.relevance_weights(x_t, censor=0.0).dropna()
        top_half = rc.relevance_weights(x_t, censor=0.5).dropna()
        assert len(top_half) == pytest.approx(0.5 * len(full), abs=2)
        # Every kept observation has relevance >= the median.
        median = full.median()
        assert (top_half >= median - 1e-9).all()

    def test_predict_returns_finite_for_linear_y(self):
        # If y = X @ beta exactly, predict should be close to the truth.
        rng = np.random.default_rng(6)
        n = 400
        X_arr = rng.standard_normal((n, 3))
        beta = np.array([1.0, -0.5, 0.25])
        y_arr = X_arr @ beta
        idx = pd.date_range("2010-01-04", periods=n, freq="B")
        X = pd.DataFrame(X_arr, columns=["a", "b", "c"], index=idx)
        y = pd.Series(y_arr, index=idx)
        rc = RelevanceCalculator(X, y)
        x_t = X.iloc[200].values
        y_hat = rc.predict(x_t, censor=0.5, weighting="exp")
        # Should be in the same ballpark as the true value.
        truth = y.iloc[200]
        assert abs(y_hat - truth) < 1.0


# ---------------------------------------------------------------------------
# PredictionGrid
# ---------------------------------------------------------------------------


class TestPredictionGrid:
    def test_grid_shape(self):
        X = _gaussian_panel(400, 4, seed=10)
        y = pd.Series(np.random.default_rng(10).standard_normal(400), index=X.index)
        grid = PredictionGrid(
            X, y, n_eval=64, n_random=5, censoring_thresholds=(0.0, 0.5),
            random_state=0,
        ).fit()
        n_subsets = len(grid.subsets_)
        n_censors = len(grid.censoring_thresholds_)
        assert len(grid.cells_) == n_subsets * n_censors
        assert grid.adj_fit_.shape == (n_subsets, n_censors)

    def test_grid_default_subsets_include_singletons_and_full(self):
        X = _gaussian_panel(400, 4, seed=11)
        y = pd.Series(np.zeros(400), index=X.index)
        grid = PredictionGrid(X, y, n_eval=32, n_random=3, random_state=0).fit()
        # Singletons.
        singletons = [s for s in grid.subsets_ if len(s) == 1]
        assert len(singletons) == 4
        # Full-feature subset.
        full = tuple(X.columns)
        assert full in grid.subsets_

    def test_grid_predictions_independent_from_eval_outcome(self):
        """Leave-one-out semantics: a cell's prediction at eval row i
        should not be affected if we relabel y[i] alone (the prediction
        averages over all rows except i)."""
        rng = np.random.default_rng(12)
        n = 200
        X = _gaussian_panel(n, 3, seed=12)
        y1 = pd.Series(rng.standard_normal(n), index=X.index)
        y2 = y1.copy()
        # Mutate y at a single row.  We pick a row that we know is in the
        # eval index for random_state=0.
        grid1 = PredictionGrid(X, y1, n_eval=50, n_random=4, random_state=0).fit()
        target_row = grid1.eval_idx_[0]
        y2.iloc[target_row] = y1.iloc[target_row] + 1e6
        grid2 = PredictionGrid(X, y2, n_eval=50, n_random=4, random_state=0).fit()
        # Predictions at all eval rows OTHER than target_row are
        # unchanged.  Predictions at target_row may change because the
        # mean/cov of the training data is rebuilt — but leave-one-out
        # excludes target_row's own y from the kernel mean.
        # Verify: prediction at target_row equals the kernel mean over
        # j != target_row, which uses y1[j] = y2[j] for j != target_row.
        for s_idx in range(len(grid1.subsets_)):
            for c_idx in range(len(grid1.censoring_thresholds_)):
                pred1 = grid1._predictions.iloc[
                    s_idx * len(grid1.censoring_thresholds_) + c_idx, 0
                ]
                pred2 = grid2._predictions.iloc[
                    s_idx * len(grid2.censoring_thresholds_) + c_idx, 0
                ]
                # Tolerance accounts for the small change in mean/cov.
                if np.isfinite(pred1) and np.isfinite(pred2):
                    assert abs(pred2 - pred1) < 0.5


# ---------------------------------------------------------------------------
# compute_rbi
# ---------------------------------------------------------------------------


class TestComputeRBI:
    def test_relevant_variable_has_positive_rbi(self):
        # Construct y to depend strongly on 'f0' only.  RBI(f0) should
        # be > RBI(noise variables).
        rng = np.random.default_rng(20)
        n = 500
        X = _gaussian_panel(n, 4, seed=20)
        y = pd.Series(X["f0"].values * 2.0 + rng.standard_normal(n) * 0.3,
                     index=X.index, name="y")
        grid = PredictionGrid(
            X, y, n_eval=128, n_random=15, random_state=0,
        ).fit()
        rbi_f0 = compute_rbi(grid, "f0")
        rbi_f1 = compute_rbi(grid, "f1")
        rbi_f2 = compute_rbi(grid, "f2")
        rbi_f3 = compute_rbi(grid, "f3")
        # f0 is the only informative variable.
        for noise in (rbi_f1, rbi_f2, rbi_f3):
            assert rbi_f0 > noise, (
                f"Informative variable should outrank noise; got "
                f"f0={rbi_f0:.3f} vs noise={noise:.3f}"
            )

    def test_tau_aligns_with_rbi(self):
        rng = np.random.default_rng(21)
        n = 400
        X = _gaussian_panel(n, 4, seed=21)
        y = pd.Series(X["f1"].values + rng.standard_normal(n) * 0.3,
                     index=X.index)
        grid = PredictionGrid(
            X, y, n_eval=128, n_random=15, random_state=0,
        ).fit()
        rbi = {c: compute_rbi(grid, c) for c in X.columns}
        tau = {c: compute_tau_statistic(grid, c) for c in X.columns}
        # The variable with the largest RBI should also have the largest
        # tau (they share a numerator).
        argmax_rbi = max(rbi, key=rbi.get)
        argmax_tau = max(tau, key=tau.get)
        assert argmax_rbi == argmax_tau == "f1"

    def test_table_sorted_by_rbi(self):
        X = _gaussian_panel(300, 4, seed=22)
        y = pd.Series(X["f0"].values, index=X.index)
        grid = PredictionGrid(
            X, y, n_eval=96, n_random=10, random_state=0,
        ).fit()
        table = compute_rbi_table(grid)
        # Index should be sorted by rbi descending.
        assert table.iloc[0].name == "f0"
        rbi_vals = table["rbi"].values
        assert (np.diff(rbi_vals) <= 1e-9).all()

    def test_unknown_variable_raises(self):
        X = _gaussian_panel(200, 3, seed=23)
        y = pd.Series(np.zeros(200), index=X.index)
        grid = PredictionGrid(X, y, n_eval=32, n_random=3, random_state=0).fit()
        with pytest.raises(KeyError):
            compute_rbi(grid, "nonexistent")


# ---------------------------------------------------------------------------
# Phase 9A — paper-faithful subset_count + deterministic_subsets
# ---------------------------------------------------------------------------


class TestPhase9ASubsetParams:
    """Verify the new subset_count and deterministic_subsets parameters
    introduced in Phase 9A behave as documented.  The Phase 5 default of
    ``n_random=15`` is preserved; new code paths gate the larger
    ``subset_count=100`` (paper-faithful) and the deterministic variant
    that removes the random-subset sampler entirely."""

    def test_n100_grid_shape(self):
        """N=100 produces 25 singletons + 100 random subsets + 1 full
        feature set + 4 censor rows.  Total cells in adj_fit_ are
        n_subsets * n_censors after dedup."""
        X = _gaussian_panel(300, 25, seed=900)
        y = pd.Series(X["f0"].values, index=X.index)
        grid = PredictionGrid(
            X, y, n_eval=24, subset_count=100, random_state=42,
        ).fit()
        # Expect at least 25 singletons + 1 full = 26 mandatory subsets,
        # plus up to 100 random extras (some may dedupe but at this
        # 25-feature scale the chance of collision is negligible).
        assert len(grid.subsets_) >= 100 + 25
        assert len(grid.subsets_) <= 100 + 25 + 1
        assert tuple(X.columns) in set(grid.subsets_), \
            "full-feature subset must be present"
        # Singletons present.
        singletons = {(c,) for c in X.columns}
        assert singletons.issubset(set(grid.subsets_))
        # Censor rows preserved.
        assert list(grid.adj_fit_.columns) == [0.0, 0.2, 0.5, 0.8]

    def test_random_state_reproducible(self):
        """Same random_state → same subsets and same eval_idx across two
        instantiations (deterministic_subsets defaults False)."""
        X = _gaussian_panel(200, 20, seed=901)
        y = pd.Series(X["f1"].values, index=X.index)
        a = PredictionGrid(X, y, n_eval=40, subset_count=30, random_state=42)
        b = PredictionGrid(X, y, n_eval=40, subset_count=30, random_state=42)
        assert a.subsets_ == b.subsets_
        np.testing.assert_array_equal(a.eval_idx_, b.eval_idx_)

    def test_random_state_changes_subsets(self):
        """Different random_state → different random subsets (modulo
        dedup), confirming the seed is actually wired through."""
        X = _gaussian_panel(200, 25, seed=902)
        y = pd.Series(X["f0"].values, index=X.index)
        a = PredictionGrid(X, y, n_eval=40, subset_count=40, random_state=1)
        b = PredictionGrid(X, y, n_eval=40, subset_count=40, random_state=999)
        # The random-extra portion of the subset list must differ.
        # Filter out the singletons and full set common to both.
        common_mandatory = set((c,) for c in X.columns) | {tuple(X.columns)}
        extras_a = [s for s in a.subsets_ if s not in common_mandatory]
        extras_b = [s for s in b.subsets_ if s not in common_mandatory]
        assert extras_a != extras_b, \
            "two different random_state values produced identical subsets"

    def test_deterministic_subsets_invariant_to_random_state(self):
        """deterministic_subsets=True yields the same subset list
        regardless of random_state — the seed only affects the
        eval-row sampler in this branch."""
        X = _gaussian_panel(200, 20, seed=903)
        y = pd.Series(X["f0"].values, index=X.index)
        a = PredictionGrid(
            X, y, n_eval=40, subset_count=30,
            deterministic_subsets=True, random_state=1,
        )
        b = PredictionGrid(
            X, y, n_eval=40, subset_count=30,
            deterministic_subsets=True, random_state=999,
        )
        assert a.subsets_ == b.subsets_, \
            "deterministic mode must not depend on random_state"
        # eval_idx may differ because the row sampler still uses
        # random_state — that is intentional.
        assert not np.array_equal(a.eval_idx_, b.eval_idx_) \
               or len(a.eval_idx_) == len(X), \
            "eval_idx should depend on random_state unless the eval " \
            "set is the full sample"

    def test_deterministic_subsets_are_lex_pairs_first(self):
        """The deterministic enumeration starts at size 2 in
        lexicographic order over the column index, so the first
        non-singleton subset is (col[0], col[1])."""
        X = _gaussian_panel(100, 6, seed=904)
        y = pd.Series(X["f0"].values, index=X.index)
        grid = PredictionGrid(
            X, y, n_eval=20, subset_count=3,
            deterministic_subsets=True, random_state=0,
        )
        cols = list(X.columns)
        # 6 singletons + 3 deterministic pairs + 1 full = 10 subsets.
        non_singleton_non_full = [
            s for s in grid.subsets_
            if len(s) > 1 and s != tuple(cols)
        ]
        assert non_singleton_non_full == [
            (cols[0], cols[1]),
            (cols[0], cols[2]),
            (cols[0], cols[3]),
        ]

    def test_subset_count_overrides_n_random(self):
        """If both subset_count and n_random are passed, subset_count
        wins."""
        X = _gaussian_panel(150, 10, seed=905)
        y = pd.Series(X["f0"].values, index=X.index)
        grid = PredictionGrid(
            X, y, n_eval=20,
            n_random=5,             # back-compat parameter
            subset_count=25,        # new, should win
            random_state=0,
        )
        # 10 singletons + 25 random + 1 full = 36 (dedup-permitting).
        assert grid.subset_count_ == 25
        assert len(grid.subsets_) >= 10 + 25
