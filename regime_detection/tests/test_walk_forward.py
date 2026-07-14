"""Unit tests for the walk-forward infrastructure.

Each test pins down the central causality contract: the value at row ``t``
must depend only on data with index strictly less than ``t`` (or strictly
less than the most recent refit position when ``step > 1``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.splits import (
    DEV_END,
    OOS_START,
    assert_no_oos_contamination,
    get_dev_test_split,
)
from regime_detection.lib.walk_forward import (
    RollingCovariance,
    RollingPCA,
    RollingStandardizer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gaussian_panel(n: int, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    return pd.DataFrame(
        rng.standard_normal((n, k)),
        index=idx,
        columns=[f"f{j}" for j in range(k)],
    )


# ---------------------------------------------------------------------------
# splits.py
# ---------------------------------------------------------------------------


class TestSplits:
    def test_dev_end_oos_start_are_one_day_apart(self):
        assert OOS_START == DEV_END + pd.Timedelta(days=1)

    def test_get_dev_test_split_disjoint_and_covers_input(self):
        df = _gaussian_panel(2_500, 3, seed=1)
        dev, oos = get_dev_test_split(df)
        assert dev.index.max() <= DEV_END
        assert oos.index.min() >= OOS_START
        # No row appears in both halves.
        overlap = dev.index.intersection(oos.index)
        assert len(overlap) == 0
        # Together they cover every row of df.
        recombined = pd.concat([dev, oos]).sort_index()
        pd.testing.assert_frame_equal(recombined, df.sort_index())

    def test_get_dev_test_split_rejects_non_datetime_index(self):
        df = pd.DataFrame({"x": [1.0, 2.0]}, index=[0, 1])
        with pytest.raises(TypeError):
            get_dev_test_split(df)

    def test_assert_no_oos_contamination_passes_for_dev_only(self):
        # Dev-only under the fixed 2020-01-01 OOS constraint must end <= 2019-12-31.
        idx = pd.date_range("2017-01-01", "2019-06-30", freq="B")
        # Should not raise.
        assert_no_oos_contamination(idx)

    def test_assert_no_oos_contamination_raises_when_oos_present(self):
        idx = pd.date_range("2018-01-01", "2022-06-30", freq="B")
        with pytest.raises(AssertionError):
            assert_no_oos_contamination(idx)

    def test_assert_no_oos_contamination_accepts_empty_index(self):
        empty = pd.DatetimeIndex([])
        assert_no_oos_contamination(empty)


# ---------------------------------------------------------------------------
# RollingStandardizer
# ---------------------------------------------------------------------------


class TestRollingStandardizer:
    def test_step1_matches_manual_trailing_stats(self):
        data = _gaussian_panel(200, 3, seed=10)
        W, MP = 30, 10
        out = RollingStandardizer(window=W, min_periods=MP).fit_transform_walk_forward(
            data, step=1,
        )

        # Pick a row well past the warm-up.
        for p in (40, 100, 199):
            fit_slice = data.iloc[p - W : p]  # rows [p-W, p-1] inclusive
            expected = (data.iloc[p] - fit_slice.mean()) / fit_slice.std(ddof=0)
            np.testing.assert_allclose(
                out.iloc[p].values, expected.values, atol=1e-12,
            )

    def test_step1_warmup_rows_are_nan(self):
        data = _gaussian_panel(100, 2, seed=11)
        out = RollingStandardizer(window=20, min_periods=15).fit_transform_walk_forward(
            data, step=1,
        )
        # Row 0 has no past data; rows 0..min_periods-1 cannot meet the
        # threshold either (rolling().shift(1) ensures the first row whose
        # trailing window has min_periods observations is row min_periods).
        assert out.iloc[:15].isna().all().all()
        # Rows from min_periods onward must be finite.
        assert out.iloc[15:].notna().all().all()

    def test_no_lookahead_when_future_is_perturbed(self):
        """The single strongest causality check: standardize twice, perturbing
        only the tail of the input.  Every row whose trailing window predates
        the perturbation must be byte-identical between the two runs."""
        data = _gaussian_panel(500, 4, seed=12)
        modify_from = 300
        data_perturbed = data.copy()
        data_perturbed.iloc[modify_from:] += 1e6  # huge shock in the future

        std = RollingStandardizer(window=60, min_periods=30)
        out_full = std.fit_transform_walk_forward(data, step=1)
        out_pert = std.fit_transform_walk_forward(data_perturbed, step=1)

        # The standardized value at row p depends on (a) the row's own raw
        # value and (b) the trailing-window stats [p-W, p-1].  For rows
        # p < modify_from BOTH inputs are unchanged, so the output must be
        # byte-identical.  Row p == modify_from has its raw value perturbed
        # even though its trailing window is still clean — that's not a
        # leak, it's just a perturbed input flowing through a causal
        # transform.
        pd.testing.assert_frame_equal(
            out_full.iloc[:modify_from],
            out_pert.iloc[:modify_from],
        )
        # And the perturbed tail must differ on at least one cell.
        assert not out_full.iloc[modify_from:].equals(out_pert.iloc[modify_from:])

    def test_step_block_mode_uses_block_start_stats(self):
        data = _gaussian_panel(252, 3, seed=13)
        W, MP, step = 60, 30, 21
        out = RollingStandardizer(
            window=W, min_periods=MP,
        ).fit_transform_walk_forward(data, step=step)

        # Pick a refit block well inside the sample.
        r = 4 * step  # refit position = 84
        fit_slice = data.iloc[r - W : r]
        expected_mean = fit_slice.mean()
        expected_std = fit_slice.std(ddof=0)

        for offset in range(step):
            row = data.iloc[r + offset]
            expected = (row - expected_mean) / expected_std
            np.testing.assert_allclose(
                out.iloc[r + offset].values, expected.values, atol=1e-12,
            )

    def test_step_block_mode_no_lookahead(self):
        """Block mode must satisfy the same prefix-invariance property as
        step=1 — every row inside block k uses stats from <= row k*step - 1.
        """
        data = _gaussian_panel(400, 3, seed=14)
        step = 21
        modify_from = 4 * step  # perturb starting at the 5th refit boundary
        data_perturbed = data.copy()
        data_perturbed.iloc[modify_from:] += 1e6

        std = RollingStandardizer(window=60, min_periods=30)
        out_full = std.fit_transform_walk_forward(data, step=step)
        out_pert = std.fit_transform_walk_forward(data_perturbed, step=step)

        # Rows in blocks ending at refit positions <= modify_from use only
        # data with index < modify_from.  The first refit affected is the
        # one at modify_from (which is the start of the next block).  So
        # rows [0, modify_from - 1] inclusive must match.
        pd.testing.assert_frame_equal(
            out_full.iloc[:modify_from],
            out_pert.iloc[:modify_from],
        )

    def test_expanding_mode_uses_all_history(self):
        data = _gaussian_panel(150, 2, seed=15)
        out = RollingStandardizer(
            window=10_000, min_periods=10, mode="expanding",
        ).fit_transform_walk_forward(data, step=1)
        # At row p, mean = data[:p].mean(), std = data[:p].std(ddof=0)
        p = 80
        expected = (data.iloc[p] - data.iloc[:p].mean()) / data.iloc[:p].std(ddof=0)
        np.testing.assert_allclose(out.iloc[p].values, expected.values, atol=1e-12)

    def test_constant_column_produces_nan(self):
        data = _gaussian_panel(100, 2, seed=16)
        data["f1"] = 7.0  # constant column → std = 0
        out = RollingStandardizer(window=20, min_periods=15).fit_transform_walk_forward(
            data, step=1,
        )
        # Constant column standardized values are all NaN past the warm-up.
        assert out.iloc[15:]["f1"].isna().all()
        # Non-constant column is finite past the warm-up.
        assert out.iloc[15:]["f0"].notna().all()

    def test_series_input_returns_series(self):
        data = _gaussian_panel(120, 1, seed=17).iloc[:, 0]
        out = RollingStandardizer(window=20, min_periods=15).fit_transform_walk_forward(
            data, step=1,
        )
        assert isinstance(out, pd.Series)
        assert out.name == data.name

    def test_invalid_init_raises(self):
        with pytest.raises(ValueError):
            RollingStandardizer(window=1)
        with pytest.raises(ValueError):
            RollingStandardizer(min_periods=1)
        with pytest.raises(ValueError):
            RollingStandardizer(window=20, min_periods=30)
        with pytest.raises(ValueError):
            RollingStandardizer(mode="cumulative")


# ---------------------------------------------------------------------------
# RollingPCA
# ---------------------------------------------------------------------------


class TestRollingPCA:
    def test_components_match_manual_fit_per_block(self):
        from sklearn.decomposition import PCA

        data = _gaussian_panel(400, 5, seed=20)
        W, MP, step, K = 60, 30, 21, 2

        rolling_pca = RollingPCA(
            n_components=K, window=W, min_periods=MP, random_state=0,
        )
        out = rolling_pca.fit_transform_walk_forward(data, step=step)

        # At refit position r=84 (block 4), components are fit on rows
        # [r-W, r) = [24, 84).  Compare transformed values of rows in the
        # block against a manually fit PCA.
        r = 4 * step
        fit_slice = data.iloc[r - W : r]
        manual = PCA(n_components=K, random_state=0).fit(fit_slice.values)
        for offset in range(step):
            row = data.iloc[r + offset].values.reshape(1, -1)
            expected = manual.transform(row).ravel()
            actual = out.iloc[r + offset].values
            # PCA can have sign ambiguity but with random_state=0 and the
            # same fit data, sklearn returns deterministic signs.
            np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_no_lookahead_when_future_is_perturbed(self):
        data = _gaussian_panel(400, 4, seed=21)
        step, W, MP, K = 21, 60, 30, 2
        modify_from = 5 * step  # 5th refit boundary

        # IMPORTANT: each call to fit_transform_walk_forward MUST reuse a
        # fresh estimator to avoid state from prior runs.  We use two
        # separate instances.
        out_full = RollingPCA(
            n_components=K, window=W, min_periods=MP, random_state=0,
        ).fit_transform_walk_forward(data, step=step)

        data_perturbed = data.copy()
        data_perturbed.iloc[modify_from:] += 1e6
        out_pert = RollingPCA(
            n_components=K, window=W, min_periods=MP, random_state=0,
        ).fit_transform_walk_forward(data_perturbed, step=step)

        # Rows in blocks whose refit position is <= modify_from use only
        # past data.  Refit positions are 0, step, 2*step, ...; the block
        # containing modify_from has refit position == modify_from, which
        # uses [modify_from - W, modify_from) — strictly past data.  So all
        # blocks whose refit_pos <= modify_from are unaffected, i.e. rows
        # [0, modify_from + step) are clean as long as their refit_pos
        # is <= modify_from.
        #
        # Conservative check: rows [0, modify_from) are guaranteed clean
        # (their refit positions are < modify_from, strictly).
        pd.testing.assert_frame_equal(
            out_full.iloc[:modify_from],
            out_pert.iloc[:modify_from],
        )

    def test_warmup_rows_are_nan(self):
        data = _gaussian_panel(200, 3, seed=22)
        K = 2
        out = RollingPCA(
            n_components=K, window=30, min_periods=20,
        ).fit_transform_walk_forward(data, step=10)
        # First refit position is 0 → fit slice empty → all NaN.
        # Next refit position is 10 → still < min_periods (20) → NaN.
        # First viable refit position is 20.
        assert out.iloc[:20].isna().all().all()
        # From row 20 onward, every block should be finite.
        assert out.iloc[20:].notna().all().all()

    def test_components_history_populated(self):
        data = _gaussian_panel(300, 4, seed=23)
        rpca = RollingPCA(n_components=2, window=60, min_periods=30, random_state=0)
        _ = rpca.fit_transform_walk_forward(data, step=21)
        assert len(rpca.components_history_) > 0
        # All keys are timestamps from the input index.
        for ts in rpca.components_history_:
            assert ts in data.index

    def test_output_columns_named_pc(self):
        data = _gaussian_panel(150, 4, seed=24)
        out = RollingPCA(
            n_components=3, window=60, min_periods=30,
        ).fit_transform_walk_forward(data, step=21)
        assert list(out.columns) == ["PC1", "PC2", "PC3"]


# ---------------------------------------------------------------------------
# RollingCovariance
# ---------------------------------------------------------------------------


class TestRollingCovariance:
    def test_returns_dict_keyed_by_refit_timestamps(self):
        data = _gaussian_panel(300, 4, seed=30)
        cov = RollingCovariance(window=60, min_periods=30).fit_transform_walk_forward(
            data, step=21,
        )
        assert isinstance(cov, dict)
        assert len(cov) > 0
        for ts, mat in cov.items():
            assert isinstance(ts, pd.Timestamp)
            assert ts in data.index
            assert mat.shape == (4, 4)
            assert list(mat.columns) == list(data.columns)
            # Symmetric.
            np.testing.assert_allclose(mat.values, mat.values.T, atol=1e-10)
            # Positive semi-definite (eigenvalues >= 0 within float tol).
            eigvals = np.linalg.eigvalsh(mat.values)
            assert eigvals.min() >= -1e-8

    def test_ledoit_wolf_intensity_in_unit_interval(self):
        data = _gaussian_panel(300, 4, seed=31)
        rc = RollingCovariance(
            window=60, min_periods=30, shrinkage="ledoit-wolf",
        )
        _ = rc.fit_transform_walk_forward(data, step=21)
        for alpha in rc.shrinkage_intensities_.values():
            assert 0.0 <= alpha <= 1.0

    def test_no_lookahead_when_future_is_perturbed(self):
        data = _gaussian_panel(400, 3, seed=32)
        step, W, MP = 21, 60, 30
        modify_from = 5 * step
        cov_full = RollingCovariance(
            window=W, min_periods=MP, shrinkage="ledoit-wolf",
        ).fit_transform_walk_forward(data, step=step)
        data_pert = data.copy()
        data_pert.iloc[modify_from:] += 1e6
        cov_pert = RollingCovariance(
            window=W, min_periods=MP, shrinkage="ledoit-wolf",
        ).fit_transform_walk_forward(data_pert, step=step)

        # Refit positions <= modify_from - step use only past data and
        # must produce identical covariance matrices.  The refit at
        # modify_from itself uses rows [modify_from - W, modify_from),
        # which are all unperturbed — so it should also match.
        for ts, mat_full in cov_full.items():
            if ts > data.index[modify_from]:
                continue
            mat_pert = cov_pert[ts]
            np.testing.assert_allclose(
                mat_full.values, mat_pert.values, atol=1e-10,
            )

    def test_cov_at_returns_active_matrix(self):
        data = _gaussian_panel(300, 3, seed=33)
        rc = RollingCovariance(window=60, min_periods=30)
        cov = rc.fit_transform_walk_forward(data, step=21)
        refit_dates = sorted(cov.keys())
        # Active matrix at the refit date itself.
        first_date = refit_dates[0]
        active = rc.cov_at(first_date)
        pd.testing.assert_frame_equal(active, cov[first_date])
        # Active matrix one day after refit is the same.
        next_day = first_date + pd.Timedelta(days=1)
        pd.testing.assert_frame_equal(rc.cov_at(next_day), cov[first_date])
        # Active matrix at the next refit date moves forward.
        second_date = refit_dates[1]
        pd.testing.assert_frame_equal(rc.cov_at(second_date), cov[second_date])
        # Before any refit, returns None.
        before_history = data.index[0] - pd.Timedelta(days=1)
        assert rc.cov_at(before_history) is None

    def test_constant_shrinkage_blends_toward_identity(self):
        rng = np.random.default_rng(34)
        n = 300
        idx = pd.date_range("2015-01-02", periods=n, freq="B")
        data = pd.DataFrame(
            rng.standard_normal((n, 3)),
            index=idx,
            columns=["a", "b", "c"],
        )
        rc = RollingCovariance(window=60, min_periods=30, shrinkage=1.0)
        cov = rc.fit_transform_walk_forward(data, step=21)
        # With shrinkage=1.0 every matrix becomes (tr(S)/k) * I — diagonal.
        for mat in cov.values():
            arr = mat.values
            off = arr - np.diag(np.diag(arr))
            np.testing.assert_allclose(off, 0.0, atol=1e-12)

    def test_empirical_covariance_when_shrinkage_none(self):
        data = _gaussian_panel(200, 2, seed=35)
        rc = RollingCovariance(window=60, min_periods=30, shrinkage=None)
        cov = rc.fit_transform_walk_forward(data, step=21)
        # Empirical covariance at refit r matches np.cov on the fit window.
        first_refit_date = sorted(cov.keys())[0]
        r = data.index.get_loc(first_refit_date)
        fit_slice = data.iloc[max(0, r - 60) : r].values
        expected = np.cov(fit_slice, rowvar=False, ddof=0)
        np.testing.assert_allclose(
            cov[first_refit_date].values, expected, atol=1e-10,
        )

    def test_invalid_shrinkage_raises(self):
        with pytest.raises(ValueError):
            RollingCovariance(shrinkage="oas")
        with pytest.raises(ValueError):
            RollingCovariance(shrinkage=1.5)
        with pytest.raises(ValueError):
            RollingCovariance(shrinkage=-0.1)


# ---------------------------------------------------------------------------
# Cross-cutting: NaN boundary behaviour
# ---------------------------------------------------------------------------


class TestNoNaNLeakage:
    """For each walk-forward class, verify that NaN appears only where the
    fit window is too short — never inside the well-defined body."""

    def test_standardizer_body_has_no_nan(self):
        data = _gaussian_panel(500, 4, seed=40)
        out = RollingStandardizer(
            window=60, min_periods=30,
        ).fit_transform_walk_forward(data, step=1)
        body = out.iloc[60:]
        assert body.notna().all().all()
        # Warm-up rows ARE NaN — verify they were not silently zero-filled.
        warm = out.iloc[:30]
        assert warm.isna().all().all()

    def test_pca_body_has_no_nan(self):
        data = _gaussian_panel(500, 4, seed=41)
        step, W, MP = 21, 60, 30
        out = RollingPCA(
            n_components=2, window=W, min_periods=MP, random_state=0,
        ).fit_transform_walk_forward(data, step=step)
        # First block with a usable refit window is the one starting at
        # refit position >= MP (positions 0 and 21 are too small here:
        # the fit window at r=21 is [0, 21) of length 21 < MP=30).
        # Smallest r with fit slice >= MP is r = 42 (length 42 >= 30).
        body = out.iloc[42:]
        assert body.notna().all().all()
