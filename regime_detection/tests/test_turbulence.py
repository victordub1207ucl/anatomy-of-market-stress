"""Unit tests for TurbulenceIndex and the regime-labelling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from regime_detection.lib.turbulence import TurbulenceIndex
from regime_detection.lib.turbulence_regimes import (
    REGIME_QUIET,
    REGIME_TURBULENT,
    classify_regimes,
    smooth_min_duration,
)


def _gaussian_panel(n: int, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    return pd.DataFrame(
        rng.standard_normal((n, k)) * 0.01,
        index=idx,
        columns=[f"f{j}" for j in range(k)],
    )


# ---------------------------------------------------------------------------
# TurbulenceIndex
# ---------------------------------------------------------------------------


class TestTurbulenceIndex:
    def test_matches_manual_mahalanobis_with_ledoit_wolf(self):
        data = _gaussian_panel(800, 5, seed=100)
        W, MP = 500, 200
        ti = TurbulenceIndex(
            lookback_days=W, min_periods=MP, shrinkage="ledoit-wolf",
            refit_every=1,
        )
        out = ti.fit_transform(data)

        # At row t, fit window is rows [t-W, t-1] inclusive (length W).
        for t in (600, 700, 799):
            fit_slice = data.iloc[t - W : t].values
            lw = LedoitWolf().fit(fit_slice)
            mu = fit_slice.mean(axis=0)
            omega_inv = np.linalg.inv(lw.covariance_)
            diff = data.iloc[t].values - mu
            expected = float(diff @ omega_inv @ diff)
            np.testing.assert_allclose(out.iloc[t], expected, atol=1e-10)

    def test_matches_manual_mahalanobis_without_shrinkage(self):
        data = _gaussian_panel(800, 5, seed=101)
        W, MP = 500, 200
        ti = TurbulenceIndex(
            lookback_days=W, min_periods=MP, shrinkage=None, refit_every=1,
        )
        out = ti.fit_transform(data)
        t = 750
        fit_slice = data.iloc[t - W : t].values
        omega = np.cov(fit_slice, rowvar=False, ddof=0)
        mu = fit_slice.mean(axis=0)
        diff = data.iloc[t].values - mu
        expected = float(diff @ np.linalg.inv(omega) @ diff)
        np.testing.assert_allclose(out.iloc[t], expected, atol=1e-10)

    def test_no_lookahead_when_future_is_perturbed(self):
        data = _gaussian_panel(800, 4, seed=102)
        modify_from = 600
        data_perturbed = data.copy()
        data_perturbed.iloc[modify_from:] += 1.0  # huge shock to all features

        ti = TurbulenceIndex(
            lookback_days=400, min_periods=200, shrinkage="ledoit-wolf",
            refit_every=1,
        )
        out_full = ti.fit_transform(data)
        ti2 = TurbulenceIndex(
            lookback_days=400, min_periods=200, shrinkage="ledoit-wolf",
            refit_every=1,
        )
        out_pert = ti2.fit_transform(data_perturbed)

        # Row t uses fit window [t-W, t-1] AND row t's own value.  For
        # t < modify_from BOTH are unchanged, so output must match.
        pd.testing.assert_series_equal(
            out_full.iloc[:modify_from],
            out_pert.iloc[:modify_from],
            check_exact=False,
            atol=1e-12,
        )
        # And the perturbed tail must differ on at least one row.
        assert not out_full.iloc[modify_from:].equals(out_pert.iloc[modify_from:])

    def test_warmup_rows_are_nan(self):
        data = _gaussian_panel(400, 3, seed=103)
        ti = TurbulenceIndex(lookback_days=100, min_periods=80, refit_every=1)
        out = ti.fit_transform(data)
        # Rows whose trailing window is shorter than min_periods are NaN.
        # The first row at which the window has >= 80 obs is row 80
        # (fit_slice = iloc[0:80], length 80 >= 80).
        assert out.iloc[:80].isna().all()
        assert out.iloc[80:].notna().all()

    def test_shrinkage_intensities_in_unit_interval(self):
        data = _gaussian_panel(600, 4, seed=104)
        ti = TurbulenceIndex(lookback_days=300, min_periods=200)
        _ = ti.fit_transform(data)
        finite = ti.shrinkage_intensities_.dropna()
        assert len(finite) > 0
        assert ((finite >= 0.0) & (finite <= 1.0)).all()

    def test_refit_every_freezes_stats_within_block(self):
        """With refit_every=21, consecutive rows inside a block share
        the same (mu, Omega) fit and therefore the same denominator term
        contribution from past stats — verify the quadratic form matches a
        single manual fit applied to multiple rows."""
        data = _gaussian_panel(500, 4, seed=105)
        W, MP, step = 200, 100, 21
        ti = TurbulenceIndex(
            lookback_days=W, min_periods=MP, shrinkage="ledoit-wolf",
            refit_every=step,
        )
        out = ti.fit_transform(data)

        r = 10 * step  # 10th refit position
        fit_slice = data.iloc[r - W : r].values
        lw = LedoitWolf().fit(fit_slice)
        mu = fit_slice.mean(axis=0)
        omega_inv = np.linalg.inv(lw.covariance_)
        for offset in range(step):
            row = data.iloc[r + offset].values
            diff = row - mu
            expected = float(diff @ omega_inv @ diff)
            np.testing.assert_allclose(out.iloc[r + offset], expected, atol=1e-10)

    def test_known_shift_in_mean_increases_turbulence(self):
        """A clear regime shift in the cross-section should produce a
        sustained turbulence spike."""
        rng = np.random.default_rng(106)
        n_quiet, n_shock, k = 600, 50, 4
        quiet = rng.standard_normal((n_quiet, k)) * 0.01
        # Big positive shift in the first two factors.
        shock = rng.standard_normal((n_shock, k)) * 0.01
        shock[:, :2] += 0.05
        data = pd.DataFrame(
            np.vstack([quiet, shock]),
            index=pd.date_range("2010-01-04", periods=n_quiet + n_shock, freq="B"),
            columns=[f"f{i}" for i in range(k)],
        )
        ti = TurbulenceIndex(lookback_days=300, min_periods=200)
        out = ti.fit_transform(data)
        # Median turbulence inside the shock window must be much higher
        # than the median in the quiet window.
        quiet_median = float(out.iloc[200:n_quiet].median())
        shock_median = float(out.iloc[n_quiet:].median())
        # Under correct Gaussian, the quadratic form has mean ≈ k=4 (chi²).
        # The shock injects a +0.05 mean shift on two factors — large vs
        # the σ=0.01 noise.  Empirically the shock median is ~4× quiet.
        assert shock_median > 3 * quiet_median, (
            f"Expected shock turbulence >> quiet; "
            f"quiet={quiet_median:.3f}, shock={shock_median:.3f}"
        )

    def test_invalid_inputs_raise(self):
        with pytest.raises(TypeError):
            TurbulenceIndex().fit_transform("not a dataframe")
        df_bad_index = pd.DataFrame(np.eye(3), index=[0, 1, 2])
        with pytest.raises(TypeError):
            TurbulenceIndex().fit_transform(df_bad_index)
        df_single_col = _gaussian_panel(100, 1, seed=107)
        with pytest.raises(ValueError):
            TurbulenceIndex().fit_transform(df_single_col)
        with pytest.raises(ValueError):
            TurbulenceIndex(lookback_days=1)
        with pytest.raises(ValueError):
            TurbulenceIndex(min_periods=1)
        with pytest.raises(ValueError):
            TurbulenceIndex(lookback_days=100, min_periods=200)
        with pytest.raises(ValueError):
            TurbulenceIndex(shrinkage="oas")
        with pytest.raises(ValueError):
            TurbulenceIndex(refit_every=0)


# ---------------------------------------------------------------------------
# classify_regimes
# ---------------------------------------------------------------------------


class TestClassifyRegimes:
    def test_threshold_is_rolling_and_causal(self):
        rng = np.random.default_rng(200)
        n = 3000
        idx = pd.date_range("2010-01-04", periods=n, freq="B")
        turb = pd.Series(rng.exponential(scale=1.0, size=n), index=idx)

        labels = classify_regimes(
            turb,
            threshold_quantile=0.75,
            rolling_quantile_window=500,
            min_periods=250,
        )
        # Manual check: at row t, threshold = 75th pct of turb[t-500:t]
        t = 1500
        window = turb.iloc[t - 500 : t]
        expected_threshold = window.quantile(0.75)
        expected_label = float(turb.iloc[t] > expected_threshold)
        assert labels.iloc[t] == expected_label

    def test_no_lookahead_when_future_is_perturbed(self):
        rng = np.random.default_rng(201)
        n = 2000
        turb = pd.Series(
            rng.exponential(scale=1.0, size=n),
            index=pd.date_range("2010-01-04", periods=n, freq="B"),
        )
        modify_from = 1500
        turb_pert = turb.copy()
        turb_pert.iloc[modify_from:] = 1e6  # huge future spike

        labels_full = classify_regimes(
            turb, threshold_quantile=0.75,
            rolling_quantile_window=500, min_periods=250,
        )
        labels_pert = classify_regimes(
            turb_pert, threshold_quantile=0.75,
            rolling_quantile_window=500, min_periods=250,
        )
        # Rows < modify_from use threshold from [t-500, t-1] which is
        # entirely past; AND turb[t] itself is unchanged.  So labels match.
        pd.testing.assert_series_equal(
            labels_full.iloc[:modify_from],
            labels_pert.iloc[:modify_from],
        )

    def test_warmup_rows_are_nan(self):
        rng = np.random.default_rng(202)
        n = 600
        turb = pd.Series(
            rng.exponential(size=n),
            index=pd.date_range("2010-01-04", periods=n, freq="B"),
        )
        labels = classify_regimes(
            turb, rolling_quantile_window=200, min_periods=100,
        )
        # Threshold at row t uses [t-200, t-1] (after shift(1)).  Need
        # >= 100 obs in that window, so first defined row is row 100.
        assert labels.iloc[:100].isna().all()
        assert labels.iloc[100:].notna().all()

    def test_labels_are_zero_or_one(self):
        rng = np.random.default_rng(203)
        n = 800
        turb = pd.Series(
            rng.exponential(size=n),
            index=pd.date_range("2010-01-04", periods=n, freq="B"),
        )
        labels = classify_regimes(turb, rolling_quantile_window=200,
                                  min_periods=100)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({REGIME_QUIET, REGIME_TURBULENT})

    def test_invalid_inputs_raise(self):
        with pytest.raises(TypeError):
            classify_regimes([1, 2, 3])
        turb = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError):
            classify_regimes(turb, threshold_quantile=0.0)
        with pytest.raises(ValueError):
            classify_regimes(turb, threshold_quantile=1.0)
        with pytest.raises(ValueError):
            classify_regimes(turb, rolling_quantile_window=1)
        with pytest.raises(ValueError):
            classify_regimes(turb, rolling_quantile_window=100, min_periods=200)


# ---------------------------------------------------------------------------
# smooth_min_duration
# ---------------------------------------------------------------------------


class TestSmoothMinDuration:
    def test_short_burst_is_squashed(self):
        # 3-day burst inside an otherwise-quiet regime is dropped when
        # min_duration = 5.
        labels = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        smoothed = smooth_min_duration(labels, min_duration=5)
        np.testing.assert_array_equal(
            smoothed.values, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        )

    def test_sustained_transition_is_accepted_on_day_min_duration(self):
        # 5-day run of 1s should accept the transition on day 5
        # (1-indexed) of the run.  The trailing 0s after the accepted
        # transition are only 2 days long, so they do not flip the regime
        # back — current stays at 1.
        labels = pd.Series([0, 0, 1, 1, 1, 1, 1, 0, 0])
        smoothed = smooth_min_duration(labels, min_duration=5)
        # Days 0, 1: still in regime 0.
        # Days 2-5: candidate streak 1..4 (< 5), output stays 0.
        # Day 6: streak reaches 5, transition accepted → current=1.
        # Days 7, 8: candidate becomes 0 but streak is only 2 (< 5) →
        # current stays at 1.
        np.testing.assert_array_equal(
            smoothed.values,
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=float),
        )

    def test_min_duration_one_is_passthrough(self):
        rng = np.random.default_rng(300)
        labels = pd.Series(rng.integers(0, 2, size=200).astype(float))
        smoothed = smooth_min_duration(labels, min_duration=1)
        pd.testing.assert_series_equal(smoothed, labels.rename("regime"))

    def test_nan_rows_remain_nan(self):
        labels = pd.Series([0.0, 0.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0])
        smoothed = smooth_min_duration(labels, min_duration=3)
        assert np.isnan(smoothed.iloc[2])
        # The NaN row resets the streak, so the run of 1s starts fresh.
        # 3 consecutive 1s after the NaN accepts the transition on the
        # 3rd one (index 5).
        np.testing.assert_array_equal(
            smoothed.values[3:],
            np.array([0, 0, 1, 1, 1], dtype=float),
        )

    def test_is_causal(self):
        """Truncating the input at any point t must not change the
        smoothed value at any row s <= t."""
        rng = np.random.default_rng(301)
        n = 500
        # Random binary labels.
        labels = pd.Series(
            (rng.uniform(size=n) > 0.6).astype(float),
            index=pd.date_range("2010-01-04", periods=n, freq="B"),
        )
        full = smooth_min_duration(labels, min_duration=5)
        truncated = smooth_min_duration(labels.iloc[:250], min_duration=5)
        pd.testing.assert_series_equal(full.iloc[:250], truncated)

    def test_invalid_inputs_raise(self):
        with pytest.raises(TypeError):
            smooth_min_duration([0, 1, 0])
        with pytest.raises(ValueError):
            smooth_min_duration(pd.Series([0, 1]), min_duration=0)
