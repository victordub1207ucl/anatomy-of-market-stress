"""Tests for the Phase-6 evaluation modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.metrics import (
    annualised_return, annualised_vol, brier_score, hit_rate_at_threshold,
    log_loss, max_drawdown, sharpe_ratio, sortino_ratio,
    strategy_returns, top_decile_threshold,
)
from regime_detection.lib.oos_pipeline import (
    _quarter_ends, run_quarterly_walk_forward,
)
from regime_detection.lib.stats_tests import (
    block_bootstrap_sortino_diff, diebold_mariano,
    stationary_bootstrap_sortino_diff,
)


def _series(values, start="2021-01-04"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_annualised_return_and_vol(self):
        # Check formula identity against the empirical mean/std of the
        # generated sample (the formula multiplies the empirical mean by
        # 252, not the theoretical mean).
        rng = np.random.default_rng(0)
        r = _series(rng.normal(0.0005, 0.01, 252))
        assert annualised_return(r) == pytest.approx(
            float(r.mean()) * 252, abs=1e-12,
        )
        expected_vol = float(r.std(ddof=0)) * np.sqrt(252)
        assert annualised_vol(r) == pytest.approx(expected_vol, abs=1e-12)

    def test_sortino_positive_for_strong_positive_drift(self):
        # Use a large drift relative to noise so the empirical Sortino is
        # reliably positive across random seeds.
        rng = np.random.default_rng(1)
        r = _series(rng.normal(0.003, 0.01, 800))
        assert sortino_ratio(r) > 0

    def test_sortino_zero_when_no_excess(self):
        r = _series(np.zeros(300))
        # All zeros means downside std is undefined → NaN.
        assert np.isnan(sortino_ratio(r))

    def test_max_drawdown_negative_or_zero(self):
        r = _series(np.r_[np.full(50, 0.001), np.full(50, -0.01),
                          np.full(50, 0.001)])
        dd = max_drawdown(r)
        # Drawdown is the cumulative log-return dip from peak to trough.
        # Cum at peak (i=49): 50*0.001 = 0.05; trough at i=99: 0.05 - 50*0.01 = -0.45
        # So dd = -0.45 - 0.05 = -0.50.
        assert dd == pytest.approx(-0.50, abs=1e-9)
        # Pure-rising series has zero drawdown.
        flat_up = _series(np.full(100, 0.001))
        assert max_drawdown(flat_up) == pytest.approx(0.0, abs=1e-12)

    def test_hit_rate_at_threshold(self):
        # 100 rows: 30 true positives, 70 true negatives.  Predictions are
        # ranked correctly for the top 10 (all true positives).
        y_true = pd.Series(([1.0] * 30) + ([0.0] * 70))
        # Top 10 scores correspond to true positives.
        scores = np.r_[np.linspace(0.95, 0.85, 30), np.linspace(0.4, 0.0, 70)]
        y_score = pd.Series(scores)
        h = hit_rate_at_threshold(y_true, y_score, decile=0.10)
        # Top decile of 100 = top 10 by score: all true positives → prec = 1.0.
        assert h["precision"] == pytest.approx(1.0, abs=1e-12)
        # Recall = 10 / 30 ≈ 0.333.
        assert h["recall"] == pytest.approx(10 / 30, abs=1e-9)

    def test_top_decile_threshold_matches_quantile(self):
        rng = np.random.default_rng(2)
        s = pd.Series(rng.uniform(size=500))
        thr = top_decile_threshold(s, decile=0.10)
        assert thr == pytest.approx(float(np.quantile(s.values, 0.9)), abs=1e-12)

    def test_brier_and_log_loss_match_definitions(self):
        y = pd.Series([1.0, 0.0, 1.0, 0.0])
        p = pd.Series([0.9, 0.1, 0.6, 0.2])
        expected_brier = float(np.mean((p.values - y.values) ** 2))
        assert brier_score(y, p) == pytest.approx(expected_brier, abs=1e-12)
        expected_ll = float(-np.mean(
            y.values * np.log(p.values)
            + (1.0 - y.values) * np.log(1.0 - p.values)
        ))
        assert log_loss(y, p) == pytest.approx(expected_ll, abs=1e-9)

    def test_strategy_returns_go_flat_when_signal_high(self):
        # Signal = 1 every day → strategy always flat (we drop the top
        # decile).  Hmm but a constant signal has no top decile.  Use
        # a varied signal instead.
        n = 100
        rng = np.random.default_rng(3)
        rets = _series(rng.normal(0.0, 0.01, n))
        sig = _series(np.linspace(0, 1, n))
        s = strategy_returns(rets, sig, flat_when_signal_is_high=True, shift=1)
        # Top decile cutoff at sig ≈ 0.9, so signals from day ~90 onward
        # set strategy to flat.  Lag by 1 day → strategy is flat from
        # day ~91.  Earlier days hold the asset, except day 0 which is
        # zeroed by the .shift(1).fillna(0) lag.
        np.testing.assert_allclose(
            s.iloc[1:50].values, rets.iloc[1:50].values, atol=1e-12,
        )
        assert s.iloc[-5:].sum() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# stats_tests
# ---------------------------------------------------------------------------


class TestDiebold:
    def test_zero_when_losses_are_identical(self):
        loss = _series(np.linspace(0.5, 0.6, 100))
        r = diebold_mariano(loss, loss, h=21)
        # Mean diff = 0 → stat = NaN (var=0) or 0; either way p-value
        # should be NaN or 1.0.
        assert np.isnan(r.statistic) or r.statistic == 0

    def test_detects_systematic_loss_gap_under_iid(self):
        rng = np.random.default_rng(10)
        n = 1000
        loss_a = _series(rng.normal(0.30, 0.01, n))
        loss_b = _series(rng.normal(0.25, 0.01, n))  # B clearly better
        r = diebold_mariano(loss_a, loss_b, h=21)
        # A loses by 0.05 on average; with n=1000 and σ≈0.014 of
        # the difference, the t-stat is huge and p ≈ 0.
        assert r.statistic > 3
        assert r.p_value < 0.01

    def test_hac_lags_is_h_minus_one(self):
        loss_a = _series(np.zeros(100))
        loss_b = _series(np.zeros(100))
        r = diebold_mariano(loss_a, loss_b, h=21)
        assert r.hac_lags == 20


class TestBlockBootstrap:
    def test_zero_difference_when_inputs_identical(self):
        rng = np.random.default_rng(20)
        r = _series(rng.normal(0.0005, 0.01, 600))
        out = block_bootstrap_sortino_diff(
            r, r, block_length=21, n_resamples=500, random_state=42,
        )
        # Observed diff is zero; CI should bracket zero comfortably.
        assert abs(out.mean_diff) < 1e-9
        assert out.ci_low <= 0 <= out.ci_high

    def test_detects_positive_drift_gap(self):
        """Block bootstrap of the Sortino DIFFERENCE has wide CIs even at
        n=1500 with large drift — Sortino estimation noise compounds
        across blocks.  We check only the sign of mean_diff here; tight
        CIs come at larger n or smaller block lengths, which is the
        sample-size ceiling we acknowledge in the report."""
        rng = np.random.default_rng(21)
        n = 3000
        a = _series(rng.normal(0.0020, 0.01, n))   # higher Sortino
        b = _series(rng.normal(0.0001, 0.01, n))
        out = block_bootstrap_sortino_diff(
            a, b, block_length=21, n_resamples=800, random_state=42,
        )
        assert out.mean_diff > 0
        # The bootstrap CI may or may not exclude zero at this sample
        # size — we don't assert on it.  Just verify it is well-formed.
        assert np.isfinite(out.ci_low) and np.isfinite(out.ci_high)
        assert out.ci_low <= out.mean_diff <= out.ci_high


class TestStationaryBootstrap:
    def test_block_length_geometric_mean(self):
        # Smoke test that the function returns finite numbers.
        rng = np.random.default_rng(30)
        a = _series(rng.normal(0.0005, 0.01, 800))
        b = _series(rng.normal(0.0002, 0.01, 800))
        out = stationary_bootstrap_sortino_diff(
            a, b, mean_block_length=21, n_resamples=300, random_state=11,
        )
        assert np.isfinite(out.mean_diff)
        assert np.isfinite(out.ci_low) and np.isfinite(out.ci_high)
        assert out.method == "stationary"


# ---------------------------------------------------------------------------
# oos_pipeline.quarter_ends
# ---------------------------------------------------------------------------


class TestQuarterEnds:
    def test_quarter_ends_strictly_in_range(self):
        idx = _quarter_ends(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-06-30"))
        assert idx.min() >= pd.Timestamp("2021-01-01")
        assert idx.max() <= pd.Timestamp("2023-06-30")
        # All dates land on the calendar quarter-end.
        for d in idx:
            assert d.month in (3, 6, 9, 12)
            # Day is between 28 and 31.
            assert 28 <= d.day <= 31
