"""Unit tests for the sliding-cutoff sensitivity module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.cutoff_sensitivity import (
    metrics_for_cutoff_run, run_sliding_cutoff, summary_statistics,
)
from regime_detection.lib.oos_pipeline import QuarterlyWalkForward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_supervised_inputs(n: int = 1500, seed: int = 0):
    """Make a synthetic (X, y, equity_returns) triple where features
    actually predict the target a little, so models converge."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-04", periods=n, freq="B")
    # Two features: a noisy signal and a noise column.
    sig = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    # Target = 1 if sig > 0 (with some noise flip).
    y_clean = (sig > 0).astype(float)
    flip_mask = rng.uniform(size=n) < 0.10
    y_clean[flip_mask] = 1 - y_clean[flip_mask]
    # Build X with a one-day lag so it's strictly causal at row t.
    X = pd.DataFrame({
        "sig_lag1": pd.Series(sig, index=idx).shift(1),
        "noise":    pd.Series(noise, index=idx).shift(1),
        "sig_ma5":  pd.Series(sig, index=idx).rolling(5).mean().shift(1),
    }, index=idx).dropna()
    y = pd.Series(y_clean, index=idx).reindex(X.index)
    # Equity returns: very mildly positive drift + noise.
    eq_ret = pd.Series(rng.normal(0.0003, 0.01, n), index=idx).reindex(X.index)
    return X, y, eq_ret


def _stub_qwf(n_oos: int = 200, pos_rate: float = 0.5,
              seed: int = 0) -> QuarterlyWalkForward:
    """A minimal QuarterlyWalkForward stand-in for the metrics-extractor
    test — no need to actually train a model."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-04", periods=n_oos, freq="B")
    y = pd.Series((rng.uniform(size=n_oos) < pos_rate).astype(float),
                  index=idx, name="y_true")
    # Predicted probability roughly correlated with truth.
    p = 0.4 * y + 0.4 * rng.uniform(size=n_oos) + 0.2
    p = pd.Series(np.clip(p, 0.001, 0.999), index=idx, name="proba")
    return QuarterlyWalkForward(
        name="stub",
        oos_predictions=p,
        oos_labels=y,
        refit_log=pd.DataFrame({"asof": [idx[100]]}),
        feature_columns=["x1"],
    )


# ---------------------------------------------------------------------------
# metrics_for_cutoff_run
# ---------------------------------------------------------------------------


class TestMetricsForCutoffRun:
    def test_returns_expected_keys_on_well_formed_input(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2022-01-04", periods=400, freq="B")
        eq = pd.Series(rng.normal(0.0005, 0.01, 400), index=idx)
        qwf = _stub_qwf(n_oos=400, pos_rate=0.5)
        m = metrics_for_cutoff_run(qwf, eq)
        for k in ("n_oos", "n_refits", "pos_rate", "auc", "brier",
                  "log_loss", "sortino_strategy", "sortino_buy_hold",
                  "sortino_lift", "max_dd_strategy", "max_dd_buy_hold",
                  "ann_ret_strategy"):
            assert k in m, f"missing key {k!r}"
        assert m["n_oos"] == 400
        assert 0.0 <= m["pos_rate"] <= 1.0
        assert 0.0 <= m["auc"] <= 1.0
        assert m["brier"] >= 0
        assert np.isfinite(m["sortino_lift"])

    def test_returns_nan_when_only_one_class(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2022-01-04", periods=200, freq="B")
        eq = pd.Series(rng.normal(0.0005, 0.01, 200), index=idx)
        qwf = _stub_qwf(n_oos=200, pos_rate=1.0)
        m = metrics_for_cutoff_run(qwf, eq)
        # With only one class, AUC and Brier are undefined.
        assert np.isnan(m["auc"])


# ---------------------------------------------------------------------------
# run_sliding_cutoff
# ---------------------------------------------------------------------------


class TestRunSlidingCutoff:
    def test_skips_cutoffs_with_too_little_oos(self):
        X, y, eq = _build_supervised_inputs(n=800)
        end = X.index.max()
        # Two cutoffs: one well inside the data (lots of OOS), one
        # within a month of the end (insufficient OOS).
        far_cutoff = end - pd.Timedelta(days=15)
        ok_cutoff = X.index[300]
        df = run_sliding_cutoff(
            X, y, eq,
            cutoff_dates=[ok_cutoff, far_cutoff],
            model_name="logistic_l1",
            purge_window=21,
            min_oos_quarters=2,
            random_state=0,
            cache_dir=None,
            verbose=False,
        )
        # Only the ok_cutoff should appear (run_quarterly_walk_forward
        # may still skip it if too few refits — the test just verifies
        # the bounds check eliminates the far_cutoff).
        assert far_cutoff not in df.index

    def test_cache_round_trip(self, tmp_path: Path):
        X, y, eq = _build_supervised_inputs(n=1200)
        cache_dir = tmp_path / "cutoff_cache"
        cutoff = X.index[400]
        # First run — populates the cache.
        df1 = run_sliding_cutoff(
            X, y, eq,
            cutoff_dates=[cutoff],
            model_name="logistic_l1",
            purge_window=21,
            min_oos_quarters=2,
            random_state=0,
            cache_dir=cache_dir,
            verbose=False,
        )
        cached_files = list(cache_dir.glob("*.json"))
        assert len(cached_files) == 1, (
            f"expected 1 cached file, got {len(cached_files)}"
        )
        # Verify the cache file payload deserialises cleanly.
        with open(cached_files[0]) as f:
            payload = json.load(f)
        assert payload["cutoff"] == str(cutoff.date())
        assert "sortino_lift" in payload

        # Second run — should pull from cache without retraining (the
        # X/y arguments are deliberately the same; if cache is honoured
        # the result is byte-identical).
        df2 = run_sliding_cutoff(
            X, y, eq,
            cutoff_dates=[cutoff],
            model_name="logistic_l1",
            purge_window=21,
            min_oos_quarters=2,
            random_state=0,
            cache_dir=cache_dir,
            verbose=False,
        )
        pd.testing.assert_frame_equal(
            df1.sort_index(axis=1), df2.sort_index(axis=1),
        )

    def test_empty_when_all_cutoffs_skipped(self):
        X, y, eq = _build_supervised_inputs(n=500)
        end = X.index.max()
        # All cutoffs within 30 days of end — all should be skipped.
        cutoffs = [end - pd.Timedelta(days=d) for d in (5, 10, 15, 20)]
        df = run_sliding_cutoff(
            X, y, eq,
            cutoff_dates=cutoffs,
            model_name="logistic_l1",
            min_oos_quarters=4,
            verbose=False,
        )
        assert df.empty


# ---------------------------------------------------------------------------
# summary_statistics
# ---------------------------------------------------------------------------


class TestSummaryStatistics:
    def test_known_distribution(self):
        idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        # 12 entries: 3 negative, 2 zero, 7 strictly positive.
        # Of the positives: 0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.20.
        # Strictly > 0.10: 0.15, 0.15, 0.20 = 3 entries.
        df = pd.DataFrame({
            "sortino_lift": [-0.2, -0.1, 0.0, 0.05, 0.10, 0.15,
                             0.20, 0.15, 0.10, 0.05, 0.0, -0.1],
            "auc": [0.55] * 12,
            "brier": [0.18] * 12,
        }, index=idx)
        df.index.name = "cutoff"
        s = summary_statistics(df, metric="sortino_lift")
        assert s["n"] == 12
        assert s["min"] == pytest.approx(-0.2, abs=1e-9)
        assert s["max"] == pytest.approx(0.20, abs=1e-9)
        assert s["frac_positive"] == pytest.approx(7 / 12, abs=1e-9)
        assert s["frac_above_0p10"] == pytest.approx(3 / 12, abs=1e-9)

    def test_returns_nan_on_empty(self):
        df = pd.DataFrame({"sortino_lift": []},
                          index=pd.DatetimeIndex([], name="cutoff"))
        s = summary_statistics(df)
        assert np.isnan(s["median"])
        assert s["n"] == 0
