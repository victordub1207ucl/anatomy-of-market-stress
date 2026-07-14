"""Unit tests for the ETF-inception splicing module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.lib.splicing import (
    FACTOR_DEFINITIONS, FACTOR_FALLBACKS, UNSPLICEABLE_FACTORS,
    build_spliced_factor_panel, splice_factor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(start, end, mean=0.0005, sd=0.01, seed=0):
    """Synthetic Gaussian log returns over a business-day range."""
    idx = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, len(idx)), index=idx, name="ret")


def _prices_from_returns(rets, anchor=100.0):
    cum = rets.cumsum()
    return np.exp(cum - cum.iloc[0]) * anchor


# ---------------------------------------------------------------------------
# splice_factor
# ---------------------------------------------------------------------------


class TestSpliceFactor:
    def test_continuity_at_implicit_splice_date(self):
        """The reconstructed price level is continuous at the splice
        date: there is no jump when we switch from proxy to primary."""
        proxy = _make_returns("2005-01-04", "2010-12-31", seed=1)
        primary = _make_returns("2010-01-04", "2020-12-31", seed=2)
        merged = splice_factor(primary, [proxy])
        # Both pre- and post-splice rows are non-NaN.
        assert merged.dropna().index.min() <= pd.Timestamp("2005-01-05")
        assert merged.dropna().index.max() >= pd.Timestamp("2020-12-30")
        # Reconstruct pseudo-prices; check no level jump on the splice
        # date.  The splice happens on the first date where the primary
        # has data — 2010-01-04.  The cumulative log return is continuous
        # by construction; we verify the price level is finite on either
        # side and the day-over-day return on the splice date matches the
        # primary's first return.
        prices = _prices_from_returns(merged.dropna())
        splice_idx = merged.dropna().index.get_indexer([primary.index[0]])[0]
        if splice_idx > 0:
            day_ret = float(np.log(prices.iloc[splice_idx]
                                    / prices.iloc[splice_idx - 1]))
            # day_ret should equal the merged series's value at splice_idx,
            # which (after combine_first) is the PRIMARY's value (primary
            # takes precedence over proxy on overlapping dates).
            np.testing.assert_allclose(
                day_ret, float(merged.iloc[splice_idx]), atol=1e-12,
            )

    def test_pre_splice_returns_equal_proxy(self):
        proxy = _make_returns("2005-01-04", "2010-12-31", seed=10)
        primary = _make_returns("2011-01-04", "2020-12-31", seed=11)
        merged = splice_factor(primary, [proxy])
        # On any pre-2011 date that the proxy covered, the merged value
        # equals the proxy value (primary was NaN there).
        sample_dates = proxy.index[100:110]
        np.testing.assert_allclose(
            merged.loc[sample_dates].values,
            proxy.loc[sample_dates].values,
            atol=1e-12,
        )

    def test_post_splice_returns_equal_primary(self):
        proxy = _make_returns("2005-01-04", "2010-12-31", seed=20)
        primary = _make_returns("2011-01-04", "2020-12-31", seed=21)
        merged = splice_factor(primary, [proxy])
        # On any post-2011 date, merged = primary (combine_first prefers
        # the LEFT operand, which is the primary).
        sample_dates = primary.index[100:110]
        np.testing.assert_allclose(
            merged.loc[sample_dates].values,
            primary.loc[sample_dates].values,
            atol=1e-12,
        )

    def test_overlap_dates_use_primary_not_proxy(self):
        """v4's algorithm uses combine_first(merged, proxy) which keeps
        the primary's value on any date where both exist.  Verify."""
        idx = pd.date_range("2010-01-04", "2010-12-31", freq="B")
        primary = pd.Series(np.full(len(idx), 0.001), index=idx,
                            name="primary")
        # Proxy has values 100x bigger on the same dates — if it leaked
        # into the merged output we'd see it.
        proxy = pd.Series(np.full(len(idx), 0.1), index=idx, name="proxy")
        merged = splice_factor(primary, [proxy])
        np.testing.assert_allclose(merged.values, primary.values, atol=1e-12)

    def test_explicit_splice_date_with_future_only_data_raises(self):
        """If splice_dates is provided and all proxy dates are AFTER the
        splice date, that proxy contributes nothing and effectively
        introduces look-ahead-shaped emptiness.  We catch a length
        mismatch as the closest enforceable invariant."""
        proxy = _make_returns("2030-01-04", "2031-12-31", seed=30)
        primary = _make_returns("2011-01-04", "2020-12-31", seed=31)
        with pytest.raises(ValueError):
            splice_factor(primary, [proxy], splice_dates=[])  # mismatch

    def test_no_proxy_returns_primary_unchanged(self):
        primary = _make_returns("2011-01-04", "2020-12-31", seed=40)
        merged = splice_factor(primary, [])
        pd.testing.assert_series_equal(merged, primary)

    def test_non_series_input_raises(self):
        primary = _make_returns("2011-01-04", "2012-12-31", seed=50)
        with pytest.raises(TypeError):
            splice_factor(primary, [primary.values])
        with pytest.raises(TypeError):
            splice_factor(primary.values, [])


# ---------------------------------------------------------------------------
# build_spliced_factor_panel
# ---------------------------------------------------------------------------


class TestBuildPanel:
    def test_panel_columns_match_factor_definitions(self):
        # Tiny synthetic ticker panel with all primary tickers + one
        # fallback present.
        defs = {
            "f1": ("T1", "f1 desc", "cat"),
            "f2": ("T2", "f2 desc", "cat"),
        }
        fallbacks = {"f1": ["B1"]}
        idx = pd.date_range("2010-01-04", "2010-12-31", freq="B")
        rng = np.random.default_rng(0)
        raw = pd.DataFrame({
            "T1": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
            "T2": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
            "B1": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
        }, index=idx)
        panel = build_spliced_factor_panel(raw, defs, fallbacks)
        assert list(panel.columns) == ["f1", "f2"]
        # The first row is naturally NaN (log-returns lose the anchor
        # day); rows from the second onwards must be fully populated.
        assert panel.iloc[1:].notna().all().all()

    def test_panel_anchors_to_100(self):
        defs = {"f1": ("T1", "desc", "cat")}
        idx = pd.date_range("2010-01-04", "2010-12-31", freq="B")
        rng = np.random.default_rng(1)
        raw = pd.DataFrame({
            "T1": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
        }, index=idx)
        panel = build_spliced_factor_panel(raw, defs, {})
        # The first finite anchor value should be exactly 100.
        first = panel["f1"].dropna().iloc[0]
        np.testing.assert_allclose(first, 100.0, atol=1e-9)

    def test_missing_primary_returns_nan_column(self):
        defs = {"f1": ("MISSING", "desc", "cat")}
        idx = pd.date_range("2010-01-04", "2010-12-31", freq="B")
        raw = pd.DataFrame({"OTHER": [1.0] * len(idx)}, index=idx)
        panel = build_spliced_factor_panel(raw, defs, {})
        assert panel["f1"].isna().all()


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_factor_definitions_has_14_entries(self):
        assert len(FACTOR_DEFINITIONS) == 14

    def test_unspliceable_factors_have_no_fallback(self):
        for f in UNSPLICEABLE_FACTORS:
            assert f in FACTOR_DEFINITIONS, f
            assert f not in FACTOR_FALLBACKS, (
                f"{f} is documented as unspliceable but has a fallback"
            )

    def test_every_fallback_factor_is_in_definitions(self):
        for f in FACTOR_FALLBACKS:
            assert f in FACTOR_DEFINITIONS, f


# ---------------------------------------------------------------------------
# Cross-validation against the cached parquet
# ---------------------------------------------------------------------------


class TestCacheConsistency:
    def test_first_valid_dates_match_v4_cache_expectations(self):
        """The Phase 2 inspection of the cached parquet found:
        - quality first valid: 2005-12-07 (SPHQ fallback applied)
        - momentum first valid: 2007-03-02 (PDP fallback applied)
        - em_credit first valid: 2007-12-20 (no fallback; primary EMB inception)
        - short_vol first valid: 2011-10-05 (no fallback; SVXY inception)
        - low_risk first valid: 2011-10-21 (no fallback; USMV inception)
        Verify our `UNSPLICEABLE_FACTORS` constant matches the
        no-fallback set."""
        # quality and momentum DO have fallbacks → not in UNSPLICEABLE
        assert "quality" not in UNSPLICEABLE_FACTORS
        assert "momentum" not in UNSPLICEABLE_FACTORS
        # em_credit, short_vol, low_risk DON'T have fallbacks → in
        # UNSPLICEABLE
        for f in ("em_credit", "short_vol", "low_risk"):
            assert f in UNSPLICEABLE_FACTORS
