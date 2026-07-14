"""Shared library modules for the regime_detection pipeline.
All re-exports from the former sub-packages are preserved here so
`from regime_detection.lib import X` keeps working.
"""

# --- from models/__init__ ---
from regime_detection.lib.event_time import (
    EventTimeConverter,
    get_overlapping_event_returns,
)
from regime_detection.lib.turbulence import TurbulenceIndex
from regime_detection.lib.turbulence_regimes import (
    classify_regimes,
    smooth_min_duration,
)

__all__ = [
    "TurbulenceIndex",
    "classify_regimes",
    "smooth_min_duration",
    "EventTimeConverter",
    "get_overlapping_event_returns",
]

# --- from evaluation/__init__ ---
from regime_detection.lib.cutoff_sensitivity import (
    metrics_for_cutoff_run, run_sliding_cutoff, summary_statistics,
)
from regime_detection.lib.metrics import (
    annualised_return, annualised_vol, hit_rate_at_threshold,
    max_drawdown, sortino_ratio, top_decile_threshold,
)
from regime_detection.lib.oos_pipeline import (
    QuarterlyWalkForward, run_quarterly_walk_forward,
)
from regime_detection.lib.stats_tests import (
    block_bootstrap_sortino_diff, diebold_mariano,
    stationary_bootstrap_sortino_diff,
)

__all__ = [
    "QuarterlyWalkForward",
    "run_quarterly_walk_forward",
    "annualised_return",
    "annualised_vol",
    "sortino_ratio",
    "max_drawdown",
    "hit_rate_at_threshold",
    "top_decile_threshold",
    "diebold_mariano",
    "block_bootstrap_sortino_diff",
    "stationary_bootstrap_sortino_diff",
    "run_sliding_cutoff",
    "metrics_for_cutoff_run",
    "summary_statistics",
]

# --- from supervised/__init__ ---
from regime_detection.lib.features import build_feature_matrix
from regime_detection.lib.pgts import PurgedGroupTimeSeriesSplit
from regime_detection.lib.targets import (
    binary_turbulence_entry,
    event_time_drawdown,
    forward_drawdown_conditional,
)

__all__ = [
    "binary_turbulence_entry",
    "event_time_drawdown",
    "forward_drawdown_conditional",
    "build_feature_matrix",
    "PurgedGroupTimeSeriesSplit",
]

# --- from utils/__init__ ---
from regime_detection.lib.splits import (
    DEV_END,
    OOS_START,
    get_dev_test_split,
    assert_no_oos_contamination,
)
from regime_detection.lib.walk_forward import (
    RollingStandardizer,
    RollingPCA,
    RollingCovariance,
)

__all__ = [
    "DEV_END",
    "OOS_START",
    "get_dev_test_split",
    "assert_no_oos_contamination",
    "RollingStandardizer",
    "RollingPCA",
    "RollingCovariance",
]

# --- from explainability/__init__ ---
from regime_detection.lib.grid_prediction import PredictionGrid
from regime_detection.lib.rbi import (
    compute_rbi,
    compute_rbi_table,
    compute_tau_statistic,
)
from regime_detection.lib.relevance import RelevanceCalculator

__all__ = [
    "RelevanceCalculator",
    "PredictionGrid",
    "compute_rbi",
    "compute_rbi_table",
    "compute_tau_statistic",
]

# --- from data/__init__ ---
from regime_detection.lib.splicing import (
    FACTOR_DEFINITIONS, FACTOR_FALLBACKS,
    UNSPLICEABLE_FACTORS,
    build_spliced_factor_panel, splice_factor,
)

__all__ = [
    "FACTOR_DEFINITIONS",
    "FACTOR_FALLBACKS",
    "UNSPLICEABLE_FACTORS",
    "splice_factor",
    "build_spliced_factor_panel",
]

# --- from benchmarks/gmm/__init__ ---
from regime_detection.lib.gmm_walk_forward import (
    WalkForwardGMM, fit_walk_forward_gmm,
)
from regime_detection.lib.label_alignment import (
    align_labels_across_refits, hungarian_permutation,
)

__all__ = [
    "WalkForwardGMM",
    "fit_walk_forward_gmm",
    "align_labels_across_refits",
    "hungarian_permutation",
]

