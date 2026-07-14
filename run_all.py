#!/usr/bin/env python3
"""Centralised runner for the v5 regime-detection project.

One command to (optionally) run the test suite, regenerate the core
analyses, and build a single RESULTS_INDEX.md that links every figure and
every plain-English checkpoint report.

Usage (from this folder, latestversion/)::

    python3 run_all.py              # tests + fast analyses (default, ~3 min)
    python3 run_all.py --full       # also the slow 49-cutoff experiments (~25 min)
    python3 run_all.py --no-tests   # skip the test suite
    python3 run_all.py --index-only # just rebuild RESULTS_INDEX.md, run nothing

Each analysis writes figures to figures/<group>/ and a writeup to
reports/checkpoints/<name>.md.  Open RESULTS_INDEX.md
afterwards to navigate everything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # the python running this script

# (label, module, group, slow?)  module=None means a special step.
STEPS = [
    ("step02: turbulence sanity check", "regime_detection.step02_turbulence_sanity_check", "turbulence", False),
    ("step14: decomposed turbulence", "regime_detection.step14_turbulence_decomposition", "decomposition", False),
    ("step16: regime-conditional causality", "regime_detection.step16_causality", "causality", False),
    ("step17: crisis typology", "regime_detection.step17_crisis_typology", "crisis_typology", False),
    ("step18: signed turbulence", "regime_detection.extras.step18_signed_turbulence", "signed_turbulence", False),
    ("step06: OOS quarterly evaluation", "regime_detection.step06_oos_evaluation", "thesis_v5", False),
    ("step12: ordinal-bucket fix", "regime_detection.extras.step12_ordinal_bucket", "thesis_v5", True),
    ("step15: composition-as-features", "regime_detection.extras.step15_composition_features", "decomposition", True),
    ("step10: sliding-cutoff robustness", "regime_detection.extras.step10_cutoff_robustness", "thesis_v5", True),
    ("step19: regime-threshold sensitivity", "regime_detection.extras.step19_threshold_sensitivity", "threshold_sensitivity", True),
    ("step20: correction-timing overlay", "regime_detection.extras.step20_correction_overlay", "correction_overlay", True),
    ("step21: event-time inference", "regime_detection.extras.step21_event_time_inference", "event_time_inference", True),
    ("step22: MSCI World benchmark", "regime_detection.extras.step22_msci_benchmark", "msci_benchmark", True),
    ("step23: absorption ratio", "regime_detection.extras.step23_absorption_ratio", "absorption_ratio", True),
    ("step24: cross-market pooling", "regime_detection.extras.step24_cross_market_pooling", "cross_market_pooling", True),
    ("step25: event-time x pooling", "regime_detection.extras.step25_eventtime_pooling", "eventtime_pooling", True),
    ("step26: typology-conditioned hedge", "regime_detection.extras.step26_typology_hedge", "typology_hedge", True),
    ("step27: interpretation on the event clock", "regime_detection.extras.step27_event_clock_interpretation", "event_clock_interpretation", True),
    ("step28: turbulence + event-time features", "regime_detection.extras.step28_eventtime_features", "eventtime_features", True),
    ("step29: event-time-feature robustness", "regime_detection.extras.step29_eventtime_feature_robustness", "eventtime_robustness", True),
    ("step30: per-market event clocks", "regime_detection.extras.step30_per_market_event_clocks", "per_market_event_clocks", True),
    ("step31: cross-universe typology", "regime_detection.extras.step31_cross_universe_typology", "cross_universe_typology", True),
    ("step32: persistence baseline", "regime_detection.step32_persistence_baseline", "persistence_baseline", True),
    ("step33: OOS composition forecast", "regime_detection.extras.step33_oos_composition", "oos_composition", True),
    ("step34: examiner robustness", "regime_detection.extras.step34_examiner_robustness", "examiner_robustness", True),
    ("step35: examiner robustness 2", "regime_detection.extras.step35_examiner_robustness2", "examiner_robustness2", True),
    ("step36: cluster-robust trend", "regime_detection.step36_cluster_robust_trend", "cluster_robust_trend", True),
    ("step37: confound + placebo", "regime_detection.step37_confound_and_placebo", "confound_placebo", True),
    ("step38: connectedness + partial-R2 + CoDA", "regime_detection.extras.step38_examiner_implementations", "examiner_implementations", True),
    ("step39: de-risk timing vs static hedge", "regime_detection.step39_derisk_timing", "derisk_timing", True),
    ("step40: correlation surprise (KT2014)", "regime_detection.extras.step40_correlation_surprise", "correlation_surprise", True),
]

# The strict-necessary pipeline: every number in the simplified report
# (docs/Anatomy_of_Market_Stress_Report.pdf) comes from these steps alone.
CORE = {
    "step02: turbulence sanity check",          # the index + validation
    "step14: decomposed turbulence",            # fingerprints
    "step16: regime-conditional causality",     # what leads the build-up
    "step17: crisis typology",                  # episodes + three types
    "step06: OOS quarterly evaluation",         # supervised timing (the null)
    "step32: persistence baseline",             # run-up vs its own past
    "step36: cluster-robust trend",             # the climb, tested at crisis level
    "step37: confound + placebo",               # placebo + fixed-covariance check
    "step39: de-risk timing vs static hedge",   # the pre-registered test
}


def run_module(module: str) -> tuple[bool, float, str]:
    t0 = time.time()
    proc = subprocess.run([PY, "-m", module], cwd=ROOT,
                          capture_output=True, text=True)
    dt = time.time() - t0
    ok = proc.returncode == 0
    tail = (proc.stdout or "")[-400:] if ok else (proc.stderr or proc.stdout or "")[-800:]
    return ok, dt, tail


def run_tests() -> bool:
    print("─" * 70)
    print("▶ Test suite (no-look-ahead invariant + all modules)")
    proc = subprocess.run([PY, "-m", "pytest", "regime_detection/tests/", "-q"],
                          cwd=ROOT, capture_output=True, text=True)
    line = [l for l in (proc.stdout or "").splitlines() if "passed" in l or "failed" in l]
    print("  " + (line[-1] if line else "(no summary)"))
    return proc.returncode == 0


def build_index() -> Path:
    """Scan figures/ and checkpoints/ and write RESULTS_INDEX.md."""
    lines = ["# Results Index", "",
             "_Auto-generated by `run_all.py`. Open the figures and the "
             "checkpoint writeups below._", ""]

    lines.append("## Checkpoint reports (plain-English findings)")
    lines.append("")
    cp_dir = ROOT / "reports" / "checkpoints"
    for md in sorted(cp_dir.glob("*.md")):
        lines.append(f"- [{md.stem}]({md.relative_to(ROOT)})")
    for extra in ["reports/step06_oos_results.md", "reports/step07_ablations.md",
                  "reports/supervisor_meeting_summary.md"]:
        if (ROOT / extra).exists():
            lines.append(f"- [{Path(extra).stem}]({extra})")
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    fig_root = ROOT / "figures"
    for group in sorted(p for p in fig_root.iterdir() if p.is_dir()):
        pngs = sorted(group.glob("*.png"))
        if not pngs:
            continue
        lines.append(f"### {group.name}")
        for png in pngs:
            lines.append(f"- `{png.relative_to(ROOT)}`")
        lines.append("")

    lines.append("## Top-level guides")
    lines.append("")
    for g in ["WHITE_PAPER.md", "00_START_HERE_MASTER_GUIDE.md",
              "PROJECT_RECAP.md", "README.md"]:
        if (ROOT / g).exists():
            lines.append(f"- [{g}]({g})")

    out = ROOT / "RESULTS_INDEX.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="include slow experiments")
    ap.add_argument("--core", action="store_true",
                    help="run only the strict-necessary pipeline behind the simplified report")
    ap.add_argument("--no-tests", action="store_true", help="skip the test suite")
    ap.add_argument("--index-only", action="store_true",
                    help="only rebuild RESULTS_INDEX.md")
    args = ap.parse_args()

    if args.index_only:
        idx = build_index()
        print(f"Wrote {idx.relative_to(ROOT)}")
        return 0

    print("=" * 70)
    print("  v5 Regime Detection — run_all")
    print("=" * 70)

    if not args.no_tests:
        if not run_tests():
            print("\n✗ Tests failed — fix before trusting results. Aborting.")
            return 1

    results = []
    for label, module, group, slow in STEPS:
        if args.core and label not in CORE:
            results.append((label, "skipped (not core)", 0.0))
            continue
        if slow and not args.full and not (args.core and label in CORE):
            results.append((label, "skipped (use --full)", 0.0))
            continue
        print("─" * 70)
        print(f"▶ {label}")
        ok, dt, tail = run_module(module)
        status = "ok" if ok else "FAILED"
        print(f"  {status} in {dt:.0f}s")
        if not ok:
            print("  ── error tail ──")
            for ln in tail.splitlines()[-12:]:
                print("   " + ln)
        results.append((label, status, dt))

    idx = build_index()

    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    for label, status, dt in results:
        t = f"{dt:.0f}s" if dt else ""
        print(f"  {status:>20s}  {t:>6s}  {label}")
    print("")
    print(f"  → Results index:  {idx.relative_to(ROOT)}")
    print(f"  → Figures:        figures/<group>/*.png")
    print(f"  → Writeups:       reports/checkpoints/*.md")
    print("")
    print("  Open the index:   open RESULTS_INDEX.md")
    any_failed = any(s == "FAILED" for _, s, _ in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
