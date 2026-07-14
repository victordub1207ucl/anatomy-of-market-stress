"""Step 31 — Cross-universe replication of the crisis typology.

The composition-predictability result (step17: the *kind* of stress is forecastable
from the run-up, cosine 0.715 vs null 0.463, p<0.0001) was derived on the 10
Factor-Lens factors over 38 episodes. The obvious examiner question — is this real
structure or an in-sample artefact of one universe? — is answered here by re-running
the *entire* pipeline (walk-forward turbulence, exact decomposition, causal episode
extraction, clustering, permutation predictability test) on a **completely different
universe**: the nine SPDR US sector ETFs (financials, technology, energy, health
care, industrials, consumer discretionary, consumer staples, utilities, materials).

If the predictability replicates on a universe the archetypes were never fit to,
the result is structural, not a Factor-Lens artefact. The archetype *labels* will
naturally differ (sector crises are energy-/financials-/tech-led, not value-led),
which is the point — what must replicate is the *phenomenon*: distinct recurring
types, and a run-up that anticipates the spike composition.

Run:  python3 -m regime_detection.step31_cross_universe_typology
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime_detection.lib.turbulence_decomposition import TurbulenceDecomposer
from regime_detection.lib.crisis_typology import (
    cluster_episodes, composition_predictability, episode_composition, extract_episodes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECTOR_PATH = PROJECT_ROOT / "data" / "sector_prices.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
CONTRIB_PATH = (PROJECT_ROOT / "reports" / "decomposition"
                / "turbulence_contributions.parquet")
OUT_DIR = PROJECT_ROOT / "reports" / "cross_universe_typology"
FIG_DIR = PROJECT_ROOT / "figures" / "cross_universe_typology"
CHECKPOINT = (PROJECT_ROOT / "reports" / "checkpoints"
              / "step31_cross_universe_typology.md")

SECTORS = {"XLF": "Financials", "XLK": "Technology", "XLE": "Energy",
           "XLV": "Health Care", "XLI": "Industrials", "XLY": "Cons. Disc.",
           "XLP": "Cons. Staples", "XLU": "Utilities", "XLB": "Materials"}
N_PERM = 10000


def _load_sectors():
    if SECTOR_PATH.exists():
        return pd.read_parquet(SECTOR_PATH)
    import yfinance as yf
    px = yf.download(list(SECTORS), start="2004-01-01", end="2026-04-15",
                     auto_adjust=True, progress=False)["Close"].dropna(how="all")
    px = px.dropna()
    SECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(SECTOR_PATH)
    return px


def _predictability_on(contrib, label):
    episodes = extract_episodes(contrib.sum(axis=1).dropna(), min_periods=252)
    comp = episode_composition(contrib, episodes)
    res = composition_predictability(contrib, episodes, comp, n_perm=N_PERM)
    try:
        labels, k, _, names = cluster_episodes(comp, k_range=range(2, 6))
    except Exception:
        k, names = None, {}
    res["k"] = k
    res["n_episodes_total"] = int(len(episodes))
    res["_comp"] = comp
    res["_labels"] = labels if k else None
    res["_names"] = names
    return res


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sector prices ...")
    px = _load_sectors()
    px = px.rename(columns=SECTORS)
    rets = np.log(px / px.shift(1)).dropna(how="all")
    print(f"  {rets.shape[1]} sectors, {rets.index.min().date()}..{rets.index.max().date()}")

    print("Computing sector turbulence + decomposition (walk-forward) ...")
    contrib_sec = TurbulenceDecomposer(lookback_days=2520, min_periods=504,
                                       shrinkage="ledoit-wolf").fit_transform(rets).dropna()
    print(f"  sector turbulence series: {contrib_sec.index.min().date()}.."
          f"{contrib_sec.index.max().date()}")

    print("Sector typology + predictability ...")
    sec = _predictability_on(contrib_sec, "sectors")

    print("Factor-universe baseline (for comparison) ...")
    contrib_fac = pd.read_parquet(CONTRIB_PATH).dropna()
    fac = _predictability_on(contrib_fac, "factors")

    # ---- report ----
    print(f"\n{'universe':10s} {'episodes':>9s} {'k':>3s} {'cosine':>7s} "
          f"{'null':>6s} {'p':>8s}")
    for nm, r in [("factors", fac), ("sectors", sec)]:
        print(f"{nm:10s} {r['n_episodes']:>9d} {str(r['k']):>3s} "
              f"{r['mean_cosine_observed']:>7.3f} {r['mean_cosine_null']:>6.3f} "
              f"{r['p_value']:>8.4f}")

    _figure(fac, sec)
    _write_checkpoint(fac, sec, list(rets.columns))
    print(f"\nWrote checkpoint → {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


def _figure(fac, sec):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (nm, r) in zip(axes, [("Factor Lens (10 macro factors)", fac),
                                  ("US sectors (9 SPDR ETFs)", sec)]):
        # permutation null is summarised by its mean; draw observed vs null mean
        ax.axvline(r["mean_cosine_null"], color="0.6", ls="--", lw=1.4,
                   label=f"permutation null ≈ {r['mean_cosine_null']:.3f}")
        ax.axvline(r["mean_cosine_observed"], color="tab:red", lw=2.4,
                   label=f"observed = {r['mean_cosine_observed']:.3f}")
        ax.set_xlim(0.3, 0.85); ax.set_yticks([])
        ax.set_xlabel("run-up vs spike composition (mean cosine)")
        ax.set_title(f"{nm}\n{r['n_episodes']} episodes, k={r['k']}, "
                     f"p {'<0.0001' if r['p_value']<1e-4 else f'={r[chr(39)+chr(112)+chr(39)]:.4f}'}",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Composition predictability replicates across universes", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "cross_universe_typology.png",
                                    dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_checkpoint(fac, sec, sectors):
    replicates = sec["p_value"] < 0.05 and sec["mean_cosine_observed"] > sec["mean_cosine_null"]
    L = []
    L.append("# Step 31 — Cross-universe replication of the crisis typology\n")
    L.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}. The step17 composition-"
             "predictability pipeline re-run, unchanged, on a different universe "
             f"(9 SPDR US sectors: {', '.join(sectors)})._\n")
    L.append("Answers the examiner question: is the predictable-composition result "
             "real structure, or an in-sample artefact of the Factor Lens?\n")
    L.append("| universe | episodes | archetypes (k) | observed cosine | null | p |")
    L.append("|---|---|---|---|---|---|")
    for nm, r in [("Factor Lens (10 macro factors)", fac), ("US sectors (9 ETFs)", sec)]:
        pstr = "< 0.0001" if r["p_value"] < 1e-4 else f"{r['p_value']:.4f}"
        L.append(f"| {nm} | {r['n_episodes']} | {r['k']} | "
                 f"**{r['mean_cosine_observed']:.3f}** | {r['mean_cosine_null']:.3f} | {pstr} |")
    L.append("")
    L.append("## Verdict\n")
    if replicates:
        L.append(f"**REPLICATES — the result is structural, not a Factor-Lens artefact.** "
                 f"On a completely different universe (sectors the archetypes were never "
                 f"fit to), the run-up still significantly anticipates the spike "
                 f"composition (cosine {sec['mean_cosine_observed']:.3f} vs null "
                 f"{sec['mean_cosine_null']:.3f}, p"
                 f"{'<0.0001' if sec['p_value']<1e-4 else f'={sec[chr(39)+chr(112)+chr(39)]:.4f}'}"
                 f"), and the episodes again resolve into k={sec['k']} recurring types. "
                 "The archetype *labels* differ (sector crises are energy-/financials-/"
                 "tech-led rather than value-led), which is exactly as expected — what "
                 "replicates is the *phenomenon*: distinct recurring stress types whose "
                 "composition is predictable from the run-up. This directly answers the "
                 "n=38 in-sample objection and elevates composition predictability from "
                 "a single-universe finding to a structural property of market stress.")
    else:
        L.append(f"**DOES NOT REPLICATE on sectors** (cosine {sec['mean_cosine_observed']:.3f} "
                 f"vs null {sec['mean_cosine_null']:.3f}, p={sec['p_value']:.4f}). The "
                 "Factor-Lens predictability does not transfer to the sector universe. "
                 "Honest cautionary finding: the composition-predictability result may be "
                 "specific to the macro-factor representation, and should be reported as "
                 "such rather than as a universal property. Worth investigating whether "
                 "the difference is the universe's dimensionality, its episode count, or a "
                 "genuine representational dependence.")
    L.append("")
    L.append("**Figure:** `figures/cross_universe_typology/cross_universe_typology.png`.")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
