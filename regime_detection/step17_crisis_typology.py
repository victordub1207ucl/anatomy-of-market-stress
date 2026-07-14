"""Crisis typology: cluster ALL high-turbulence episodes by composition.

Extracts every high-turbulence episode 2007-2026 (causal rolling-quantile
threshold), characterises each by its per-factor turbulence-composition
fingerprint, clusters them into an empirical taxonomy of stress types,
and tests whether spike composition is predictable from the composition
of incipient turbulence (the "what kind of stress is coming" question).

Run::

    python3 -m regime_detection.step17_crisis_typology
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime_detection.lib.crisis_typology import (
    cluster_episodes, composition_predictability, episode_composition,
    extract_episodes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
CONTRIB_PATH = (PROJECT_ROOT / "reports" / "decomposition"
                / "turbulence_contributions.parquet")
OUT_DIR = PROJECT_ROOT / "reports" / "crisis_typology"
FIG_DIR = PROJECT_ROOT / "figures" / "crisis_typology"
CHECKPOINT_PATH = (PROJECT_ROOT / "reports"
                   / "checkpoints" / "step17_crisis_typology.md")

NAMED_CRISES = {
    "2008 GFC":         ("2008-09-01", "2009-03-31"),
    "2011 Euro/US dg":  ("2011-08-01", "2011-10-31"),
    "2015 China/oil":   ("2015-08-01", "2016-02-29"),
    "2018 Volmageddon": ("2018-02-01", "2018-02-28"),
    "2018 Q4":          ("2018-10-01", "2018-12-31"),
    "2020 COVID":       ("2020-02-15", "2020-05-31"),
    "2022 inflation":   ("2022-01-01", "2022-12-31"),
}


def _annotate_named(episodes: pd.DataFrame) -> pd.Series:
    """Map each episode to an overlapping named crisis (or '')."""
    out = []
    for _, ep in episodes.iterrows():
        tag = ""
        for name, (s, e) in NAMED_CRISES.items():
            if ep["start"] <= pd.Timestamp(e) and ep["end"] >= pd.Timestamp(s):
                tag = name
                break
        out.append(tag)
    return pd.Series(out, index=episodes.index)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading turbulence + decomposition ...")
    turb = pd.read_parquet(TURB_PATH)["turbulence"]
    if not CONTRIB_PATH.exists():
        raise SystemExit("Run run_turbulence_decomposition first "
                         f"(missing {CONTRIB_PATH})")
    contrib = pd.read_parquet(CONTRIB_PATH)

    print("Extracting high-turbulence episodes (causal rolling q90) ...")
    # min_periods=252 (not 504): the turbulence series starts 2007-12, so a
    # 504-day threshold warm-up would exclude the GFC from the taxonomy.
    episodes = extract_episodes(turb, min_periods=252)
    episodes["named"] = _annotate_named(episodes).values
    print(f"  → {len(episodes)} episodes, "
          f"{episodes['n_days'].sum()} total days, "
          f"median length {episodes['n_days'].median():.0f}d")

    comp = episode_composition(contrib, episodes)
    episodes = episodes[episodes["start"].isin(comp.index)].reset_index(drop=True)

    print("Clustering compositions (Ward + silhouette) ...")
    labels, k, Z, names = cluster_episodes(comp)
    episodes["cluster"] = labels.values
    episodes["type"] = episodes["cluster"].map(names)
    print(f"  → k={k} types: "
          + "; ".join(f"{names[c]} (n={int((labels==c).sum())})"
                      for c in sorted(names)))
    for _, ep in episodes[episodes["named"] != ""].iterrows():
        print(f"     {ep['named']:18s} → {ep['type']}")

    print("Testing composition predictability (pre-spike → spike) ...")
    pred = composition_predictability(contrib, episodes, comp)
    print(f"  → mean cosine {pred['mean_cosine_observed']:.3f} vs null "
          f"{pred['mean_cosine_null']:.3f}, p={pred['p_value']:.4f} "
          f"(n={pred['n_episodes']})")

    episodes.to_csv(OUT_DIR / "episodes.csv", index=False)
    comp.to_csv(OUT_DIR / "compositions.csv")

    # ---- figure 1: composition heatmap sorted by cluster ----
    order = episodes.sort_values(["cluster", "start"]).index
    M = comp.iloc[order]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.18 * len(M))))
    im = ax.imshow(M.values, aspect="auto", cmap="Reds", vmin=0,
                   vmax=np.nanpercentile(M.values, 98))
    ax.set_xticks(range(len(M.columns)))
    ax.set_xticklabels(M.columns, rotation=40, ha="right", fontsize=8)
    ylabels = []
    for i in order:
        ep = episodes.loc[i]
        tag = f"  ◀ {ep['named']}" if ep["named"] else ""
        ylabels.append(f"{ep['start']:%Y-%m} [{ep['type']}]{tag}")
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_title(f"Stress-episode taxonomy — {len(M)} episodes, "
                 f"k={k} types (share of episode turbulence)")
    fig.colorbar(im, ax=ax, label="composition share")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "episode_taxonomy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: timeline by type + predictability null ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for j, c in enumerate(sorted(names)):
        sub = episodes[episodes["cluster"] == c]
        ax.scatter(sub["start"], sub["mean_turb"],
                   s=20 + 4 * sub["n_days"], color=palette[j],
                   alpha=0.75, label=names[c])
    for _, ep in episodes[episodes["named"] != ""].iterrows():
        ax.annotate(ep["named"], (ep["start"], ep["mean_turb"]),
                    fontsize=6, rotation=30, xytext=(2, 6),
                    textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylabel("episode mean d_t (log)")
    ax.set_title("Stress episodes over time, coloured by type "
                 "(size = duration)")
    ax.legend(fontsize=7)

    ax = axes[1]
    rng = np.random.default_rng(0)
    ax.hist([pred["mean_cosine_null"]], bins=1, alpha=0)  # keep axes simple
    ax.axvline(pred["mean_cosine_null"], color="grey", ls="--",
               label=f"null mean {pred['mean_cosine_null']:.3f}")
    ax.axvline(pred["mean_cosine_observed"], color="#c0392b", lw=2,
               label=f"observed {pred['mean_cosine_observed']:.3f} "
                     f"(p={pred['p_value']:.4f})")
    ax.set_xlim(0, 1)
    ax.set_title("Is spike composition predictable from the 21d run-up?")
    ax.set_xlabel("mean cosine(pre-spike composition, spike composition)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "typology_timeline_predictability.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved figures to {FIG_DIR}")

    # ---- checkpoint ----
    centroids = comp.groupby(labels.values).mean()
    centroids.index = [names[c] for c in centroids.index]
    sig = "**YES**" if pred["p_value"] < 0.05 else "**NO**"
    md = [
        "# CHECKPOINT — crisis typology from decomposed turbulence",
        "",
        f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC",
        "",
        "## What was done",
        f"Extracted **{len(episodes)} high-turbulence episodes** (causal "
        "rolling-q90 threshold, gaps ≤5d merged, ≥5d long) over "
        f"{episodes['start'].min():%Y-%m} → {episodes['end'].max():%Y-%m}, "
        "characterised each by its per-factor turbulence-composition "
        f"fingerprint, and clustered (Ward, silhouette-selected k={k}).",
        "",
        "## The taxonomy",
        "",
        centroids.round(3).to_markdown(),
        "",
        "Named crises map to: "
        + "; ".join(f"{ep['named']} → {ep['type']}"
                    for _, ep in episodes[episodes['named'] != ''].iterrows()),
        "",
        "![taxonomy](../../../figures/crisis_typology/episode_taxonomy.png)",
        "![timeline](../../../figures/crisis_typology/typology_timeline_predictability.png)",
        "",
        "## Is the *kind* of stress predictable?",
        "",
        f"Cosine similarity between the 21-day pre-spike composition and "
        f"the realised spike composition: **{pred['mean_cosine_observed']:.3f}** "
        f"vs permutation null {pred['mean_cosine_null']:.3f} "
        f"(p = {pred['p_value']:.4f}, n = {pred['n_episodes']}). "
        f"Composition predictability: {sig}.",
        "",
        "## Why this matters",
        "Phase step15_composition_features showed composition does not improve *when* "
        "prediction. This asks the complementary, actionable question: "
        "*what kind* of stress — which determines **which hedge** to hold.",
    ]
    CHECKPOINT_PATH.write_text("\n".join(md))
    print(f"  → wrote {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
