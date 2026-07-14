"""Build (and execute) notebooks/step05_rbi_explainability.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "step05_rbi_explainability.ipynb"


HEADER_MD = """\
# 05 — Relevance-Based Importance

Phase 5 of the thesis pivot.  Replaces the legacy TreeSHAP explainer with
**Relevance-Based Importance** (Czasonis, Kritzman & Turkington 2024).
TreeSHAP is kept as the benchmark.

The notebook covers:

1. Per-feature RBI ranking + comparison against mean |SHAP| from the
   Phase-4 random forest.
2. Heatmap of the prediction-grid adjusted-fit (the substrate of RBI).
3. Per-regime RBI: which features matter in *quiet* vs *turbulent*
   stretches.
4. Time series of the top-3 RBI features over the dev period, so the
   thesis can pin specific features to specific crises.

All RBI Mahalanobis statistics are fit on the **dev partition only**
(index ≤ DEV_END = 2019-12-31; fixed 2020-01-01 OOS constraint); no
full-sample statistics leak into the importance ranking.
"""

SETUP = """\
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "regime_detection").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

plt.rcParams.update({
    "figure.figsize": (12, 4),
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

art_dir = PROJECT_ROOT / "reports" / "explainability"
out_dir = PROJECT_ROOT / "figures" / "explainability"
out_dir.mkdir(parents=True, exist_ok=True)

rbi = pd.read_parquet(art_dir / "rbi_values.parquet").sort_values(
    "rbi", ascending=False,
)
per_regime = pd.read_parquet(art_dir / "rbi_per_regime.parquet")
grid_adj_fit = pd.read_parquet(art_dir / "grid_adj_fit.parquet")
print(f"loaded {len(rbi)} feature rankings")
print(f"per-regime tables: {sorted(per_regime['regime'].unique().tolist())}")
"""

PLOT1_MD = """\
## 1. RBI vs |SHAP| — feature ranking

Two horizontal-bar panels: RBI on the left, mean |SHAP| from the
Phase-4 random forest on the right.  Features are listed in **RBI order**
so the visual reads: "this is how RBI ranks them; this is what SHAP
would have said".
"""

PLOT1_CODE = """\
fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)

# Sort by RBI ascending so the most important feature appears at the top.
ordered = rbi.sort_values("rbi", ascending=True)
axes[0].barh(ordered.index, ordered["rbi"], color="#1f77b4", alpha=0.85)
axes[0].set_xlabel("RBI (adj_fit difference)")
axes[0].set_title("Relevance-Based Importance")
axes[0].grid(axis="x", alpha=0.3)

# Same ordering on the SHAP panel for direct comparison.
shap_vals = ordered["mean_abs_shap"].fillna(0.0)
axes[1].barh(ordered.index, shap_vals, color="#ff7f0e", alpha=0.85)
axes[1].set_xlabel("mean |SHAP| (Phase-4 random forest)")
axes[1].set_title("TreeSHAP (benchmark)")
axes[1].grid(axis="x", alpha=0.3)

# Highlight features whose ranks disagree by 5 or more.
rank_diff = ordered["shap_rank"] - ordered["rbi_rank"]
for i, name in enumerate(ordered.index):
    diff = rank_diff.loc[name]
    if np.isfinite(diff) and abs(diff) >= 8:
        axes[0].annotate(f"Δrank={int(diff):+d}",
                         xy=(ordered.loc[name, 'rbi'], i),
                         xytext=(4, 0), textcoords="offset points",
                         color="#d62728", fontsize=8, va="center")

fig.suptitle("Feature ranking — RBI vs mean |SHAP|  "
             "(both computed on the dev partition)", fontsize=11)
fig.tight_layout()
fig.savefig(out_dir / "fig1_rbi_vs_shap.png", dpi=120)
plt.show()
"""

PLOT2_MD = """\
## 2. Rank correlation between RBI and SHAP

Spearman ρ of the two rankings.  If ρ is high but a handful of variables
disagree, RBI is picking up *conditional* importance that SHAP misses.
"""

PLOT2_CODE = """\
from scipy import stats as sp_stats
mask = rbi["mean_abs_shap"].notna()
rho, p_val = sp_stats.spearmanr(rbi.loc[mask, "rbi"],
                                rbi.loc[mask, "mean_abs_shap"])
print(f"Spearman ρ(RBI rank, |SHAP| rank) = {rho:.3f}  (p = {p_val:.4f})")
print(f"\\nVariables where |Δrank| ≥ 8:")
big_disagree = rbi[(rbi['shap_rank'] - rbi['rbi_rank']).abs() >= 8].sort_values('rbi_rank')
print(big_disagree[['rbi', 'rbi_rank', 'mean_abs_shap', 'shap_rank']].round(4).to_string())

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(rbi["rbi_rank"], rbi["shap_rank"], s=40, color="#1f77b4", alpha=0.8)
for name, row in rbi.iterrows():
    if abs(row["rbi_rank"] - row["shap_rank"]) >= 6:
        ax.annotate(name, xy=(row["rbi_rank"], row["shap_rank"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color="#d62728")
ax.plot([0, len(rbi)], [0, len(rbi)], color="grey", lw=0.8, ls="--",
        label="identity")
ax.set_xlabel("RBI rank")
ax.set_ylabel("|SHAP| rank")
ax.set_title(f"Rank-comparison scatter  (Spearman ρ = {rho:.3f})")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "fig2_rank_scatter.png", dpi=120)
plt.show()
"""

PLOT3_MD = """\
## 3. Adjusted-fit heatmap

Rows = variable subsets (singletons listed first, full-feature subset
last).  Columns = censoring thresholds (0, 0.2, 0.5, 0.8).  Each cell is
the leave-one-out R² of the relevance-weighted predictor on the
evaluation sample.  Reading: which subsets *gain* the most when censoring
to the most-relevant neighbours, and which lose.
"""

PLOT3_CODE = """\
fig, ax = plt.subplots(figsize=(8, 12))
data = grid_adj_fit.copy()
im = ax.imshow(data.values, aspect="auto", cmap="viridis",
               interpolation="nearest", vmin=0.0,
               vmax=max(0.05, float(np.nanmax(data.values))))
ax.set_xticks(range(len(data.columns)))
ax.set_xticklabels([f"censor={c:.1f}" for c in data.columns])
ax.set_yticks(range(len(data.index)))
ax.set_yticklabels(data.index, fontsize=7)
ax.set_title("Adjusted-fit grid (prediction sample, leave-one-out R²)")
fig.colorbar(im, ax=ax, fraction=0.04)
fig.tight_layout()
fig.savefig(out_dir / "fig3_adj_fit_heatmap.png", dpi=120)
plt.show()
"""

PLOT4_MD = """\
## 4. Per-regime RBI — what matters when

Per the Phase-2 smoothed turbulence regime: subset the prediction sample
to *quiet* (regime = 0) or *turbulent* (regime = 1) rows and recompute
the RBI ranking.  This isolates which features carry conditional
information — the principal claim of the RBI paper that SHAP cannot
recover.
"""

PLOT4_CODE = """\
quiet  = per_regime[per_regime["regime"] == 0].set_index("variable").sort_values("rbi", ascending=True)
turbu  = per_regime[per_regime["regime"] == 1].set_index("variable").sort_values("rbi", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
axes[0].barh(quiet.index,  quiet["rbi"],  color="#1f77b4", alpha=0.85)
axes[0].set_xlabel("RBI (regime = 0, quiet)")
axes[0].set_title("Quiet regime — top features")
axes[0].grid(axis="x", alpha=0.3)

axes[1].barh(turbu.index,  turbu["rbi"],  color="#d62728", alpha=0.85)
axes[1].set_xlabel("RBI (regime = 1, turbulent)")
axes[1].set_title("Turbulent regime — top features")
axes[1].grid(axis="x", alpha=0.3)
fig.suptitle("Per-regime variable importance — features in same RBI order each panel",
             fontsize=11)
fig.tight_layout()
fig.savefig(out_dir / "fig4_per_regime_rbi.png", dpi=120)
plt.show()

# Print the regime-conditional swap: which variables moved most in rank
# between quiet and turbulent.
combined = pd.DataFrame({
    "rbi_quiet": quiet["rbi"],
    "rbi_turbulent": turbu["rbi"],
})
combined["rank_quiet"] = combined["rbi_quiet"].rank(ascending=False)
combined["rank_turbulent"] = combined["rbi_turbulent"].rank(ascending=False)
combined["rank_swing"] = combined["rank_quiet"] - combined["rank_turbulent"]
print("\\nLargest regime-conditional rank swings (positive = more important in turbulent):")
print(combined.dropna().sort_values("rank_swing").head(8).round(3).to_string())
print()
print("(negative swing = feature de-emphasised in turbulent regime)")
"""

PLOT5_MD = """\
## 5. Conclusions

Persisted artefacts (under `artifacts/explainability/`):

* `rbi_values.parquet` — feature ranking with `rbi`, `tau`,
  `informativeness`, `mean_abs_shap`, and both ranks.
* `rbi_per_regime.parquet` — per-regime breakdown.
* `grid_adj_fit.parquet` — full subset × censor adjusted-fit matrix.
* `rbi_run_meta.json` — run configuration (n_subsets, n_eval, R²,
  shrinkage intensity).

Figures (under `outputs/explainability/`):

* `fig1_rbi_vs_shap.png`
* `fig2_rank_scatter.png`
* `fig3_adj_fit_heatmap.png`
* `fig4_per_regime_rbi.png`

The headline thesis claim — *RBI captures conditional importance that
SHAP underweights* — is testable from the Spearman ρ and the rank
swings printed above.  Variables that move many positions between RBI
and SHAP, or between quiet and turbulent regimes, are the candidates to
spotlight in the thesis narrative.
"""


def make_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                        "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb.cells = [
        nbformat.v4.new_markdown_cell(HEADER_MD),
        nbformat.v4.new_code_cell(SETUP),
        nbformat.v4.new_markdown_cell(PLOT1_MD),
        nbformat.v4.new_code_cell(PLOT1_CODE),
        nbformat.v4.new_markdown_cell(PLOT2_MD),
        nbformat.v4.new_code_cell(PLOT2_CODE),
        nbformat.v4.new_markdown_cell(PLOT3_MD),
        nbformat.v4.new_code_cell(PLOT3_CODE),
        nbformat.v4.new_markdown_cell(PLOT4_MD),
        nbformat.v4.new_code_cell(PLOT4_CODE),
        nbformat.v4.new_markdown_cell(PLOT5_MD),
    ]
    return nb


def main() -> None:
    nb = make_notebook()
    print(f"executing notebook ({len(nb.cells)} cells) ...")
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
