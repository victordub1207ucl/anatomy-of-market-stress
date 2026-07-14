"""Regime-conditional causality: how the factor causal network differs
between turbulent and calm markets, and what drives turbulence escalation.

Descriptive (full-sample) characterisation conditioned on v5's *causal*
turbulence regime — an interpretation/causality contribution, not an OOS
predictor.  See ``explainability/causality.py`` for the look-ahead contract.

Run::

    python3 -m regime_detection.step16_causality

Outputs:
  * ``figures/causality/regime_causal_networks.png`` — turbulent vs calm
    factor Granger networks, side by side (shared layout).
  * ``figures/causality/turbulence_drivers.png`` — which factors Granger-
    cause turbulence escalation (full sample + per regime).
  * ``outputs/causality/*.csv`` — edge lists + driver tables.
  * ``reports/checkpoints/step16_causality.md`` — the writeup.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from regime_detection.lib.causality import (
    granger_causality_matrix, regime_conditional_granger,
    granger_drivers_of_turbulence,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "causality"
FIG_DIR = PROJECT_ROOT / "figures" / "causality"
CHECKPOINT_PATH = (PROJECT_ROOT / "reports"
                   / "checkpoints" / "step16_causality.md")

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MAX_LAG = 3
FDR_Q = 0.05


TOP_K = 12  # edges drawn per network panel


def _strength_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-edge Granger strength (-log10 BH-p) + within-regime percentile rank.

    At q<0.05 almost every edge is 'significant' (the sample is large), so
    significance is uninformative; we rank edges by Granger strength
    instead.  The within-regime percentile rank makes the cross-regime
    differential robust to the calm/turbulent power asymmetry (calm has ~5x
    the observations, hence systematically smaller p-values)."""
    s = df[["causing_factor", "caused_factor", "p_value_adj"]].copy()
    s["strength"] = -np.log10(s["p_value_adj"].clip(lower=1e-300))
    s["rank"] = s["strength"].rank(pct=True)
    return s.set_index(["causing_factor", "caused_factor"])


def _draw_network(ax, edges, pos, title: str, color="#c0392b"):
    """edges: iterable of (causing, caused, width)."""
    G = nx.DiGraph()
    G.add_nodes_from(FACTORS_10)
    for causing, caused, w in edges:
        G.add_edge(causing, caused, w=w)
    out_deg = dict(G.out_degree())
    node_sizes = [250 + 320 * out_deg.get(n, 0) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color="#cfe2f3", edgecolors="#2c3e50")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    if G.edges():
        widths = [G[u][v]["w"] for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=widths, edge_color=color,
            arrowsize=12, alpha=0.75, connectionstyle="arc3,rad=0.08",
            node_size=node_sizes)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return out_deg


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data ...")
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    rets = np.log(prices / prices.shift(1)).dropna(how="any")
    tdf = pd.read_parquet(TURB_PATH)
    turb, reg01 = tdf["turbulence"], tdf["regime_smoothed"]
    regime = reg01.map({0.0: "calm", 1.0: "turbulent"})
    common = rets.index.intersection(regime.dropna().index)
    rets, regime = rets.loc[common], regime.loc[common]
    print(f"  → {rets.shape}, calm {int((regime=='calm').sum())}d / "
          f"turbulent {int((regime=='turbulent').sum())}d")

    # ---- Analysis 1: regime-conditional factor causal network ----
    print("Regime-conditional Granger (episode-segmented, Fisher-combined) ...")
    rcg = regime_conditional_granger(rets, regime, max_lag=MAX_LAG, fdr_q=FDR_Q)
    S = {r: _strength_frame(df) for r, df in rcg.items()}
    for r, df in rcg.items():
        df.to_csv(OUT_DIR / f"granger_{r}.csv", index=False)

    calm_s, turb_s = S["calm"], S["turbulent"]
    # Differential by within-regime rank (sample-size robust).
    diff = (turb_s["rank"] - calm_s["rank"]).rename("d_rank")
    diff_str = pd.concat([diff,
                          turb_s["strength"].rename("turb_strength"),
                          calm_s["strength"].rename("calm_strength")], axis=1)

    def _top(frame_strength, k=TOP_K):
        top = frame_strength.sort_values("strength", ascending=False).head(k)
        wmax = top["strength"].max()
        return [(i[0], i[1], 0.6 + 3.5 * (row.strength / wmax))
                for i, row in top.iterrows()]

    calm_edges = _top(calm_s)
    turb_edges = _top(turb_s)
    # rewiring: edges most strengthened in turbulent vs calm (positive d_rank)
    rew = diff_str.sort_values("d_rank", ascending=False).head(TOP_K)
    rew_edges = [(i[0], i[1], 0.8 + 4.0 * row.d_rank)
                 for i, row in rew.iterrows()]

    pos = nx.circular_layout(sorted(FACTORS_10))
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    od_calm = _draw_network(axes[0], calm_edges, pos,
                            f"Calm regime — top {TOP_K} causal links",
                            color="#2980b9")
    od_turb = _draw_network(axes[1], turb_edges, pos,
                            f"Turbulent regime — top {TOP_K} causal links",
                            color="#c0392b")
    _draw_network(axes[2], rew_edges, pos,
                  f"Stress rewiring — top {TOP_K} links that\n"
                  "strengthen most in turbulence (by rank)", color="#8e44ad")
    fig.suptitle("Factor Granger-causal network by regime  "
                 "(arrow x→y: x Granger-causes y; edge width = Granger strength)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regime_causal_networks.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {FIG_DIR / 'regime_causal_networks.png'}")

    stats = {
        "calm": {"out_degree": od_calm,
                 "top_links": [(i[0], i[1]) for i, _ in
                               calm_s.sort_values("strength", ascending=False)
                               .head(5).iterrows()]},
        "turbulent": {"out_degree": od_turb,
                      "top_links": [(i[0], i[1]) for i, _ in
                                    turb_s.sort_values("strength", ascending=False)
                                    .head(5).iterrows()]},
        "rewiring": [(i[0], i[1]) for i, _ in rew.head(6).iterrows()],
    }
    diff_str.sort_values("d_rank", ascending=False).to_csv(
        OUT_DIR / "rewiring_differential.csv")

    # ---- Analysis 2: drivers of turbulence escalation ----
    print("Granger drivers of turbulence escalation (full sample + per regime) ...")
    drivers = granger_drivers_of_turbulence(
        rets, turb, max_lag=5, fdr_q=FDR_Q, regime_labels=regime)
    drivers.to_csv(OUT_DIR / "turbulence_drivers.csv", index=False)
    full = drivers[drivers["scope"] == "full_sample"].copy()
    full["neglog10p"] = -np.log10(full["p_value_adj"].clip(lower=1e-300))
    full = full.sort_values("neglog10p", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#c0392b" if s else "#bdc3c7" for s in full["significant"]]
    ax.barh(full["factor"], full["neglog10p"], color=colors)
    ax.axvline(-np.log10(0.05), color="grey", ls="--", lw=1,
               label="BH q=0.05")
    ax.set_xlabel(r"$-\log_{10}$ (BH-adjusted Granger p)")
    ax.set_title("What drives turbulence escalation\n"
                 "(factor returns Granger-causing $\\Delta\\log d_t$, full sample)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "turbulence_drivers.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {FIG_DIR / 'turbulence_drivers.png'}")

    write_checkpoint(stats, drivers)
    print(f"  → wrote {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def _hub_summary(out_deg: dict, k: int = 3) -> str:
    if not out_deg:
        return "—"
    top = sorted(out_deg.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return ", ".join(f"{n} ({d})" for n, d in top if d > 0) or "—"


def _fmt_links(links):
    return ", ".join(f"{a}→{b}" for a, b in links) or "—"


def write_checkpoint(stats, drivers):
    calm, turb = stats["calm"], stats["turbulent"]
    full = drivers[drivers["scope"] == "full_sample"]
    sig_drivers = full[full["significant"]].sort_values("p_value_adj")

    md = [
        "# CHECKPOINT — regime-conditional causality (clean v5 port)",
        "",
        f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC",
        "",
        "## What this is",
        "A descriptive characterisation of the factor causal structure, "
        "conditioned on v5's **causal** turbulence regime — ports the strong "
        "idea from v4's `causal.py` (regime-conditional Granger with "
        "contiguous-episode segmentation + Fisher combination) onto leak-free "
        "foundations. Interpretation/causality contribution, not an OOS "
        "predictor.",
        "",
        "## Method note",
        "At q<0.05 almost every directed edge is 'significant' (the sample is "
        "large), so significance is uninformative as a network filter. We "
        "therefore rank edges by **Granger strength** (-log10 BH-p) and report "
        "the strongest links per regime, plus a **rewiring** view: edges that "
        "rise most in *within-regime rank* from calm to turbulent (rank-based "
        "to neutralise the ~5x calm/turbulent sample-size asymmetry).",
        "",
        "## 1. Factor causal network by regime",
        "",
        f"- **Calm — top causal hubs (out-degree):** "
        f"{_hub_summary(calm['out_degree'])}; strongest links: "
        f"{_fmt_links(calm['top_links'])}.",
        f"- **Turbulent — top hubs:** {_hub_summary(turb['out_degree'])}; "
        f"strongest links: {_fmt_links(turb['top_links'])}.",
        f"- **Stress rewiring (links strengthening most in turbulence):** "
        f"{_fmt_links(stats['rewiring'])}.",
        "",
        "![networks](../../../figures/causality/regime_causal_networks.png)",
        "",
        "## 2. What drives turbulence escalation",
        "",
        "Factor returns Granger-causing the change in log-turbulence "
        "(full sample, BH-FDR q<0.05), ranked by strength:",
        "",
        sig_drivers[["factor", "p_value_raw", "p_value_adj"]].to_markdown(index=False)
        if not sig_drivers.empty else "_none significant_",
        "",
        "![drivers](../../../figures/causality/turbulence_drivers.png)",
        "",
        "## Caveats",
        "- Granger causality is predictive lead-lag, not structural "
        "intervention causality. The PC-algorithm DAG (v4's `estimate_dag`) "
        "was **not** ported: `causal-learn` is not installed — offer to add it "
        "as a structural complement.",
        "- Regime labels are causal, but the Granger estimation is full-sample "
        "within each regime — a stylised-fact characterisation, deliberately "
        "not a walk-forward predictor.",
    ]
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text("\n".join(md))


if __name__ == "__main__":
    raise SystemExit(main())
