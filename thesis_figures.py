"""Generator for the figures used in the thesis.

Rebuilds every figure referenced by the LaTeX (fig_*.png) into thesisdrafts/figures/
in one consistent publication style, from the project's real data outputs. Fixes the
three reviewer-flagged issues: the disowned 5e-21 on the boundary figure, the illegible
causal-network and taxonomy panels, and the "Step N" pipeline titles.

Run:  python3 thesis_figures.py [--out DIR]
"""
from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

# Paths are resolved against the repository root, where this script lives.
# --out writes the PNGs somewhere else (e.g. a LaTeX figures/ directory).
import argparse as _argparse
ROOT = Path(__file__).resolve().parent
DATA = ROOT
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--out", default=str(ROOT / "figures" / "thesis"))
OUT = Path(_ap.parse_known_args()[0].out)
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
})

FAC = ["equity","rates","credit","commodities","em_equity","fx_usd",
       "inflation","value","quality","vix"]
NAME = {"equity":"equity","rates":"rates","credit":"credit","commodities":"commodities",
        "em_equity":"EM eq","fx_usd":"USD","inflation":"inflation","value":"value",
        "quality":"quality","vix":"VIX"}
PAL = dict(zip(FAC, plt.get_cmap("tab10").colors))
RED, GREEN, BLUE, GREY = "#b2182b", "#1b7837", "#2166ac", "#777777"

def rd(p):  # read parquet/csv, resolved against the pipeline's output tree
    p = DATA / p
    return pd.read_parquet(p) if str(p).endswith("parquet") else pd.read_csv(p)

def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  ✓ {name}.png")

CRISES = {  # display name -> (start, end)
    "2008 GFC": ("2008-09-01","2009-03-31"), "2011 euro": ("2011-07-15","2011-10-31"),
    "2015 China/oil": ("2015-08-01","2016-02-29"), "2018 Volmageddon": ("2018-02-01","2018-02-28"),
    "2018 Q4": ("2018-10-01","2018-12-31"), "2020 COVID": ("2020-02-20","2020-04-15"),
    "2022 inflation": ("2022-01-01","2022-12-31"),
}

# ---------------------------------------------------------------- 1.1 boundary
def fig_boundary():
    fig, ax = plt.subplots(figsize=(12, 6)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
    ax.add_patch(plt.Rectangle((0.2,0.8),4.5,8.4, fc="#fdecea", ec="none"))
    ax.add_patch(plt.Rectangle((5.3,0.8),4.5,8.4, fc="#e9f3ec", ec="none"))
    ax.plot([5.0,5.0],[0.8,9.2], ls=(0,(6,4)), color="0.3", lw=2.5)
    fig.suptitle("The predictability boundary: what a stress signal can and cannot forecast",
                 fontsize=14, fontweight="bold", y=0.99)
    ax.text(2.45,9.55,"TRIGGER", ha="center", fontsize=20, fontweight="bold", color=RED)
    ax.text(2.45,9.0,"exogenous  ·  not forecastable", ha="center", fontsize=12, color=RED)
    ax.text(7.55,9.55,"ANATOMY", ha="center", fontsize=20, fontweight="bold", color=GREEN)
    ax.text(7.55,9.0,"endogenous  ·  forecastable", ha="center", fontsize=12, color=GREEN)
    AMBER = "#b8860b"
    L = [("Timing of equity corrections","AUC ≈ 0.48  (≈ chance)", RED),
         ("Direction of stress, for timing","signed turbulence adds nothing", RED),
         ("Risk-adjusted return edge","none, once COVID is held out", RED)]
    # the entry claim is marginal: its interval contains the naive-persistence floor,
    # so it is drawn straddling the divider in amber rather than inside the green panel.
    R = [("What KIND of stress (composition)","cosine 0.715, p<10⁻⁴ · replicates & holds OOS", GREEN),
         ("Which factors LEAD it","VIX Granger-leads — strongest, most stable", GREEN)]
    ys = [7.3, 5.7, 4.1]
    for (t,s,c),y in zip(L,ys):
        ax.add_patch(Circle((4.3,y+0.18),0.12, color=c))
        ax.text(4.0,y+0.18,t, ha="right", va="center", fontsize=13.5, fontweight="bold")
        ax.text(4.0,y-0.32,s, ha="right", va="center", fontsize=10.5, color="0.35")
    for (t,s,c),y in zip(R,ys[:2]):
        ax.add_patch(Circle((5.7,y+0.18),0.12, color=c))
        ax.text(6.0,y+0.18,t, ha="left", va="center", fontsize=13.5, fontweight="bold")
        ax.text(6.0,y-0.32,s, ha="left", va="center", fontsize=10.5, color="0.35")
    # marginal claim, on its own row centred on the divider
    ymarg = 2.45
    ax.add_patch(Circle((5.0,ymarg+0.18),0.12, color=AMBER))
    ax.text(5.0,ymarg+0.62,"That turbulence is forming", ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=AMBER)
    ax.text(5.0,ymarg-0.26,"entry AUC ≈ 0.71, but the interval [0.62, 0.81]",
            ha="center", va="center", fontsize=10.5, color="0.35")
    ax.text(5.0,ymarg-0.62,"contains the 0.63 naive-persistence floor — marginal",
            ha="center", va="center", fontsize=10.5, color="0.35")
    ax.annotate("", xy=(9.0,1.35), xytext=(1.0,1.35), arrowprops=dict(arrowstyle="-|>", color="0.4", lw=2))
    ax.text(0.5,1.0,"less predictable", fontsize=10.5, color="0.4")
    ax.text(9.2,1.0,"more predictable", ha="right", fontsize=10.5, color="0.4")
    save(fig, "fig_predictability_boundary")

# ---------------------------------------------------------------- 2.1 turbulence + regime
def fig_turbulence_regime():
    ts = rd("reports/turbulence/turbulence_series.parquet")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(ts.index, ts["turbulence"], color=BLUE, lw=0.8)
    reg = ts["regime_smoothed"].fillna(0).values if "regime_smoothed" in ts else ts["regime"].fillna(0).values
    ax.fill_between(ts.index, 0, ts["turbulence"].max()*1.05, where=reg>0.5,
                    color=RED, alpha=0.12, step="mid", lw=0)
    ax.set_ylim(0, ts["turbulence"].quantile(0.999)*1.1)
    ax.set_ylabel("turbulence $d_t$"); ax.set_xlabel("")
    for lbl,(s,_) in {"2008 GFC":CRISES["2008 GFC"],"2011":CRISES["2011 euro"],
                      "COVID":CRISES["2020 COVID"],"2022":CRISES["2022 inflation"]}.items():
        x = pd.Timestamp(s)
        if x >= ts.index.min():
            ax.text(x, ax.get_ylim()[1]*0.92, lbl, fontsize=9, color="0.4", rotation=0)
    ax.set_title("Walk-forward turbulence index with turbulent-regime shading")
    save(fig, "fig_turbulence_regime")

# ---------------------------------------------------------------- 2.2 density + qq
def fig_density_qq():
    from scipy import stats
    r = rd("reports/event_time/returns_by_time_base.parquet").dropna()
    cal = stats.zscore(r["calendar_21d_return"]); ev = stats.zscore(r["event_return"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    grid = np.linspace(-6, 6, 300)
    for d,lab,c in [(cal,"calendar (21d)",GREY),(ev,"event time",BLUE)]:
        kde = stats.gaussian_kde(d); ax[0].plot(grid, kde(grid), color=c, lw=2, label=lab)
    ax[0].plot(grid, stats.norm.pdf(grid), color="k", ls=":", lw=1.2, label="normal")
    ax[0].set_xlim(-6,6); ax[0].set_yscale("log"); ax[0].set_ylim(1e-4, 1)
    ax[0].set_title("Return density (standardised, log scale)"); ax[0].legend()
    for d,lab,c in [(cal,"calendar (21d)",GREY),(ev,"event time",BLUE)]:
        osm,osr = stats.probplot(d, dist="norm")[0]; ax[1].plot(osm, osr, ".", ms=3, color=c, label=lab, alpha=0.6)
    lim=[-6,6]; ax[1].plot(lim,lim,"k:",lw=1.2); ax[1].set_xlim(lim); ax[1].set_ylim(-12,12)
    ax[1].set_xlabel("theoretical quantiles"); ax[1].set_ylabel("sample quantiles")
    ax[1].set_title("Q–Q vs normal"); ax[1].legend()
    fig.suptitle("Calendar vs event-time returns — excess kurtosis 8.81 → 0.57 (−94%)", y=1.02, fontsize=12.5)
    save(fig, "fig_density_qq")

# ---------------------------------------------------------------- 4.1 drivers over time
def fig_turbulence_drivers():
    c = rd("reports/decomposition/turbulence_contributions.parquet").clip(lower=0)
    cm = c.rolling(21, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.stackplot(cm.index, [cm[f].values for f in FAC],
                 colors=[PAL[f] for f in FAC], labels=[NAME[f] for f in FAC], lw=0)
    ax.set_ylabel("turbulence contribution (21d mean)"); ax.set_xlim(cm.index.min(), cm.index.max())
    ax.set_ylim(0, cm.sum(axis=1).quantile(0.995))
    ax.legend(ncol=5, loc="upper left", fontsize=9)
    ax.set_title("Per-factor decomposition of turbulence over time")
    save(fig, "fig_turbulence_drivers")

# ---------------------------------------------------------------- 4.2 crisis fingerprints
def fig_crisis_drivers():
    # Shares are the EXACT SIGNED shares m_i/sum(m), matching Table 4.2. An earlier
    # version clipped negatives to zero before normalising, which diluted every
    # dominant factor (Volmageddon read 65% VIX against the table's 79%).
    # Windows come from the pipeline, not a local copy, which had drifted on 2011
    # and COVID and was the other half of the disagreement.
    c = rd("reports/decomposition/turbulence_contributions.parquet")
    import re as _re
    _s14 = (DATA/"regime_detection"/"step14_turbulence_decomposition.py").read_text()
    _blk = _re.search(r"CRISIS_WINDOWS\s*=\s*\{(.*?)\}", _s14, _re.S).group(1)
    windows = {m.group(1): (m.group(2), m.group(3)) for m in
               _re.finditer(r'"([^"]+)":\s*\("(\d{4}-\d\d-\d\d)",\s*"(\d{4}-\d\d-\d\d)"\)', _blk)}
    label = {"2011 Euro/US dg": "2011 euro", "2018 Q4 selloff": "2018 Q4"}
    comp = {}
    for name,(s,e) in windows.items():
        seg = c.loc[s:e]
        if seg.empty: continue
        m = seg.mean(); comp[label.get(name,name)] = (m/m.sum())
    df = pd.DataFrame(comp).T[FAC]
    fig, ax = plt.subplots(figsize=(12, 5.4), layout="constrained")
    y = np.arange(len(df)); pos = np.zeros(len(df)); neg = np.zeros(len(df))
    for f in FAC:
        v = df[f].values; p_ = np.clip(v,0,None); n_ = np.clip(v,None,0)
        ax.barh(y, p_, left=pos, color=PAL[f], label=NAME[f])
        ax.barh(y, n_, left=neg, color=PAL[f])   # negatives stack leftward
        pos += p_; neg += n_
    for i,cr in enumerate(df.index):
        top = df.loc[cr].idxmax()
        ax.text(pos[i]+0.012, y[i], f"{NAME[top]} {df.loc[cr,top]*100:.0f}%",
                va="center", ha="left", fontsize=9.5, fontweight="bold", color="#333333")
    ax.axvline(0, color="black", lw=1.0, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(df.index); ax.invert_yaxis()
    ax.set_xlim(neg.min()-0.03, pos.max()+0.13)
    ax.set_xlabel("share of episode turbulence  (negative = factor moved with the usual structure)")
    ax.grid(axis="y", visible=False)
    # Legend goes BELOW the plot so the title owns the top strip: with ten
    # entries the two overprint each other if they share a row.
    _h,_l = ax.get_legend_handles_labels()
    fig.legend(_h, _l, ncol=10, loc="outside lower center", fontsize=9.5, columnspacing=1.1)
    fig.suptitle("Per-crisis decomposition fingerprints", fontsize=13.5)
    save(fig, "fig_crisis_drivers")

# ---------------------------------------------------------------- 4.3 causal networks (legible)
def _draw_net(ax, edges, color, title, sink=None):
    n=len(FAC); ang=np.linspace(np.pi/2, np.pi/2-2*np.pi, n, endpoint=False)
    pos={f:(np.cos(a),np.sin(a)) for f,a in zip(FAC,ang)}
    w = edges["w"]/edges["w"].max()
    for _,r in edges.iterrows():
        x0,y0=pos[r["src"]]; x1,y1=pos[r["dst"]]
        ax.add_patch(FancyArrowPatch((x0*0.86,y0*0.86),(x1*0.86,y1*0.86),
            connectionstyle="arc3,rad=0.12", arrowstyle="-|>", mutation_scale=12,
            lw=0.8+3.2*(r["w"]/edges["w"].max()), color=color, alpha=0.55))
    for f,(x,y) in pos.items():
        big = (f==sink)
        ax.add_patch(Circle((x,y), 0.16 if big else 0.13,
            fc=(color if big else "white"), ec=color, lw=2, zorder=3))
        ax.text(x*1.34, y*1.34, NAME[f], ha="center", va="center", fontsize=11,
                fontweight="bold" if big else "normal")
    ax.set_xlim(-1.6,1.6); ax.set_ylim(-1.6,1.6); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title(title, fontsize=12.5)

def fig_causal_networks():
    rw = rd("reports/causality/rewiring_differential.csv")
    rw = rw.rename(columns={"causing_factor":"src","caused_factor":"dst"})
    rw = rw[rw["src"].isin(FAC) & rw["dst"].isin(FAC)]
    calm = rw.nlargest(11,"calm_strength").assign(w=lambda d:d["calm_strength"])
    turb = rw.nlargest(11,"turb_strength").assign(w=lambda d:d["turb_strength"])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 6.2))
    _draw_net(ax[0], calm, BLUE, "Calm regime — strongest directed leads")
    _draw_net(ax[1], turb, RED, "Turbulent regime — network reorients toward USD", sink="fx_usd")
    fig.suptitle("Regime-conditional factor dependency network  (arrow = Granger lead; width = strength)",
                 y=0.99, fontsize=12.5)
    save(fig, "fig_causal_networks")

# ---------------------------------------------------------------- typology helpers
def _episodes():
    import sys; sys.path.insert(0, str(ROOT))
    from regime_detection.lib.crisis_typology import (
        extract_episodes, episode_composition, cluster_episodes)
    c = rd("reports/decomposition/turbulence_contributions.parquet").dropna()
    ep = extract_episodes(c.sum(axis=1).dropna(), min_periods=252)
    comp = episode_composition(c, ep)
    labels,k,_,names = cluster_episodes(comp, k_range=range(2,6))
    return c, ep, comp, labels, k, names

# ---------------------------------------------------------------- 5.1 taxonomy (legible: archetype centroids)
def fig_episode_taxonomy():
    c, ep, comp, labels, k, names = _episodes()
    fig, ax = plt.subplots(figsize=(12, 5))
    order = sorted(np.unique(labels))
    arche = []
    for ci in order:
        cent = comp[labels==ci].mean()[FAC]; arche.append(cent)
    bw = 0.8/len(order)
    x = np.arange(len(FAC))
    def archname(cent):
        top = cent.idxmax()
        if top == "vix": return "volatility-led"
        if top == "equity": return "equity-led"
        if top in ("value","commodities","em_equity"): return "mixed / unclassified"
        return f"{NAME[top]}-led"
    for j,(ci,cent) in enumerate(zip(order,arche)):
        cnt = int((labels==ci).sum())
        ax.bar(x + j*bw - 0.4 + bw/2, cent.values, bw, label=f"{archname(cent)}  (n={cnt})",
               color=plt.get_cmap("Set2").colors[j])
    ax.set_xticks(x); ax.set_xticklabels([NAME[f] for f in FAC])
    ax.set_ylabel("mean composition share"); ax.legend(loc="upper left")
    ax.set_title("Two recognisable archetypes and a mixed remainder — "
                 f"centroid composition of {len(comp)} episodes")
    save(fig, "fig_episode_taxonomy")

# ---------------------------------------------------------------- 5.2 typology timeline
def fig_typology_timeline():
    c, ep, comp, labels, k, names = _episodes()
    idx = c.index; rows=[]
    for start in comp.index:
        loc = idx.searchsorted(start)
        seg = c.iloc[max(0,loc-21):loc].dropna(how="all"); v=seg.mean(axis=0)
        if v.sum()<=0: continue
        pre=(v/v.sum()).values; real=comp.loc[start].values
        cos = float(np.dot(pre,real)/(np.linalg.norm(pre)*np.linalg.norm(real)))
        rows.append((start,cos))
    t=pd.DataFrame(rows,columns=["date","cos"])
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.axhspan(0.30,0.60, color=GREY, alpha=0.18, label="permutation-null band")
    ax.axhline(0.463, color=GREY, ls="--", lw=1, label="null mean 0.463")
    ax.scatter(t["date"], t["cos"], color=GREEN, s=30, zorder=3)
    ax.axhline(t["cos"].mean(), color=GREEN, lw=1.5, label=f"observed mean {t['cos'].mean():.3f}")
    ax.set_ylim(0,1); ax.set_ylabel("run-up vs spike cosine"); ax.legend(loc="lower left", ncol=2)
    ax.set_title("Composition predictability over time — the run-up anticipates the spike (p < 10⁻⁴)")
    save(fig, "fig_typology_timeline")

# ---------------------------------------------------------------- 5.3 cross-universe
def fig_cross_universe():
    data = [("Factor Lens (10 macro factors)", 38, 0.715, 0.463),
            ("US sectors (9 SPDR ETFs)", 47, 0.777, 0.615)]
    # No suptitle: the LaTeX caption carries the title, and a suptitle at y=1.02
    # overprinted the two-line subplot titles under savefig bbox="tight".
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for a,(nm,n,obs,null) in zip(ax, data):
        a.bar([0],[obs], width=0.5, color=GREEN, label=f"observed {obs:.3f}")
        a.bar([1],[null], width=0.5, color=GREY, alpha=0.5, label="shuffled pairings")
        a.axhline(null, color=GREY, ls="--", lw=1.4)
        a.text(0, obs+0.02, f"{obs:.3f}", ha="center", va="bottom",
               fontsize=11, fontweight="bold", color=GREEN)
        a.text(1, null+0.02, f"{null:.3f}", ha="center", va="bottom",
               fontsize=11, color="#555555")
        a.set_xticks([0,1]); a.set_xticklabels(["run-up → episode","null"]); a.set_ylim(0,0.9)
        a.set_title(f"{nm}\n{n} episodes  ·  $p < 0.0001$", pad=10)
        a.set_ylabel("mean cosine similarity"); a.legend(loc="upper right")
    fig.suptitle("Composition predictability replicates across universes", fontsize=13.5)
    save(fig, "fig_cross_universe")

# ---------------------------------------------------------------- 5.4 persistence baseline
def fig_persistence_baseline():
    F = dict(runup=0.715, m1=0.608, m2=0.580, m3=0.536, unc=0.643, null=0.463)
    S = dict(runup=0.777, m1=0.725, m2=0.741, m3=0.718, unc=0.762, null=0.615)
    # No suptitle (caption carries it). Reference lines labelled inline rather
    # than via a bottom-left legend, which printed over the dashed null line.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for a,(d,nm,verdict,pp) in zip(ax,[(F,"Factor Lens ($n=38$)","anticipation","$p = 0.003$"),
                                       (S,"US sectors ($n=47$)","persistence","$p = 0.080$")]):
        xs=[0,1,2,3]; ys=[d["runup"],d["m1"],d["m2"],d["m3"]]
        a.plot(xs,ys,"o-",color=BLUE,lw=2,zorder=3)
        a.scatter([0],[d["runup"]],color=RED,zorder=5,s=80)
        a.axhline(d["unc"],color=GREEN,ls=":",lw=1.6,zorder=2)
        a.axhline(d["null"],color=GREY,ls="--",lw=1.4,zorder=2)
        lo = min(d["null"], min(ys)) - 0.035; hi = max(d["runup"], d["unc"]) + 0.05
        a.set_ylim(lo, hi); pad = (hi-lo)*0.018
        a.text(3.22, d["unc"]+pad,  f"unconditional {d['unc']:.3f}",
               color=GREEN, fontsize=9.5, va="bottom", ha="right")
        a.text(3.22, d["null"]+pad, f"cross-episode null {d['null']:.3f}",
               color="#555555", fontsize=9.5, va="bottom", ha="right")
        a.set_xticks(xs); a.set_xticklabels(["run-up","1m back","2m back","3m back"])
        a.set_xlim(-0.25, 3.25)
        a.set_ylabel("mean cosine to the eventual episode")
        a.set_title(f"{nm} — {verdict}\nrun-up $-$ 1m carry-forward, {pp}", pad=10)
    fig.suptitle("Persistence baseline: does the run-up beat carrying forward\nan earlier composition?", fontsize=13.5)
    save(fig, "fig_persistence_baseline")

# ---------------------------------------------------------------- 6.2 sliding cutoff
def _l1_cutoffs():
    # the correction-overlay 5%-target 49-cutoff distribution (median -0.087, 20% positive)
    return rd("reports/correction_overlay/sliding_5pct.parquet")["sortino_lift"].dropna().values

def fig_sliding_cutoff():
    v = _l1_cutoffs()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(v, bins=18, color=BLUE, alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(np.median(v), color=RED, ls="--", lw=1.6, label=f"median {np.median(v):.3f}")
    ax.set_xlabel("Sortino lift over buy-and-hold"); ax.set_ylabel("count")
    ax.set_title(f"Sliding-cutoff robustness — {len(v)} train/test boundaries "
                 f"({100*(v>0).mean():.0f}% positive)"); ax.legend()
    save(fig, "fig_sliding_cutoff")

# ---------------------------------------------------------------- 6.3 ordinal bucket
def fig_ordinal_bucket():
    nb = rd("reports/robustness/nbuckets_sweep.parquet")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    rows = list(nb.index)
    stats = [[nb.loc[r,"min"],nb.loc[r,"q25"],nb.loc[r,"median"],nb.loc[r,"q75"],nb.loc[r,"max"]] for r in rows]
    bx = ax.boxplot(stats, vert=True, widths=0.55, patch_artist=True,
                    medianprops=dict(color="k", lw=1.8), showfliers=False, whis=[0,100])
    cols=[GREY,GREEN,GREEN,GREEN]
    for patch,c in zip(bx["boxes"],cols): patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticklabels(["binary\n(1 threshold)","ordinal\nn=3","ordinal\nn=5","ordinal\nn=10"])
    ax.set_ylabel("Sortino lift across 49 boundaries (logistic-L1)")
    ax.set_title("A representation fix: binary flag vs ordinal severity bucket")
    save(fig, "fig_ordinal_bucket")

# ---------------------------------------------------------------- 6.4 signed turbulence
def fig_signed_turbulence():
    dev = dict(total=0.768, loss=0.692, gain=0.693); oos = dict(total=0.588, loss=0.553, gain=0.529)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x=np.arange(3); w=0.38
    ax.bar(x-w/2,[dev["total"],dev["loss"],dev["gain"]],w,label="development (≤2019)",color=BLUE)
    ax.bar(x+w/2,[oos["total"],oos["loss"],oos["gain"]],w,label="out-of-sample (2020+)",color=RED,alpha=0.8)
    ax.axhline(0.5,color="k",ls=":",lw=1,label="chance")
    ax.set_xticks(x); ax.set_xticklabels(["total $d_t$","loss-side","gain-side"])
    ax.set_ylim(0.45,0.85); ax.set_ylabel("AUC for worst forward 21-day return"); ax.legend()
    ax.set_title("Signed turbulence: direction adds nothing")
    save(fig, "fig_signed_turbulence")

# ---------------------------------------------------------------- 6.5 event-time inference
def fig_event_time_inference():
    try:
        ev = rd("reports/event_time_inference/calendar_vs_eventtime.csv")
    except Exception:
        ev = None
    cal = dict(auc=0.482, pos=20); evt = dict(auc=0.436, pos=43)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].bar([0,1],[cal["auc"],evt["auc"]],color=[GREY,BLUE],width=0.5)
    ax[0].axhline(0.5,color="k",ls=":",lw=1); ax[0].set_ylim(0.4,0.55)
    ax[0].set_xticks([0,1]); ax[0].set_xticklabels(["calendar","event clock"]); ax[0].set_ylabel("point-classifier AUC")
    ax[0].set_title("Worse classifier on the event clock")
    ax[1].bar([0,1],[cal["pos"],evt["pos"]],color=[GREY,BLUE],width=0.5)
    ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["calendar","event clock"]); ax[1].set_ylabel("% of 49 cutoffs positive")
    ax[1].set_title("Yet a more robust strategy distribution")
    fig.suptitle("The event clock changes the statistics, not the forecastability", y=1.02, fontsize=12.5)
    save(fig, "fig_event_time_inference")

# ---------------------------------------------------------------- 6.6 per-market clocks
def fig_per_market_clocks():
    pm = rd("reports/per_market_event_clocks/per_market_event_clocks.csv").set_index(pm_index(rd))
    labels = {"calendar":"calendar\nhorizon","global_clock":"single global\nevent clock",
              "per_market_clock":"per-market\nevent clocks"}
    order=["calendar","global_clock","per_market_clock"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y=[pm.loc[k,"basket_lift"] for k in order]
    lo=[pm.loc[k,"basket_lift"]-pm.loc[k,"ci_low"] for k in order]
    hi=[pm.loc[k,"ci_high"]-pm.loc[k,"basket_lift"] for k in order]
    ax.errorbar(range(3), y, yerr=[lo,hi], fmt="o", color=BLUE, capsize=5, ms=9, lw=1.5)
    ax.axhline(0,color="k",lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([labels[k] for k in order])
    ax.set_ylabel("pooled basket Sortino lift (95% CI)")
    ax.set_title("Per-market clocks recover the global clock's damage — but only to calendar parity")
    save(fig, "fig_per_market_clocks")

def pm_index(_):  # the per-market csv has unnamed first col
    return pd.read_csv(DATA/"reports/per_market_event_clocks/per_market_event_clocks.csv").columns[0]

# ---------------------------------------------------------------- 6.1 correction overlay
def fig_correction_overlay():
    """Install step20's own figure rather than rebuilding it here.

    This used to re-derive all four panels from scratch, which silently drifted
    from the analysis: it charged transaction costs as an amortised daily drag
    over [0,5,10,15,20] bps instead of per-switch over [0,5,10,20], and it drew
    the reliability panel from the *turbulence-entry* predictions while titling
    it with the *correction* target's Brier scores. The figure that ships with
    the thesis must be the one step20 produces from the run it reports.
    """
    src = DATA / "figures" / "correction_overlay" / "correction_overlay.png"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing — run `python3 -m regime_detection.extras."
            "step20_correction_overlay` before regenerating thesis figures.")
    shutil.copyfile(src, OUT / "fig_correction_overlay.png")
    print("  fig_correction_overlay  <- step20 output (copied)")

# ---------------------------------------------------------------- 5.x worked run-up example
def fig_runup_example():
    """Two episodes side by side: the run-up composition against what arrived.

    Picks the best- and worst-anticipated episodes by cosine, so the figure shows
    both what the Visibility result looks like when it works and what it looks
    like when it fails. The failure case is the COVID onset, which is the point.
    """
    import sys
    sys.path.insert(0, str(DATA))
    from regime_detection.lib.crisis_typology import extract_episodes, episode_composition

    LEAD = 21
    c = rd("reports/decomposition/turbulence_contributions.parquet").dropna()
    ep = extract_episodes(c.sum(axis=1).dropna(), min_periods=252)
    comp = episode_composition(c, ep)

    cases = []
    for i, (_, e) in enumerate(ep.iterrows()):
        pos = c.index.get_indexer([e["start"]])[0]
        if pos < LEAD:
            continue
        pv = c.iloc[pos - LEAD:pos].mean()
        pv = pv / pv.sum()
        sv = comp.iloc[i]
        cos = float(pv @ sv / (np.linalg.norm(pv) * np.linalg.norm(sv)))
        cases.append((e["start"], cos, pv, sv))
    cases.sort(key=lambda r: r[1])
    worst, best = cases[0], cases[-1]

    labels = ["equity", "rates", "credit", "commod.", "EM eq", "USD",
              "inflation", "value", "quality", "VIX"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    x = np.arange(len(labels))
    w = 0.38
    for ax, (start, cos, pv, sv), tag in [
            (axes[0], best, "anticipated"), (axes[1], worst, "not anticipated")]:
        ax.bar(x - w / 2, pv.values, w, label="run-up (21 trading days before)",
               color=BLUE, alpha=0.85)
        ax.bar(x + w / 2, sv.values, w, label="episode as realised",
               color=RED, alpha=0.85)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"episode of {start:%d %b %Y} — {tag}\ncosine = {cos:+.3f}",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("composition share")
    axes[0].legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_runup_example")


# ---------------------------------------------------------------- 5.x convergence ladder
def fig_convergence_ladder():
    """The Sharpening result as a picture: similarity to the realised spike
    composition at four lookback windows, for the factor universe, the sector
    universe and 400 non-crisis placebo anchors."""
    rungs = ["3m back\n(63\u201384d)", "2m back\n(42\u201363d)", "1m back\n(21\u201342d)", "run-up\n(0\u201321d)"]
    x = np.arange(4)
    factor  = [0.536, 0.580, 0.608, 0.715]
    sectors = [0.718, 0.741, 0.725, 0.777]
    placebo = [0.685, 0.691, 0.707, 0.763]
    uncond  = 0.643

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x, factor,  "-o", color=RED,   lw=2.2, ms=7, label="Factor Lens (38 episodes)")
    ax.plot(x, sectors, "-s", color=BLUE,  lw=1.8, ms=6, label="US sectors (47 episodes)")
    ax.plot(x, placebo, "--^", color=GREY, lw=1.8, ms=6, label="placebo (400 non-crisis dates)")
    ax.axhline(uncond, color=GREEN, ls=":", lw=1.6)
    ax.annotate("unconditional baseline (0.643)", xy=(0.05, uncond), xytext=(0.05, uncond - 0.028),
                color=GREEN, fontsize=9)
    ax.annotate("", xy=(3.0, factor[3]), xytext=(0.0, factor[0]),
                arrowprops=dict(arrowstyle="-", ls="", color="none"))
    ax.text(1.25, 0.548, "slope $-0.056$ per rung", color=RED, fontsize=9.5)
    ax.text(0.06, 0.700, "slope $-0.019$ per rung", color=GREY, fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(rungs, fontsize=9)
    ax.set_ylabel("mean cosine similarity to the realised spike")
    ax.set_xlabel("window used as the predictor, moving towards the episode \u2192")
    ax.set_ylim(0.50, 0.81)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    ax.set_title("Composition converges towards the spike; the placebo does not",
                 fontsize=13, pad=12)
    save(fig, "fig_convergence_ladder")


# ---------------------------------------------------------------- 6.7 archetype hedges
def fig_archetype_hedges():
    h = rd("data/hedge_assets.parquet")
    def tot(asset,s,e):
        seg=h[asset].loc[s:e].dropna(); return 100*(seg.iloc[-1]/seg.iloc[0]-1)
    eps=[("2020 COVID\n(Feb–Apr)","2020-02-19","2020-04-30"),
         ("2022 inflation\n(full year)","2022-01-01","2022-12-31")]
    assets=[("bonds (TLT)","TLT",BLUE),("gold (GLD)","GLD","#d4a017"),("equity (SPY)","SPY",GREY)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); x=np.arange(len(eps)); w=0.26
    for j,(lab,a_,col) in enumerate(assets):
        vals=[tot(a_,s,e) for _,s,e in eps]
        ax.bar(x+(j-1)*w, vals, w, label=lab, color=col)
        for xi,v in zip(x+(j-1)*w,vals):
            ax.text(xi, v+(1 if v>=0 else -1)*1.5, f"{v:+.0f}%", ha="center",
                    va="bottom" if v>=0 else "top", fontsize=9)
    ax.axhline(0,color="k",lw=1); ax.set_xticks(x); ax.set_xticklabels([e[0] for e in eps])
    ax.set_ylabel("total return over episode (%)"); ax.legend()
    ax.set_title("The hedge that works flips by hedge regime")
    save(fig, "fig_archetype_hedges")

# ---------------------------------------------------------------- 6.8 oos equity curves
def fig_oos_equity_curves():
    # real strategy curves persisted by step26 (equity-protected, de-risk gate + routing)
    d = rd("reports/typology_hedge/oos_curves.parquet")
    meta = {"buy_and_hold":("buy-and-hold (SPY)",GREY,"-",1.8),
            "blind_bonds":("blind: always bonds",BLUE,":",1.3),
            "blind_gold":("blind: always gold","#d4a017",":",1.3),
            "split_bond_gold":("static 50/50 split",GREEN,"--",1.4),
            "composition_aware":("composition-aware (routed)",RED,"-",2.2)}
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axvspan(pd.Timestamp("2020-02-19"),pd.Timestamp("2020-04-30"),color=RED,alpha=0.06)
    ax.axvspan(pd.Timestamp("2022-01-01"),pd.Timestamp("2022-12-31"),color="#d4a017",alpha=0.06)
    for k,(lab,c,ls,lw) in meta.items():
        if k not in d.columns: continue
        s = 100*(np.exp(d[k].cumsum())-1)
        ax.plot(s.index, s, color=c, ls=ls, lw=lw, label=f"{lab}  ({s.iloc[-1]:+.0f}%)")
    ax.axhline(0,color="k",lw=0.8); ax.set_ylabel("cumulative return (%)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Out-of-sample hedging strategies, 2020–2026 — always-gold won this path; "
                 "routing's value is robustness")
    save(fig, "fig_oos_equity_curves")

# ---------------------------------------------------------------- run all
FIGS = [fig_boundary, fig_turbulence_regime, fig_density_qq, fig_turbulence_drivers,
        fig_crisis_drivers, fig_causal_networks, fig_episode_taxonomy, fig_typology_timeline,
        fig_runup_example, fig_convergence_ladder, fig_cross_universe,
        fig_persistence_baseline, fig_sliding_cutoff, fig_ordinal_bucket,
        fig_signed_turbulence, fig_event_time_inference, fig_per_market_clocks,
        fig_correction_overlay, fig_archetype_hedges, fig_oos_equity_curves]

if __name__ == "__main__":
    print(f"Writing {len(FIGS)} figures → {OUT}")
    ok=0
    for f in FIGS:
        try: f(); ok+=1
        except Exception as e:
            import traceback; print(f"  ✗ {f.__name__}: {e}"); traceback.print_exc()
    print(f"\nDone: {ok}/{len(FIGS)} figures.")
