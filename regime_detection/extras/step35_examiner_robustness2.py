"""Step 35 — Two more examiner-grade robustness tests (round 6).

A. VOLATILITY-MATCHED HEDGE BENCHMARK. The constructive hedge result (§6.5.5) reduces
   drawdown but loses to buy-and-hold and always-gold on the realised path; an examiner
   will say the drawdown reduction is just *lower exposure* (de-risking), not skill. Test:
   scale buy-and-hold down to the same realised volatility as the composition-aware overlay
   and compare drawdowns. If the overlay's drawdown is still lower than the vol-matched
   buy-and-hold, the benefit is more than de-risking; if not, it is exposure alone.

B. CLUSTER-BOOTSTRAP ON THE OOS ANTICIPATION. The OOS anticipation rests on a paired
   carry-forward difference of +0.102 (p=0.039) over 24 post-2019 episodes that are NOT
   independent (2020-2022 legs cluster). Resampling whole *clusters* of episodes (legs of
   the same meta-crisis) instead of episodes gives an honest CI and an effective-n.

Run:  python3 -m regime_detection.step35_examiner_robustness2
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

from regime_detection.lib.crisis_typology import extract_episodes, episode_composition

ROOT = Path(__file__).resolve().parents[2]
CURVES = ROOT / "reports" / "typology_hedge" / "oos_curves.parquet"
CONTRIB = ROOT / "reports" / "decomposition" / "turbulence_contributions.parquet"
CHECKPOINT = ROOT / "reports" / "checkpoints" / "step35_examiner_robustness2.md"
OOS_START = pd.Timestamp("2020-01-01"); LEAD = 21; N_BOOT = 20000; SEED = 0


def _sortino(r):
    r = np.asarray(r); dn = r[r < 0]
    dd = dn.std(ddof=1) if len(dn) > 1 else np.nan
    return float(np.nan if not dd else r.mean() / dd * np.sqrt(252))

def _maxdd(r):
    eq = (1 + pd.Series(r)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def test_volmatch():
    c = pd.read_parquet(CURVES)            # daily returns per strategy
    oa, bh = c["composition_aware"], c["buy_and_hold"]
    w = oa.std() / bh.std()                # exposure that vol-matches B&H to the overlay
    bh_m = w * bh
    out = {"w_exposure": float(w)}
    for nm, r in [("buy-and-hold", bh), ("buy-and-hold, vol-matched", bh_m),
                  ("composition-aware overlay", oa), ("static 50/50 split", c["split_bond_gold"])]:
        out[nm] = {"ann_vol": float(r.std()*np.sqrt(252)), "sortino": _sortino(r), "max_dd": _maxdd(r)}
    # does the overlay beat the vol-matched B&H on drawdown?
    out["overlay_dd"] = out["composition-aware overlay"]["max_dd"]
    out["volmatched_bh_dd"] = out["buy-and-hold, vol-matched"]["max_dd"]
    out["beats_volmatched"] = out["overlay_dd"] > out["volmatched_bh_dd"]  # less negative = shallower DD
    return out


def test_cluster_bootstrap():
    c = pd.read_parquet(CONTRIB).dropna()
    ep = extract_episodes(c.sum(axis=1).dropna(), min_periods=252)
    comp = episode_composition(c, ep); idx = c.index
    def cos(a, b):
        n = np.linalg.norm(a)*np.linalg.norm(b); return np.dot(a, b)/n if n > 0 else np.nan
    starts, d = [], []
    for s in comp.index:
        if s < OOS_START: continue
        loc = idx.searchsorted(s); R = comp.loc[s].values
        ru = c.iloc[max(0, loc-LEAD):loc].dropna(how="all").mean(axis=0)
        cf = c.iloc[max(0, loc-2*LEAD):loc-LEAD].dropna(how="all").mean(axis=0)
        if ru.sum() <= 0 or cf.sum() <= 0: continue
        starts.append(s); d.append(cos((ru/ru.sum()).values, R) - cos((cf/cf.sum()).values, R))
    starts = pd.DatetimeIndex(starts); d = np.array(d); n = len(d)
    # cluster by gaps > 90 trading days between consecutive episode starts (= meta-crisis legs)
    gaps = np.diff(starts.values).astype("timedelta64[D]").astype(int)
    cl = np.concatenate([[0], np.cumsum(gaps > 90)])
    clusters = [np.where(cl == k)[0] for k in np.unique(cl)]
    rng = np.random.default_rng(SEED)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(clusters), len(clusters))
        idxs = np.concatenate([clusters[p] for p in pick])
        means[b] = d[idxs].mean()
    ci = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    # design effect / effective n via cluster sizes
    sizes = np.array([len(x) for x in clusters])
    deff = 1 + (sizes.mean() - 1) * 0.5  # rough intra-cluster correlation ~0.5 assumption
    return {"n": n, "n_clusters": len(clusters), "cluster_sizes": sizes.tolist(),
            "obs_paired": float(d.mean()), "cluster_ci": ci, "eff_n": float(n/deff)}


def main() -> int:
    vm = test_volmatch(); cb = test_cluster_bootstrap()
    L = ["# Step 35 — Examiner robustness, round 6\n", f"_Generated {datetime.now():%Y-%m-%d %H:%M}._\n"]
    # A
    L.append("## A. Volatility-matched hedge benchmark\n")
    L.append(f"Buy-and-hold scaled to exposure **w = {vm['w_exposure']:.2f}** matches the overlay's "
             "realised volatility. If the overlay's drawdown is shallower than this vol-matched "
             "buy-and-hold, the benefit exceeds mere de-risking.\n")
    L.append("| strategy | ann. vol | Sortino | max drawdown |")
    L.append("|---|--:|--:|--:|")
    for nm in ["buy-and-hold", "buy-and-hold, vol-matched", "static 50/50 split", "composition-aware overlay"]:
        s = vm[nm]; L.append(f"| {nm} | {s['ann_vol']:.1%} | {s['sortino']:.2f} | {s['max_dd']:.1%} |")
    verdict = ("EXCEEDS de-risking" if vm["beats_volmatched"] else "is NOT separable from de-risking")
    L.append(f"\n**Verdict — the drawdown benefit {verdict}.** The overlay's max drawdown is "
             f"{vm['overlay_dd']:.1%} versus {vm['volmatched_bh_dd']:.1%} for a buy-and-hold dialled "
             f"down to the same volatility. " + (
             "Because the overlay's drawdown is shallower than the equally-volatile buy-and-hold, the "
             "drawdown reduction is not explained by lower average exposure alone — there is genuine "
             "timing/composition content in *when* and *where* it de-risks."
             if vm["beats_volmatched"] else
             "Because the equally-volatile buy-and-hold achieves a comparable or shallower drawdown, the "
             "overlay's drawdown reduction is largely lower average exposure, not skill — the honest claim "
             "is 'robustness to which hedge wins', not a drawdown edge over simple de-risking.") + "\n")
    # B
    L.append("## B. Cluster-bootstrap on the OOS anticipation (+0.102)\n")
    L.append(f"The {cb['n']} post-2019 factor episodes form **{cb['n_clusters']} independent meta-crisis "
             f"clusters** (sizes {cb['cluster_sizes']}); the effective independent sample is roughly "
             f"{cb['eff_n']:.0f}, not {cb['n']}. Resampling whole clusters:\n")
    L.append(f"- observed paired difference (run-up − carry-forward): **{cb['obs_paired']:+.3f}**")
    L.append(f"- 95% cluster-bootstrap CI: **[{cb['cluster_ci'][0]:+.3f}, {cb['cluster_ci'][1]:+.3f}]**\n")
    excl = cb["cluster_ci"][0] > 0
    L.append(f"**Verdict — {'survives' if excl else 'marginal under'} clustering.** " + (
        "The cluster-bootstrap CI excludes zero, so the OOS anticipation is not an artefact of treating "
        "dependent episodes as independent." if excl else
        "The cluster-bootstrap CI includes zero: on the held-out window the anticipation is real in the "
        "point estimate but not separable from chance once episode dependence is respected. The honest "
        "abstract claim is that OOS anticipation is *established at marginal significance on a small, "
        "dependent set*, while distinctiveness/persistence is the bulletproof result.") + "\n")
    CHECKPOINT.write_text("\n".join(L) + "\n"); print("\n".join(L))
    print(f"\nWrote → {CHECKPOINT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
