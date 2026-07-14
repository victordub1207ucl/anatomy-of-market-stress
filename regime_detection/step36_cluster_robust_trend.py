"""Step 36 — Cluster-robust convergence trend (the flagship, re-tested honestly).

The anticipation claim's load-bearing evidence is the full-sample convergence trend:
Page's L p=0.007 and a within-episode slope sign-flip p=0.011 over 38 episodes. But those
tests treat all 38 episodes as exchangeable units, while the OOS analysis (step35) showed the
episodes collapse into a handful of meta-crisis clusters with correlated convergence ladders.
The same dependence that voided the OOS CI applies to the full sample, so the effective N for
Page/slope is well below 38 and p=0.007 is optimistic.

This re-runs the trend inference respecting cluster structure: aggregate to the meta-crisis
cluster level (each cluster contributes one mean slope), then (i) a cluster sign-flip
permutation p, (ii) a cluster-bootstrap CI on the mean slope, (iii) a one-sample test on the
cluster-level slopes. If the trend survives at the cluster level, anticipation stands; if not,
it degrades to "suggestive" and the spine rests on distinctiveness.

Run:  python3 -m regime_detection.step36_cluster_robust_trend
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

from regime_detection.lib.crisis_typology import extract_episodes, episode_composition

ROOT = Path(__file__).resolve().parents[1]
CONTRIB = ROOT / "reports" / "decomposition" / "turbulence_contributions.parquet"
CHECKPOINT = ROOT / "reports" / "checkpoints" / "step36_cluster_robust_trend.md"
LEAD, GAP_DAYS, N_PERM, SEED = 21, 90, 20000, 0


def main() -> int:
    c = pd.read_parquet(CONTRIB).dropna()
    ep = extract_episodes(c.sum(axis=1).dropna(), min_periods=252)
    comp = episode_composition(c, ep); idx = c.index
    def cos(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b); return np.dot(a, b) / n if n > 0 else np.nan

    starts, slopes = [], []
    dist = np.array([0, 1, 2, 3.0])
    for s in comp.index:
        loc = idx.searchsorted(s); R = comp.loc[s].values; row = []
        ok = True
        for k in range(4):
            seg = c.iloc[max(0, loc - (k + 1) * LEAD):loc - k * LEAD].dropna(how="all").mean(axis=0)
            if seg.sum() <= 0:
                ok = False; break
            row.append(cos((seg / seg.sum()).values, R))
        if not ok:
            continue
        starts.append(s)
        slopes.append(np.polyfit(dist, row, 1)[0])      # <0 = converges toward spike
    starts = pd.DatetimeIndex(starts); slopes = np.array(slopes); n = len(slopes)

    # meta-crisis clusters: gaps > GAP_DAYS calendar days between consecutive episode starts
    gaps = np.diff(starts.values).astype("timedelta64[D]").astype(int)
    cl = np.concatenate([[0], np.cumsum(gaps > GAP_DAYS)])
    clusters = [np.where(cl == k)[0] for k in np.unique(cl)]
    cl_slopes = np.array([slopes[ix].mean() for ix in clusters])    # one unit per cluster
    K = len(clusters)

    rng = np.random.default_rng(SEED)
    # naive (episode-level) for contrast
    naive_p = float((np.abs((slopes * rng.choice([-1, 1], (N_PERM, n))).mean(1)) >= abs(slopes.mean())).mean())
    # (i) cluster sign-flip permutation on cluster-mean slopes
    obs = cl_slopes.mean()
    signs = rng.choice([-1, 1], (N_PERM, K))
    cl_perm_p = float((np.abs((signs * cl_slopes).mean(1)) >= abs(obs)).mean())
    # (ii) cluster bootstrap CI on the mean cluster slope
    boot = np.array([cl_slopes[rng.integers(0, K, K)].mean() for _ in range(N_PERM)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    # (iii) one-sample t / Wilcoxon on K cluster slopes
    t_p = float(stats.ttest_1samp(cl_slopes, 0).pvalue)
    try:
        w_p = float(stats.wilcoxon(cl_slopes).pvalue)
    except Exception:
        w_p = np.nan

    survives = cl_perm_p < 0.05 and ci[1] < 0
    L = ["# Step 36 — Cluster-robust convergence trend\n", f"_Generated {datetime.now():%Y-%m-%d %H:%M}._\n"]
    L.append(f"38-episode convergence slope (cosine-to-spike vs lookback distance; negative = "
             f"converges toward the spike). Episode-level mean slope **{slopes.mean():+.4f}** "
             f"(naive sign-flip p = {naive_p:.4f}, the optimistic number).\n")
    L.append(f"The {n} episodes form **{K} meta-crisis clusters** (sizes {[len(x) for x in clusters]}). "
             "Aggregating to one mean slope per cluster and testing at the cluster level:\n")
    L.append(f"- cluster-mean slope: **{obs:+.4f}**")
    L.append(f"- cluster sign-flip permutation p: **{cl_perm_p:.4f}**")
    L.append(f"- cluster-bootstrap 95% CI: **[{ci[0]:+.4f}, {ci[1]:+.4f}]**")
    L.append(f"- one-sample t-test on {K} cluster slopes: p = {t_p:.4f}; Wilcoxon p = {w_p:.4f}\n")
    L.append("## Verdict\n")
    if survives:
        L.append(f"**Survives clustering.** Even with the effective sample reduced to {K} independent "
                 f"meta-crises, the convergence trend is significant (cluster sign-flip p = {cl_perm_p:.4f}, "
                 f"bootstrap CI excludes zero). The anticipation claim's in-sample evidence is robust to "
                 "the dependence that voids the OOS test; the honest line is 'anticipation established "
                 "in-sample (cluster-robust), marginal out-of-sample.'")
    else:
        L.append(f"**Degrades under clustering.** At the cluster level ({K} independent meta-crises) the "
                 f"trend is no longer significant (cluster sign-flip p = {cl_perm_p:.4f}, bootstrap CI "
                 f"[{ci[0]:+.4f}, {ci[1]:+.4f}] includes zero). The Page/slope p-values were optimistic "
                 "because they treated correlated within-meta-crisis episodes as independent. The honest "
                 "claim becomes: anticipation is **suggestive both in- and out-of-sample**, not "
                 "established; the spine rests on distinctiveness and persistence, which are cluster-robust "
                 "(they replicate across a different universe). The convergence ladder is reported as "
                 "*descriptive* — consistent with sharpening but not separable from clustered persistence.")
    CHECKPOINT.write_text("\n".join(L) + "\n"); print("\n".join(L))
    print(f"\nWrote → {CHECKPOINT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
