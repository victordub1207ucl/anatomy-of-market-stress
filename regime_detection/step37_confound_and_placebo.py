"""Step 37 — Two deep robustness tests (round 9).

A. FIXED-COVARIANCE DISTINCTIVENESS. The contribution is c_i = delta_i (A delta)_i with
   A = Omega^{-1} on a trailing window. Run-up and spike of one episode share an essentially
   identical A (windows offset by days); the cross-episode permutation null pairs compositions
   computed under *different* A's (episodes years apart). So the observed>null gap could partly
   reflect "same A imposes a similar contribution geometry", not only return-direction
   persistence. Re-run the permutation with a SINGLE FIXED full-sample A for every episode: if
   the gap survives, distinctiveness is genuine and not an A-matching artefact.

B. PLACEBO / PSEUDO-EPISODE CONTROL for the convergence ladder. Under pure autocorrelation,
   ANY date shows a backward convergence ladder (nearer windows more correlated with the
   endpoint). Draw random non-crisis dates, build matched run-up/ladders, and compare the
   pseudo convergence slope to the real episode slope. If real is materially steeper, there is
   sharpening beyond ambient persistence; if equal, the ladder is just autocorrelation.

Run:  python3 -m regime_detection.step37_confound_and_placebo
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf

from regime_detection.lib.crisis_typology import extract_episodes, episode_composition

ROOT = Path(__file__).resolve().parents[1]
FACT = ["equity","rates","credit","commodities","em_equity","fx_usd","inflation","value","quality","vix"]
CONTRIB = ROOT / "reports" / "decomposition" / "turbulence_contributions.parquet"
TURB = ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
CHECKPOINT = ROOT / "reports" / "checkpoints" / "step37_confound_and_placebo.md"
LEAD, N_PERM, SEED = 21, 10000, 0


def _cos(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b); return np.dot(a, b)/n if n > 0 else np.nan


def fixed_A_contributions():
    px = pd.read_parquet(ROOT/"data"/"factor_prices.parquet")[FACT]
    rets = np.log(px/px.shift(1)).dropna()
    mu = rets.mean().values
    A = np.linalg.inv(LedoitWolf().fit(rets.values).covariance_)   # single full-sample A
    D = rets.values - mu
    C = D * (D @ A)                                                 # c_i = delta_i (A delta)_i
    return pd.DataFrame(C, index=rets.index, columns=FACT)


def comp_and_runups(contrib, episodes):
    comp = episode_composition(contrib, episodes); idx = contrib.index
    runup, R = [], []
    for s in comp.index:
        loc = idx.searchsorted(s)
        v = contrib.iloc[max(0, loc-LEAD):loc].dropna(how="all").mean(axis=0)
        if v.sum() <= 0: continue
        runup.append((v/v.sum()).values); R.append(comp.loc[s].values)
    return np.asarray(runup), np.asarray(R)


def perm_test(P, R, rng):
    n = len(R); obs = float(np.nanmean([_cos(P[i], R[i]) for i in range(n)]))
    null = np.array([np.nanmean([_cos(P[i], R[pm[i]]) for i in range(n)])
                     for pm in (rng.permutation(n) for _ in range(N_PERM))])
    return obs, float(null.mean()), float((null >= obs).mean())


def main() -> int:
    rng = np.random.default_rng(SEED)
    turb = pd.read_parquet(TURB)["turbulence"]
    episodes = extract_episodes(turb.dropna(), min_periods=252)
    roll = pd.read_parquet(CONTRIB).dropna()
    fixed = fixed_A_contributions()

    # A. distinctiveness under rolling vs fixed A
    Pr, Rr = comp_and_runups(roll, episodes); obs_r, null_r, p_r = perm_test(Pr, Rr, np.random.default_rng(SEED))
    Pf, Rf = comp_and_runups(fixed, episodes); obs_f, null_f, p_f = perm_test(Pf, Rf, np.random.default_rng(SEED))

    # B. placebo convergence slope (rolling A): real episodes vs random non-crisis dates
    idx = roll.index; dist = np.array([0,1,2,3.])
    def slope_at(loc):
        row = []
        for k in range(4):
            seg = roll.iloc[max(0,loc-(k+1)*LEAD):loc-k*LEAD].dropna(how="all").mean(axis=0)
            if seg.sum() <= 0: return None
            row.append(_cos((seg/seg.sum()).values, _R))
        return np.polyfit(dist, row, 1)[0]
    real_sl = []
    for s in episode_composition(roll, episodes).index:
        loc = idx.searchsorted(s); _R = episode_composition(roll, episodes).loc[s].values
        sl = slope_at(loc)
        if sl is not None: real_sl.append(sl)
    real_sl = np.array(real_sl)
    # pseudo dates: not within 84 trading days of any episode start
    ep_locs = np.array([idx.searchsorted(s) for s in episodes["start"]])
    valid = np.arange(4*LEAD, len(idx))
    bad = np.concatenate([np.arange(l-4*LEAD, l+4*LEAD) for l in ep_locs])
    pool = np.setdiff1d(valid, bad)
    pseudo_sl = []
    for loc in rng.choice(pool, size=min(400, len(pool)), replace=False):
        _R = (roll.iloc[loc:loc+1].values[0]); _R = _R/_R.sum() if _R.sum()>0 else _R
        sl = slope_at(loc)
        if sl is not None and np.isfinite(sl): pseudo_sl.append(sl)
    pseudo_sl = np.array(pseudo_sl)
    # is real steeper (more negative) than pseudo?
    diff = real_sl.mean() - pseudo_sl.mean()
    pooled = np.concatenate([real_sl, pseudo_sl]); nA = len(real_sl)
    permd = np.array([(lambda x: x[:nA].mean()-x[nA:].mean())(rng.permutation(pooled)) for _ in range(N_PERM)])
    p_placebo = float((permd <= diff).mean())   # one-sided: real more negative

    surv_A = p_f < 0.05 and obs_f > null_f
    L = ["# Step 37 — Fixed-covariance distinctiveness + placebo control\n",
         f"_Generated {datetime.now():%Y-%m-%d %H:%M}._\n"]
    L.append("## A. Distinctiveness: rolling vs single fixed covariance\n")
    L.append("| covariance | observed | null | p |"); L.append("|---|--:|--:|--:|")
    L.append(f"| rolling A (as in thesis) | {obs_r:.3f} | {null_r:.3f} | {'<0.0001' if p_r<1e-4 else f'{p_r:.4f}'} |")
    L.append(f"| **fixed full-sample A** | **{obs_f:.3f}** | {null_f:.3f} | {'<0.0001' if p_f<1e-4 else f'{p_f:.4f}'} |")
    L.append(f"\n**Verdict — {'GENUINE (not an A-matching artefact)' if surv_A else 'CONFOUNDED'}.** "
             + ("Recomputing every episode's composition under one fixed full-sample covariance — so "
                "run-ups and shuffled spikes share the identical A — the observed cosine still clears the "
                "permutation null decisively. The distinctiveness is a property of the return-direction "
                "composition, not of time-varying covariance matching." if surv_A else
                "Under a fixed A the observed-vs-null gap collapses, so a material part of the headline "
                "distinctiveness gap was covariance-window matching, not return persistence. Lead "
                "distinctiveness with the carry-forward framing, which is clean of this confound.") + "\n")
    L.append("## B. Placebo: is the convergence ladder special to crises?\n")
    L.append(f"- real episode mean slope: **{real_sl.mean():+.4f}** (n={len(real_sl)})")
    L.append(f"- pseudo (random non-crisis dates) mean slope: **{pseudo_sl.mean():+.4f}** (n={len(pseudo_sl)})")
    L.append(f"- real − pseudo: {diff:+.4f}; permutation p (real steeper): **{p_placebo:.4f}**\n")
    steeper = p_placebo < 0.05 and diff < 0
    L.append(f"**Verdict — {'sharpening beyond ambient persistence' if steeper else 'ladder is largely ambient autocorrelation'}.** "
             + ("Crisis run-ups converge on their spike materially faster than random windows converge on "
                "an arbitrary endpoint, so the sharpening is specific to episodes, not a universal "
                "autocorrelation artefact." if steeper else
                "Random non-crisis windows show a convergence ladder of similar steepness, so the ladder "
                "is mostly ambient composition autocorrelation; the defensible claim stays the weaker one "
                "(the spike's signature is legible in the run-up), not 'sharpening beyond persistence'.") + "\n")
    CHECKPOINT.write_text("\n".join(L)+"\n"); print("\n".join(L))
    print(f"\nWrote → {CHECKPOINT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
