"""Step 32 — Persistence baseline for the composition-predictability flagship.

The flagship result (step17/step31) is that the *run-up* composition (the 21 days
strictly before an episode) anticipates the spike composition: cosine 0.715 vs a
permutation null of 0.463, p<1e-4, replicated on sectors. The examiner objection
(viva-grade) is that this may be **persistence, not anticipation**: an episode is
*defined* by turbulence crossing its trailing q90, so the 21-day "run-up" is already
rising-turbulence days, and the cross-episode permutation null only establishes that
compositions are episode-specific and sticky — not that the run-up *adds* predictive
information over simply carrying forward a recent known composition.

This step settles it. For each episode it compares the run-up predictor against
**carry-forward** predictors taken from windows progressively further back (the month
before the run-up, the month before that, ...) and against an **unconditional**
"average episode" baseline. The decisive test is *paired*: does cos(run-up, spike)
exceed cos(carry-forward, spike), episode by episode? A sign-flip permutation test and
an episode bootstrap quantify it.

Interpretation:
  * run-up >> carry-forward (paired, significant)  -> genuine ANTICIPATION / convergence;
    the flagship "predictable" claim survives as stated.
  * run-up ~ carry-forward (no paired difference)   -> PERSISTENCE dominates; composition
    is sticky months out and the run-up adds nothing. Reframe the headline to
    "persistent and distinctive" (still novel, still replicated) — not "anticipated".
  * monotone rise toward the spike across lookbacks  -> convergence (weak anticipation).

Run:  python3 -m regime_detection.step32_persistence_baseline
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
from regime_detection.lib.crisis_typology import episode_composition, extract_episodes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRIB_PATH = (PROJECT_ROOT / "reports" / "decomposition"
                / "turbulence_contributions.parquet")
SECTOR_PATH = PROJECT_ROOT / "data" / "sector_prices.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "persistence_baseline"
FIG_DIR = PROJECT_ROOT / "figures" / "persistence_baseline"
CHECKPOINT = (PROJECT_ROOT / "reports" / "checkpoints"
              / "step32_persistence_baseline.md")

SECTORS = {"XLF": "Financials", "XLK": "Technology", "XLE": "Energy",
           "XLV": "Health Care", "XLI": "Industrials", "XLY": "Cons. Disc.",
           "XLP": "Cons. Staples", "XLU": "Utilities", "XLB": "Materials"}
LEAD = 21          # window length (trading days) — matches step17 lead_window
N_BACK = 3         # how many carry-forward windows to trace (1..3 months back)
N_PERM = 10000
SEED = 0


def _cos_rows(A, B):
    na = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return (A * B).sum(axis=1) / np.where(na == 0, np.nan, na)


def baselines(contrib: pd.DataFrame, *, min_periods: int = 252) -> dict:
    """Run-up vs carry-forward vs unconditional, on one universe."""
    turb = contrib.sum(axis=1).dropna()
    episodes = extract_episodes(turb, min_periods=min_periods)
    comp = episode_composition(contrib, episodes)       # realised spike composition R
    idx = contrib.index

    preds = {k: [] for k in range(N_BACK + 1)}
    R, kept = [], []
    for start in comp.index:
        loc = idx.searchsorted(start)
        vecs, ok = {}, True
        for k in range(N_BACK + 1):                     # k=0 run-up, k>=1 carry-forward
            hi, lo = loc - k * LEAD, max(0, loc - (k + 1) * LEAD)
            seg = contrib.iloc[lo:hi].dropna(how="all")
            v = seg.mean(axis=0)
            if len(seg) < 5 or float(v.sum()) <= 0:
                ok = False
                break
            vecs[k] = (v / v.sum()).values
        if not ok:
            continue
        for k in range(N_BACK + 1):
            preds[k].append(vecs[k])
        R.append(comp.loc[start].values)
        kept.append(start)

    R = np.asarray(R)
    P = {k: np.asarray(preds[k]) for k in preds}
    U = np.tile(R.mean(axis=0) / R.mean(axis=0).sum(), (len(R), 1))   # "average episode"
    n = len(R)

    cos = {f"runup" if k == 0 else f"carry_{k}m": _cos_rows(P[k], R)
           for k in range(N_BACK + 1)}
    cos["uncond"] = _cos_rows(U, R)

    rng = np.random.default_rng(SEED)

    # cross-episode permutation null for the run-up (reproduces the step17 test)
    perm = np.array([np.nanmean(_cos_rows(P[0], R[rng.permutation(n)]))
                     for _ in range(N_PERM)])
    runup_p = float((perm >= np.nanmean(cos["runup"])).mean())

    # decisive paired test: run-up vs 1-month-back carry-forward
    d = cos["runup"] - cos["carry_1m"]
    d = d[~np.isnan(d)]
    obs_d = float(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null_d = (signs * d).mean(axis=1)
    paired_p = float((np.abs(null_d) >= abs(obs_d)).mean())
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_PERM)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    return {
        "n": n,
        "means": {name: float(np.nanmean(c)) for name, c in cos.items()},
        "runup_null": float(perm.mean()),
        "runup_p": runup_p,
        "paired_obs": obs_d,
        "paired_p": paired_p,
        "paired_ci": ci,
        "paired_frac_pos": float((d > 0).mean()),
        "_cos": cos,
    }


def _load_sector_contrib() -> pd.DataFrame:
    px = pd.read_parquet(SECTOR_PATH).rename(columns=SECTORS)
    rets = np.log(px / px.shift(1)).dropna(how="all")
    return TurbulenceDecomposer(lookback_days=2520, min_periods=504,
                                shrinkage="ledoit-wolf").fit_transform(rets).dropna()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Factor universe ...")
    fac = baselines(pd.read_parquet(CONTRIB_PATH).dropna())
    print("Sector universe ...")
    sec = baselines(_load_sector_contrib())

    order = ["runup"] + [f"carry_{k}m" for k in range(1, N_BACK + 1)] + ["uncond"]
    labels = {"runup": "run-up (0-21d)", "carry_1m": "carry-fwd (21-42d)",
              "carry_2m": "carry-fwd (42-63d)", "carry_3m": "carry-fwd (63-84d)",
              "uncond": "unconditional"}
    for nm, r in [("FACTORS", fac), ("SECTORS", sec)]:
        print(f"\n=== {nm}  (n={r['n']} episodes) ===")
        for key in order:
            print(f"  {labels[key]:24s} mean cosine to spike = {r['means'][key]:.3f}")
        print(f"  run-up cross-episode null = {r['runup_null']:.3f} "
              f"(p {'<0.0001' if r['runup_p']<1e-4 else f'={r[chr(39).join(['','runup_p',''])]:.4f}'})")
        print(f"  PAIRED run-up - carry_1m  = {r['paired_obs']:+.4f}  "
              f"[95% CI {r['paired_ci'][0]:+.3f}, {r['paired_ci'][1]:+.3f}]  "
              f"sign-flip p = {r['paired_p']:.4f}  "
              f"({100*r['paired_frac_pos']:.0f}% of episodes positive)")

    _figure(fac, sec, order, labels)
    _write_checkpoint(fac, sec, order, labels)
    print(f"\nWrote checkpoint -> {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


def _verdict(r) -> str:
    sig = r["paired_p"] < 0.05 and r["paired_obs"] > 0 and r["paired_ci"][0] > 0
    if sig:
        return "ANTICIPATION"
    return "PERSISTENCE"


def _figure(fac, sec, order, labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (nm, r) in zip(axes, [("Factor Lens", fac), ("US sectors", sec)]):
        xs = list(range(len(order)))
        ys = [r["means"][k] for k in order]
        ax.plot(xs[:-1], ys[:-1], "o-", color="tab:blue", lw=2)
        ax.axhline(r["means"]["uncond"], color="tab:green", ls=":",
                   label=f"unconditional = {r['means']['uncond']:.3f}")
        ax.axhline(r["runup_null"], color="0.6", ls="--",
                   label=f"cross-episode null = {r['runup_null']:.3f}")
        ax.scatter([0], [r["means"]["runup"]], color="tab:red", zorder=5, s=60)
        ax.set_xticks(xs[:-1])
        ax.set_xticklabels(["run-up", "1m back", "2m back", "3m back"][:len(order) - 1],
                           fontsize=8)
        ax.set_ylabel("mean cosine to spike composition")
        ax.set_title(f"{nm} (n={r['n']}) — {_verdict(r)}\n"
                     f"paired run-up - 1m = {r['paired_obs']:+.3f}, p={r['paired_p']:.3f}",
                     fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Persistence baseline: does the run-up beat carrying forward "
                 "an earlier composition?", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "persistence_baseline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_checkpoint(fac, sec, order, labels):
    L = ["# Step 32 — Persistence baseline for the composition flagship\n"]
    L.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Tests whether the run-up "
             "composition *anticipates* the spike, or merely reflects the **persistence** "
             "of a slow-moving composition — the central viva objection to the flagship "
             "result._\n")
    L.append("For each episode the run-up predictor (0–21d before the spike) is compared, "
             "**paired**, against carry-forward predictors from earlier windows (21–42d, "
             "42–63d, 63–84d back) and an unconditional 'average episode' baseline. The "
             "decisive quantity is whether the run-up beats the 1-month-back carry-forward "
             "episode by episode (sign-flip permutation test + episode bootstrap CI).\n")
    L.append("| universe | n | run-up | carry 1m | carry 2m | carry 3m | uncond | "
             "run-up − 1m (paired) | sign-flip p | 95% CI | verdict |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    for nm, r in [("Factor Lens", fac), ("US sectors", sec)]:
        m = r["means"]
        L.append(f"| {nm} | {r['n']} | **{m['runup']:.3f}** | {m['carry_1m']:.3f} | "
                 f"{m['carry_2m']:.3f} | {m['carry_3m']:.3f} | {m['uncond']:.3f} | "
                 f"{r['paired_obs']:+.3f} | {r['paired_p']:.4f} | "
                 f"[{r['paired_ci'][0]:+.3f}, {r['paired_ci'][1]:+.3f}] | "
                 f"**{_verdict(r)}** |")
    L.append("")
    L.append("## Reading\n")
    L.append("- **run-up vs unconditional** — how much *episode-specific* information any "
             "pre-spike window carries over 'the average crisis'. A large gap = "
             "compositions are distinctive (the cross-universe replication already "
             "established this).")
    L.append("- **run-up vs carry-forward (paired)** — the decisive test. If the run-up "
             "does *not* significantly beat a composition measured a month earlier, the "
             "predictability is **persistence**, not anticipation, and the flagship must "
             "be reframed.\n")
    L.append("## Verdict\n")
    for nm, r in [("Factor Lens", fac), ("US sectors", sec)]:
        v = _verdict(r)
        if v == "ANTICIPATION":
            L.append(f"- **{nm}: ANTICIPATION.** The run-up beats the 1-month-back "
                     f"carry-forward by {r['paired_obs']:+.3f} (paired sign-flip "
                     f"p={r['paired_p']:.4f}, 95% CI "
                     f"[{r['paired_ci'][0]:+.3f}, {r['paired_ci'][1]:+.3f}], "
                     f"{100*r['paired_frac_pos']:.0f}% of episodes positive). The "
                     "composition genuinely *converges* toward the spike as it approaches; "
                     "the 'predictable composition' claim survives as stated.")
        else:
            L.append(f"- **{nm}: PERSISTENCE.** The run-up does **not** significantly beat "
                     f"a composition measured a month earlier (paired Δ={r['paired_obs']:+.3f}, "
                     f"sign-flip p={r['paired_p']:.4f}, 95% CI "
                     f"[{r['paired_ci'][0]:+.3f}, {r['paired_ci'][1]:+.3f}]). The "
                     "predictability is driven by the **persistence** of a slow-moving, "
                     "episode-distinctive composition, not by anticipation of the spike. "
                     "Honest reframing required: composition is *persistent and distinctive* "
                     "(and replicates out-of-universe) — which is still novel and useful for "
                     "hedge pre-positioning — but it is not a *forecast* of a discontinuity.")
    L.append("")
    L.append("**Figure:** `figures/persistence_baseline/persistence_baseline.png`.")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
