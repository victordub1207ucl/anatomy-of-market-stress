"""Step 33 — Is composition predictability out-of-sample?

The predictability boundary (Ch.7) places composition predictability (an in-sample
structural statistic computed over the full 2009-2026 history) opposite the strict-OOS
timing nulls (post-2019). A reviewer rightly notes this mixes non-comparable
measurements: "composition predictable, timing not" is partly "an in-sample descriptive
statistic beats an out-of-sample forecast", which is almost guaranteed by construction.

This step removes the asymmetry. The run-up -> spike test is *already* causal per episode
(the run-up strictly precedes the spike), so the only thing needed for a genuine OOS test
is to evaluate it on episodes whose spike falls in the held-out era (>= 2020-01-01) — the
same window over which the timing nulls are computed. We report the cosine, the
cross-episode permutation null, and the persistence baseline (run-up vs one-month-back
carry-forward, paired) for the OOS-only episodes, alongside the dev (<2020) episodes for
contrast, on both universes.

Run:  python3 -m regime_detection.step33_oos_composition
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from regime_detection.lib.turbulence_decomposition import TurbulenceDecomposer
from regime_detection.lib.crisis_typology import episode_composition, extract_episodes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRIB_PATH = (PROJECT_ROOT / "reports" / "decomposition"
                / "turbulence_contributions.parquet")
SECTOR_PATH = PROJECT_ROOT / "data" / "sector_prices.parquet"
CHECKPOINT = (PROJECT_ROOT / "reports" / "checkpoints"
              / "step33_oos_composition.md")
SECTORS = {"XLF": "Financials", "XLK": "Technology", "XLE": "Energy",
           "XLV": "Health Care", "XLI": "Industrials", "XLY": "Cons. Disc.",
           "XLP": "Cons. Staples", "XLU": "Utilities", "XLB": "Materials"}
OOS_START = pd.Timestamp("2020-01-01")
LEAD, N_PERM, SEED = 21, 10000, 0


def _cos(A, B):
    na = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return (A * B).sum(axis=1) / np.where(na == 0, np.nan, na)


def _vectors(contrib, comp):
    """run-up (0-21d) and carry-forward (21-42d) predictors + realised R per episode."""
    idx = contrib.index
    runup, carry, R, starts = [], [], [], []
    for start in comp.index:
        loc = idx.searchsorted(start)
        ru = contrib.iloc[max(0, loc - LEAD):loc].dropna(how="all").mean(axis=0)
        cf = contrib.iloc[max(0, loc - 2 * LEAD):loc - LEAD].dropna(how="all").mean(axis=0)
        if ru.sum() <= 0 or cf.sum() <= 0:
            continue
        runup.append((ru / ru.sum()).values)
        carry.append((cf / cf.sum()).values)
        R.append(comp.loc[start].values)
        starts.append(start)
    return (np.asarray(runup), np.asarray(carry), np.asarray(R),
            pd.DatetimeIndex(starts))


def _eval(P_runup, P_carry, R, rng):
    """cosine, permutation null/p, and paired run-up vs carry-forward."""
    n = len(R)
    obs = float(np.nanmean(_cos(P_runup, R)))
    null = np.array([np.nanmean(_cos(P_runup, R[rng.permutation(n)]))
                     for _ in range(N_PERM)])
    p = float((null >= obs).mean())
    d = _cos(P_runup, R) - _cos(P_carry, R)
    d = d[~np.isnan(d)]
    obs_d = float(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    paired_p = float((np.abs((signs * d).mean(axis=1)) >= abs(obs_d)).mean())
    return {"n": n, "cosine": obs, "null": float(null.mean()), "p": p,
            "paired": obs_d, "paired_p": paired_p}


def analyse(contrib, label):
    turb = contrib.sum(axis=1).dropna()
    episodes = extract_episodes(turb, min_periods=252)
    comp = episode_composition(contrib, episodes)
    P0, P1, R, starts = _vectors(contrib, comp)
    rng = np.random.default_rng(SEED)
    out = {}
    for split, mask in [("dev (<2020)", starts < OOS_START),
                        ("oos (>=2020)", starts >= OOS_START),
                        ("full", np.ones(len(starts), bool))]:
        if mask.sum() < 5:
            out[split] = {"n": int(mask.sum()), "cosine": np.nan, "null": np.nan,
                          "p": np.nan, "paired": np.nan, "paired_p": np.nan}
            continue
        out[split] = _eval(P0[mask], P1[mask], R[mask], np.random.default_rng(SEED))
    return out


def main() -> int:
    fac = analyse(pd.read_parquet(CONTRIB_PATH).dropna(), "factors")
    px = pd.read_parquet(SECTOR_PATH).rename(columns=SECTORS)
    sec_contrib = TurbulenceDecomposer(lookback_days=2520, min_periods=504,
                                       shrinkage="ledoit-wolf").fit_transform(
        np.log(px / px.shift(1)).dropna(how="all")).dropna()
    sec = analyse(sec_contrib, "sectors")

    for nm, r in [("FACTORS", fac), ("SECTORS", sec)]:
        print(f"\n=== {nm} ===")
        print(f"{'split':14s} {'n':>3s} {'cosine':>7s} {'null':>6s} {'p':>8s} "
              f"{'runup-carry':>11s} {'paired_p':>8s}")
        for split in ["dev (<2020)", "oos (>=2020)", "full"]:
            d = r[split]
            ps = "<1e-4" if (d["p"] == d["p"] and d["p"] < 1e-4) else (
                f"{d['p']:.4f}" if d["p"] == d["p"] else "n/a")
            pps = f"{d['paired_p']:.4f}" if d["paired_p"] == d["paired_p"] else "n/a"
            cos = f"{d['cosine']:.3f}" if d["cosine"] == d["cosine"] else "n/a"
            nul = f"{d['null']:.3f}" if d["null"] == d["null"] else "n/a"
            pr = f"{d['paired']:+.3f}" if d["paired"] == d["paired"] else "n/a"
            print(f"{split:14s} {d['n']:>3d} {cos:>7s} {nul:>6s} {ps:>8s} "
                  f"{pr:>11s} {pps:>8s}")

    _write(fac, sec)
    print(f"\nWrote -> {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


def _write(fac, sec):
    L = ["# Step 33 — Is composition predictability out-of-sample?\n"]
    L.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}. The run-up→spike test is causal "
             "per episode (run-up strictly precedes the spike); restricting it to episodes "
             "whose spike falls in the held-out era (≥ 2020-01-01) makes it a genuine OOS "
             "forecast, on the same window as the timing nulls._\n")
    L.append("| universe | split | n | run-up cosine | null | p | run-up − carry (paired) | paired p |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for nm, r in [("Factor Lens", fac), ("US sectors", sec)]:
        for split in ["dev (<2020)", "oos (≥2020)" if False else "oos (>=2020)", "full"]:
            d = r[split]
            ps = "< 0.0001" if (d["p"] == d["p"] and d["p"] < 1e-4) else (
                f"{d['p']:.4f}" if d["p"] == d["p"] else "n/a")
            pps = f"{d['paired_p']:.4f}" if d["paired_p"] == d["paired_p"] else "n/a"
            cos = f"**{d['cosine']:.3f}**" if d["cosine"] == d["cosine"] else "n/a"
            nul = f"{d['null']:.3f}" if d["null"] == d["null"] else "n/a"
            pr = f"{d['paired']:+.3f}" if d["paired"] == d["paired"] else "n/a"
            L.append(f"| {nm} | {split.replace('>=','≥')} | {d['n']} | {cos} | {nul} | "
                     f"{ps} | {pr} | {pps} |")
    L.append("")
    fo = fac["oos (>=2020)"]
    L.append("## Verdict\n")
    held = fo["p"] == fo["p"] and fo["p"] < 0.05 and fo["cosine"] > fo["null"]
    if held:
        L.append(f"**Composition predictability holds out-of-sample.** On the {fo['n']} "
                 f"held-out-era factor episodes (spike ≥ 2020), the run-up still anticipates "
                 f"the spike composition (cosine {fo['cosine']:.3f} vs null {fo['null']:.3f}, "
                 f"p {'<0.0001' if fo['p']<1e-4 else f'={fo[chr(39)+chr(112)+chr(39)]:.4f}'}), "
                 "so the boundary no longer mixes an in-sample structural statistic with "
                 "OOS predictive failures — both sides of the boundary now have a strict-OOS "
                 "leg. The timing nulls and the composition positive are measured on the same "
                 "held-out window.")
    else:
        L.append(f"**Underpowered / weaker OOS.** On the {fo['n']} held-out-era factor "
                 f"episodes the run-up cosine is {fo['cosine']:.3f} vs null {fo['null']:.3f} "
                 f"(p={fo['p']:.4f}); the held-out sample is small. The honest reading is that "
                 "composition predictability is established in-sample/structurally and "
                 "replicates across universes, but the held-out-era episode count limits a "
                 "clean single-era OOS confirmation — §7.3 must state that the boundary mixes "
                 "an in-sample structural result with OOS predictive ones.")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
