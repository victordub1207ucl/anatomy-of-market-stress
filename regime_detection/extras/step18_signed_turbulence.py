"""Directional turbulence: does loss-side turbulence predict drawdowns
better than total turbulence?

Splits d_t exactly into ``turb_loss`` (factors deviating in their stress
direction; VIX sign-flipped) and ``turb_gain``, then compares the three
series as univariate predictors of the forward 21-day equity drawdown,
on the dev (=<2019-12-31) and OOS (>=2020) windows separately.

Run::

    python3 -m regime_detection.step18_signed_turbulence
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from regime_detection.lib.turbulence_signed import SignedTurbulence
from regime_detection.lib.targets import forward_drawdown_conditional
from regime_detection.lib.splits import DEV_END

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "signed_turbulence"
FIG_DIR = PROJECT_ROOT / "figures" / "signed_turbulence"
CHECKPOINT_PATH = (PROJECT_ROOT / "reports"
                   / "checkpoints" / "step18_signed_turbulence.md")

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
HORIZON = 21
CRASH_QUANTILE = 0.05  # dev-window quantile defining a "crash" label


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading prices, computing signed turbulence (flip=vix) ...")
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    rets = np.log(prices / prices.shift(1)).dropna(how="any")
    signed = SignedTurbulence(flip=("vix",)).fit_transform(rets)
    signed.to_parquet(OUT_DIR / "signed_turbulence.parquet")
    fin = signed.dropna()
    err = (fin["turb_loss"] + fin["turb_gain"] - fin["turbulence"]).abs().max()
    print(f"  → {len(fin)} rows; partition check max err {err:.2e}")
    assert err < 1e-9

    # Forward 21d equity drawdown (worst forward cumulative log return).
    fwd_dd = forward_drawdown_conditional(prices["equity"], horizon=HORIZON)
    fwd_dd = fwd_dd.reindex(fin.index)
    crash_thr = float(fwd_dd.loc[:DEV_END].quantile(CRASH_QUANTILE))
    crash = (fwd_dd < crash_thr).astype(float)
    print(f"  → crash threshold (dev {CRASH_QUANTILE:.0%} fwd-dd): "
          f"{crash_thr:+.4f}; OOS crash rate "
          f"{crash.loc[DEV_END:].mean():.1%}")

    signals = {
        "turbulence (total)": fin["turbulence"],
        "turb_loss": fin["turb_loss"],
        "turb_gain": fin["turb_gain"],
        "loss_share": (fin["turb_loss"] / fin["turbulence"]).clip(-2, 2),
    }
    rows = []
    for split, sl in [("dev", slice(None, DEV_END)), ("oos", slice(DEV_END, None))]:
        y_dd = fwd_dd.loc[sl].dropna()
        y_cr = crash.reindex(y_dd.index)
        for name, s in signals.items():
            x = s.reindex(y_dd.index)
            ok = x.notna() & y_dd.notna()
            rho = spearmanr(x[ok], y_dd[ok]).statistic
            auc = (roc_auc_score(y_cr[ok], x[ok])
                   if y_cr[ok].nunique() > 1 else np.nan)
            rows.append({"split": split, "signal": name,
                         "spearman_vs_fwd_dd": rho, "auc_crash": auc,
                         "n": int(ok.sum())})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "predictive_comparison.csv", index=False)
    print(res.round(4).to_string(index=False))

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    zoom = fin.loc["2019-10":"2020-12"]
    ax.plot(zoom.index, zoom["turb_loss"], color="#c0392b", lw=1.0,
            label="turb_loss (stress-side)")
    ax.plot(zoom.index, zoom["turb_gain"], color="#27ae60", lw=1.0,
            label="turb_gain")
    ax.plot(zoom.index, zoom["turbulence"], color="#7f8c8d", lw=0.8,
            ls="--", label="total d_t")
    ax.set_title("COVID window — loss vs gain side of turbulence")
    ax.legend(fontsize=8)

    ax = axes[1]
    piv = res.pivot(index="signal", columns="split", values="auc_crash")
    piv = piv.loc[list(signals)]
    x = np.arange(len(piv)); w = 0.36
    ax.bar(x - w/2, piv["dev"], w, color="#1f77b4", label="dev")
    ax.bar(x + w/2, piv["oos"], w, color="#e67e22", label="OOS")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"AUC — forward {HORIZON}d crash")
    ax.set_title("Crash-prediction AUC by signal")
    ax.legend(fontsize=9)
    fig.suptitle("Directional (signed) turbulence", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "signed_turbulence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {FIG_DIR / 'signed_turbulence.png'}")

    # ---- checkpoint ----
    def _get(split, sig, col):
        return float(res[(res.split == split) & (res.signal == sig)][col].iloc[0])

    d_loss_dev, d_loss_oos = _get("dev", "turb_loss", "auc_crash"), _get("oos", "turb_loss", "auc_crash")
    d_tot_dev, d_tot_oos = _get("dev", "turbulence (total)", "auc_crash"), _get("oos", "turbulence (total)", "auc_crash")
    wins = (d_loss_dev > d_tot_dev) + (d_loss_oos > d_tot_oos)
    if wins == 2:
        verdict = (f"**SUPPORTED.** turb_loss beats total turbulence on crash "
                   f"AUC in BOTH windows (dev {d_tot_dev:.3f}->{d_loss_dev:.3f}; "
                   f"OOS {d_tot_oos:.3f}->{d_loss_oos:.3f}). The sign split "
                   "carries real directional information.")
    elif wins == 1:
        verdict = (f"**MIXED.** turb_loss beats total in one window only "
                   f"(dev {d_tot_dev:.3f} vs {d_loss_dev:.3f}; OOS "
                   f"{d_tot_oos:.3f} vs {d_loss_oos:.3f}).")
    else:
        verdict = (f"**NOT SUPPORTED.** Total turbulence matches or beats "
                   f"turb_loss in both windows (dev {d_tot_dev:.3f} vs "
                   f"{d_loss_dev:.3f}; OOS {d_tot_oos:.3f} vs {d_loss_oos:.3f}).")

    md = [
        "# CHECKPOINT — directional (signed) turbulence",
        "",
        f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC",
        "",
        "## Idea",
        "Mahalanobis turbulence is sign-blind. We split d_t **exactly** into "
        "`turb_loss` (contributions from factors deviating in their stress "
        "direction; VIX flipped so a vol spike is stress-side) and "
        "`turb_gain`, the cross-sectional analogue of realized "
        "semivariance. Hypothesis: loss-side turbulence predicts forward "
        f"{HORIZON}d equity drawdowns better than total d_t.",
        "",
        "## Predictive comparison (univariate)",
        "",
        res.round(4).to_markdown(index=False),
        "",
        "![signed](../../../figures/signed_turbulence/signed_turbulence.png)",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Notes",
        "- Exact partition: turb_loss + turb_gain == d_t (checked at runtime).",
        "- Causal: same walk-forward machinery as TurbulenceIndex.",
        "- Crash label threshold fixed on dev only.",
    ]
    CHECKPOINT_PATH.write_text("\n".join(md))
    print(f"  → wrote {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
