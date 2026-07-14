"""Phase 9F robustness — is the ordinal-bucket cutoff fix smooth in n_buckets?

The 9F headline used n_buckets=5.  To pre-empt "why 5?", sweep
n_buckets in {3, 5, 10} and confirm the cutoff-robustness improvement is
smooth across granularities rather than a cherry-picked single value.

logistic_l1 only (the configuration the bucket helps), full 2007-12+
window, the same 49-cutoff sliding test.  n=5 is reused from the 9F cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_detection.lib.cutoff_sensitivity import (
    run_sliding_cutoff, summary_statistics,
)
from regime_detection.lib.targets import binary_turbulence_entry
from regime_detection.lib.turbulence_bucket import (
    build_feature_matrix_with_bucket,
)
from regime_detection.lib.splits import DEV_END, OOS_START

ROOT = Path(__file__).resolve().parents[2]
FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
OUT = ROOT / "reports" / "robustness" / "nbuckets_sweep.parquet"
CACHE = ROOT / ".cache" / "phase9f_nbuckets"


def main() -> int:
    prices = pd.read_parquet(ROOT / "data" / "factor_prices.parquet"
                             ).sort_index()[FACTORS_10]
    tdf = pd.read_parquet(ROOT / "reports" / "turbulence"
                          / "turbulence_series.parquet")
    turb, binreg = tdf["turbulence"], tdf["regime_smoothed"]
    eq = np.log(prices["equity"] / prices["equity"].shift(1))
    thr = float(turb.loc[:DEV_END].dropna().quantile(0.90))
    y = binary_turbulence_entry(turb, threshold=thr, horizon=21)
    sched = list(pd.date_range("2018-06-30", "2022-06-30", freq="ME"))

    # binary baseline (from 9C cache)
    v9c = pd.read_parquet(ROOT / "reports" / "robustness"
                          / "sliding_cutoff.parquet")
    binr = v9c[v9c["model"] == "logistic_l1"].set_index("cutoff")
    s_bin = summary_statistics(binr, "sortino_lift")
    print(f"binary (n=1 threshold):  median={s_bin['median']:+.3f}  "
          f"frac_pos={s_bin['frac_positive']:.0%}")

    rows = [{"setting": "binary (1 threshold)", "n_buckets": 1,
             **s_bin}]
    for nb in (3, 5, 10):
        X = build_feature_matrix_with_bucket(
            prices, turb, binreg, factors=FACTORS_10, n_buckets=nb)
        df = run_sliding_cutoff(
            X, y, eq, cutoff_dates=sched, model_name="logistic_l1",
            purge_window=21, min_oos_quarters=4, random_state=0,
            cache_dir=CACHE / f"nb{nb}", verbose=False)
        s = summary_statistics(df, "sortino_lift")
        print(f"ordinal n_buckets={nb:>2d}:      median={s['median']:+.3f}  "
              f"frac_pos={s['frac_positive']:.0%}  "
              f"q25={s['q25']:+.3f}  q75={s['q75']:+.3f}")
        rows.append({"setting": f"ordinal n={nb}", "n_buckets": nb, **s})

    out = pd.DataFrame(rows).set_index("setting")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print(f"\nsaved {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
