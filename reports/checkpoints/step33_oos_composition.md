# Step 33 — Is composition predictability out-of-sample?

_Generated 2026-06-23 17:39. The run-up→spike test is causal per episode (run-up strictly precedes the spike); restricting it to episodes whose spike falls in the held-out era (≥ 2020-01-01) makes it a genuine OOS forecast, on the same window as the timing nulls._

| universe | split | n | run-up cosine | null | p | run-up − carry (paired) | paired p |
|---|---|--:|--:|--:|--:|--:|--:|
| Factor Lens | dev (<2020) | 14 | **0.743** | 0.608 | 0.0012 | +0.115 | 0.0276 |
| Factor Lens | oos (≥2020) | 24 | **0.698** | 0.475 | < 0.0001 | +0.102 | 0.0389 |
| Factor Lens | full | 38 | **0.715** | 0.463 | < 0.0001 | +0.107 | 0.0032 |
| US sectors | dev (<2020) | 22 | **0.771** | 0.593 | < 0.0001 | +0.077 | 0.1477 |
| US sectors | oos (≥2020) | 25 | **0.781** | 0.660 | < 0.0001 | +0.029 | 0.3548 |
| US sectors | full | 47 | **0.777** | 0.615 | < 0.0001 | +0.052 | 0.0798 |

## Verdict

**Composition predictability holds out-of-sample.** On the 24 held-out-era factor episodes (spike ≥ 2020), the run-up still anticipates the spike composition (cosine 0.698 vs null 0.475, p <0.0001), so the boundary no longer mixes an in-sample structural statistic with OOS predictive failures — both sides of the boundary now have a strict-OOS leg. The timing nulls and the composition positive are measured on the same held-out window.
