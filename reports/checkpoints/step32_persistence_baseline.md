# Step 32 — Persistence baseline for the composition flagship

_Generated 2026-06-23 16:38. Tests whether the run-up composition *anticipates* the spike, or merely reflects the **persistence** of a slow-moving composition — the central viva objection to the flagship result._

For each episode the run-up predictor (0–21d before the spike) is compared, **paired**, against carry-forward predictors from earlier windows (21–42d, 42–63d, 63–84d back) and an unconditional 'average episode' baseline. The decisive quantity is whether the run-up beats the 1-month-back carry-forward episode by episode (sign-flip permutation test + episode bootstrap CI).

| universe | n | run-up | carry 1m | carry 2m | carry 3m | uncond | run-up − 1m (paired) | sign-flip p | 95% CI | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| Factor Lens | 38 | **0.715** | 0.608 | 0.580 | 0.536 | 0.643 | +0.107 | 0.0032 | [+0.043, +0.173] | **ANTICIPATION** |
| US sectors | 47 | **0.777** | 0.725 | 0.741 | 0.718 | 0.762 | +0.052 | 0.0798 | [-0.002, +0.110] | **PERSISTENCE** |

## Reading

- **run-up vs unconditional** — how much *episode-specific* information any pre-spike window carries over 'the average crisis'. A large gap = compositions are distinctive (the cross-universe replication already established this).
- **run-up vs carry-forward (paired)** — the decisive test. If the run-up does *not* significantly beat a composition measured a month earlier, the predictability is **persistence**, not anticipation, and the flagship must be reframed.

## Verdict

- **Factor Lens: ANTICIPATION.** The run-up beats the 1-month-back carry-forward by +0.107 (paired sign-flip p=0.0032, 95% CI [+0.043, +0.173], 68% of episodes positive). The composition genuinely *converges* toward the spike as it approaches; the 'predictable composition' claim survives as stated.
- **US sectors: PERSISTENCE.** The run-up does **not** significantly beat a composition measured a month earlier (paired Δ=+0.052, sign-flip p=0.0798, 95% CI [-0.002, +0.110]). The predictability is driven by the **persistence** of a slow-moving, episode-distinctive composition, not by anticipation of the spike. Honest reframing required: composition is *persistent and distinctive* (and replicates out-of-universe) — which is still novel and useful for hedge pre-positioning — but it is not a *forecast* of a discontinuity.

**Figure:** `figures/persistence_baseline/persistence_baseline.png`.
