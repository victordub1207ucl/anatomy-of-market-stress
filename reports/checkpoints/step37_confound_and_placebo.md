# Step 37 — Fixed-covariance distinctiveness + placebo control

_Generated 2026-06-27 18:53._

## A. Distinctiveness: rolling vs single fixed covariance

| covariance | observed | null | p |
|---|--:|--:|--:|
| rolling A (as in thesis) | 0.715 | 0.463 | <0.0001 |
| **fixed full-sample A** | **0.715** | 0.474 | <0.0001 |

**Verdict — GENUINE (not an A-matching artefact).** Recomputing every episode's composition under one fixed full-sample covariance — so run-ups and shuffled spikes share the identical A — the observed cosine still clears the permutation null decisively. The distinctiveness is a property of the return-direction composition, not of time-varying covariance matching.

## B. Placebo: is the convergence ladder special to crises?

- real episode mean slope: **-0.0562** (n=38)
- pseudo (random non-crisis dates) mean slope: **-0.0188** (n=400)
- real − pseudo: -0.0374; permutation p (real steeper): **0.0062**

**Verdict — sharpening beyond ambient persistence.** Crisis run-ups converge on their spike materially faster than random windows converge on an arbitrary endpoint, so the sharpening is specific to episodes, not a universal autocorrelation artefact.

