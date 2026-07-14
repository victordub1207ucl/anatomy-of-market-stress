# Step 36 — Cluster-robust convergence trend

_Generated 2026-06-27 17:34._

38-episode convergence slope (cosine-to-spike vs lookback distance; negative = converges toward the spike). Episode-level mean slope **-0.0562** (naive sign-flip p = 0.0114, the optimistic number).

The 38 episodes form **14 meta-crisis clusters** (sizes [1, 4, 1, 1, 3, 4, 2, 8, 7, 2, 2, 1, 1, 1]). Aggregating to one mean slope per cluster and testing at the cluster level:

- cluster-mean slope: **-0.0589**
- cluster sign-flip permutation p: **0.0218**
- cluster-bootstrap 95% CI: **[-0.1038, -0.0172]**
- one-sample t-test on 14 cluster slopes: p = 0.0240; Wilcoxon p = 0.0419

## Verdict

**Survives clustering.** Even with the effective sample reduced to 14 independent meta-crises, the convergence trend is significant (cluster sign-flip p = 0.0218, bootstrap CI excludes zero). The anticipation claim's in-sample evidence is robust to the dependence that voids the OOS test; the honest line is 'anticipation established in-sample (cluster-robust), marginal out-of-sample.'
