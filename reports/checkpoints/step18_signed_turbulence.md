# CHECKPOINT — directional (signed) turbulence

**Generated:** 2026-06-18 13:51 UTC

## Idea
Mahalanobis turbulence is sign-blind. We split d_t **exactly** into `turb_loss` (contributions from factors deviating in their stress direction; VIX flipped so a vol spike is stress-side) and `turb_gain`, the cross-sectional analogue of realized semivariance. Hypothesis: loss-side turbulence predicts forward 21d equity drawdowns better than total d_t.

## Predictive comparison (univariate)

| split   | signal             |   spearman_vs_fwd_dd |   auc_crash |    n |
|:--------|:-------------------|---------------------:|------------:|-----:|
| dev     | turbulence (total) |              -0.1767 |      0.7681 | 3039 |
| dev     | turb_loss          |              -0.1294 |      0.6917 | 3039 |
| dev     | turb_gain          |              -0.1128 |      0.693  | 3039 |
| dev     | loss_share         |              -0.02   |      0.5117 | 3039 |
| oos     | turbulence (total) |              -0.0585 |      0.5875 | 1581 |
| oos     | turb_loss          |              -0.0506 |      0.5529 | 1581 |
| oos     | turb_gain          |              -0.013  |      0.5293 | 1581 |
| oos     | loss_share         |              -0.0273 |      0.5101 | 1581 |

![signed](../../../figures/signed_turbulence/signed_turbulence.png)

## Verdict

**NOT SUPPORTED.** Total turbulence matches or beats turb_loss in both windows (dev 0.768 vs 0.692; OOS 0.588 vs 0.553).

## Notes
- Exact partition: turb_loss + turb_gain == d_t (checked at runtime).
- Causal: same walk-forward machinery as TurbulenceIndex.
- Crash label threshold fixed on dev only.