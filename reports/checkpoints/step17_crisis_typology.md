# CHECKPOINT — crisis typology from decomposed turbulence

**Generated:** 2026-06-18 13:51 UTC

## What was done
Extracted **38 high-turbulence episodes** (causal rolling-q90 threshold, gaps ≤5d merged, ≥5d long) over 2011-08 → 2026-03, characterised each by its per-factor turbulence-composition fingerprint, and clustered (Ward, silhouette-selected k=3).

## The taxonomy

|                       |   equity |   rates |   credit |   commodities |   em_equity |   fx_usd |   inflation |   value |   quality |   vix |
|:----------------------|---------:|--------:|---------:|--------------:|------------:|---------:|------------:|--------:|----------:|------:|
| value+commodities-led |    0.07  |   0.106 |    0.062 |         0.159 |       0.132 |    0.068 |       0.063 |    0.16 |     0.115 | 0.065 |
| vix-led               |    0.153 |   0.088 |    0.026 |         0.104 |       0.017 |    0.05  |       0.025 |   -0.07 |     0.096 | 0.51  |
| equity-led            |    0.569 |   0.064 |    0.032 |         0.093 |      -0.001 |    0.074 |       0.064 |   -0.14 |     0.221 | 0.024 |

Named crises map to: 2011 Euro/US dg → vix-led; 2015 China/oil → vix-led; 2015 China/oil → value+commodities-led; 2015 China/oil → vix-led; 2018 Volmageddon → vix-led; 2018 Q4 → vix-led; 2018 Q4 → vix-led; 2018 Q4 → equity-led; 2020 COVID → value+commodities-led; 2020 COVID → value+commodities-led; 2022 inflation → equity-led; 2022 inflation → equity-led; 2022 inflation → value+commodities-led; 2022 inflation → equity-led; 2022 inflation → value+commodities-led; 2022 inflation → value+commodities-led

![taxonomy](../../../figures/crisis_typology/episode_taxonomy.png)
![timeline](../../../figures/crisis_typology/typology_timeline_predictability.png)

## Is the *kind* of stress predictable?

Cosine similarity between the 21-day pre-spike composition and the realised spike composition: **0.715** vs permutation null 0.463 (p = 0.0000, n = 38). Composition predictability: **YES**.

## Why this matters
Phase step15_composition_features showed composition does not improve *when* prediction. This asks the complementary, actionable question: *what kind* of stress — which determines **which hedge** to hold.