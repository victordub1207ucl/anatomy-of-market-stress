# Checkpoints — one results writeup per pipeline step

*Files are named after the `scripts/stepNN_*` driver that produces or documents them. `step00a/b` precede the numbered pipeline (audit + infrastructure); `framing_decision.md` and `thesis_draft_log.md` are decision/log documents without a script.*

Brief end-of-phase summaries: what was implemented, what tests pass,
deviations from the original spec (and why), and open questions for the
user. Generated at the close of each phase from the canonical reports
under `reports/` and the live state of `regime_detection/` and
`outputs/`.

## Index

| step (historical phase) | title | tests | report |
|---:|---|---:|---|
| [0](step00a_lookahead_audit.md) | Look-ahead audit | n/a | [audit_lookahead.md](../../docs/audit_lookahead.md) |
| [1](step00b_walk_forward_infrastructure.md) | Walk-forward infrastructure | 29 | — |
| [2](step01_turbulence.md) | Mahalanobis turbulence index | 19 | [02 notebook](../../notebooks/step01_turbulence_diagnostics.ipynb) |
| [3](step03_event_time.md) | Event-time conversion | 21 | [03 notebook](../../notebooks/step03_event_time_diagnostics.ipynb) |
| [4](step04_supervised_baseline.md) | Supervised forecasting | 16 | [04 supervised results](../step04_supervised_results.md) |
| [5](step05_rbi_explainability.md) | Relevance-Based Importance | 14 | [05 notebook](../../notebooks/step05_rbi_explainability.ipynb) |
| [6](step06_oos_evaluation.md) | Strict OOS evaluation | 15 | [06 OOS results](../step06_oos_results.md) |
| [7](step07_ablations.md) | Benchmarks and ablations | — | [07 ablations](../step07_ablations.md) |
| [8](thesis_draft_log.md) | Thesis regeneration | — | [thesis v5 PDF](../../thesis/v5/regime_detection_thesis_v5.pdf) |

**Total tests across phases: 114/114 passing.**

## Reading order
Phases 0 → 8 are sequential; each builds on the artefacts of the
previous. The audit (Phase 0) motivates everything that follows. The
Phase 7 checkpoint contains the three thesis limitations most worth
acting on; the Phase 8 checkpoint summarises which open questions
materially affect the headline thesis claim.

## Convention going forward
Each new phase should add a `phase{N}_{short_name}.md` checkpoint
before the phase is considered done. The file should be short (one
page), enumerate exactly the four sections above, and list every
open question explicitly so they do not get lost between sessions.

**Phase 9 onward** — checkpoints live at
[`regime_detection/reports/checkpoints/`](../../regime_detection/reports/checkpoints/)
per the Phase 9 brief's instruction. Phases 0–8 remain here in
`checkpoints/` as a historical record. The index will be updated to
link both locations once Phase 9 closes.
