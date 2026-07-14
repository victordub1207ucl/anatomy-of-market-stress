# CHECKPOINT — Phase 8: Thesis Deliverables

## What was implemented
- [`thesis/v5/regime_detection_thesis_v5.tex`](../../thesis/v5/regime_detection_thesis_v5.tex) — 1,583-line LaTeX source matching the v4 thesis's depth (was an initial 600-line skeleton — rewritten on request).
- [`thesis/v5/citations.bib`](../../thesis/v5/citations.bib) — 22 BibLaTeX entries including the three pivot references (Kritzman–Li 2010 Skulls, CKT 2023 Event Time, CKT 2024 RBI) plus the carry-over v4 set (Two Sigma factor lens, Two Sigma regime modelling, Hamilton 1989, López de Prado 2018, Lo 2002, Jobson-Korkie, Memmel) and the new infrastructure refs (Ledoit-Wolf 2004, Diebold-Mariano 1995, Newey-West 1987, Politis-Romano 1994, SHAP, TreeSHAP).
- [`thesis/v5/regime_detection_thesis_v5.pdf`](../../thesis/v5/regime_detection_thesis_v5.pdf) — **24 pages**, 2.4 MB; compiles cleanly via `pdflatex → biber → pdflatex × 2`.
- Two Sigma exhibit palette carried over from v4 (`regSteady/regWOI/regCrisis/regInfl/regBull`) plus new turbulence-specific colours.
- `\graphicspath` includes 4 directories so the thesis pulls figures from `figures/thesis_v5/` + `outputs/{turbulence,event_time,supervised}/` without copying.

## Build verification
- **0 missing references, 0 missing citations.**
- **11 figures bound** — every `\includegraphics` resolves.
- 12 minor overfull-hbox warnings (worst 58.6 pt ≈ 2 cm); cosmetic, no impact on readability. The initial 158-pt warning (architecture table) was fixed by switching to `p{}` column widths.

## Structure (matches v4's depth)
1. Introduction (problem, the audit pivot, philosophy, contributions)
2. Literature grounding (with the three new pivot refs prominent)
3. System architecture (8-phase package map)
4. Data foundation (10-factor subset rationale)
5. Walk-forward infrastructure (causality contract)
6. Mahalanobis turbulence (primary regime detector)
7. Event-time conversion (load-bearing 94 % kurtosis result)
8. Supervised forecasting on turbulence targets
9. Relevance-Based Importance (with the across-window stability finding flagged)
10. Results (Phase 6 headline + per-regime + per-year + DM + bootstrap)
11. Ablations and robustness (Phase 7 — three limitations prominent)
12. Critical evaluation (solid / honest / what sets this apart)
13. Conclusion and future directions
- Appendix A: Phase-0 audit summary by module
- Appendix B: TreeSHAP benchmark
- Appendix C: Reproducibility cookbook (8 commands to rebuild everything)

## Deviations from spec
1. **Brief said "copy from v4"** — no v4 .tex file existed on disk. Created v5 from scratch, then rewrote it to match the style and depth of the v4 source the user pasted in chat.
2. **biblatex + biber** (modern), not the v4's manual `\thebibliography{99}` (legacy bibtex-style). biblatex gives proper authoryear citations and lets biber handle the `.bbl`. Functionally equivalent for the reader.
3. **No PPTX or DOCX generated**, per the brief's explicit instruction ("those come after the next supervisor meeting confirms the methodology direction").
4. **No supplementary appendix figures from the v4 pipeline** (matrix profile, motif, granger heatmaps, regime weight heatmap) — those belong to the retired v4 modules and rebuilding them in the new pipeline was out of scope. The thesis still has 11 figures, all from the Phase-2/3/6 new pipeline.

## Open questions for the user
- Add a chapter-style supplementary appendix with figures from the v4 pipeline (Granger, motif, regime weights) for context, rebuilt under the new walk-forward discipline? Would add ~2 weeks of work to rebuild those visualisations causally.
- Add a glossary of acronyms (PGTS, LW, RBI, OOS, DM, HAC, …) — useful for a thesis examiner who is not also a quant?
- Switch citation style from `authoryear` to `numeric` to save vertical space, or keep author-year for legibility?
- Cap the figure-style fixes for the remaining 12 cosmetic overfull-hboxes (mostly inside `\fbox{\parbox{...}}` blocks with long `\texttt{}` strings), or accept them since they don't visibly affect the rendered PDF?
