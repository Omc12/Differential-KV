# Paper TODO

## Done
- [x] Repository audit (active = MLX runtime; SRL gated off in benchmarks).
- [x] Architecture reconstruction from implementation (notes/architecture_reconstruction.md).
- [x] Resolved measurement conflict: REPORT "active" = exact decode, not compressed. User
      chose: honest full picture + mx_peak headline (notes/DECISIONS.md).
- [x] Clean paired re-measurement: compressed + exact ablation (4k-64k), dense (4k-32k).
- [x] Figures F1-F5 (architecture/dataflow) + G1-G5 (data) generators.
- [x] Tables T2/T3/T5 generated from JSON.
- [x] Technical report (main.tex) + conference (conference.tex), shared sources.
- [x] Reproducibility appendix.

## Needs author attention before submission
- [ ] **Verify bibliography metadata** (arXiv ids / venues) in bibliography/references.bib —
      entries are accurate in title/author/year but some identifiers were intentionally omitted.
- [ ] **Compile**: no system LaTeX on the build machine. Compile on Overleaf/arXiv:
      `pdflatex main && bibtex main && pdflatex main && pdflatex main` (same for conference).
- [ ] Add author names/affiliations (currently blank / "Anonymous").
- [ ] Optional: dense 64k cell (was OOM/swap-thrash in the prior full run; not re-run here —
      stated analytically). Run if a clean OOM datapoint is wanted.

## Future experiments (paper calls these out, not blocking)
- [ ] Rank/fidelity and residual-budget/fidelity sweep (the central open problem).
- [ ] Fixed-seed SVD for run-to-run determinism.
- [ ] Hand-written fused decode kernel (Metal / CUDA-Triton) — slot into G3/G5 + T3.
- [ ] Broader models/sizes, perplexity, batched throughput, SRL-module evaluation.
