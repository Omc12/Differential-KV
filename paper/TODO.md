# Paper TODO

## Done (2026-06-30 rebuild — corrected data + residual design)
- [x] Re-audit against current code: residual mechanism + residual-key router + top-K routing +
      adaptive decode policy (auto@16k) are now the defaults; SVD seeded (deterministic).
- [x] Found & fixed the wrong data: prior `active_modes_sweep.json` was PRE-FIX (compressed
      missed the needle); compression ratio was overstated 10× (residuals omitted) → true 2.85×.
- [x] Corrected analytic accounting (`measure_active.py::analytic_kv_bytes` counts residuals+min/max).
- [x] Clean re-measurement, contention-free (`run_paper_measurements.sh`):
      ablation 4k–32k (compressed+exact), 64k reach, residual-budget sweep.
- [x] Rewrote all sections for the residual design + corrected numbers (abstract, intro,
      background, overview, compression, runtime, decode, impl, evaluation, analysis,
      limitations, conclusion) + appendices + notation.
- [x] Regenerated figures F1–F5 (now show residuals/routing/adaptive) + graphs G1–G6
      (added G6 residual trade-off) + tables T2/T3/T4/T5 from the clean JSON.

## Needs author attention before submission
- [ ] **Compile** (no system LaTeX on the build machine): on Overleaf/arXiv run
      `pdflatex main && bibtex main && pdflatex main && pdflatex main` (same for conference).
- [ ] **Verify bibliography identifiers** (arXiv ids / venues) in `bibliography/references.bib`.
- [ ] Add author names/affiliations (currently blank / "Anonymous").
- [ ] Optional: dense 64k cell (analytic only here — dense OOMs before 64k on this device).

## Future experiments (paper calls these out, not blocking)
- [ ] Hand-written fused decode kernel (Metal / CUDA-Triton) — slot into G3/G5 + T3 (placeholders ready).
- [ ] Content-adaptive residual budget (raise ratio; address the 1M-token residual memory ceiling).
- [ ] Broader models/sizes, perplexity, multi-needle/varied-depth recall, batched throughput, SRL eval.
