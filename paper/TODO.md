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
- [x] **Compile** — both documents build clean with `tectonic -X compile {main,conference}.tex`
      (0 undefined refs/cites, 0 overfull boxes in the conference build).
- [x] **Verify bibliography identifiers** — arXiv ids checked against arxiv.org 2026-07-25;
      KIVI upgraded to its ICML'24 journal-ref. Venues not stated by arXiv were left as preprints.
- [x] Add author names/affiliations.
- [x] Dense 64k cell — measured, not analytic: the optimized dense engine *does* reach 64k
      (821 s prefill). The "dense OOMs before 64k" note is retracted.

## Open before any camera-ready
- [ ] **Weight-matched PyTorch baseline.** The current one runs unquantized fp16
      (`Qwen/Qwen2.5-1.5B-Instruct`) because `transformers` cannot load the MLX int4 checkpoint;
      the paper now says so explicitly, but a quantized PyTorch baseline would be the better
      experiment.
- [ ] **Fix the PyTorch prefill timer.** `benchmarks/bench_worker.py::run_dense` stops the prefill
      timer with no `torch.mps.synchronize()`, so prefill work is charged to decode (prefill
      *falls* 0.83 s @4k → 0.50 s @8k). Documented as a caveat in §8; should be re-measured.
- [ ] **Commit the E7–E12 result JSONs.** `run_multi_needle_mlx.py`, `run_llama3b_mlx.py`,
      `run_signal_ablation_mlx.py`, `run_lego_mem_mlx.py` emit files that are not in the repo, so
      T7–T10 are hand-transcribed and not machine-verifiable. Their throughput columns are prompt
      tokens ÷ end-to-end time (now labelled "Prompt tok/s"), not decode tok/s.
- [ ] **Re-run E5/E6 under the current decode configuration** so their absolute tok/s line up with
      Table 3 instead of needing a "not comparable across tables" note.
- [ ] Broaden eval (RULER/LongBench, cache-compression baselines) per §10 limitations.

## Future experiments (paper calls these out, not blocking)
- [ ] Hand-written fused decode kernel (Metal / CUDA-Triton) — slot into G3/G5 + T3 (placeholders ready).
- [ ] Content-adaptive residual budget (raise ratio; address the 1M-token residual memory ceiling).
- [ ] Broader models/sizes, perplexity, multi-needle/varied-depth recall, batched throughput, SRL eval.
