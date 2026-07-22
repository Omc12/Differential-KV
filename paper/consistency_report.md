# Consistency Report — DKV Active-Runtime Paper (clean rebuild 2026-07-07)

Verifies every figure, table, equation, and claim against the implementation and the **clean**
measured outputs. Supersedes the 2026-06-30 report (which certified rank-16, decode-cache-off,
partly memory-contended data). "Code" = `ACTIVE_RUNTIME/serving/mlx_dkv_wrapper.py` unless noted.

## 0. What changed in this rebuild (and why)
- **Config corrected to what the active CLI actually runs**: preset `mid`, `serving_mode=balanced`,
  **rank 32** (the `mid` default — only `low` drops to 16; the earlier rank-16 runs were wrong),
  int4, `DKV_COMPRESSED_DECODE=1` (force the full DKV sparse path at every context),
  `DKV_DECODE_CACHE=1`, `DKV_SPARSE_PREFILL=1`, `DKV_SPARSE_BIAS=auto`,
  `DKV_MAX_RESIDUAL=128`, seed 1234.
- **Contamination removed**: the primary sweep is now measured one isolated process at a time with
  nothing else running (no figure rendering / compiles concurrent). The earlier "dense-32k prefill
  ≈ 200 s" and depressed dense-tps figures were an artifact of concurrent CPU load; the clean run
  gives stable, reproducible numbers (dense-32k prefill ≈ 78 s, close to the independent 06-30 clean
  value of 73 s).
- **Figures — correctness pass done, aesthetic pass pending.** All data graphs (G1–G7) and
  architecture diagrams (F1–F6) have been regenerated from the CLEAN rank-32 data, so the embedded
  figures now agree with the tables and prose (decode/prefill/footprint curves, block-budget
  numbers 178 KiB / 1.44×, rank shapes [255,32] / [2,32,128], projected-query dim ℝ³²). What is
  still owed is the *presentation-quality aesthetic* pass the project owner will drive via Fable
  (industry-deck styling, specific shades of blue, body text always in black); the current figures
  are functional but not that final styling. Data graphs cover the 4k–32k head-to-head range where
  both engines run; the 64k reach (active-only, dense OOM) lives in Table~3 and the prose.

## 1. Design claims ↔ code (verified by reading the source)
| Paper claim | Code location | Status |
|---|---|---|
| Block = anchor (token 0) + delta | `compress_deferred_prefill_blocks` | ✓ |
| Joint K/V, V-rescaled to K RMS, row-normalized randomized truncated SVD, seeded, adaptive rank | same (`DKV_V_SCALE`, `compress_mlx_block_batched`, seed 1234, 99.9% energy) | ✓ |
| Exact residuals = top-`max_residual` highest joint-error tokens | `compress_deferred_prefill_blocks` (`argsort(errors)[-max_res:]`) | ✓ |
| `max_residual` default 128; key min/max stored for router | `__init__`, scatter loop | ✓ |
| Decode top-K routing K=16, residual-key router default | `execute_decode_attention`, `_block_relevance_residual` | ✓ |
| Query scored in low-rank space (no dense-K reconstruction) | `compute_decode_attention_static` | ✓ |
| Residuals + recency = exact branch; flash-style LSE merge, NaN-guarded | same | ✓ |
| Decompress-and-cache decode (bit-exact, re-route every N) | `_execute_decode_cache` (`DKV_DECODE_CACHE`) | ✓ |
| Block-sparse prefill (sink + routed + window + self) | `_sparse_prefill_attend` (`DKV_SPARSE_PREFILL`) | ✓ |
| Prefill→decode native-cache release | `MLXQwenModel.__call__` (`_prefill_caches.pop`, `mx.clear_cache`) | ✓ |
| Bounded pre-allocated pool, M=256 blocks | `_create_empty_session` | ✓ |
| rank 32 under `mid` preset | `cli.py` (`args.rank` default 32; only `low`→16) | ✓ |
| SRL/factual store gated off | `get_srl_state`→None, `DKV_FACTUAL_STORE=0` | ✓ |

## 2. Data provenance (single clean dataset)
- Primary: `benchmarks/results/clean_{active,dense}_{4096..65536}.json` — one back-to-back sweep,
  isolated per cell, Apple M3 8.6 GB, Qwen2.5-1.5B int4, greedy gen=128, deterministic NIAH prompt.
- Ablations: `paper/generated/active_modes_fresh.json` (compressed vs exact decode, 4k–32k) and
  `paper/generated/residual_sweep_fresh.json` (R = 0/8/16/32/64/128 @16k).
- Body-text numbers are LaTeX macros emitted by `make_facts.py` from these files, so prose cannot
  drift from the data.

## 3. Tables ↔ generators (regenerate after data lands)
T1 config (code dims) · T2 per-block budget (code dims, rank 32; R=128 & R=64) · T3 main results
(clean primary) · T4 residual sweep (measured) · T5 per-run detail (clean primary) · T6 decode-mode
ablation (measured). All emitted by `make_tables.py`; no hand-typed numbers.

## 4. Honesty checks
- Decode-throughput cost stated plainly (DKV decode is slower than a dense cache that fits).
- The "dense OOMs at 32k" claim from older drafts is RETRACTED — dense completes to 64k here.
- Compression ratio reported at the measured config (rank 32): lower than the old rank-16 figure;
  both residual presets shown.
- Memory: analytic KV state is bounded and smaller; measured allocator peak is a wash / slightly
  higher for DKV at this model size — stated explicitly.
- CUDA/Triton = future-work placeholder; no CUDA number plotted or stated.

## 5. Status
- [x] Clean rank-32 primary sweep (4k–32k) + 64k reach (active ✓ / dense OOM) measured.
- [x] Clean decode-mode ablation (4k–32k) and residual-budget sweep (@16k) measured.
- [x] `make_facts.py` + `make_tables.py` regenerated from the clean data.
- [x] Figures regenerated at rank 32 (correctness); prose/tables/figures mutually consistent.
- [x] `main.tex` (31 pp) and `conference.tex` compile with tectonic; no undefined refs/cites.
- [ ] FINAL aesthetic figure pass (Fable): industry-deck styling, blue shades, black body text.
- [ ] (Pre-camera-ready) verify arXiv ids in `references.bib`; broaden eval (RULER/LongBench,
  multi-model, baselines) per §10 limitations.

## 6. Headline measured numbers (clean, of record)
Apple M3 / 8.6 GB, Qwen2.5-1.5B int4, mid/balanced/rank-32, COMPRESSED_DECODE=1, greedy gen=128:
| ctx | DKV prefill s | DKV tok/s | dense prefill s | dense tok/s | needle |
|----|----|----|----|----|----|
| 4k | 6.6 | 19.9 | 5.1 | 65.7 | ✓/✓ |
| 8k | 13.6 | 18.4 | 11.8 | 55.3 | ✓/✓ |
| 16k | 28.2 | 18.7 | 27.8 | 47.0 | ✓/✓ |
| 32k | 58.5 | 17.0 | 77.9 | 35.7 | ✓/✓ |
| 64k | 928 | 8.6 | **OOM** | — | ✓ / n-a |
Per-block compression (rank 32): R=128 → 1.44×, R=64 → 2.25×. Residual sweep @16k: needle ✓ for
R=8..128, ratio 3.80×→1.40×. Decode ablation @16k: compressed 18.8 vs exact 30.3 tok/s (both ✓).
