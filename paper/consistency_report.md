# Consistency Report — DiffKV Active-Runtime Paper (clean rebuild 2026-07-07)

Verifies every figure, table, equation, and claim against the implementation and the **clean**
measured outputs. Supersedes the 2026-06-30 report (which certified rank-16, decode-cache-off,
partly memory-contended data). "Code" = `ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py` unless noted.

## 0. What changed in this rebuild (and why)
- **Config corrected to what the active CLI actually runs**: preset `mid`, `serving_mode=balanced`,
  **rank 32** (the `mid` default — only `low` drops to 16; the earlier rank-16 runs were wrong),
  int4, `DIFFKV_COMPRESSED_DECODE=1` (force the full DiffKV sparse path at every context),
  `DIFFKV_DECODE_CACHE=1`, `DIFFKV_SPARSE_PREFILL=1`, `DIFFKV_SPARSE_BIAS=auto`,
  `DIFFKV_MAX_RESIDUAL=128`, seed 1234.
- **Contamination removed**: the primary sweep is now measured one isolated process at a time with
  nothing else running (no figure rendering / compiles concurrent). The earlier "dense-32k prefill
  ≈ 200 s" and depressed dense-tps figures were an artifact of concurrent CPU load; the clean run
  gives stable, reproducible numbers (dense-32k prefill ≈ 78 s, close to the independent 06-30 clean
  value of 73 s).
- **Figures are deferred**: per the project owner, all data graphs and diagrams will be regenerated
  in a final presentation-quality pass (shades of blue, black text). The current `paper/figures/`
  PNG/PDFs still carry rank-16 shapes/KiB in a few captions/labels and MUST be regenerated at rank
  32 before release. The prose and tables in this build are already rank-32-correct (they read the
  code dims / measured JSON via macros).

## 1. Design claims ↔ code (verified by reading the source)
| Paper claim | Code location | Status |
|---|---|---|
| Block = anchor (token 0) + delta | `compress_deferred_prefill_blocks` | ✓ |
| Joint K/V, V-rescaled to K RMS, row-normalized randomized truncated SVD, seeded, adaptive rank | same (`DIFFKV_V_SCALE`, `compress_mlx_block_batched`, seed 1234, 99.9% energy) | ✓ |
| Exact residuals = top-`max_residual` highest joint-error tokens | `compress_deferred_prefill_blocks` (`argsort(errors)[-max_res:]`) | ✓ |
| `max_residual` default 128; key min/max stored for router | `__init__`, scatter loop | ✓ |
| Decode top-K routing K=16, residual-key router default | `execute_decode_attention`, `_block_relevance_residual` | ✓ |
| Query scored in low-rank space (no dense-K reconstruction) | `compute_decode_attention_static` | ✓ |
| Residuals + recency = exact branch; flash-style LSE merge, NaN-guarded | same | ✓ |
| Decompress-and-cache decode (bit-exact, re-route every N) | `_execute_decode_cache` (`DIFFKV_DECODE_CACHE`) | ✓ |
| Block-sparse prefill (sink + routed + window + self) | `_sparse_prefill_attend` (`DIFFKV_SPARSE_PREFILL`) | ✓ |
| Prefill→decode native-cache release | `MLXQwenModel.__call__` (`_prefill_caches.pop`, `mx.clear_cache`) | ✓ |
| Bounded pre-allocated pool, M=256 blocks | `_create_empty_session` | ✓ |
| rank 32 under `mid` preset | `cli.py` (`args.rank` default 32; only `low`→16) | ✓ |
| SRL/factual store gated off | `get_srl_state`→None, `DIFFKV_FACTUAL_STORE=0` | ✓ |

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
- Decode-throughput cost stated plainly (DiffKV decode is slower than a dense cache that fits).
- The "dense OOMs at 32k" claim from older drafts is RETRACTED — dense completes to 64k here.
- Compression ratio reported at the measured config (rank 32): lower than the old rank-16 figure;
  both residual presets shown.
- Memory: analytic KV state is bounded and smaller; measured allocator peak is a wash / slightly
  higher for DiffKV at this model size — stated explicitly.
- CUDA/Triton = future-work placeholder; no CUDA number plotted or stated.

## 5. TODO before release
- [ ] Regenerate ALL figures at rank 32 in the final visual pass (blue shades, black text).
- [ ] Re-run `make_facts.py` + `make_tables.py` after the clean ablation/64k cells finish.
- [ ] Compile `main.tex` and `conference.tex` with tectonic; fix any float/overfull warnings.
- [ ] Verify each `\ref`/`\eqref` resolves and each `\cite` key exists in `references.bib`.
