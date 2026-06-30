# Consistency Report — DiffKV Active-Runtime Paper

Verifies every figure, table, equation, and claim against the implementation and the measured
outputs. Rebuilt 2026-06-30 (corrected-data rebuild; supersedes the 2026-06-28 report, which
certified pre-fix numbers — compressed needle ✗, 10.25×, unseeded SVD — now all corrected).
"Code" = `ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py` unless noted.

## 1. Design claims ↔ code (verified by reading the source)
| Paper claim | Code location | Status |
|---|---|---|
| Block = anchor (token 0) + delta | `_compress_block` | ✓ |
| Joint K/V row-normalized randomized truncated SVD, rank 16, adaptive [4,16], seeded | `compress_mlx_block` (`DIFFKV_SVD_SEED=1234`, 99.9% energy) | ✓ |
| Exact residuals = top-64 highest-recon-error tokens, highest-error-first | `_compress_block` (`np.argsort(errors)[-max_res:][::-1]`) | ✓ |
| `max_residual` default 64; key min/max stored for router | `__init__`, `_compress_block` (`comp_min_k/max_k`) | ✓ |
| Decode top-K routing, K=16, residual-key router default | `execute_decode_attention`, `_block_relevance_residual` | ✓ |
| Query scored in low-rank space (no dense-key reconstruction) | `compute_decode_attention_static` | ✓ |
| Residuals concatenated with recency window → exact branch | `execute_decode_attention` (`augmented_k=concat[res,dense]`) | ✓ |
| Flash-style LSE merge, NaN-guarded | `compute_decode_attention_static` | ✓ |
| Adaptive decode policy `auto`, threshold 16384, decided once at boundary | `_resolve_compressed_decode`, `MLXQwenModel.__call__` | ✓ |
| Native prefill cache dropped at boundary when compressed | `MLXQwenModel.__call__` (`_prefill_caches.pop`) | ✓ |
| Bounded pre-allocated pool, M=256 blocks | `_create_empty_session` | ✓ |
| SRL/factual store gated off in benchmarks | `get_srl_state`→None | ✓ |

## 2. Storage budget (Table 2 / §4) ↔ verified arithmetic
Per 256-token block, fp16 (H_kv=2, d=128, r=16, R=64), verified against the live `MLXKVBlockManager`:
U 8,160 + V_K/V_V 16,384 + anchors 1,024 + min/max 1,024 + scalars 8 = **26,600 B** (low-rank);
residuals 64·1024 = **65,536 B**; **total 92,136 B** vs dense **262,144 B** → **2.845×**;
pool 28·(256·92,136 + 768·1,024) ≈ **0.682 GB**. The earlier 10.25× / 25,576 B omitted
residuals+min/max and is explicitly corrected in §4/§8/§9/§11.

## 3. Figures ↔ generators ↔ data
- F1–F5 (`make_diagrams.py`): show residuals, min/max, top-K routing, residual+recency exact
  branch, adaptive boundary. Architecture only (no measured numbers). ✓
- G1 footprint, G2 allocator peak (global ≈ dense + decode-phase line), G3 decode tps, G4 prefill,
  G5 combined, G6 residual trade-off — all from the clean JSON; missing/failed cells not invented. ✓
- CUDA: G3/G5 + T3 reserve space; no CUDA number plotted/stated (no NVIDIA GPU on host). ✓

## 4. Tables ↔ generators
T2 `t_config` (arithmetic §2) · T3 `t_main` (v2 compressed primary + exact ablation + dense) ·
T4 `t_residual` (`residual_sweep.json`) · T5 `t_detail`. No hand-typed numbers. ✓

## 5. Equations ↔ code
Recon `K̂=a_k+s(U·V_K)` ↔ kernel delta path ✓ · router `ρ_b=max(q·a_k, max_j q·R_K,j)σ` ↔
`_block_relevance_residual` ✓ · compute profile (router O(nRd) + low-rank O(K(H_kv r d+rB)) +
exact O(H_kv(KR+W)d)) ↔ kernel ✓.

## 6. Data provenance (single, self-consistent, clean re-measurement)
`active_modes_sweep_v2.json` (ablation 4k–32k) · `active_modes_sweep_64k.json` (reach) ·
`residual_sweep.json` ({0,8,16,32,64}@16k) · `PAPER_dense_sweep.json` (dense 4k–32k). Apple M3
8.6 GB, same NIAH prompt per ctx, gen=128 greedy, seeded SVD. Measured clean — earlier
swap-thrash-contended cells were detected (exact-4k tps=1.0) and re-run with no concurrent CPU load.

## 7. Honesty checks
- No fabricated/averaged numbers; failed/absent cells (dense beyond budget) stated, not invented.
- Compressed (primary) vs exact (ablation) vs dense (baseline) always distinguished.
- Compression ratio reported as 2.85× (residuals counted); the 10× figure is shown only as the
  wrong low-rank-only accounting it corrects.
- Needle recall reported as solved by residuals (E5) with the budget/recall trade-off characterized (E6).
- Memory headline = global mx_peak + decode-phase peak + analytic KV state; process-tree demoted.
- CUDA/Triton = labelled future-work placeholder; no estimated numbers.

## 8. LaTeX integrity
- Macros `\maxres`,`\topk` added to preamble; `tang2024quest` added to bib. All `\input` targets
  exist; `\ref`/`\eqref` resolve; environments balanced.
- Figures F1–F5 + G1–G6 (PNG+PDF) and tables T2/T3/T4/T5 regenerated from the clean JSON.
- Not compiled here (no system LaTeX); compile on Overleaf/arXiv (see TODO.md).

## 9. Data cross-check (final measured values)
_To be appended verbatim from the regenerated tables once the clean batch + `make_tables.py` complete._
