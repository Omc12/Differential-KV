# Consistency Report

Every quantitative claim in the paper was cross-checked against the measured JSON
(`paper/generated/active_modes_sweep.json`, `benchmarks/results/PAPER_dense_sweep.json`)
or derived from the model dimensions in code. Verified 2026-06-28.

## Data provenance (single self-consistent dataset)
- **active (compressed + exact)**: `paper/scripts/measure_active.py`, one isolated
  subprocess per cell, Apple M3 / 8.6 GB, Qwen2.5-1.5B int4, greedy 128-token decode.
- **dense**: `benchmarks/run_bench.py --engines dense`, same host/prompts/metric.
- REPORT.md's old "active" numbers (exact-decode mislabel) are **not** used anywhere.

## Numbers verified against JSON
| Claim (paper) | Source | Status |
|---|---|---|
| Compressed decode tok/s 8.0/8.1/8.2/8.0/6.8 (4k–64k) | active JSON | ✓ |
| Exact decode tok/s 42.8/40.3/35.2/24.2/17.6 | active JSON | ✓ |
| Dense tok/s 66.9/61.7/47.3/30.1 (4k–32k) | dense JSON | ✓ |
| Dense prefill 5.1/11.1/27.4/92.3 s | dense JSON | ✓ |
| Needle: compressed ✗ all ctx; exact ✓ all ctx; dense ✓ to 32k | both JSON | ✓ |
| KV store occupied 0.028/0.040/0.066/0.111/0.203 GB | active JSON (analytic) | ✓ |
| Dense full-KV 0.117/0.235/0.473/0.942/1.881 GB | active JSON (analytic) | ✓ |
| Footprint ratio 4.1×→9.3× | computed (1.881/0.203=9.27; 0.117/0.028=4.14) | ✓ (fixed from 9.4) |
| Per-block 25,576 B vs 262,144 B = 10.25× | dims (Hkv2,d128,B256,r16) | ✓ |
| Decode-phase MLX peak 1.70/1.70/1.70/2.03/2.97 GB (compressed) | active JSON | ✓ |
| Reaches 65,615 tokens; pool cap 65,536+768 | dims + prompt_ref | ✓ |
| Per-token dense KV 28.7 KB; 1.88 GB@64k; 0.47 GB@16k | 4·L·Hkv·d = 28,672 B/tok | ✓ |

## Architecture/claims verified against implementation
- active runtime = MLX wrapper (hf_diffkv_wrapper.py:1220 rebind). ✓
- SRL/factual module inert (get_srl_state→None, _session_srl never populated). ✓
- block_size 256, rank 16, recency 512 (buffer 768), max_blocks 256, fp16 store. ✓
- compress_mlx_block: anchor + row-normalized joint K/V randomized truncated SVD, adaptive
  rank [4,16], scale, re-applied norms. ✓ (Alg. 1 matches code)
- decode kernel scores in low-rank space (q·V_K then ·Uᵀ), LSE merge, NaN guards. ✓ (Alg. 3)
- prefill→decode boundary drops native cache under compressed mode. ✓
- compressed path non-deterministic (unseeded rSVD) — stated as limitation. ✓

## Honesty checks
- No fabricated or averaged numbers; failed/absent cells (dense 64k) stated, not invented.
- Compressed (primary) vs exact (ablation) vs dense (baseline) always distinguished.
- Memory headline = mx_peak + analytic KV state; process-tree metric demoted (per decision).
- CUDA/Triton = labelled future-work placeholder; no estimated numbers.

## LaTeX integrity
- All `\input` targets exist; all `\ref`/`\eqref` resolve to a `\label`; environments balanced.
- Figures F1–F5 (diagrams) + G1–G5 (data) present as PNG+PDF; tables T2/T3/T5 generated.
- Not compiled here (no system LaTeX); compile on Overleaf/arXiv (see TODO.md).
