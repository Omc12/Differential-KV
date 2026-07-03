# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-07-03T18:31:46  |  Decode tokens/test: 128
- Per-test timeout: 1500.0s · RAM kill-cap: 7.2 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `active` — DiffKV active runtime (MLX int4)
- `dense` — Dense (mlx_lm int4, full KV)

## Prefill time (s)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| active | 6.61 | 13.31 | 42.29 |
| dense | 5.08 | 12.48 | 39.36 |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| active | 9.3 | 5.7 | 2.9 |
| dense | 71.8 | 30.5 | 35.7 |

## Peak memory (GB)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| active | 3.50 | 3.96 | 4.30 |
| dense | 2.63 | 4.05 | 5.89 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| active | 4k | ok | 4096 | 128 | 6.61 | 9.3 | 3.50 | 1.18 | 1.77 | - | 25.7 |
| dense | 4k | ok | 4096 | 128 | 5.08 | 71.8 | 2.63 | 1.11 | 1.68 | Y | 11.9 |
| active | 8k | ok | 8192 | 128 | 13.31 | 5.7 | 3.96 | 1.27 | 1.89 | - | 41.9 |
| dense | 8k | ok | 8192 | 128 | 12.48 | 30.5 | 4.05 | 0.39 | 1.79 | Y | 24.2 |
| active | 16k | ok | 16490 | 128 | 42.29 | 2.9 | 4.30 | 1.45 | 2.25 | - | 95.1 |
| dense | 16k | ok | 16490 | 128 | 39.36 | 35.7 | 5.89 | 0.47 | 2.01 | Y | 54.7 |
