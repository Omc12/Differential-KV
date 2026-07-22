# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-07-22T01:41:33  |  Decode tokens/test: 128
- Per-test timeout: 2400.0s · RAM kill-cap: 8.2 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `active` — DiffKV active runtime (MLX int4)
- `dense` — Dense (mlx_lm int4, full KV)

## Prefill time (s)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| active | 6.15 | 17.37 | 54.41 | 93.82 | 224.94 |
| dense | 4.99 | 18.83 | 36.84 | 75.99 | 998.44 |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| active | 33.4 | 26.2 | 33.4 | 30.5 | 22.9 |
| dense | 72.4 | 12.8 | 18.7 | 37.9 | 17.2 |

## Peak memory (GB)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| active | 2.17 | 2.28 | 2.83 | 3.71 | 5.63 |
| dense | 2.74 | 4.16 | 6.00 | 6.00 | 6.00 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| active | 4k | ok | 4096 | 128 | 6.15 | 33.4 | 2.17 | 0.76 | 1.55 | Y | 15.5 |
| dense | 4k | ok | 4096 | 128 | 4.99 | 72.4 | 2.74 | 0.89 | 1.68 | Y | 14.1 |
| active | 8k | ok | 8192 | 128 | 17.37 | 26.2 | 2.28 | 1.32 | 1.67 | Y | 28.0 |
| dense | 8k | ok | 8192 | 128 | 18.83 | 12.8 | 4.16 | 1.34 | 1.79 | Y | 35.9 |
| active | 16k | ok | 16490 | 128 | 54.41 | 33.4 | 2.83 | 1.54 | 2.20 | Y | 63.9 |
| dense | 16k | ok | 16490 | 128 | 36.84 | 18.7 | 6.00 | 1.54 | 2.03 | Y | 50.7 |
| active | 32k | ok | 32865 | 128 | 93.82 | 30.5 | 3.71 | 1.55 | 3.12 | Y | 104.8 |
| dense | 32k | ok | 32865 | 128 | 75.99 | 37.9 | 6.00 | 1.55 | 2.45 | Y | 86.8 |
| active | 64k | ok | 65615 | 128 | 224.94 | 22.9 | 5.63 | 1.56 | 5.04 | Y | 236.9 |
| dense | 64k | ok | 65615 | 128 | 998.44 | 17.2 | 6.00 | 1.55 | 3.23 | Y | 1013.0 |
