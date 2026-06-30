# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-06-30T01:22:42  |  Decode tokens/test: 128
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `native` — DiffKV native (C++, GGUF Q4_K_M)
- `active` — DiffKV active runtime (MLX int4)
- `dense` — Dense (mlx_lm int4, full KV)
- `ollama` — Ollama / llama.cpp (GGUF Q4_K_M)

## Prefill time (s)

| Engine | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| native | 6.13 | 12.87 | 33.96 | 14189.80 |
| active | 8.84 | 18.98 | 44.80 | 117.67 |
| dense | 5.11 | 11.20 | 32.14 | **CRASH** |
| ollama | 4.83 | 11.06 | 3396.06 | 87.04 |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| native | 11.1 | 8.5 | 4.9 | 1.9 |
| active | 39.9 | 36.6 | 9.4 | 7.7 |
| dense | 65.1 | 61.2 | 15.1 | **CRASH** |
| ollama | 66.2 | 60.3 | 0.1 | 1000000.0 |

## Peak memory (GB)

| Engine | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| native | 2.26 | 2.43 | 3.17 | 4.86 |
| active | 3.31 | 3.44 | 3.90 | 4.84 |
| dense | 2.62 | 4.05 | 5.89 | **CRASH** |
| ollama | 1.29 | 1.46 | 1.62 | 2.44 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 4k | ok | 4096 | 18 | 6.13 | 11.1 | 2.26 | 2.26 | — | - | 10.8 |
| active | 4k | ok | 4096 | 128 | 8.84 | 39.9 | 3.31 | 1.10 | 2.15 | Y | 17.1 |
| dense | 4k | ok | 4096 | 128 | 5.11 | 65.1 | 2.62 | 1.39 | 1.68 | Y | 11.7 |
| ollama | 4k | ok | 4096 | 128 | 4.83 | 66.2 | 1.29 | 1.29 | — | Y | 8.8 |
| native | 8k | ok | 8192 | 18 | 12.87 | 8.5 | 2.43 | 2.34 | — | - | 18.0 |
| active | 8k | ok | 8192 | 128 | 18.98 | 36.6 | 3.44 | 1.98 | 2.27 | Y | 27.1 |
| dense | 8k | ok | 8192 | 128 | 11.20 | 61.2 | 4.05 | 1.39 | 1.79 | Y | 18.7 |
| ollama | 8k | ok | 8192 | 128 | 11.06 | 60.3 | 1.46 | 1.46 | — | Y | 15.1 |
| native | 16k | ok | 16490 | 18 | 33.96 | 4.9 | 3.17 | 2.58 | — | - | 40.8 |
| active | 16k | ok | 16490 | 128 | 44.80 | 9.4 | 3.90 | 1.92 | 2.51 | Y | 63.3 |
| dense | 16k | ok | 16490 | 128 | 32.14 | 15.1 | 5.89 | 1.40 | 2.03 | Y | 50.6 |
| ollama | 16k | ok | 16490 | 128 | 3396.06 | 0.1 | 1.62 | 1.62 | — | Y | 75.5 |
| native | 32k | ok | 32865 | 33 | 14189.80 | 1.9 | 4.86 | 2.24 | — | - | 183.6 |
| active | 32k | ok | 32865 | 128 | 117.67 | 7.7 | 4.84 | 1.18 | 2.98 | Y | 139.3 |
| dense | 32k | crash | — | — | — | — | 5.89 | 1.40 | — |  | 95.5 |
| ollama | 32k | ok | 32767 | 1 | 87.04 | 1000000.0 | 2.44 | 2.04 | — | - | 91.8 |
