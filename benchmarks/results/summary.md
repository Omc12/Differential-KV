# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-06-22T06:48:32  |  Decode tokens/test: 128
- Per-test timeout: 1500.0s · RAM kill-cap: 7.2 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `native` — DiffKV native (C++, GGUF Q4_K_M)
- `active` — DiffKV active runtime (MLX int4)
- `dense` — Dense (mlx_lm int4, full KV)
- `ollama` — Ollama / llama.cpp (GGUF Q4_K_M)

## Prefill time (s)

| Engine | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| native | 5.86 | 13.80 | 35.24 | **OOM** | **SKIPPED** | **SKIPPED** |
| active | 8.14 | 17.43 | 39.44 | 100.57 | 295.94 | **OOM** |
| dense | 5.19 | 11.46 | 27.87 | 75.75 | **OOM** | **SKIPPED** |
| ollama | 5.01 | 11.21 | 29.52 | 85.54 | 78.75 | 86.69 |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| native | 27.7 | 23.3 | 18.5 | **OOM** | **SKIPPED** | **SKIPPED** |
| active | 44.3 | 40.8 | 36.1 | 28.0 | 16.7 | **OOM** |
| dense | 64.7 | 56.6 | 45.5 | 34.3 | **OOM** | **SKIPPED** |
| ollama | 63.1 | 61.7 | 51.9 | 1000000.0 | 1000000.0 | 1000000.0 |

## Peak memory (GB)

| Engine | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| native | 2.50 | 3.37 | 5.04 | **OOM** | **SKIPPED** | **SKIPPED** |
| active | 2.86 | 2.86 | 3.18 | 4.21 | 6.02 | **OOM** |
| dense | 2.76 | 4.31 | 5.94 | 5.89 | **OOM** | **SKIPPED** |
| ollama | 1.35 | 1.48 | 1.75 | 2.22 | 2.24 | 2.21 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 4k | ok | 4096 | 60 | 5.86 | 27.7 | 2.50 | 2.40 | — | - | 11.2 |
| active | 4k | ok | 4096 | 128 | 8.14 | 44.3 | 2.86 | 2.03 | 1.60 | Y | 15.9 |
| dense | 4k | ok | 4096 | 128 | 5.19 | 64.7 | 2.76 | 1.39 | 1.68 | Y | 11.5 |
| ollama | 4k | ok | 4096 | 128 | 5.01 | 63.1 | 1.35 | 1.33 | — | Y | 8.9 |
| native | 8k | ok | 8192 | 60 | 13.80 | 23.3 | 3.37 | 2.35 | — | - | 19.6 |
| active | 8k | ok | 8192 | 128 | 17.43 | 40.8 | 2.86 | 2.06 | 1.63 | Y | 25.4 |
| dense | 8k | ok | 8192 | 128 | 11.46 | 56.6 | 4.31 | 1.39 | 1.79 | Y | 18.7 |
| ollama | 8k | ok | 8192 | 128 | 11.21 | 61.7 | 1.48 | 1.46 | — | Y | 15.2 |
| native | 16k | ok | 16490 | 128 | 35.24 | 18.5 | 5.04 | 2.93 | — | - | 45.2 |
| active | 16k | ok | 16490 | 128 | 39.44 | 36.1 | 3.18 | 2.35 | 1.87 | Y | 48.1 |
| dense | 16k | ok | 16490 | 128 | 27.87 | 45.5 | 5.94 | 1.40 | 2.03 | Y | 36.9 |
| ollama | 16k | ok | 16490 | 128 | 29.52 | 51.9 | 1.75 | 1.73 | — | Y | 34.0 |
| native | 32k | oom | — | — | — | — | 7.22 | 3.37 | — |  | 68.7 |
| active | 32k | ok | 32865 | 128 | 100.57 | 28.0 | 4.21 | 2.17 | 2.35 | Y | 111.0 |
| dense | 32k | ok | 32865 | 128 | 75.75 | 34.3 | 5.89 | 1.41 | 2.45 | Y | 87.2 |
| ollama | 32k | ok | 32767 | 1 | 85.54 | 1000000.0 | 2.22 | 2.20 | — | - | 87.5 |
| native | 64k | skipped | — | — | — | — | — | — | — |  | — |
| active | 64k | ok | 65615 | 128 | 295.94 | 16.7 | 6.02 | 1.97 | 3.28 | Y | 308.9 |
| dense | 64k | oom | — | — | — | — | 5.89 | 1.41 | — |  | 501.4 |
| ollama | 64k | ok | 32767 | 1 | 78.75 | 1000000.0 | 2.24 | 2.22 | — | - | 81.0 |
| native | 128k | skipped | — | — | — | — | — | — | — |  | — |
| active | 128k | oom | — | — | — | — | 6.11 | 2.03 | — |  | 1295.8 |
| dense | 128k | skipped | — | — | — | — | — | — | — |  | — |
| ollama | 128k | ok | 32767 | 1 | 86.69 | 1000000.0 | 2.21 | 2.20 | — | - | 88.9 |
