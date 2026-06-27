# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-06-27T19:23:54  |  Decode tokens/test: 50
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `native` — DiffKV native (C++, GGUF Q4_K_M)

## Prefill time (s)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| native | 5.82 | 13.23 | 33.36 | 98.73 | **OOM** |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| native | 20.8 | 19.6 | 11.2 | 6.3 | **OOM** |

## Peak memory (GB)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| native | 2.29 | 2.35 | 3.23 | 4.33 | **OOM** |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 4k | ok | 4096 | 15 | 5.82 | 20.8 | 2.29 | 2.29 | — | - | 9.7 |
| native | 8k | ok | 8192 | 24 | 13.23 | 19.6 | 2.35 | 2.24 | — | - | 19.2 |
| native | 16k | ok | 16490 | 18 | 33.36 | 11.2 | 3.23 | 2.45 | — | - | 37.9 |
| native | 32k | ok | 32865 | 27 | 98.73 | 6.3 | 4.33 | 2.74 | — | - | 107.2 |
| native | 64k | oom | — | — | — | — | 7.65 | 3.41 | — |  | 350.4 |
