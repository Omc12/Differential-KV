# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-06-25T17:40:17  |  Decode tokens/test: 128
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `native` — DiffKV native (C++, GGUF Q4_K_M)

## Prefill time (s)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| native | 6.37 | 14.93 | 41.11 |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| native | 7.1 | 3.4 | 2.1 |

## Peak memory (GB)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| native | 2.34 | 3.33 | 5.52 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 4k | ok | 4096 | 128 | 6.37 | 7.1 | 2.34 | 1.68 | — | - | 33.5 |
| native | 8k | ok | 8192 | 15 | 14.93 | 3.4 | 3.33 | 1.92 | — | - | 26.6 |
| native | 16k | ok | 16490 | 69 | 41.11 | 2.1 | 5.52 | 2.64 | — | - | 82.0 |
