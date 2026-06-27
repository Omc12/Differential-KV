# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-06-27T19:45:07  |  Decode tokens/test: 50
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `native` — DiffKV native (C++, GGUF Q4_K_M)

## Prefill time (s)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| native | 5.88 | 13.65 | 33.66 |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| native | 20.5 | 19.3 | 10.2 |

## Peak memory (GB)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| native | 2.09 | 2.33 | 3.21 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 4k | ok | 4096 | 15 | 5.88 | 20.5 | 2.09 | 1.95 | — | - | 10.5 |
| native | 8k | ok | 8192 | 24 | 13.65 | 19.3 | 2.33 | 2.15 | — | - | 19.8 |
| native | 16k | ok | 16490 | 18 | 33.66 | 10.2 | 3.21 | 2.32 | — | - | 38.8 |
