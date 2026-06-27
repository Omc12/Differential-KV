# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-06-27T20:02:45  |  Decode tokens/test: 50
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `native` — DiffKV native (C++, GGUF Q4_K_M)

## Prefill time (s)

| Engine | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| native | 5.65 | 13.06 | 33.28 | 93.77 |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| native | 7.2 | 4.2 | 1.6 | 1.2 |

## Peak memory (GB)

| Engine | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| native | 2.26 | 2.26 | 3.05 | 4.96 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 4k | ok | 4096 | 18 | 5.65 | 7.2 | 2.26 | 2.26 | — | - | 11.3 |
| native | 8k | ok | 8192 | 43 | 13.06 | 4.2 | 2.26 | 2.25 | — | Y | 26.1 |
| native | 16k | ok | 16490 | 18 | 33.28 | 1.6 | 3.05 | 2.20 | — | - | 47.9 |
| native | 32k | ok | 32865 | 50 | 93.77 | 1.2 | 4.96 | 2.62 | — | Y | 138.8 |
