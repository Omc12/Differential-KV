# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-06-30T16:30:48  |  Decode tokens/test: 64
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `native` — DiffKV native (C++, GGUF Q4_K_M)

## Prefill time (s)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| native | 5.77 | 13.01 | 33.40 | 99.65 | **OOM** |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| native | 30.2 | 23.2 | 14.3 | 6.6 | **OOM** |

## Peak memory (GB)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| native | 2.25 | 2.55 | 3.65 | 5.42 | **OOM** |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 4k | ok | 4096 | 64 | 5.77 | 30.2 | 2.25 | 2.25 | — | Y | 10.9 |
| native | 8k | ok | 8192 | 64 | 13.01 | 23.2 | 2.55 | 2.48 | — | Y | 18.5 |
| native | 16k | ok | 16490 | 64 | 33.40 | 14.3 | 3.65 | 2.99 | — | Y | 40.7 |
| native | 32k | ok | 32865 | 64 | 99.65 | 6.6 | 5.42 | 3.36 | — | Y | 118.8 |
| native | 64k | oom | — | — | — | — | 7.61 | 3.78 | — |  | 369.7 |
