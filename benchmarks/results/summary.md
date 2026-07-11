# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-07-11T19:19:17  |  Decode tokens/test: 1
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `native` — DiffKV native (C++, GGUF Q4_K_M)

## Prefill time (s)

| Engine | 32k |
|---|---|
| native | 47.67 |

## Decode throughput (tok/s)

| Engine | 32k |
|---|---|
| native | 1.0 |

## Peak memory (GB)

| Engine | 32k |
|---|---|
| native | 5.92 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 32k | ok | 32865 | 2 | 47.67 | 1.0 | 5.92 | 4.67 | — | - | 56.4 |
