# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-06-28T21:58:33  |  Decode tokens/test: 64
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `active` — DiffKV active runtime (MLX int4)

## Prefill time (s)

| Engine | 16k |
|---|---|
| active | 40.31 |

## Decode throughput (tok/s)

| Engine | 16k |
|---|---|
| active | 8.1 |

## Peak memory (GB)

| Engine | 16k |
|---|---|
| active | 3.66 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| active | 16k | ok | 16490 | 64 | 40.31 | 8.1 | 3.66 | 1.62 | 2.03 | - | 52.5 |
