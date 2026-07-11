# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-07-11T19:01:09  |  Decode tokens/test: 128
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `active` — DiffKV active runtime (MLX int4)

## Prefill time (s)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| active | 4.83 | 10.84 | 23.45 |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| active | 33.1 | 31.3 | 31.2 |

## Peak memory (GB)

| Engine | 4k | 8k | 16k |
|---|---|---|---|
| active | 2.90 | 3.13 | 3.27 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| active | 4k | ok | 4096 | 128 | 4.83 | 33.1 | 2.90 | 1.40 | 1.69 | Y | 12.3 |
| active | 8k | ok | 8192 | 128 | 10.84 | 31.3 | 3.13 | 1.41 | 1.76 | Y | 19.6 |
| active | 16k | ok | 16490 | 128 | 23.45 | 31.2 | 3.27 | 1.40 | 1.93 | Y | 32.0 |
