# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-07-12T00:15:52  |  Decode tokens/test: 128
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `active` — DiffKV active runtime (MLX int4)

## Prefill time (s)

| Engine | 32k | 64k |
|---|---|---|
| active | 48.06 | 101.15 |

## Decode throughput (tok/s)

| Engine | 32k | 64k |
|---|---|---|
| active | 29.2 | 25.5 |

## Peak memory (GB)

| Engine | 32k | 64k |
|---|---|---|
| active | 3.56 | 4.04 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| active | 32k | ok | 32865 | 128 | 48.06 | 29.2 | 3.56 | 1.41 | 2.19 | Y | 58.1 |
| active | 64k | ok | 65615 | 128 | 101.15 | 25.5 | 4.04 | 1.41 | 2.71 | Y | 110.9 |
