# DiffKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)

- Host: Oms-MacBook-Pro-2.local · 8.6 GB RAM · Apple M3
- Started: 2026-07-04T17:41:12  |  Decode tokens/test: 128
- Per-test timeout: 1800.0s · RAM kill-cap: 7.5 GB
- All numbers measured; failed cells show the failure reason (OOM/timeout/crash), never fabricated.
- **Memory** = peak of per-process `max(phys_footprint, RSS)` summed over the engine's process tree, sampled at 20 Hz. Rationale: MLX (active/dense) keeps weights/KV in Metal buffers counted by `phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap their GGUF weights, counted by RSS but not `phys_footprint`. Taking the larger per process avoids undercounting either family. The `(RSS GB)` column is plain resident set, for reference. ollama is measured on its `llama-server` process tree.

**Engines**
- `active` — DiffKV active runtime (MLX int4)
- `dense` — Dense (mlx_lm int4, full KV)

## Prefill time (s)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| active | 6.10 | 12.98 | 30.55 | 79.35 | 598.89 |
| dense | 4.97 | 11.06 | 27.06 | 73.02 | 927.16 |

## Decode throughput (tok/s)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| active | 15.2 | 11.8 | 9.5 | 7.0 | 3.4 |
| dense | 70.3 | 62.6 | 51.2 | 37.4 | 15.6 |

## Peak memory (GB)

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| active | 3.53 | 4.09 | 4.48 | 5.18 | 6.38 |
| dense | 2.63 | 4.09 | 5.94 | 5.89 | 5.89 |

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| active | 4k | ok | 4096 | 128 | 6.10 | 15.2 | 3.53 | 2.11 | 1.81 | Y | 18.9 |
| dense | 4k | ok | 4096 | 128 | 4.97 | 70.3 | 2.63 | 1.39 | 1.68 | Y | 11.2 |
| active | 8k | ok | 8192 | 128 | 12.98 | 11.8 | 4.09 | 2.08 | 1.96 | Y | 28.3 |
| dense | 8k | ok | 8192 | 128 | 11.06 | 62.6 | 4.09 | 1.40 | 1.79 | Y | 17.8 |
| active | 16k | ok | 16490 | 128 | 30.55 | 9.5 | 4.48 | 2.10 | 2.38 | Y | 48.7 |
| dense | 16k | ok | 16490 | 128 | 27.06 | 51.2 | 5.94 | 1.40 | 2.03 | Y | 35.2 |
| active | 32k | ok | 32865 | 128 | 79.35 | 7.0 | 5.18 | 2.40 | 3.09 | Y | 102.5 |
| dense | 32k | ok | 32865 | 128 | 73.02 | 37.4 | 5.89 | 1.40 | 2.45 | Y | 82.0 |
| active | 64k | ok | 65615 | 128 | 598.89 | 3.4 | 6.38 | 2.40 | 4.40 | Y | 642.0 |
| dense | 64k | ok | 65615 | 128 | 927.16 | 15.6 | 5.89 | 1.40 | 3.23 | Y | 941.4 |
