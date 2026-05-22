# Benchmark Results

> Update this file with real measured results after each benchmark run.

## Methodology

All benchmarks run on single GPU. Baseline = dense HF inference (no DiffKV).

## Format

```
Date:        YYYY-MM-DD
Model:       Qwen2-7B
GPU:         RTX 4090 / A100 / etc.
Context:     25K tokens

| Metric          | Baseline | DiffKV | Delta |
|-----------------|----------|--------|-------|
| Peak VRAM (GB)  |          |        |       |
| Prefill (s)     |          |        |       |
| Decode (tok/s)  |          |        |       |
| Quality (PPL)   |          |        |       |
```
