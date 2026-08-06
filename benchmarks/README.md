# Long-context engine benchmark

Compares four ways of running **Qwen2.5-1.5B-Instruct (4-bit)** across context
lengths **4k → 128k**, measuring **prefill time, decode throughput, and peak
memory**, with OOM-aware early stopping (once an engine fails at a context
length it is skipped at larger ones).

| engine   | what it is                                  | weights        |
|----------|---------------------------------------------|----------------|
| `native` | DKV C++ / ggml binary                    | GGUF Q4_K_M    |
| `active` | DKV ACTIVE_RUNTIME (MLX) — compressed KV | MLX int4       |
| `dense`  | plain `mlx_lm` with a **full** KV cache     | MLX int4       |
| `ollama` | llama.cpp via the ollama server             | GGUF Q4_K_M    |

`dense` shares the exact int4 weights + MLX engine with `active`, so
`active` vs `dense` isolates the DKV compression algorithm.

## Files
- `bench_common.py` — builds the per-length NIAH chat prompt (exact token count).
- `bench_worker.py` — runs ONE (engine, ctx) measurement in isolation; prints JSON.
- `prose_fact_recall.py` — proper-noun prose recall evaluation benchmark (non-digit entity retrieval across 8k–64k).
- `tool_calling_agent_eval.py` — multi-turn tool-calling & jagged agent state recall benchmark (JSON schemas, SQL queries, tool outputs).
- `run_bench.py` — orchestrator: sweep, 20 Hz process-tree memory sampler
  (`max(phys_footprint, RSS)`; ollama also reads `/api/ps size_vram`), per-test
  timeout + RAM kill-cap, OOM/skip logic, incremental JSON + `summary.md`.
- `make_report.py` — post-processes `results/results_latest.json` → `REPORT.md`
  (relabels ollama's `trunc@32k` cells, adds findings + caveats).

## Run
```bash
# full sweep (≈45–90 min on an 8 GB M3; many high-context cells OOM/skip)
./dkv_venv/bin/python3 benchmarks/run_bench.py \
    --contexts 4096 8192 16384 32768 65536 131072 \
    --gen 128 --timeout 1500 --ram-cap-gb 7.2

# regenerate the polished report from the latest results
./dkv_venv/bin/python3 benchmarks/make_report.py
```
Subset example: `--engines active dense --contexts 4096 16384`.

Requires the ollama server running (`qwen2.5:1.5b-instruct` pulled), the native
binary at `dkv_native/build/dkv_native`, the GGUF at
`dkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf`, and the MLX model
`mlx-community/Qwen2.5-1.5B-Instruct-4bit` (auto-downloaded by mlx_lm).

## Outputs (in `results/`)
- `summary.md` — auto table written live during the sweep.
- `REPORT.md` (in `benchmarks/`) — curated report with findings & caveats.
- `results_<timestamp>.json` / `results_latest.json` — full machine-readable data.
- `log_<engine>_<ctx>.txt` — per-cell stdout/stderr for audit.
- `fig_memory.png`, `fig_decode_tps.png`, `fig_prefill.png`, `fig_combined.png` — plots.

```bash
# regenerate plots from existing results
./dkv_venv/bin/python3 benchmarks/plot_graphs.py
```

## Honesty notes
Every number is measured one engine at a time. Failed cells record the reason
(OOM / timeout / skip), never a fabricated value. See the **Caveats** section of
`REPORT.md` — in particular, on this 8 GB box the MLX engines past 32k fail via
swap-thrash (killed) rather than a clean allocator OOM, and `active` 128k fit in
memory (6.1 GB) but was killed for an impractically slow prefill.
