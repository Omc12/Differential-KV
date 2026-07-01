# Differential KV Cache — ACTIVE_RUNTIME

> **What it is:** a **sparse KV-cache inference runtime** for transformer models. Instead of keeping
> the full dense KV history, DiffKV compresses each 256-token block to an anchor token (exact) + a
> rank-16 low-rank delta (`U @ Vᵀ`) + the top-64 highest-error tokens kept **exact** (residuals), so
> long-context inference runs in a fraction of the memory. A top-K residual router attends only the
> most relevant blocks per decode step.

---

## Backends (read this first)

There are **two** implementations of the same concepts; which one runs depends on the platform:

| Backend | Runs on | Entry | Notes |
|---|---|---|---|
| **MLX** (this doc's focus) | **Apple Silicon** | `serving/mlx_diffkv_wrapper.py` | On macOS, `DiffKVHFWrapper` is aliased to `MLXDiffKVWrapper`. Uses `MLXKVBlockManager`. |
| **PyTorch / Triton** | Linux + CUDA | `serving/hf_diffkv_wrapper.py` → `native_core/kv_runtime_manager.py` | Full SRL/chunk-graph path; Triton fused decode (`native_core/sparse_decode/triton_fused_decode.py`). |

> ⚠️ **MLX is Apple-only.** There is no fast portable CPU/Windows Python sparse path. For non-Apple,
> use the PyTorch/CUDA backend or the C++ native binary in `diffkv_native/`.

---

## Decode paths & the factual store (current behavior)

**Decode routing** — `DIFFKV_COMPRESSED_DECODE`:
- `auto` (default): exact full-KV dense decode **below 16k** tokens, DiffKV sparse decode **at/above
  16k** (`DIFFKV_COMPRESSED_MIN_CTX`). Below the threshold the DiffKV kernel does not engage.
- `1` = always sparse · `0` = always dense.

**Accuracy:** on realistic prompts, sparse decode recalls facts exactly (single-needle and spread
multi-entity both pass). The one weak spot is **content-dense blocks** (e.g. a table with many
near-identical facts crammed into one 256-token block), where the residual budget overflows and
digits can corrupt — that's what the optional factual store targets.

**Factual store (opt-in, `DIFFKV_FACTUAL_STORE=1`, default OFF):** builds an entity/value index at
the prefill→decode boundary and biases decode toward the queried entity's own fact span (positional
query→value linking over an inverted index). It **helps only dense-fact/table retrieval** (e.g. a
crammed table: bare-value recall 1/5 → 4/5); on ordinary prose plain sparse is already correct, so
the store adds ~32% decode cost for no gain there. Keep it off unless you specifically need
table/multi-entity fact extraction.

---

## Environment flags (quick reference)

| flag | default | meaning |
|---|---|---|
| `DIFFKV_COMPRESSED_DECODE` | `auto` | `1`/`0`/`auto` — force sparse / force dense / threshold |
| `DIFFKV_COMPRESSED_MIN_CTX` | `16384` | auto sparse threshold (tokens) |
| `DIFFKV_MAX_RESIDUAL` | `64` | exact residual tokens kept per block (memory ↔ accuracy) |
| `DIFFKV_TOPK_BLOCKS` | `16` | top-K compressed blocks attended per decode step |
| `DIFFKV_ROUTER` | `residual` | block router: `residual` (tight) or `minmax` (cheap) |
| `DIFFKV_FACTUAL_STORE` | `0` | enable factual store / entity binding (dense-fact retrieval; opt-in) |
| `DIFFKV_FACTUAL_MAX_OCC` | `4` | max total occurrences for a query token to anchor positional linking |
| `DIFFKV_FACTUAL_WINDOW` | `40` | token window for the nearest fact span to an anchor |
| `DIFFKV_FACTUAL_IDF_MIN` | `3.0` | min block-IDF for a positional anchor (bypassed for ≤2-occ tokens) |
| `DIFFKV_RESIDUAL_EXCLUDE_SVD` | `0` | drop residual positions from the SVD pool (experimental, marginal) |
| `DIFFKV_TELEMETRY` | `0` | print `[SRL]`/`[FACTUAL]` build lines |
| `DIFFKV_FACTUAL_DBG` | `0` | verbose factual-store traces (`[FDBG]`, `[FENTRY]`, per-token) |

Factual-store heuristic constants are tuned against the relational eval and may need adjustment on
other document shapes.

---

## Quick start

**MLX (Apple Silicon), one-shot generation:**
```bash
# from repo root; diffkv_venv has mlx installed
diffkv_venv/bin/python paper/scripts/measure_active.py \
    --single compressed,16384 --gen 40 --ctx 16384 --out /tmp/x.json
```

**Serve (PyTorch backend) with Open WebUI:**
```bash
python -m serving.openai_compatible_api_gateway \
    --model Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000 --serving-mode balanced
# Open WebUI → Settings → Connections → OpenAI API:  URL http://localhost:8000/v1  Key: none
```
See **[docs/open_webui_integration.md](docs/open_webui_integration.md)** for the full guide.

---

## Benchmarks & evals

| script | what it measures |
|---|---|
| `paper/scripts/measure_active.py` | live decode tps / prefill / peak mem / KV composition (MLX) |
| `benchmarks/niah_recall.py --bench` | single-needle recall under compression |
| `benchmarks/relational_ab.py` | **multi-entity binding** A/B (`exact`/`sparse`/`sparse_factual`; `--natural`, `--spread`) |

```bash
# multi-entity fact binding, factual store on, realistic layout:
diffkv_venv/bin/python benchmarks/relational_ab.py --mode sparse_factual --natural --spread --target 6000 --gen 16
```
`relational_ab.py` reports both exact-key and `n_num_correct` (bare value) so binding accuracy is
separated from generation quality.

---

## Architecture

```
ACTIVE_RUNTIME/
├── serving/
│   ├── mlx_diffkv_wrapper.py            ← MLX runtime: patched attention, MLXKVBlockManager,
│   │                                       compressed decode kernel, factual store (opt-in)
│   ├── hf_diffkv_wrapper.py             ← PyTorch/CUDA wrapper (aliases to MLX on macOS)
│   ├── openai_compatible_api_gateway.py ← FastAPI OpenAI-compatible server
│   └── batch_engine.py                  ← continuous-batching decode loop
├── native_core/
│   ├── kv_runtime_manager.py           ← PyTorch/CUDA session/block/compression orchestrator
│   ├── compression/lowrank.py          ← SVD low-rank delta compression
│   ├── srl/
│   │   ├── factual_store.py            ← entity/value spans (boundary-aware segmentation)
│   │   ├── inverted_index.py           ← token → position index (important_vocab + IDF)
│   │   ├── session_srl_state.py        ← per-session SRL state
│   │   └── factual_alignment.py        ← VSL logit masking / bias helpers
│   └── sparse_decode/triton_fused_decode.py  ← Triton fused decode (CUDA)
└── docs/open_webui_integration.md
```

---

## Status & known gaps

- **Decode is overhead/dispatch-bound** on MLX (per-layer Python orchestration), not FLOP-bound —
  micro-opts give %, real parity with dense needs a fused Metal / batched-layer decode.
- **Sparse-from-start not yet default:** below ~24k, sparse is slower and uses *more* (pre-allocated)
  memory than dense, so `auto` waits. Flipping needs the decode-speed work first.
- **Factual store:** works on realistic + dense-table retrieval (opt-in); remaining gap is verbatim
  emission of shared synthetic-code prefixes (a VSL in-order-emission tuning item).

See **[../HANDOFF_ACTIVE_MLX_2026-07-01.md](../HANDOFF_ACTIVE_MLX_2026-07-01.md)** for the full
current-state writeup, benchmark numbers, and prioritized next steps.

---

## Running tests

```bash
diffkv_venv/bin/python -m pytest tests/test_diffkv_kernel_parity.py -q   # kernel parity gate
diffkv_venv/bin/python -m pytest tests/ -v                               # full suite
```
