# Differential KV Cache — ACTIVE_RUNTIME

> **What it is:** a **sparse KV-cache inference runtime** for transformer models. Instead of keeping
> the full dense KV history, DKV compresses each 256-token block to an anchor token (exact) + a
> rank-16 low-rank delta (`U @ Vᵀ`) + the top-64 highest-error tokens kept **exact** (residuals), so
> long-context inference runs in a fraction of the memory. A top-K residual router attends only the
> most relevant blocks per decode step.

---

## Backends (read this first)

There are **two** implementations of the same concepts; which one runs depends on the platform:

| Backend | Runs on | Entry | Notes |
|---|---|---|---|
| **MLX** (this doc's focus) | **Apple Silicon** | `serving/mlx_dkv_wrapper.py` | On macOS, `DKVHFWrapper` is aliased to `MLXDKVWrapper`. Uses `MLXKVBlockManager`. |
| **PyTorch / Triton** | Linux + CUDA | `serving/hf_dkv_wrapper.py` → `native_core/kv_runtime_manager.py` | Full SRL/chunk-graph path; Triton fused decode (`native_core/sparse_decode/triton_fused_decode.py`). |

> ⚠️ **MLX is Apple-only.** There is no fast portable CPU/Windows Python sparse path. For non-Apple,
> use the PyTorch/CUDA backend or the C++ native binary in `dkv_native/`.

---

## Decode paths & the factual store (current behavior)

**Decode routing** — `DKV_COMPRESSED_DECODE`:
- `1` (**default**): DKV **sparse decode from token 1** — the architecture is always exercised.
  Trade-off: at short context this is slower than fused dense (~16 vs ~36 tps @4k) and pre-allocates
  the block pool, with no accuracy change; the memory/reach win is at long context.
- `auto` (**opt-in adaptive**): dense below `DKV_COMPRESSED_MIN_CTX` (16k), sparse above — avoids the
  short-context regression when raw short-prompt throughput matters more than always engaging DKV.
- `0` = always dense (exact full-KV).

**Accuracy:** on realistic prompts, sparse decode recalls facts exactly (single-needle and spread
multi-entity both pass). The one weak spot is **content-dense blocks** (e.g. a table with many
near-identical facts crammed into one 256-token block), where the residual budget overflows and
digits can corrupt — that's what the optional factual store targets.

**Factual store (opt-in, `DKV_FACTUAL_STORE=1`, default OFF):** builds an entity/value index at
the prefill→decode boundary and biases decode toward the queried entity's own fact span (positional
query→value linking over an inverted index). It **helps only dense-fact/table retrieval** (e.g. a
crammed table: bare-value recall 1/5 → 4/5); on ordinary prose plain sparse is already correct, so
the store adds ~32% decode cost for no gain there. Keep it off unless you specifically need
table/multi-entity fact extraction.

---

## Environment flags (quick reference)

| flag | default | meaning |
|---|---|---|
| `DKV_COMPRESSED_DECODE` | **`1`** | sparse-always (default, from token 1); `auto` = adaptive opt-in; `0` = dense |
| `DKV_COMPRESSED_MIN_CTX` | `16384` | auto sparse threshold (tokens) |
| `DKV_MAX_RESIDUAL` | `64` | exact residual tokens kept per block (memory ↔ accuracy) |
| `DKV_TOPK_BLOCKS` | derived: **CUDA** `max(16, 4096 // block_size)` = `16`; **MLX** `max(2, 4096 // block_size)` = `4` at `block_size=1024` | top-K compressed blocks attended per decode step. `K × block_size` is a routed-TOKEN budget (4096). The floors differ **on purpose**: K trades synthesis quality against the decode cache it sizes, and that cache is on for MLX but gated off on CUDA (`DKV_DECODE_CACHE_CUDA=0`). Measured on CUDA, K=4 vs K=16: peak VRAM identical, decode latency CI contains 0, needles 9/9 both, synthesis −16.2 points. So CUDA keeps 16. `0` = attend every block; set `32` for synthesis-shaped work. |
| `DKV_ROUTER` | `residual` | block router: `residual` (tight) or `minmax` (cheap) |
| `DKV_ROUTE_TOPP` | `0` (off) | **EXPERIMENTAL — not a default, and measured not to be worth making one.** Adaptive K: instead of a fixed top-K, take blocks until they cover this share of the softmaxed relevance mass, clamped to `[max(DKV_ROUTE_TOPP_KMIN, k_eff), N]`. `0` disables it and the fixed top-K runs. See the table below before enabling. |
| `DKV_ROUTE_TOPP_KMIN` | `4` | Floor for the above. A rule with no floor selects one block on an over-confident score and nothing downstream recovers the context it skipped. |
| `DKV_ROUTE_SCORE` | `max` | **EXPERIMENTAL.** How a block's keys are aggregated into one relevance score: `max` (upper bound on its best key) or `lse` (log-sum-exp ≈ the attention mass it would receive). `lse` exists because a max cannot be thresholded sensibly — but it is bounded by `max + log(R)`, so it only reorders blocks already within ~log(R) of each other. |
| `DKV_ROUTE_TOPP_STATS` | `0` | Print the K the adaptive rule actually chose (count, mean, median, p90, min, max) at exit. An adaptive rule's cost is an *outcome*, not a setting, so comparing it against a fixed K without this compares two different budgets. |
| `DKV_FACTUAL_STORE` | `0` | enable factual store / entity binding (dense-fact retrieval; opt-in) |
| `DKV_FACTUAL_MAX_OCC` | `4` | max total occurrences for a query token to anchor positional linking |
| `DKV_FACTUAL_WINDOW` | `40` | token window for the nearest fact span to an anchor |
| `DKV_FACTUAL_IDF_MIN` | `3.0` | min block-IDF for a positional anchor (bypassed for ≤2-occ tokens) |
| `DKV_RESIDUAL_EXCLUDE_SVD` | `0` | drop residual positions from the SVD pool (experimental, marginal) |
| `DKV_TELEMETRY` | `0` | print `[SRL]`/`[FACTUAL]` build lines |
| `DKV_FACTUAL_DBG` | `0` | verbose factual-store traces (`[FDBG]`, `[FENTRY]`, per-token) |

Factual-store heuristic constants are tuned against the relational eval and may need adjustment on
other document shapes.

---

## Quick start

**MLX (Apple Silicon), one-shot generation:**
```bash
# from repo root; dkv_venv has mlx installed
dkv_venv/bin/python paper/scripts/measure_active.py \
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
dkv_venv/bin/python benchmarks/relational_ab.py --mode sparse_factual --natural --spread --target 6000 --gen 16
```
`relational_ab.py` reports both exact-key and `n_num_correct` (bare value) so binding accuracy is
separated from generation quality.

---

## Architecture

```
ACTIVE_RUNTIME/
├── serving/
│   ├── mlx_dkv_wrapper.py            ← MLX runtime: patched attention, MLXKVBlockManager,
│   │                                       compressed decode kernel, factual store (opt-in)
│   ├── hf_dkv_wrapper.py             ← PyTorch/CUDA wrapper (aliases to MLX on macOS)
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

- **Decode is dispatch-bound** on MLX (per-layer Python orchestration), not FLOP-bound — WS1 sync
  removal captured the safe +20%; router knobs give only ~% (within noise). Real short-ctx parity with
  dense needs a **fused Metal / batched-layer decode** kernel.
- **Sparse-from-start is now the default** (`DKV_COMPRESSED_DECODE=1`). Known cost at short context:
  slower than fused dense and pre-allocates the full block pool (memory scales to the prompt is a
  deferred follow-up); use `DKV_COMPRESSED_DECODE=auto` if short-prompt throughput matters more.
- **Factual store:** binding works on realistic (4/4) + dense-table (4/5 value) retrieval (opt-in);
  remaining gap is verbatim emission of shared synthetic-code prefixes (a VSL in-order-emission item).

See **[../HANDOFF_ACTIVE_MLX_2026-07-01.md](../HANDOFF_ACTIVE_MLX_2026-07-01.md)** for the full
current-state writeup, benchmark numbers, and prioritized next steps.

---

## Running tests

```bash
dkv_venv/bin/python -m pytest tests/test_dkv_kernel_parity.py -q   # kernel parity gate
dkv_venv/bin/python -m pytest tests/ -v                               # full suite
```
