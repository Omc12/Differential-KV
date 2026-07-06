# Differential-KV

A sparse KV-cache inference runtime for long-context LLM inference on constrained
hardware (dev target: M3 Mac, 8 GB unified memory, Qwen2.5-1.5B-Instruct).

**Core idea:** split the KV cache into fixed-size micro-blocks. Each block is compressed
to an **anchor** token (kept exact) + a low-rank **SVD delta** (`U @ V.T`, rank 16, int8 U
with per-row scales) + up to 64 exact **residual** rows (the tokens the SVD reconstructs
worst, selected by boosted joint K/V error + IDF rarity). Decode attends
anchors + deltas + residuals + a dense recency window instead of the full sequence,
merging the sparse and dense halves with a logsumexp combine.

## Quickstart (macOS / Apple Silicon)

```bash
git clone --recurse-submodules <repo-url> && cd Differential-KV
make setup      # create venv + install Python deps
make chat       # interactive DiffKV chat (downloads the model on first run)
```

- `make serve` — run an OpenAI-compatible API at `http://localhost:8000` instead.
- `make test` — run the NIAH recall guardrail (8k + 16k).
- `make` (no target) — list all commands. Pick a model with `make chat MODEL=<hf-id>`.

The best decode config (fast dense for short prompts, DiffKV sparse ≥8k, decompress-and-cache
decode) is applied automatically — see [`ACTIVE_RUNTIME/serving/decode_config.py`](ACTIVE_RUNTIME/serving/decode_config.py).
**Linux/CUDA:** `make setup` installs the base deps; see [`BUILD.md`](BUILD.md) for the CUDA
extras (triton, cuSOLVER/cuBLAS) and the native `-DGGML_CUDA=ON` build (`make native`).

## Two implementations

| | `ACTIVE_RUNTIME/` | `diffkv_native/` |
|---|---|---|
| Language | Python | C++17 |
| Backend (macOS) | MLX (Apple Silicon) | forked llama.cpp/ggml, Metal + CPU |
| Backend (Linux) | PyTorch + Triton (CUDA) | CPU / CUDA |
| Models | HuggingFace / mlx-community | GGUF |
| Status | Reference accuracy: NIAH `--bench` 4/4 exact at 4k–32k | Honest NIAH sweep 3/6; remaining gap is a decode logit-margin issue (see `PLAN_NEW_DIRECTIONS.md` §D7) |

## Build & run

See **`BUILD.md`** for both engines (CPython extension build for ACTIVE_RUNTIME, CMake
for the native engine, including restoring the vendored llama.cpp fused-op commits from
`diffkv_native/third_party/diffkv-fused-op.bundle`).

## Benchmarks / guardrails

Run from the repo root inside `diffkv_venv`:

```bash
# Kernel parity oracle (MLX)
python -m pytest ACTIVE_RUNTIME/tests/test_diffkv_kernel_parity.py -q

# Needle-in-a-haystack recall — ALWAYS use --bench (the hard prompt) for real claims
cd benchmarks && python niah_recall.py --bench --ctx 4096 8192 16384 32768 \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit

# Multi-entity relational binding
cd benchmarks && python relational_ab.py --mode sparse --natural --spread

# Native engine honest sweep (do NOT sanitize the digit filler — it is the test)
cd diffkv_native/tests && ./test_niah_native.sh

# Native kernel byte-parity selftest
DIFFKV_SELFTEST=1 diffkv_native/build/diffkv_native <model.gguf> "x"
```

Memory/perf claims: `paper/scripts/measure_active.py` (MLX) and
`diffkv_native/monitor_memory_native.py` (native RSS).

## Key environment knobs

| Var | Default | Meaning |
|---|---|---|
| `DIFFKV_COMPRESSED_DECODE` | `auto` (engages ≥16k) | MLX sparse decode on/off/auto |
| `DIFFKV_CACHE_LIMIT_GB` | `1` | MLX buffer-cache cap (halves long-prefill peak RAM) |
| `DIFFKV_TOPK_BLOCKS` | `16` | blocks routed per decode step (both engines) |
| `DIFFKV_MAX_RESIDUAL` | `64` | exact residual rows per block (native) |
| `DIFFKV_SVD_SEED` | `1234` | rSVD determinism — keep set or parity tests flake |
| `DIFFKV_NATIVE_ATTN` | off | native fused ggml attention path (experimental, slower) |
| `DIFFKV_CB_ROUTE_ALL` | on | native decode routes over all resident blocks (fix for the anchor_screen selection bug) |
| `DIFFKV_FUSED_DECODE` | `0` (off) | EXPERIMENTAL Metal decode kernel (MLX). **Broken on the canonical bench as of 2026-07-03: garbage output at 9.8 tps — do not enable** (see AUDIT_SEVENTH_PASS_AND_OPPORTUNITIES.md §3.3) |
| `DIFFKV_CB_GQA_ROUTE` | on | GQA query head-averaging in the native routing loop (engages only when blocks > TOPK; accuracy-neutral in measured cells) |
| `DIFFKV_PROFILE_CB` | `0` | Log layer-wise routing, readback, GPU, and total attention latency |

## Optimization & Performance Milestones (July 2026)

> ⚠️ **Audit note (2026-07-03, seventh pass):** the C1 numbers below did NOT reproduce on
> the canonical `niah_recall.py --bench` harness — `DIFFKV_FUSED_DECODE=1` at 4k produced
> garbage output at 9.8 tps (default path: exact recall at 19.4 tps). See
> `AUDIT_SEVENTH_PASS_AND_OPPORTUNITIES.md` §1/§3.3 before trusting this section.

### 1. Custom Metal Decode Kernel Parallelization (C1) — *claims not reproduced; kernel disabled by default*
- **Design:** Redesigned the threadgroup layout to use exactly 256 threads (matching the 256 block size) and leveraged threadgroup-shared memory to store and project queries, intermediate weights, and outputs.
- **Claimed speedup (unreproduced):** 67.7 TPS at 4k, 55.5 TPS at 16k, 100% recall to 32k — measured only via a private script, not the canonical bench.

### 2. Native Decode attention callback GQA Routing (C2)
- **Design:** Implemented query head-averaging across GQA groups, reducing routing loop iterations 7x (from 28 down to 4 heads).
- **Speedup:** Reduced callback routing latency **8x** (from **$14.3\text{ms}$** down to **$0.35\text{ms}$** per layer) with **100% identical** prediction outputs.

### 3. Native Prefill SVD Draining (C3)
- **Design:** Offloaded SVD calculations to a background thread pool with lowest `QOS_CLASS_UTILITY` settings, ensuring background compression does not block the GPU prompt prefill thread.

## Where things stand / who to read next

- `PLAN_FABLE5_OPTIMIZATION.md` — audited optimization plan with status tags.
- `PLAN_NEW_DIRECTIONS.md` — current work plan; §D7 is the open native-accuracy frontier.
- `SESSION_REPORT_FABLE5.md` — measured session logs (what was tried, what's rejected,
  with numbers — check here before re-proposing an idea).
- `docs/ANTIGRAVITY_LOG_2026-07.md` — the July 2026 Antigravity execution log.
