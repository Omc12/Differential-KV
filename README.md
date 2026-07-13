# Differential-KV

A sparse KV-cache inference runtime for long-context LLM inference on constrained
hardware (dev target: M3 Mac, 8 GB unified memory, Qwen2.5-1.5B-Instruct).

**Core idea:** split the KV cache into fixed-size micro-blocks (size $B_s = 256$ tokens). Each block is compressed to an **anchor** token (kept exact) + a joint $K\!\mid\!V$ low-rank **SVD delta** (rank 32, fp16 coefficients $U$ and bases $V_K, V_V$) + up to 128 exact **residual** rows (the tokens the SVD reconstructs worst, selected by joint reconstruction error). Decode attends anchors + deltas + residuals + a dense recency window instead of the full sequence, scoring queries in the low-rank subspace and merging the sparse and dense halves with a flash-style logsumexp combine.

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
| Status | Reference accuracy: NIAH `--bench` 4/4 exact at 4k–32k; reaches 64k (needle recovered exactly; dense baseline OOMs) | Honest NIAH sweep 6/6 (both Q8_0 and Q4_K_M); GQA-routed & decompress-and-cache decode fully verified |

## Build & run (Boosted Performance Paths)

For optimal execution speed, DiffKV provides compiled C++ and hardware-accelerated paths:
* **`ACTIVE_RUNTIME` C++ Extension (`diffkv_core`):** Boosts the Python/PyTorch runtime using Accelerate (macOS) or CUDA (Linux). Silently falls back to pure Python/MLX if not compiled.
* **`diffkv_native` C++ Engine:** A high-performance standalone C++ implementation using a forked llama.cpp/ggml runtime.

See **[BUILD.md](file:///Users/omchimurkar1/Desktop/Differential-KV/BUILD.md)** for detailed build and compilation instructions for both engines (including how to restore the vendored llama.cpp fused-op commits).

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
| `DIFFKV_COMPRESSED_DECODE` | `auto` (engages ≥8k) | MLX sparse decode on/off/auto |
| `DIFFKV_ENGAGE_THRESHOLD` | `~8k` (min of 8192 and the memory-budget cap) | native: context length at which sparse decode engages. ~8k default (matches MLX). Viable because the sparse exact-recency window is now block-aligned — an earlier off-by-a-partial-block seam double-counted boundary tokens and made document reproduction degenerate into loops; with that fixed, a ~12k paper paste reproduces cleanly at the default recency (temp 0 and 0.7) and NIAH stays 6/6. Raise it (e.g. very large) to prefer dense throughput below ~24k, where dense is a touch faster |
| `DIFFKV_HIGH_QUALITY_ROUTING` | `0` (fast bounded-K) | **cross-runtime** (native + MLX + CUDA). Applies **when sparse decode is engaged.** `0`/unset = fast bounded-K pruning: attend the top relevant compressed blocks only — context-independent (~flat tps), NIAH 6/6 + 3/3 multi-fact validated. `1` = High-Quality: attend all blocks (native/MLX) + dynamic 2-hop graph candidate routing |
| `DIFFKV_CACHE_LIMIT_GB` | `1` | MLX buffer-cache cap (halves long-prefill peak RAM) |
| `DIFFKV_TOPK_BLOCKS` | `16` | blocks routed per decode step (both engines) |
| `DIFFKV_MLX_PARITY` | off | native low-level attend-all override (isolated A/B benchmarking; `DIFFKV_HIGH_QUALITY_ROUTING` is the user-facing toggle) |
| `DIFFKV_MAX_RESIDUAL` | `128` | exact residual rows per block (default 128; knob for memory ↔ accuracy) |
| `DIFFKV_SVD_SEED` | `1234` | rSVD determinism — keep set or parity tests flake |
| `DIFFKV_NATIVE_ATTN` | off | native fused ggml attention path (experimental, slower) |
| `DIFFKV_CB_ROUTE_ALL` | on | native decode routes over all resident blocks (fix for the anchor_screen selection bug) |
| `DIFFKV_FUSED_DECODE` | `0` (off) | EXPERIMENTAL Metal decode kernel (MLX). **Broken on the canonical bench as of 2026-07-03: garbage output at 9.8 tps — do not enable** |
| `DIFFKV_CB_GQA_ROUTE` | on | GQA query head-averaging in the native routing loop (engages only when blocks > TOPK; accuracy-neutral in measured cells) |
| `DIFFKV_PROFILE_CB` | `0` | Log layer-wise routing, readback, GPU, and total attention latency |
| `DIFFKV_EARLY_LAYER_RANK_BOOST` | `0` (off) | Enable rank boosting (2x base rank) for early layers (first 15% of the network) to improve syntactic representation |
| `DIFFKV_MAX_RANK_EARLY` | `0` (auto) | Cap for early-layer boosted rank (0 = auto-selects 2x base rank) |

### Algorithmic Rank-Boosting Paths (Accuracy Protection)

To preserve model accuracy under high compression, the Python active runtime supports two dynamic rank-boosting paths:
1. **Early-Layer Rank Boosting:** Boosts SVD rank by up to 2× for the first 15% of layers in the network to safeguard syntax representation. Enable this by setting `DIFFKV_EARLY_LAYER_RANK_BOOST=1`.
2. **Content-Based Rank Boosting:** Automatically detects if a 256-token KV-cache block contains digit patterns, mathematical formula markers (e.g., LaTeX tags like `$$`, `\sum`, `\sqrt`), or key definition patterns (e.g., "is defined as", "refers to"). If detected, the runtime dynamically boosts the SVD rank for that block by 1.5× to protect critical context.

## Measured Evaluation (from the Paper)

All experimental numbers are measured on a single host: **Apple M3 with 8.6 GB of unified memory**, running **Qwen2.5-1.5B-Instruct at int4** via `mlx_lm`. DiffKV runs in its default serving configuration: compressed sparse decode, decompress-and-cache, block-sparse prefill, rank $r=32$, residual budget $R=128$, top-$K=16$, and residual-key router. 

### 1. Main Results (DiffKV vs. Dense baseline)

Comparing both engines on the **exact same quantized weights** creates a clean ablation of the cache representation:

| Context Length | Runtime / Engine | Prefill Time (s) | Decode Speed (tok/s) | Peak Allocator Memory (GB) | Needle Recalled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4k** | **DiffKV (Compressed)** | 6.6s | 19.9 | 1.74 GB | **Yes** |
| | Dense Baseline | 5.1s | 65.7 | 1.68 GB | **Yes** |
| **8k** | **DiffKV (Compressed)** | 13.6s | 18.4 | 1.89 GB | **Yes** |
| | Dense Baseline | 11.8s | 55.3 | 1.79 GB | **Yes** |
| **16k** | **DiffKV (Compressed)** | 28.2s | 18.7 | 2.36 GB | **Yes** |
| | Dense Baseline | 27.8s | 47.0 | 2.03 GB | **Yes** |
| **32k** | **DiffKV (Compressed)** | 58.5s | 17.0 | 3.12 GB | **Yes** |
| | Dense Baseline | 77.9s | 35.7 | 2.45 GB | **Yes** |
| **64k** | **DiffKV (Compressed)** | 928s | 8.6 | 4.63 GB | **Yes** |
| | Dense Baseline | *OOM* | *OOM* | *OOM* | *OOM* |

> [!NOTE]
> **Prefill Crossover:** DiffKV prefill scales sub-quadratically ($O(L \cdot K)$ vs. $O(L^2)$) due to block-sparse prefill. While streaming SVD compression adds a fixed overhead at short context, DiffKV prefill crossovers and beats the dense baseline at 32k ($1.33\times$ faster: 58.5s vs 77.9s).
> 
> **Reach Advantage:** At 64k context, the dense baseline runs out of memory (OOMs) during prefill on the 8.6 GB host. DiffKV completes successfully with the needle recovered exactly, demonstrating its memory-bounded pool advantage.

### 2. Residual-Budget Sweep (at 16K context)

The residual budget $R$ acts as an explicit memory-speed-accuracy dial. When $R$ is reduced, block store sizes fall (increasing the compression ratio), and decode speed rises, while passcode recall is preserved:

| Residuals ($R$) | Needle Recall | Decode Speed (tok/s) | KV Cache Store Size (GB) | Block Compression vs. Dense |
| :---: | :---: | :---: | :---: | :---: |
| **8** | **Yes** | 21.4 | 0.124 GB | $3.80\times$ |
| **16** | **Yes** | 21.1 | 0.139 GB | $3.41\times$ |
| **32** | **Yes** | 20.5 | 0.167 GB | $2.83\times$ |
| **64** | **Yes** | 21.5 | 0.224 GB | $2.11\times$ |
| **128 (Default)** | **Yes** | 19.6 | 0.338 GB | $1.40\times$ |

### 3. Per-Block Storage Budget (256-token block)

Below is the layout of one $B_s = 256$ token block in memory. The low-rank core is fixed, while the residual budget $R$ controls the size of the block:

| Component | Dimensions / Shape | Bytes | Note |
| :--- | :--- | :--- | :--- |
| **$U$ coefficients** | $[255, 16]$ | 16,320 B | Low-rank core |
| **$V_K, V_V$ bases** | $[2, 16, 128]$ | 32,768 B | Low-rank core |
| **Anchors $a_k, a_v$** | $[2, 128]$ | 1,024 B | Exact block reference |
| **Key Min/Max** | $[2, 128]$ | 1,024 B | Decode router bounds |
| **Scalars / Metadata** | scale, seq_len | 8 B | Per-block control |
| **Low-Rank Core Total** | | **51,144 B** | **49.9 KiB (Fixed)** |
| Exact residuals ($R=128$) | $[128, 2, 128]$ | 131,072 B | 128.0 KiB |
| Exact residuals ($R=64$) | $[64, 2, 128]$ | 65,536 B | 64.0 KiB |
| **DiffKV Block ($R=128$)** | | **182,216 B** | **177.9 KiB ($1.44\times$ compression)** |
| **DiffKV Block ($R=64$)** | | **116,680 B** | **113.9 KiB ($2.25\times$ compression)** |
| **Dense Block** | $[256, 2, 128] \times 2$ | **262,144 B** | **256.0 KiB ($1.00\times$)** |

## Optimization & Performance Milestones (July 2026)

> ⚠️ **Audit note (2026-07-03, seventh pass):** the C1 numbers below did NOT reproduce on
> the canonical `niah_recall.py --bench` harness — `DIFFKV_FUSED_DECODE=1` at 4k produced
> garbage output at 9.8 tps (default path: exact recall at 19.4 tps). Treat C1 as historical
> context only; the kernel stays disabled by default (`DIFFKV_FUSED_DECODE=0`).

### 1. Custom Metal Decode Kernel Parallelization (C1) — *claims not reproduced; kernel disabled by default*
- **Design:** Redesigned the threadgroup layout to use exactly 256 threads (matching the 256 block size) and leveraged threadgroup-shared memory to store and project queries, intermediate weights, and outputs.
- **Claimed speedup (unreproduced):** 67.7 TPS at 4k, 55.5 TPS at 16k, 100% recall to 32k — measured only via a private script, not the canonical bench.

### 2. Native Decode attention callback GQA Routing (C2)
- **Design:** Implemented query head-averaging across GQA groups, reducing routing loop iterations 7x (from 28 down to 4 heads).
- **Speedup:** Reduced callback routing latency **8x** (from **$14.3\text{ms}$** down to **$0.35\text{ms}$** per layer) with **100% identical** prediction outputs.

### 3. Native Prefill SVD Draining (C3)
- **Design:** Offloaded SVD calculations to a background thread pool with lowest `QOS_CLASS_UTILITY` settings, ensuring background compression does not block the GPU prompt prefill thread.

## Where things stand / who to read next

- `docs/ANTIGRAVITY_LOG_2026-07.md` — the July 2026 Antigravity execution log.
- Older plan/handoff/session-report docs were superseded and removed from the tree
  (`git log --grep=handoff` / `--grep=D7` for that history); the open native-accuracy
  frontier is the native NIAH sweep gap noted in the table above.
