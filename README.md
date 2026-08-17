# Differential-KV (DKV): Anchor + Low-Rank KV-Cache Compression

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21539110.svg)](https://doi.org/10.5281/zenodo.21539110)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey.svg)](BUILD.md)
[![Backend](https://img.shields.io/badge/Backend-MLX%20%7C%20PyTorch%20%7C%20CUDA%20%7C%20C%2B%2B17-blueviolet.svg)](ACTIVE_RUNTIME/README.md)

**Differential-KV (DKV)** is a sparse KV-cache inference runtime designed for high-efficiency, memory-bounded long-context Large Language Model (LLM) inference across Apple Silicon (MLX) and CUDA GPUs.

ResearchGate: https://www.researchgate.net/publication/410864213

Author: **Om Chimurkar** (Newton School of Technology, Rishihood University)  
Paper: https://doi.org/10.5281/zenodo.21539110

***Try a demo here***: https://huggingface.co/spaces/Om1232/Differential-KV

---

## 📌 Core Architecture & Paper Overview

The Key-Value (KV) cache is the primary memory bottleneck in long-context LLM inference: its memory footprint scales linearly $O(L)$ with sequence length $L$, causing commodity hardware to run out of memory (OOM) during prefill or generation long before model weights exhaust VRAM.

Differential-KV addresses this by partitioning the KV cache into fixed micro-blocks of size $B_s = 256$ tokens and decomposing each block into four complementary components:

1. **Anchor Token ($a_k, a_v$):** The first token in the block, preserved in exact precision to serve as an anchor reference.
2. **Joint $K \mid V$ Low-Rank SVD Delta:** A truncated Singular Value Decomposition (SVD) of rank $r = 32$ (with layer-adaptive variation: early layers $0.75r$, mid layers $1.5r$, late layers $0.5r$) capturing shared structural variation across key and value projections ($U \in \mathbb{R}^{(B_s-1) \times r}$, $V_K, V_V \in \mathbb{R}^{r \times d_{\text{head}}}$).
3. **Exact Residual Tokens ($R = 128$ default budget):** Tokens with high reconstruction error or key semantic structures (digits, mathematical formulas, entity names, relational connectives) kept uncompressed.
4. **Bounded Dense Recency Window:** A sliding window holding the most recent sequence tokens uncompressed.

### Decode & Prefill Innovations
* **Low-Rank Space Query Scoring:** Scores queries directly in the low-rank subspace ($O(r \cdot d_{\text{head}})$ dot products per block) without ever decompressing or materializing the full Key matrix $K$.
* **Flash-Style LSE Combine:** Merges sparse low-rank/residual attention scores with the dense recency window using numerically stable fp32 log-sum-exp (LSE) accumulation and fp16 operands.
* **Sub-Quadratic Prefill:** Uses block-sparse attention during prefill, reducing prompt processing complexity from $O(L^2)$ down to $O(L \cdot K)$, enabling prefill crossover where Differential-KV prefilling outperforms dense baselines at long contexts ($\ge 32\text{k}$).

---

## 🚀 Quickstart

### macOS / Apple Silicon (MLX)
```bash
# 1. Clone repository with submodules
git clone --recurse-submodules https://github.com/Omc12/Differential-KV.git
cd Differential-KV

# 2. Setup virtual environment and dependencies
make setup

# 3. Launch interactive DKV Chat CLI (downloads default model Qwen/Qwen2.5-1.5B-Instruct on first run)
make chat

# Or specify a custom model:
make chat MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

### Linux / CUDA
```bash
# 1. Setup python virtual environment
make setup

# 2. Build CUDA native extensions or Triton kernel paths
# See BUILD.md for detailed instructions on cuSOLVER/cuBLAS & Triton requirements
```

### Useful Commands (`Makefile`)
| Command | Description |
|---|---|
| `make setup` | Creates `dkv_venv` Python virtualenv & installs required packages |
| `make chat` | Starts interactive terminal CLI in Direct Mode |
| `make serve` | Launches OpenAI-compatible REST API gateway on `http://localhost:8000` |
| `make test` | Runs needle-in-a-haystack (NIAH) recall guardrail tests at 8k & 16k context |
| `make native` | Compiles high-performance C++ engine (`dkv_native`) with Metal/CUDA support |

---

## 🖥️ CLI Commands & Operating Modes

The Differential-KV terminal interface (`ACTIVE_RUNTIME/serving/cli.py`) supports two primary operational modes: **Direct Execution Mode** and **Client-Server Mode**.

### 1. Operational Modes

#### Direct Mode (Local Inference)
Runs model weights directly in-process via MLX or PyTorch:
```bash
python ACTIVE_RUNTIME/serving/cli.py --model Qwen/Qwen2.5-1.5B-Instruct --preset mid --serving-mode balanced
```

#### Client-Server Mode (Remote / API Gateway)
Connects the CLI UI to a running `make serve` API server instance:
```bash
# Terminal 1: Start API server
make serve MODEL=Qwen/Qwen2.5-1.5B-Instruct

# Terminal 2: Connect CLI client
python ACTIVE_RUNTIME/serving/cli.py --api-url http://localhost:8000/v1
```

### 2. CLI Options & Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID or local directory path |
| `--api-url` | `str` | `None` | API Gateway base URL. When provided, runs CLI in Client Mode |
| `--serving-mode` | `choice` | `balanced` | KV Cache strategy: `lightweight`, `balanced`, `performance`, `long-context`, `fused-sparse` |
| `--preset` | `choice` | `mid` | Fidelity preset: `low`, `mid`, `high`, `ultra` — see the ladder below |
| `--fastdc` | `flag` | `False` | CUDA only. Prioritises decode over prefill (CUDA-graph decode path). Opt-in: worth ~1.2% at 16k and nothing at 32k |
| `--rank` | `int` | `32` | SVD rank for KV compression (capped at $d_{\text{head}}$) |
| `--micro-block-size` | `int` | `256` | Number of tokens per compressed KV micro-block ($B_s = 256$) |
| `--batch-size` | `int` | `4` | Maximum continuous batching size for engine |
| `--load-in-4bit` | `flag` | `False` | Load model weights using 4-bit NF4 quantization |
| `--load-in-8bit` | `flag` | `False` | Load model weights using 8-bit quantization |
| `--max-tokens` | `int` | `16384` | Maximum tokens to generate per response |
| `--temperature` | `float` | `0.7` | Sampling temperature |
| `--top-p` | `float` | `0.9` | Top-p nucleus sampling probability |
| `--repetition-penalty` | `float` | `1.15` | Repetition penalty factor |
| `--draft-model` | `str` | `None` | Optional draft model ID for speculative decoding |
| `--max-resident-sessions` | `int` | `4` | Maximum active resident chat sessions held in VRAM |

### 3. The Preset Ladder

| preset | keys stored | energy / rank | residual $R$ | pick it when |
|---|---|---|---|---|
| `low` | rotated | 0.999 / 32 | 40 | memory is the binding constraint |
| **`mid`** (default) | rotated | 0.9999 / 64 | 40 | general use — the fast default |
| `high` | rotated | 0.99999 / 128 | 128 | you want the largest fidelity budget |
| **`ultra`** | **unrotated** | 0.9999 / 64 | 40 | **digits, codes, tables, or confusable content** |

**`ultra` is exactly `mid` plus one change: keys are stored un-rotated.** Same rank,
same energy, same residual budget — so it costs the rotation and nothing else.

Two things worth knowing before choosing:

* **For digit- or table-heavy work you want `ultra`, not `high`.** `high` stores
  rotated keys like `mid` does and does not fix that class of error; its larger
  rank and residual budget are not what is failing.
* **`ultra` is not free, and its cost depends on your model.** Rotating at read
  is paid on every *attended* layer, so a hybrid model (Qwen3.5-2B: 6 of 24
  attended) pays ~11% of decode while a dense-attention model (Qwen2.5-1.5B: 28
  of 28) pays ~27%. Check your model's attended-layer count before budgeting.
  `DKV_ROTATED_POOL=0` takes the same trade on any preset.

---

## ⚙️ Key Environment Knobs

Differential-KV runtime behaviors can be fine-tuned using environment variables:

| Variable | Default | Scope | Description |
|---|---|---|---|
| `DKV_COMPRESSED_DECODE` | `auto` | MLX | Controls sparse decode (`on`, `off`, `auto`). Auto engages sparse decode at sequence lengths $\ge 8\text{k}$. |
| `DKV_ENGAGE_THRESHOLD` | Budget-gated | Native / MLX | Context length at which sparse decode engages. Keeps dense decode for short contexts where dense fits. |
| `DKV_HIGH_QUALITY_ROUTING` | `0` | Cross-runtime | `0` = Fast bounded-K pruning (attends top-$K$ blocks, context-independent speed). `1` = High-Quality routing (dynamic candidate routing). |
| `DKV_CACHE_LIMIT_GB` | `1` | MLX | Buffer-cache allocation cap in GB (halves peak prefill RAM). |
| `DKV_TOPK_BLOCKS` | `16` | Both engines | Number of compressed micro-blocks routed per decode step. |
| `DKV_MAX_RESIDUAL` | `128` MLX / `40` CUDA | Both engines | Exact residual token rows per block. CUDA `mid`/`ultra`/`low` use 40, `high` keeps 128; MLX is flat 128. The budget measured **inert** on four benchmarks including a digit-table one, and the pool allocates these slots on *every* block, so 40 is a straight saving. |
| `DKV_ROTATED_POOL` | `1` (`0` on `ultra`) | Both engines | `1` stores POST-RoPE keys (fast reads). `0` stores PRE-RoPE and rotates at read: **exact dense parity on retrieval and digit recall**, at 43–137% of decode depending on attended-layer count. |
| `DKV_DETERMINISTIC` | `0` | CUDA | `1` forces a deterministic SDPA backend. Greedy decode is **not** reproducible at long context without it — the cause is the decode attention's reduction, not compression. **Turn this on for any run you intend to compare.** |
| `DKV_FAST_DECODE` | `0` | CUDA | Same as `--fastdc`. Routed CUDA-graph decode, gated per session to whether a graph will actually be captured. |
| `DKV_SVD_SEED` | `1234` | MLX | SVD random state seed for deterministic compression. |
| `DKV_EARLY_LAYER_RANK_BOOST` | `0` | MLX | Set `1` to boost SVD rank ($2\times$) in early layers ($\le 15\%$ network depth) for syntactic protection. |
| `DKV_MAX_RANK_EARLY` | `0` | MLX | Cap for early layer boosted rank ($0$ = auto-selects $2\times$ base rank). |
| `DKV_PROFILE_CB` | `0` | Both engines | Enable layer-wise routing, GPU kernel, and readback latency profiling logs. |
| `DKV_CB_GQA_ROUTE` | `on` | Native | Grouped-Query Attention (GQA) head-averaging in native routing loop ($8\times$ latency reduction). |
| `DKV_CB_ROUTE_ALL` | `on` | Native | Forces routing across all resident blocks to prevent candidate screening drops. |
| `DKV_FUSED_DECODE` | `0` | MLX | Experimental Metal decode kernel ($0$ = disabled; keep disabled for benchmark accuracy). |

---

## 🏗️ Dual-Engine Architecture

| Feature | `ACTIVE_RUNTIME/` (Python Overlay) | `dkv_native/` (C++ Engine) |
|---|---|---|
| **Language** | Python 3.10+ (with optional C++ `dkv_core` extension) | Pure C++17 |
| **Backends** | MLX (Apple Silicon) / PyTorch + Triton (CUDA) | forked `llama.cpp` / `ggml` (Metal & CUDA) |
| **Model Format** | HuggingFace Transformers / `mlx-community` | GGUF |
| **Primary Target** | Research, rapid iteration, serving gateway | Production edge deployment, minimal host overhead |
| **Status** | Reference accuracy (4k–64k NIAH 100% exact recall) | Experimental / Work-In-Progress |

---

## 📊 Measured Paper Benchmarks

All benchmark results are empirically measured on a single host: **Apple M3 with 8.6 GB unified memory**, evaluating **Qwen2.5-1.5B-Instruct (int4)** using rank $r=32$, residual budget $R=128$, micro-block size $B_s=256$, top-$K=16$, and residual-key router (raw logs preserved in `benchmarks/results/results_latest.json`).

### 0. CUDA Runtime — Preset Accuracy and the Cost of `ultra`

Separate host and runtime from the MLX numbers below: **RTX-class CUDA GPU,
Qwen3.5-2B, 32k context**, `transformers` 5.14.1 / `torch` 2.11.0+cu130. **A dense
control was run in the same process generation as every DKV number** — absolute
scores here are not comparable to scores taken in another environment, and this
project has been burned by exactly that (see the note under the accuracy table).

**Accuracy — `ultra` reaches dense exactly on both metrics that can resolve anything**

| benchmark | dense | `ultra` | `mid` |
|---|---|---|---|
| linkbench (multi-hop retrieval), 48 seeds | 23/48 | **23/48** | 21/48 |
| tablebench (exact 4-digit recall), 24 seeds | 24/24 | **24/24** | 14/24 |
| needle sweep, 9 depths × 2 models | — | 9/9 | 9/9 |

> **Do not compare these absolutes against older numbers in this repo.** An
> environment change (most likely a `transformers`/`torch` update) roughly halved
> *every* linkbench arm — **including dense, which shares no DKV code**. The
> *relationships* are what hold, and they reproduce: `ultra == dense`, `mid` below
> both. Always run a dense arm alongside, in the same environment.

**What closes the gap — and what does not**

The whole `mid`→`ultra` difference is the un-rotated pool. Everything else was
measured and is inert on `tablebench` (all 14/24, unchanged from `mid`):

| lever tried | result |
|---|---|
| residual budget $R$ = 40 vs 128 | no change (also no change on 3 prose benchmarks) |
| attend **every** block (`DKV_TOPK_BLOCKS=0`) | no change → not a routing problem |
| tier residuals on error tail, not median | no change |
| **un-rotated pool (`ultra`)** | **14/24 → 24/24, i.e. dense parity** |

**Speed — what `ultra` costs**, paired and in-process, 8 rounds, 32k:

| model | attended layers | `mid` | `ultra` | cost |
|---|---|---|---|---|
| Qwen3.5-2B | 6 of 24 | 38.63 ms/tok | 43.02 ms/tok | **+11.5%** |
| Qwen2.5-1.5B | 28 of 28 | 51.28 ms/tok | 65.10 ms/tok | **+26.5%** |

The cost is the rotation applied at read on every attended layer, so it scales
with attended-layer count — **a figure for one model tells you nothing about
another.** That is why `mid` stays rotated by default and `ultra` is opt-in.

### 1. Context Length Sweep (DKV vs. Dense Baselines)

| Context Length | Runtime / Engine | Prefill Time (s) | Decode Speed (tok/s) | Peak Allocator Memory (GB) | Needle Recalled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4k** | **DKV Active (Compressed)** | 6.16s | 33.3 | 1.55 GB | **Yes** |
| | MLX Dense Baseline | 5.00s | 68.9 | 1.68 GB | **Yes** |
| | PyTorch Dense Baseline | 0.83s | 3.6 | 8.05 GB | **Yes** |
| **8k** | **DKV Active (Compressed)** | 13.37s | 31.4 | 1.67 GB | **Yes** |
| | MLX Dense Baseline | 11.08s | 54.5 | 1.79 GB | **Yes** |
| | PyTorch Dense Baseline | 0.48s | 3.5 | 8.05 GB | **Yes** |
| **16k** | **DKV Active (Compressed)** | 29.77s | 29.7 | 2.20 GB | **Yes** |
| | MLX Dense Baseline | 27.04s | 48.2 | 2.03 GB | **Yes** |
| | PyTorch Dense Baseline | *OOM* | *OOM* | *OOM* (8.58 GB) | *OOM* |
| **32k** | **DKV Active (Compressed)** | 72.33s | 27.2 | 3.12 GB | **Yes** |
| | MLX Dense Baseline | 75.00s | 37.5 | 2.45 GB | **Yes** |
| | PyTorch Dense Baseline | *OOM* | *OOM* | *OOM* (8.58 GB) | *OOM* |
| **64k** | **DKV Active (Compressed)** | **477.12s** | **21.4** | 5.04 GB | **Yes** |
| | MLX Dense Baseline | **820.98s** | **20.2** | 3.23 GB | **Yes** |
| | PyTorch Dense Baseline | *OOM* | *OOM* | *OOM* (8.58 GB) | *OOM* |

> 💡 **Key Benchmark Takeaways:**
> - **Prefill Crossover at 64k:** At extreme context ($64\text{k}$), DKV prefilling beats MLX Dense prefilling by **$1.72\times$** (477.1s vs 821.0s), demonstrating sub-quadratic scaling ($O(L \cdot K)$).
> - **Baseline Differentiation:** Standard PyTorch Dense (`AutoModelForCausalLM`) suffers an Out-Of-Memory (OOM) failure at $\ge 16\text{k}$ context on an 8.6 GB memory host, whereas MLX unified memory optimization enables MLX Dense to complete 64k.
> - **Needle Recall & Heuristic-Protection Transparency:**
>   - **Pattern-Protected NIAH Needle Recall (Digits/Codes):** 100% exact recall (protected by `streaming_sparse_ingest.py` content-aware regex safeguards).
>   - **Unassisted Prose Entity Recall (`benchmarks/prose_fact_recall.py`):** ~60% recall under default unassisted SVD compression. Toggle regex rules via `DKV_DISABLE_REGEX_HEURISTICS=1`.

### 2. Residual Budget ($R$) Trade-Off Sweep (16k Context)

The residual budget $R$ acts as an explicit memory-speed-accuracy dial:

| Residuals ($R$) | Passcode Recall | Decode Speed (tok/s) | KV Cache Store Size (GB) | Compression Ratio vs. Dense |
| :---: | :---: | :---: | :---: | :---: |
| **8** | **Yes** | 21.4 | 0.124 GB | $3.80\times$ |
| **16** | **Yes** | 21.1 | 0.139 GB | $3.41\times$ |
| **32** | **Yes** | 20.5 | 0.167 GB | $2.83\times$ |
| **64** | **Yes** | 21.5 | 0.224 GB | $2.11\times$ |
| **128 (Default)** | **Yes** | 19.6 | 0.338 GB | $1.40\times$ |

### 3. Per-Block Memory Breakdown ($B_s = 256$ Tokens)

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
| **DKV Block ($R=128$)** | | **182,216 B** | **177.9 KiB ($1.44\times$ compression)** |
| **DKV Block ($R=64$)** | | **116,680 B** | **113.9 KiB ($2.25\times$ compression)** |
| **Dense Block** | $[256, 2, 128] \times 2$ | **262,144 B** | **256.0 KiB ($1.00\times$)** |

### 4. THUDM LongBench & NVIDIA RULER Benchmark Suites (32k Context)

Evaluated on **Qwen2.5-1.5B-Instruct (int4)** under up to $32,768$ context length using DKV active runtime ($r=32$, $R=128$, $B_s=256$):

#### NVIDIA RULER Benchmark Suite
| Task Category | Task | Evaluated Metric | DKV Score |
| :--- | :--- | :--- | :---: |
| **Multi-Document QA** | `qa_2` | Exact Match / Substring | **100.0%** |
| **Single-Needle Recall** | `niah_single` | Needle Retrieval | **100.0%** |
| **Multi-Key Recall** | `niah_multikey` | Key-Value Retrieval | **100.0%** |
| **Multi-Value Recall** | `niah_multivalue` | Value Association | **100.0%** |
| **Variable Tracking** | `variable_tracking` | Chain-of-Thought Variable | **100.0%** |
| **Common Words Extraction** | `cwe` | Frequency Extraction | **100.0%** |

#### THUDM LongBench Comprehensive Suite
| Category | Task | Metric | DKV Score |
| :--- | :--- | :--- | :---: |
| **Single-Doc QA** | `qasper` | F1 Score | **87.90%** |
| | `multifieldqa-en` | F1 Score | **62.61%** |
| | `narrativeqa` | F1 Score | **24.56%** |
| **Multi-Doc QA** | `2wikimultihopqa` | F1 Score | **9.68%** |
| | `hotpotqa` | F1 Score | **5.11%** |
| | `musique` | F1 Score | **4.97%** |
| **Summarization** | `multi_news` | ROUGE-L | **26.48** |
| | `gov_report` | ROUGE-L | **20.38** |
| | `qmsum` | ROUGE-L | **7.79** |
| **Code Completion** | `repobench-p` | Code Sim | **30.00%** |
| | `lcc` | Code Sim | **24.33%** |
| **Synthetic & Few-Shot** | `passage_count` | Accuracy | **65.05%** |
| | `samsum` | ROUGE-L | **58.99** |
| | `passage_retrieval-en` | Rank Accuracy | **1.58%** |
| | `trec` | Accuracy | **2.56%** |
| | `triviaqa` | F1 Score | **0.13%** |

> 💡 **Routing & Pruning Insight (`DKV_TOPK_BLOCKS=16`):**
> - **High-Efficiency Pruning Trade-off:** The default benchmark setting (`DKV_TOPK_BLOCKS=16`) routes only 16 micro-blocks ($16 \times 256 = 4,096$ tokens) per decode step out of the full 32,768 ($32\text{k}$) context window, pruning **87.5%** of sequence blocks to achieve $>30$ tok/s decode speed on unified memory.
> - **Multi-Hop Retrieval Scaling:** For complex multi-document reasoning tasks (such as `2wikimultihopqa`, `hotpotqa`, and `musique`) where evidence is scattered across multiple distant passages, increasing the candidate budget to `DKV_TOPK_BLOCKS=32` or `64` ($8\text{k}$–$16\text{k}$ attended tokens) expands evidence gathering and further boosts multi-hop recall.

---

## ⚡ Advanced Systems Features & Safeguards

### 1. Algorithmic Rank-Boosting
- **Early-Layer Boosting:** Boosts SVD rank by up to $2\times$ in the first 15% of network layers to safeguard syntactic representations (`DKV_EARLY_LAYER_RANK_BOOST=1`).
- **Content-Aware Dynamic Boosting:** Automatically detects micro-blocks containing numerical data, mathematical formulas (e.g., LaTeX `$$`, `\sum`), or formal definition patterns, boosting block SVD rank by $1.5\times$ on-the-fly.

### 2. Multi-Signal Residual Selection
To ensure critical factual information is retained, residuals are selected using three combined IDF-weighted priority signals:
- **Owner-Capture:** Entity names accompanying high reconstruction error tokens.
- **Edge-Capture:** Relational connectives with potential low-rank key collision.
- **Coverage Bonus:** Enforces uniform spread across block token positions to prevent localized error clustering.

### 3. Tiered Offloading & Asynchronous Prefetch (kTransformers-Inspired)
- **Tiered CPU-GPU KV Offloading:** Maintains a heat score for each micro-block, evicting cold blocks to pinned host RAM when GPU pool utilization exceeds 80%.
- **Step-Ahead Async Prefetch:** Background prefetching retrieves cold blocks into GPU memory before the subsequent decode step touch point, hiding PCIe transfer latency.

---

## 🧪 Testing & Verification

Execute test suites from repository root using `dkv_venv`:

```bash
# 1. Kernel Parity Oracle Test (MLX)
pytest ACTIVE_RUNTIME/tests/test_dkv_kernel_parity.py -q

# 2. Needle-in-a-Haystack Recall Benchmark (Canonical)
cd benchmarks && python niah_recall.py --bench --ctx 4096 8192 16384 32768 \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit

# 3. Multi-Entity Relational Binding Test
cd benchmarks && python relational_ab.py --mode sparse --natural --spread

# 4. Native C++ Engine Honest Sweep
cd dkv_native/tests && ./test_niah_native.sh
```

---

## 📖 Citation & References

If you use Differential-KV in your research or project, please cite:

```bibtex
@misc{chimurkar_2026_21539110,
  author       = {Chimurkar, Om},
  title        = {DKV: Anchor + Low-Rank Differential KV-Cache
                   Compression for Scalable Long-Context Inference},
  month        = jul,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.21539110},
  url          = {https://doi.org/10.5281/zenodo.21539110},
}
```

Persistent DOI: https://doi.org/10.5281/zenodo.21539110

For build instructions and native compilation details, see **[BUILD.md](BUILD.md)**.

---

## ⚖️ License

Differential-KV is released under the **MIT License** — see [LICENSE](LICENSE).

Vendored third-party code keeps its own license and is **not** covered by the root LICENSE:

| Component | Path | License |
|---|---|---|
| `llama.cpp` / `ggml` | `dkv_native/third_party/llama.cpp/` | MIT — [LICENSE](dkv_native/third_party/llama.cpp/LICENSE) (© 2023–2026 The ggml authors) |
| `nlohmann/json` | `dkv_native/third_party/llama.cpp/licenses/` | MIT — [LICENSE-jsonhpp](dkv_native/third_party/llama.cpp/licenses/LICENSE-jsonhpp) |

Both are MIT and therefore compatible with this project's license; their copyright
notices are retained in place, as MIT requires.
