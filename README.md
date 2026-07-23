# Differential-KV (DKV): Anchor + Low-Rank KV-Cache Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey.svg)](BUILD.md)
[![Backend](https://img.shields.io/badge/Backend-MLX%20%7C%20PyTorch%20%7C%20CUDA%20%7C%20C%2B%2B17-blueviolet.svg)](ACTIVE_RUNTIME/README.md)

**Differential-KV (DKV)** is a sparse KV-cache inference runtime designed for high-efficiency, memory-bounded long-context Large Language Model (LLM) inference across Apple Silicon (MLX) and CUDA GPUs.

Author: **Om Chimurkar** (Newton School of Technology, Rishihood University)  
Technical Report: [paper/main.pdf](paper/main.pdf)

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
| `--preset` | `choice` | `mid` | Hardware optimization preset: `low`, `mid`, `high` |
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
| `DKV_MAX_RESIDUAL` | `128` | Both engines | Number of exact residual token rows stored per block ($R = 128$). |
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
| **Status** | Reference accuracy (4k–64k NIAH 100% exact recall) | Verified NIAH sweep (6/6 Q8_0 & Q4_K_M) |

---

## 📊 Measured Paper Benchmarks

All benchmark results are empirically measured on a single host: **Apple M3 with 8.6 GB unified memory**, evaluating **Qwen2.5-1.5B-Instruct (int4)** using rank $r=32$, residual budget $R=128$, micro-block size $B_s=256$, top-$K=16$, and residual-key router.

### 1. Context Length Sweep (DKV vs. Dense Baselines)

| Context Length | Runtime / Engine | Prefill Time (s) | Decode Speed (tok/s) | Peak Allocator Memory (GB) | Needle Recalled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4k** | **DKV (Compressed)** | 6.6s | 19.9 | 1.74 GB | **Yes** |
| | Dense Baseline | 5.1s | 65.7 | 1.68 GB | **Yes** |
| **8k** | **DKV (Compressed)** | 13.6s | 18.4 | 1.89 GB | **Yes** |
| | Dense Baseline | 11.8s | 55.3 | 1.79 GB | **Yes** |
| **16k** | **DKV (Compressed)** | 28.2s | 18.7 | 2.36 GB | **Yes** |
| | Dense Baseline | 27.8s | 47.0 | 2.03 GB | **Yes** |
| **32k** | **DKV (Compressed)** | 58.5s | 17.0 | 3.12 GB | **Yes** |
| | Dense Baseline | 77.9s | 35.7 | 2.45 GB | **Yes** |
| **64k** | **DKV (Compressed)** | 928s | 8.6 | 4.63 GB | **Yes** |
| | Dense Baseline | *OOM* | *OOM* | *OOM* | *OOM* |

> 💡 **Key Benchmark Takeaways:**
> - **Prefill Crossover ($\ge 32\text{k}$):** Thanks to block-sparse prefill scaling $O(L \cdot K)$, Differential-KV prefilling beats the dense baseline at 32k context ($1.33\times$ faster prefill: 58.5s vs 77.9s).
> - **64k Reach Advantage:** The dense full-KV baseline suffers an Out-Of-Memory (OOM) failure at 64k context on the 8.6 GB memory host. Differential-KV completes 64k inference with 100% exact needle recovery within bounded memory.

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

If you use Differential-KV in your research or project, please cite the technical report:

```bibtex
@article{chimurkar2026differentialkv,
  title={Differential-KV: Anchor + Low-Rank Differential KV-Cache Compression for Scalable Long-Context Inference},
  author={Chimurkar, Om},
  journal={Technical Report, Newton School of Technology, Rishihood University},
  year={2026},
  url={https://github.com/Omc12/Differential-KV}
}
```

For build instructions and native compilation details, see **[BUILD.md](BUILD.md)**.
