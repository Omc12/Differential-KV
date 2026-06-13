# Differential KV — Architecture Report

**Date:** 2026-06-13  
**Codebase phase:** Phase 28+ (ACTIVE_RUNTIME) / Production (diffkv_native)

---

## Overview

DiffKV is a sparse KV-cache inference runtime for transformer models. Rather than retaining the full dense KV history per token, it compresses each micro-block of tokens into:

- **Anchor token** — one full-precision (fp16) K/V pair stored exactly.
- **Low-rank delta** — the remaining tokens represented as `U @ V.T` (U int8-quantized, V fp16), computed via truncated SVD.

This yields ~5–10× VRAM compression at the cost of approximate attention for non-anchor tokens. A Semantic Routing Layer (SRL) selects which compressed blocks to attend to on each decode step, so the attention cost stays sub-linear in context length.

The repository contains two parallel implementations with shared concepts:

| Directory | Language | Status | Purpose |
|-----------|----------|--------|---------|
| `ACTIVE_RUNTIME/` | Python + PyTorch + Triton | Research prototype | Rapid iteration, HuggingFace model support |
| `diffkv_native/` | C++17 + llama.cpp (ggml) | Production binary | Metal/CPU serving, GGUF models |

---

## Main Subsystems

### 1. Block Management

**Core abstraction: `KVBlock`** ([ACTIVE_RUNTIME/native_core/kv_runtime_manager.py:104](ACTIVE_RUNTIME/native_core/kv_runtime_manager.py), [diffkv_native/native_core/streaming_sparse_ingest.hpp](diffkv_native/native_core/streaming_sparse_ingest.hpp))

Each block stores a slice of the session's KV history. Block lifecycle:

```
ACCUMULATING → SUBMITTED → CPU_COMPRESSED → COMPRESSED (GPU pool)
                                            ↘ PAGED (CPU RAM)
```

| State | VRAM held | CPU held |
|-------|-----------|----------|
| ACCUMULATING | dense K/V (active window) | — |
| SUBMITTED | dense K/V (being compressed) | — |
| CPU_COMPRESSED | anchor only | U, V (CPU) |
| COMPRESSED | U, V, anchor in pool slot | — |
| PAGED | — | U, V, anchor (CPU) |

**NativeBlockPool** ([diffkv_native/runtime/native_block_pool.hpp](diffkv_native/runtime/native_block_pool.hpp), Python: [ACTIVE_RUNTIME/runtime/native_block_pool.py](ACTIVE_RUNTIME/runtime/native_block_pool.py))  
Contiguous GPU tensor pool holding all compressed block data in flat arrays:
- `U` — [max_blocks, S_max, rank] int8
- `U_scale` — [max_blocks] fp16  
- `VK`, `VV` — [max_blocks, rank, kv_heads × head_dim] fp16  
- `anchors_K`, `anchors_V` — [max_blocks, kv_heads × head_dim] fp16  
- `seq_lens`, `scales`, `desc_matrix` — per-slot metadata

**StreamingSparseIngestManager** ([ACTIVE_RUNTIME/native_core/streaming_sparse_ingest.py](ACTIVE_RUNTIME/native_core/streaming_sparse_ingest.py), [diffkv_native/native_core/streaming_sparse_ingest.hpp](diffkv_native/native_core/streaming_sparse_ingest.hpp))  
Accumulates raw K/V from prefill, fires SVD jobs once a micro-block fills.

**PagedKVStore** ([diffkv_native/native_core/paging/paged_kv_store.hpp](diffkv_native/native_core/paging/paged_kv_store.hpp), [ACTIVE_RUNTIME/native_core/paging/paged_kv_store.py](ACTIVE_RUNTIME/native_core/paging/paged_kv_store.py))  
LRU GPU→CPU eviction when GPU budget is exceeded. Tracks access times per block for eviction decisions.

---

### 2. Compression Pipeline

**Entry point:** `StreamingSparseIngestManager.ingest_chunk()` → `AsyncCompressor.submit()`

**`AsyncCompressor`** ([diffkv_native/native_core/compression/async_compressor.hpp](diffkv_native/native_core/compression/async_compressor.hpp))  
Thread-pool backed job queue (max 16 384 pending jobs). Background workers call `compress_sync()`. On MPS/Metal, async is disabled and compression runs on the main thread for thread-safety.

**`compress_lowrank` / `process_job`** ([ACTIVE_RUNTIME/native_core/compression/lowrank.py](ACTIVE_RUNTIME/native_core/compression/lowrank.py))  
Truncated SVD per micro-block:
1. Pick anchor = first token (highest norm, acts as attention sink).
2. Compute delta: `K_delta[t] = K[t] - anchor_K`, `V_delta[t] = V[t] - anchor_V`.
3. Stack into `[2*kv_heads*head_dim, block_size]` matrix.
4. Truncated SVD → `U [block_size, rank]`, `S`, `Vt [rank, feat_dim]`.
5. Absorb `S` into `V = Vt * S[:, None]`; quantize `U` to int8 with per-block scale.
6. Optional factual split: semantic components `U_sem` (int4) vs. factual components `U_fact` (fp16).

**Per-layer adaptive rank schedule** ([ACTIVE_RUNTIME/native_core/kv_runtime_manager.py:44](ACTIVE_RUNTIME/native_core/kv_runtime_manager.py))  
Early layers (0–15%): full base_rank. Mid layers (15–79%): 0.75× base_rank (min 6). Final layers (79–100%): 0.5× base_rank (min 8). Rationale: early layers have broader distributions needing more rank; later layers are concentrated.

---

### 3. Semantic Routing Layer (SRL)

The SRL runs at decode time to select the K most relevant compressed blocks for each token. It replaces full attention over all blocks with a pruned, scored subset.

**`SemanticIndex`** ([diffkv_native/native_core/srl/semantic_index.hpp](diffkv_native/native_core/srl/semantic_index.hpp), [ACTIVE_RUNTIME/native_core/srl/attention_cache.py](ACTIVE_RUNTIME/native_core/srl/attention_cache.py))  
ANN search over 64-dim descriptor vectors. Each block's descriptor = `W_proj @ mean_Q` computed at ingest time. W_proj is a fixed random projection (Xavier normal, row-normalized).

**`ChunkGraph`** ([diffkv_native/native_core/srl/chunk_graph.hpp](diffkv_native/native_core/srl/chunk_graph.hpp))  
Block-to-block similarity graph built after prefill. Used for 2-hop graph propagation during routing. Stores:
- `neighbors` flat array `[N × max_degree]`
- `parent_landmarks` + `parent_to_children_tensor` (hierarchical routing)
- `cluster_centers_tensor` + `role_mapping_tensor` (concentric routing)

**`InvertedTokenIndex`** ([diffkv_native/native_core/srl/inverted_index.hpp](diffkv_native/native_core/srl/inverted_index.hpp))  
Token_id → slot_ids mapping. IDF-weighted lexical scoring. Rare term always-index pass ensures high-IDF (≥3.0) terms are never missed by the frequency cap. Decays by positional distance.

**`FactualExactStore`** ([diffkv_native/native_core/srl/factual_store.hpp](diffkv_native/native_core/srl/factual_store.hpp), [ACTIVE_RUNTIME/native_core/srl/factual_store.py](ACTIVE_RUNTIME/native_core/srl/factual_store.py))  
Stores dense K/V for rare/factual token spans. Built using an "Eagle lookback score" (causal key self-similarity) to identify salient tokens. Allows exact attention over factual sequences even though those tokens' blocks are compressed.

**`SessionSRLState`** ([diffkv_native/native_core/srl/session_srl_state.hpp](diffkv_native/native_core/srl/session_srl_state.hpp))  
Per-session container holding:
- SemanticIndex, ChunkGraph, InvertedTokenIndex, FactualExactStore
- Ordered slot IDs (chronological), sink blocks (always-included)
- Adaptive-K EMA miss-rate signal and k_multiplier
- Per-step slot cache, token windows, dynamic anchors

**`FactualAlignment` (VSL)** ([diffkv_native/native_core/srl/factual_alignment.hpp](diffkv_native/native_core/srl/factual_alignment.hpp))  
Verifiable Sequence Locking. During decode, after factual sequences are retrieved, constrains next-token logits to allowed tokens (factual sequence continuations + helper words). Resets after 4 consecutive "helper" tokens to avoid locking on function words.

**`QueryRouter`** ([diffkv_native/native_core/srl/query_router.hpp](diffkv_native/native_core/srl/query_router.hpp))  
10-step pipeline per decode step:
1. Compute `q_desc` via `W_proj @ mean_Q`
2. `adaptive_k`: entropy-based K budget (k_min=20 to k_max=200)
3. Semantic: concentric routing (cluster centers → around → outer) or hierarchical (landmarks → children) or flat ANN
4. Topic-switch detection: `best_sem_score < 0.25`
5. Lexical: IDF-weighted slot scoring
6. Rare lexical: high-IDF (≥2.0) term exact matches
7. Graph expansion: 2-hop propagation from semantic+lexical seeds
8. Recency: last k_recency slots
9. Merge all channels (sink > semantic > rare_lex > graph > lexical > recency)
10. Two-level gate: anchor dot-product reranking with age penalty

---

### 4. Attention Execution

**Prefill** ([diffkv_native/src/main.cpp:230](diffkv_native/src/main.cpp))  
Chunked processing (default 512 tokens/chunk). Each chunk uses `ggml_flash_attn_ext` (Flash Attention) attending to all prior chunks' K/V concatenated (exact causal attention). Raw K (pre-RoPE) and raw V are exported per-layer for ingestion.

**Decode — dense fast-path** (< `engage_threshold`, default 2048 tokens)  
`ggml_flash_attn_ext` with full dense K/V past tensors. Bypasses SRL entirely.

**Decode — sparse path** (≥ 2048 tokens, [diffkv_native/src/main.cpp:384](diffkv_native/src/main.cpp))  
For each decode step:
1. Layer 0: compute `q_desc`, semantic top-K + host-computed slots → `selected_slots [K]`
2. All layers: RoPE on Q, then `ggml_map_custom3` callback dispatching to Metal/CPU kernel  
3. Kernel: for each slot, compute anchor score + low-rank delta scores (q_proj via VK rows, then weighted by U coefficients), combined softmax, weighted sum of anchor V + U@VV contributions
4. Residual + FFN continue normally

**Metal shader** ([diffkv_native/native_core/diffkv_core/metal/diffkv_decode.metal](diffkv_native/native_core/diffkv_core/metal/diffkv_decode.metal))  
Custom Metal kernel for Apple Silicon, compiled to `.metallib` and embedded.

**Python Triton fallback** ([ACTIVE_RUNTIME/native_core/sparse_decode/triton_fused_decode.py](ACTIVE_RUNTIME/native_core/sparse_decode/triton_fused_decode.py))  
Research implementation using Triton `@triton.jit` kernels (fused Project-Then-Attend).

---

### 5. Serving Layer

**ACTIVE_RUNTIME (Python)**

| File | Role |
|------|------|
| [ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py](ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py) | FastAPI server, OpenAI `/v1/chat/completions` + `/v1/models` |
| [ACTIVE_RUNTIME/serving/batch_engine.py](ACTIVE_RUNTIME/serving/batch_engine.py) | Continuous batching decode loop |
| [ACTIVE_RUNTIME/serving/hf_diffkv_wrapper.py](ACTIVE_RUNTIME/serving/hf_diffkv_wrapper.py) | HuggingFace model + KVRuntimeManager monkey-patch wiring |
| [ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py) | Apple MLX backend wrapper |
| [ACTIVE_RUNTIME/runtime/diffkv_attention.py](ACTIVE_RUNTIME/runtime/diffkv_attention.py) | HF attention monkey-patch (prefill + decode routing) |

**diffkv_native (C++)**

| File | Role |
|------|------|
| [diffkv_native/src/main.cpp](diffkv_native/src/main.cpp) | Main inference binary (prefill + decode loop, interactive stdin/stdout protocol) |
| [diffkv_native/serving/batch_engine.hpp/.cpp](diffkv_native/serving/batch_engine.hpp) | `DiffKVBatchEngine` — single-worker thread processing `BatchRequest` queue |
| [diffkv_native/serving/production_session_manager.hpp/.cpp](diffkv_native/serving/production_session_manager.hpp) | Multi-session lifecycle management |
| [diffkv_native/serving/openai_compatible_api_gateway.hpp/.cpp](diffkv_native/serving/openai_compatible_api_gateway.hpp) | C++ OpenAI-compatible HTTP gateway |
| [diffkv_native/serving/cli.py](diffkv_native/serving/cli.py) | Python CLI wrapper — spawns C++ binary as subprocess, communicates via `__READY__`/`__RESPONSE__`/`__FINISH__` sentinels on stdin/stdout |

---

## Memory Flow

```
User Prompt
    │
    ▼
Tokenize (llama_tokenize / HF tokenizer)
    │
    ▼
PREFILL PHASE (chunked, 256–2048 tokens/chunk)
    │
    ├── ggml_flash_attn_ext (full causal attention)
    │       └── Export raw K (pre-RoPE), raw V per layer
    │
    ▼
StreamingSparseIngestManager.ingest_chunk()
    │
    ├── Accumulate K/V into KVBlock (ACCUMULATING)
    │
    └── When block full (micro_block_size tokens):
            │
            ├── AsyncCompressor.submit() → background SVD
            │       ├── anchor = first token
            │       ├── delta = K/V - anchor K/V
            │       ├── SVD → U (int8), V (fp16)
            │       └── Write → NativeBlockPool slot (COMPRESSED)
            │
            └── If MPS: compress_sync() on main thread
                │
                └── finalize_compressed_blocks() GPU upload
    │
    ▼
finalize_srl_index()  [after all prefill blocks compressed]
    │
    ├── SemanticIndex (ANN over 64-dim descriptors)
    ├── ChunkGraph (block similarity + hierarchy)
    ├── InvertedTokenIndex (lexical)
    └── FactualExactStore (salient token spans)
    │
    ▼
DECODE LOOP (autoregressive, one token per step)
    │
    ├── [< 2048 tokens]: dense ggml_flash_attn_ext (full K/V)
    │
    └── [≥ 2048 tokens]: SRL sparse path
            │
            ├── QueryRouter.route_query()
            │       ├── q_desc = W_proj @ mean_Q
            │       ├── adaptive_k (entropy → K budget)
            │       ├── Semantic → Lexical → Graph → Recency
            │       └── Two-level gate reranking → selected_slots [K]
            │
            ├── For each layer: Metal/CPU custom kernel
            │       ├── For each slot in selected_slots:
            │       │       ├── anchor score: q · anchor_K
            │       │       ├── q_proj = q · VK rows (shape [rank])
            │       │       └── token scores = q_proj · U_t (per token)
            │       └── Softmax over all (anchor + token) scores → weighted V
            │
            ├── Residual + SwiGLU FFN (standard)
            │
            ├── Sample next token (temp/top-p)
            │
            └── Update SRL state:
                    ├── Register new token in inverted index
                    ├── Update recent_generated_tokens window
                    └── Update dynamic anchors (every 4 generated tokens)
    │
    ▼
Stream output tokens to caller
    │
    ▼
Session cleanup: clear_session() → free NativeBlockPool slots → GC
```

---

## Key Entry Points

| Entry Point | File | Purpose |
|-------------|------|---------|
| C++ inference binary | [diffkv_native/src/main.cpp:845](diffkv_native/src/main.cpp) | `main()` — model load, prefill, decode loop, interactive stdin |
| Python research server | [ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py](ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py) | FastAPI server, OpenAI-compatible |
| CLI (direct mode) | [diffkv_native/serving/cli.py:596](diffkv_native/serving/cli.py) | Spawns C++ binary subprocess, interactive terminal |
| CLI (client mode) | [diffkv_native/serving/cli.py:460](diffkv_native/serving/cli.py) | HTTP client to running gateway |
| KV session manager (Python) | [ACTIVE_RUNTIME/native_core/kv_runtime_manager.py:386](ACTIVE_RUNTIME/native_core/kv_runtime_manager.py) | `KVRuntimeManager.__init__` |
| KV session manager (C++) | [diffkv_native/native_core/kv_runtime_manager.hpp:16](diffkv_native/native_core/kv_runtime_manager.hpp) | `KVRuntimeManager` class |
| SRL index build | [ACTIVE_RUNTIME/native_core/kv_runtime_manager.py:737](ACTIVE_RUNTIME/native_core/kv_runtime_manager.py) | `finalize_srl_index()` |
| SRL query routing | [diffkv_native/native_core/srl/query_router.hpp:219](diffkv_native/native_core/srl/query_router.hpp) | `route_query()` |
| Decode graph (C++) | [diffkv_native/src/main.cpp:384](diffkv_native/src/main.cpp) | `build_decode_graph()` |
| Prefill graph (C++) | [diffkv_native/src/main.cpp:105](diffkv_native/src/main.cpp) | `build_prefill_graph()`, `build_prefill_ctx_graph()` |

---

## Important Files

### ACTIVE_RUNTIME (Python / Research)

| File | Importance |
|------|-----------|
| [ACTIVE_RUNTIME/native_core/kv_runtime_manager.py](ACTIVE_RUNTIME/native_core/kv_runtime_manager.py) | Master orchestrator: block lifecycle, SRL index build, session management |
| [ACTIVE_RUNTIME/native_core/streaming_sparse_ingest.py](ACTIVE_RUNTIME/native_core/streaming_sparse_ingest.py) | Streaming ingest pipeline — replaces old O(N²) batch approach |
| [ACTIVE_RUNTIME/native_core/compression/lowrank.py](ACTIVE_RUNTIME/native_core/compression/lowrank.py) | Core SVD compression algorithm |
| [ACTIVE_RUNTIME/native_core/srl/query_router.py](ACTIVE_RUNTIME/native_core/srl/query_router.py) | SRL 10-step routing pipeline (Python reference) |
| [ACTIVE_RUNTIME/native_core/srl/session_srl_state.py](ACTIVE_RUNTIME/native_core/srl/session_srl_state.py) | SessionSRLState — aggregation of all routing indices |
| [ACTIVE_RUNTIME/native_core/srl/factual_store.py](ACTIVE_RUNTIME/native_core/srl/factual_store.py) | FactualExactStore — salience-based exact-attention spans |
| [ACTIVE_RUNTIME/runtime/diffkv_attention.py](ACTIVE_RUNTIME/runtime/diffkv_attention.py) | HF attention monkey-patch (prefill/decode dispatch) |
| [ACTIVE_RUNTIME/serving/batch_engine.py](ACTIVE_RUNTIME/serving/batch_engine.py) | Continuous-batching decode loop |

### diffkv_native (C++ / Production)

| File | Importance |
|------|-----------|
| [diffkv_native/src/main.cpp](diffkv_native/src/main.cpp) | 2650-line main: prefill graph, decode graph, interactive loop |
| [diffkv_native/native_core/kv_runtime_manager.hpp](diffkv_native/native_core/kv_runtime_manager.hpp) | C++ KVRuntimeManager interface |
| [diffkv_native/native_core/kv_runtime_manager.cpp](diffkv_native/native_core/kv_runtime_manager.cpp) | KVRuntimeManager implementation |
| [diffkv_native/native_core/srl/query_router.hpp](diffkv_native/native_core/srl/query_router.hpp) | Full C++ SRL routing (header-only, ~725 lines) |
| [diffkv_native/native_core/srl/session_srl_state.hpp](diffkv_native/native_core/srl/session_srl_state.hpp) | SessionSRLState C++ struct (adaptive-K, EQA-DR, segment IDs) |
| [diffkv_native/native_core/srl/factual_store.hpp](diffkv_native/native_core/srl/factual_store.hpp) | FactualExactStore C++ struct |
| [diffkv_native/native_core/srl/factual_alignment.hpp](diffkv_native/native_core/srl/factual_alignment.hpp) | VSL verifiable sequence locking |
| [diffkv_native/native_core/srl/inverted_index.hpp](diffkv_native/native_core/srl/inverted_index.hpp) | InvertedTokenIndex (build + search, header-only ~340 lines) |
| [diffkv_native/runtime/diffkv_attention.hpp/.cpp](diffkv_native/runtime/diffkv_attention.hpp) | Custom attention op callback wired into ggml |
| [diffkv_native/native_core/compression/async_compressor.hpp](diffkv_native/native_core/compression/async_compressor.hpp) | Background SVD thread pool |
| [diffkv_native/native_core/diffkv_core/metal/diffkv_decode.metal](diffkv_native/native_core/diffkv_core/metal/diffkv_decode.metal) | Apple Metal shader for sparse decode |
| [diffkv_native/serving/cli.py](diffkv_native/serving/cli.py) | Python CLI: subprocess protocol + client mode |
| [diffkv_native/CMakeLists.txt](diffkv_native/CMakeLists.txt) | Build system |

---

## Potential Areas of Technical Debt

### 1. Full KV Reset on Every Multi-Turn Conversation (Critical)
**Location:** [diffkv_native/src/main.cpp:1097–1107](diffkv_native/src/main.cpp)

Every turn re-prefills the entire conversation from token 0. The comment explicitly marks decompression-based prefix injection as "future work." This means O(N) prefill cost per turn even when the user adds one sentence. Performance degrades quadratically with conversation length.

```cpp
// Each turn: reset compressed pool, re-prefill full prompt from 0, decode.
// Speed improvement from prefix-skipping requires implementing decompression-
// based prior context injection (future work).
```

---

### 2. Domain-Specific Token Hardcoding in Production SRL
**Location:** [diffkv_native/native_core/srl/session_srl_state.hpp:290–296](diffkv_native/native_core/srl/session_srl_state.hpp)

Specific tokens like `"EP2"`, `"hermitian"`, `"diabolic"`, `"conical"`, `"branch"` receive a +5.0 score boost in `setup_sas_and_eqa()`. These appear to be physics/math terms from a specific benchmark and are now baked into the production SRL state struct. This will cause incorrect salience scoring on unrelated domains.

```cpp
if (word == "1" || word == "2" || word == "3" || word == "ep2" || word == "ep3" ||
    word == "hermitian" || word == "diabolic" || word == "conical" || word == "branch") {
    score += 5.0f;
}
```

---

### 3. Stop-Word List Duplicated in Three Places
**Locations:**
- [ACTIVE_RUNTIME/native_core/kv_runtime_manager.py:427–436](ACTIVE_RUNTIME/native_core/kv_runtime_manager.py)
- [diffkv_native/src/main.cpp:910–928](diffkv_native/src/main.cpp)
- [diffkv_native/native_core/srl/factual_alignment.hpp:31–56](diffkv_native/native_core/srl/factual_alignment.hpp) (as ALLOWED_HELPER_WORDS)

Three separate hardcoded lists with no shared source. They are not identical — the helper-words list in factual_alignment.hpp is significantly larger (60+ words). Divergence between them will cause subtle routing differences between Python and C++ paths.

---

### 4. Untrained Semantic Projection Matrix
**Location:** [ACTIVE_RUNTIME/native_core/kv_runtime_manager.py:576–579](ACTIVE_RUNTIME/native_core/kv_runtime_manager.py), [diffkv_native/src/main.cpp:977–989](diffkv_native/src/main.cpp)

`W_proj` is fixed random (fixed seed 42 in C++). While the Johnson-Lindenstrauss lemma guarantees distance preservation, a random projection is not tuned to the model's semantic structure. A learned or PCA-derived projection from model activations could improve SRL retrieval quality without additional inference cost.

---

### 5. Absolute Paths Hardcoded in CLI
**Location:** [diffkv_native/serving/cli.py:601–609](diffkv_native/serving/cli.py)

```python
model_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"
```

Machine-specific absolute paths are baked into production CLI defaults. Will silently fail on any other machine. GGUF model files are also committed to the repository (`qwen2.5-0.5b-instruct.gguf`, `qwen2.5-1.5b-instruct-q8_0.gguf`).

---

### 6. Mixed Python/C++ in Production Directory
**Location:** `diffkv_native/serving/`, `diffkv_native/native_core/*/` Python `__pycache__/`

The `diffkv_native/serving/` directory contains both C++ production code and Python research files (`hf_diffkv_wrapper.py`, `mlx_diffkv_wrapper.py`, `openai_compatible_api_gateway.py`). `__pycache__` directories exist inside the C++ module tree. The architectural boundary between research and production is blurred.

---

### 7. No Test Suite for C++ Native Code
**Observation:** `diffkv_native/` has no `tests/` directory.

The only verification is `verify_attention_cpu()` in [diffkv_native/src/main.cpp:606](diffkv_native/src/main.cpp) — a one-off CPU reference check run during development. `ACTIVE_RUNTIME/tests/` has proper pytest coverage. The C++ production binary has none. Regressions in the Metal kernel or SVD compression will be invisible without manual testing.

---

### 8. AsyncCompressor Job Silently Dropped on Overflow
**Location:** [diffkv_native/native_core/compression/async_compressor.hpp:71](diffkv_native/native_core/compression/async_compressor.hpp)

`MAX_QUEUE_SIZE = 16384`. When exceeded, `submit()` increments `queue_overflows_` and returns false — but callers in the ingest pipeline do not check the return value. Dropped compression jobs mean tokens silently remain uncompressed or their blocks are never finalized, which could corrupt session state without visible errors.

---

### 9. SRL Routing Thresholds Are Magic Numbers Without Documentation
**Location:** [diffkv_native/native_core/srl/session_srl_state.hpp:107–113](diffkv_native/native_core/srl/session_srl_state.hpp)

```cpp
if (recent_miss_rate > 0.4f)
    k_multiplier = std::min(k_multiplier * 1.2f, 3.0f);
else
    k_multiplier = std::max(k_multiplier * 0.99f, 1.0f);
```

The values 0.4, 1.2, 3.0, 0.99, alpha=0.05 are empirically chosen without ablation documentation. The EMA miss-rate signal is plausible but its relationship to actual recall quality is unvalidated in the codebase.

---

### 10. KV Capture Accumulation Still O(N) for Python Path
**Location:** [ACTIVE_RUNTIME/native_core/kv_runtime_manager.py:963–966](ACTIVE_RUNTIME/native_core/kv_runtime_manager.py)

Despite the `_prefill_kv_capture` dict being used for FactualExactStore, it still accumulates K/V across all prefill chunks via `torch.cat`. For a 32K-token prefill with 28 layers, this creates 28 growing tensors concatenated 64 times each — an O(N²) memory spike that the README describes as eliminated but is still present in the factual capture path.

---

### 11. Archive Directory Contains Active Code References
**Location:** `ACTIVE_RUNTIME/archive/`

The archive contains old test files (`test_batching.py`, `test_long_context.py`, etc.) and compiled distribution snapshots under `dist/`. These reference old API shapes (`triton_diffkv.py`, `production_session_manager.py`). If a developer searches the repo for an API, they will find outdated implementations here that no longer match the live code.

---

### 12. Block Pool Size Calculation Complexity
**Location:** [ACTIVE_RUNTIME/native_core/kv_runtime_manager.py:505–557](ACTIVE_RUNTIME/native_core/kv_runtime_manager.py)

Pool size is computed through a chain of serving-mode heuristics, MPS detection, preset detection, and multiple min/max clamps. The code acknowledges (with a comment) that the old formula over-allocated by 8×. The current formula is better but still relies on `avg_block_sz = max(32, min(micro_block_size, 64))` as a stand-in for the actual distribution. No telemetry exists to validate whether the pool size is appropriate at runtime without enabling `DIFFKV_TELEMETRY=1`.

---

## Environment Variables Reference

The system is configured entirely through environment variables — there is no config file:

| Variable | Default | Effect |
|----------|---------|--------|
| `DIFFKV_USE_GPU` | `0` | Enable Metal/GPU backend in C++ |
| `DIFFKV_GPU_BUDGET_GB` | `2.0` | GPU memory budget for block pool |
| `DIFFKV_MICRO_BLOCK_SIZE` | `64` | Tokens per compressed block |
| `DIFFKV_PREFILL_CHUNK_SIZE` | `512` | Tokens per prefill forward pass |
| `DIFFKV_PRESET` | `mid` | `low`/`mid`/`high` (affects chunk size) |
| `DIFFKV_MAX_CONTEXT_SLOTS` | auto | Override number of pool slots |
| `DIFFKV_SRL_K_SEM` | `32` | Semantic channel K |
| `DIFFKV_SRL_K_LEX` | `8` | Lexical channel K |
| `DIFFKV_SRL_K_GRAPH` | `8` | Graph channel K |
| `DIFFKV_SRL_K_RECENCY` | `8` | Recency channel K |
| `DIFFKV_SRL_K_KEEP` | `16` | Final slot count after gate |
| `DIFFKV_SRL_K_MIN` | `20` | Adaptive-K lower bound |
| `DIFFKV_SRL_K_MAX` | `200` | Adaptive-K upper bound |
| `DIFFKV_TEMPERATURE` | `0.7` | Sampling temperature |
| `DIFFKV_TOP_P` | `0.9` | Top-p nucleus sampling |
| `DIFFKV_REPETITION_PENALTY` | `1.15` | Repetition penalty coefficient |
| `DIFFKV_VERBOSE` | unset | Enable verbose stderr logging |
| `DIFFKV_TELEMETRY` | `0` | Enable block-state telemetry |
| `DIFFKV_MPS_APPROXIMATE_ATTN` | `1` on Apple | Enable MPS approximate attention |
| `DIFFKV_EARLY_LAYER_RANK_BOOST` | `0` | Boost rank for early transformer layers |
| `DIFFKV_SEED` | random | Fix sampling seed for reproducibility |
